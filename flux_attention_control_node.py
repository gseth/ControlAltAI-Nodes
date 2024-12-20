import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from einops import rearrange
import comfy.model_management as model_management
from comfy.ldm.modules import attention as comfy_attention
from comfy.ldm.flux import math as flux_math
from comfy.ldm.flux import layers as flux_layers
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from functools import partial

# Protected xformers import
try:
    from xformers.ops import memory_efficient_attention as xattention
    has_xformers = True
except ImportError:
    has_xformers = False
    xattention = None

class FluxAttentionControl:
    def __init__(self):
        self.original_attention = comfy_attention.optimized_attention
        self.original_flux_attention = flux_math.attention
        self.original_flux_layers_attention = flux_layers.attention
        if not has_xformers:
            print("\n" + "="*70)
            print("\033[94mControlAltAI-Nodes: This node requires xformers to function.\033[0m")
            print("\033[33mPlease check \"xformers_instructions.txt\" in ComfyUI\\custom_nodes\\ControlAltAI-Nodes for how to install XFormers\033[0m")
            print("="*70 + "\n")
        print("FluxAttentionControl initialized")         

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "condition": ("CONDITIONING",),
                "latent_dimensions": ("LATENT",),
                "region1": ("REGION",),
                "number_of_regions": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "display": "Number of Regions"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "display": "Enable Regional Control"
                }),
                "feather_radius1": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "Feather Radius for Region 1"
                }),
            },
            "optional": {
                "region2": ("REGION",),
                "feather_radius2": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "Feather Radius for Region 2"
                }),
                "region3": ("REGION",),
                "feather_radius3": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "Feather Radius for Region 3"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING",)
    RETURN_NAMES = ("model", "conditioning",)
    FUNCTION = "apply_attention_control"
    CATEGORY = "ControlAltAI Nodes/Flux Region"

    def generate_region_mask(self, region: Dict, width: int, height: int, feather_radius: float) -> Image.Image:
        if region.get('bbox') is not None:
            x1, y1, x2, y2 = region['bbox']
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            mask = Image.new('L', (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle([x1_px, y1_px, x2_px, y2_px], fill=255)
            if feather_radius > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
            print(f'Generating masks with {width}x{height} and [{x1}, {y1}, {x2}, {y2}], feather_radius={feather_radius}')
            return mask
        elif region.get('mask') is not None:
            mask = region['mask'][0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask)
            mask = mask.resize((width, height))
            if feather_radius > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
            return mask
        else:
            raise Exception('Unknown region type')

    def generate_test_mask(self, masks: List[Image.Image], height: int, width: int):
        hH, hW = int(height) // 16, int(width) // 16
        print(f'{width} {height} -> {hW} {hH}')
        
        lin_masks = []
        for mask in masks:
            mask = mask.convert('L')
            mask = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255.0  # Normalize to 0-1
            mask = F.interpolate(mask, (hH, hW), mode='bilinear', align_corners=False).flatten()
            lin_masks.append(mask)
        return lin_masks, hH, hW

    def prepare_attention_mask(self, lin_masks: List[torch.Tensor], region_strengths: List[float], Nx: int, emb_size: int, emb_len: int):
        """Prepare attention mask for three regions with per-region strengths."""
        total_len = emb_len + Nx
        n_regs = len(lin_masks)
        
        # Initialize attention mask and scales
        cross_mask = torch.zeros(total_len, total_len)
        q_scale = torch.ones(total_len)
        k_scale = torch.ones(total_len)
        
        # Indices for embeddings
        main_prompt_start = 0
        main_prompt_end = emb_size

        # Subprompt indices
        subprompt_starts = [emb_size * (i + 1) for i in range(n_regs)]
        subprompt_ends = [emb_size * (i + 2) for i in range(n_regs)]
        
        # Initialize position masks
        position_masks = torch.stack(lin_masks)  # Shape: [n_regs, Nx]

        # Normalize masks so that overlapping areas sum to 1
        position_masks_sum = position_masks.sum(dim=0)
        position_masks_normalized = position_masks / (position_masks_sum + 1e-8)

        # Build attention masks and scales
        for i in range(n_regs):
            sp_start = subprompt_starts[i]
            sp_end = subprompt_ends[i]
            mask_i = position_masks_normalized[i]

            # Scale embeddings based on mask and per-region strength
            strength = region_strengths[i]
            q_scale[sp_start:sp_end] = mask_i.mean() * strength
            k_scale[sp_start:sp_end] = mask_i.mean() * strength

            # Create mask including tokens and positions
            m_with_tokens = torch.cat([torch.ones(emb_len), mask_i])
            mb = m_with_tokens > 0.0  # Include positions where mask > 0

            # Block attention between positions not in mask and subprompt
            cross_mask[~mb, sp_start:sp_end] = 1
            cross_mask[sp_start:sp_end, ~mb] = 1

            # Block attention between positions in region and main prompt
            positions_idx = (mask_i > 0.0).nonzero(as_tuple=True)[0] + emb_len
            cross_mask[positions_idx[:, None], main_prompt_start:main_prompt_end] = 1
            cross_mask[main_prompt_start:main_prompt_end, positions_idx[None, :]] = 1

            # Block attention between subprompts
            for j in range(n_regs):
                if i != j:
                    other_sp_start = subprompt_starts[j]
                    other_sp_end = subprompt_ends[j]
                    cross_mask[sp_start:sp_end, other_sp_start:other_sp_end] = 1
                    cross_mask[other_sp_start:other_sp_end, sp_start:sp_end] = 1

        # Ensure self-attention is allowed
        cross_mask.fill_diagonal_(0)
        
        # Prepare scales for GPU
        q_scale = q_scale.reshape(1, 1, -1, 1).cuda()
        k_scale = k_scale.reshape(1, 1, -1, 1).cuda()
        
        return cross_mask, q_scale, k_scale

    def xformers_attention(self, q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
                            attn_mask: Optional[Tensor] = None,
                            mask: Optional[Tensor] = None) -> Tensor:  # Added mask parameter
        q, k = flux_math.apply_rope(q, k, pe)
        q = rearrange(q, "B H L D -> B L H D")
        k = rearrange(k, "B H L D -> B L H D")
        v = rearrange(v, "B H L D -> B L H D")
        
        # Use attn_mask if provided, otherwise use the mask parameter
        attention_bias = attn_mask if attn_mask is not None else mask
        
        if attention_bias is not None:
            x = xattention(q, k, v, attn_bias=attention_bias)
        else:
            x = xattention(q, k, v)
            
        x = rearrange(x, "B L H D -> B L (H D)")
        return x

    def apply_attention_control(self,
                              model: object,
                              condition: List,
                              latent_dimensions: Dict,
                              region1: Dict,
                              number_of_regions: int,
                              enabled: bool,
                              feather_radius1: float = 0.0,
                              region2: Optional[Dict] = None,
                              feather_radius2: Optional[float] = 0.0,
                              region3: Optional[Dict] = None,
                              feather_radius3: Optional[float] = 0.0):

        # Extract dimensions and embeddings first (moved before enabled check)
        latent = latent_dimensions["samples"]
        bs_l, n_ch, lH, lW = latent.shape
        text_emb = condition[0][0].clone()
        clip_emb = condition[0][1]['pooled_output'].clone()
        bs, emb_size, emb_dim = text_emb.shape
        iH, iW = lH * 8, lW * 8

        if not enabled:
            # Restore original attention functions
            flux_math.attention = self.original_flux_attention
            flux_layers.attention = self.original_flux_layers_attention
            print("Regional control disabled. Restored original attention functions.")
            return (model, condition)  # Return original condition when disabled

        if enabled and not has_xformers:
            raise RuntimeError("Xformers is required for this node when enabled. Please install xformers.")

        print(f'Region attention Node enabled: {enabled}, regions: {number_of_regions}')

        # Extract dimensions and embeddings
        latent = latent_dimensions["samples"]
        bs_l, n_ch, lH, lW = latent.shape
        text_emb = condition[0][0].clone()
        clip_emb = condition[0][1]['pooled_output'].clone()
        bs, emb_size, emb_dim = text_emb.shape
        iH, iW = lH * 8, lW * 8

        # Process active regions
        subprompts_embeds = []
        masks = []
        region_strengths = []
        
        # Collect regions and feather radii
        regions = [region1, region2, region3]
        feather_radii = [feather_radius1, feather_radius2, feather_radius3]

        for idx, region in enumerate(regions[:number_of_regions]):
            if region is not None and region.get('conditioning') is not None:
                # Get 'strength' from region or default to 1.0
                strength = region.get('strength', 1.0)
                region_strengths.append(strength)
                subprompt_emb = region['conditioning'][0][0]
                subprompts_embeds.append(subprompt_emb)
                # Use per-region feather_radius
                feather_radius = feather_radii[idx] if feather_radii[idx] is not None else 0.0
                masks.append(self.generate_region_mask(region, iW, iH, feather_radius))
            else:
                print(f"Region {idx+1} is None or has no conditioning")

        if not subprompts_embeds:
            print("No active regions with conditioning found.")
            # Restore original attention functions
            flux_math.attention = self.original_flux_attention
            flux_layers.attention = self.original_flux_layers_attention
            return (model, condition)
        
        n_regs = len(subprompts_embeds)
        
        # Generate attention components
        lin_masks, hH, hW = self.generate_test_mask(masks, iH, iW)
        Nx = int(hH * hW)
        emb_len = emb_size * (n_regs + 1)  # +1 for main prompt
        
        # Create attention mask
        attn_mask, q_scale, k_scale = self.prepare_attention_mask(
            lin_masks, region_strengths, Nx, emb_size, emb_len)

        # Format for xFormers
        device = torch.device('cuda')
        attn_dtype = torch.bfloat16 if model_management.should_use_bf16(device=device) else torch.float16
        
        if attn_mask is not None:
            print(f'Applying attention masks: torch.Size([{attn_mask.shape[0]}, {attn_mask.shape[1]}])')
            L = attn_mask.shape[0]
            H = 24  # Number of heads in FLUX model
            pad = (8 - L % 8) % 8  # Ensure pad is between 0 and 7
            pad_L = L + pad
            mask_out = torch.zeros([bs, H, pad_L, pad_L], dtype=attn_dtype, device=device)
            mask_out[:, :, :L, :L] = attn_mask.to(device, dtype=attn_dtype)
            attn_mask = mask_out[:, :, :pad_L, :pad_L]

        # Prepare final mask
        attn_mask_bool = attn_mask > 0.5
        attn_mask.masked_fill_(attn_mask_bool, float('-inf'))
        
        # Override attention
        attn_mask_arg = attn_mask if enabled else None
        override_attention = partial(self.xformers_attention, attn_mask=attn_mask_arg)
        flux_math.attention = override_attention
        flux_layers.attention = override_attention

        # Create extended conditioning
        extended_condition = torch.cat([text_emb] + subprompts_embeds, dim=1)
        
        return (model, [[extended_condition, {'pooled_output': clip_emb}]])

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FluxAttentionControl": FluxAttentionControl
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAttentionControl": "Flux Attention Control"
}