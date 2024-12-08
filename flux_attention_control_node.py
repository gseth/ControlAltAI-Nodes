import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from einops import rearrange
import comfy.model_management as model_management
from comfy.ldm.modules import attention as comfy_attention
from comfy.ldm.flux import math as flux_math
from comfy.ldm.flux import layers as flux_layers
from xformers.ops import memory_efficient_attention as xattention
import numpy as np
from PIL import Image
from functools import partial

class FluxAttentionControl:
    has_xformers = False
    try:
        from xformers.ops import memory_efficient_attention as xattention
        has_xformers = True
    except ImportError:
        print("\n" + "="*70)
        print("\033[94mControlAltAI-Nodes: This node requires xformers to function.\033[0m")
        print("\033[33mPlease check \"xformers_instructions.txt\" in ComfyUI\\custom_nodes\\ControlAltAI-Nodes for how to install XFormers\033[0m")
        print("="*70 + "\n")

    def __init__(self):
        self.original_attention = comfy_attention.optimized_attention
        self.original_flux_attention = flux_math.attention
        self.original_flux_layers_attention = flux_layers.attention
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
                    "max": 4,
                    "step": 1,
                    "display": "Number of Regions"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "display": "Enable Regional Control"
                }),
            },
            "optional": {
                "region2": ("REGION",),
                "region3": ("REGION",),
                "region4": ("REGION",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING",)
    RETURN_NAMES = ("model", "conditioning",)
    FUNCTION = "apply_attention_control"
    CATEGORY = "conditioning/flux"

    def generate_region_mask(self, region: Dict, width: int, height: int) -> Image.Image:
        if region.get('bbox') is not None:
            x1, y1, x2, y2 = region['bbox']
            mask = Image.new('L', (width, height), 0)
            mask_arr = np.array(mask)
            mask_arr[int(y1*height):int(y2*height), int(x1*width):int(x2*width)] = 255
            print(f'Generating masks with {width}x{height} and [{x1}, {y1}, {x2}, {y2}]')
            return Image.fromarray(mask_arr)
        elif region.get('mask') is not None:
            mask = region['mask'][0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask)
            return mask.resize((width, height))
        else:
            raise Exception('Unknown region type')

    def generate_test_mask(self, masks: List[Image.Image], height: int, width: int):
        hH, hW = int(height) // 16, int(width) // 16
        print(f'{width} {height} -> {hW} {hH}')
        
        lin_masks = []
        for mask in masks:
            mask = mask.convert('L')
            mask = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255
            mask = F.interpolate(mask, (hH, hW), mode='nearest-exact').flatten()
            lin_masks.append(mask)
        return lin_masks, hH, hW

    def prepare_attention_mask(self, lin_masks: List[torch.Tensor], reg_embeds: List[torch.Tensor],
                               Nx: int, emb_size: int, emb_len: int):
        """Prepare attention mask for specific numbers of regions (1, 2, or 3)"""
        total_len = emb_len + Nx
        cross_mask = torch.zeros(total_len, total_len)
        q_scale = torch.ones(total_len)
        k_scale = torch.ones(total_len)
        
        n_regs = len(lin_masks)
        
        # Indices for embeddings
        main_prompt_start = 0
        main_prompt_end = emb_size

        # Subprompt indices
        subprompt_starts = [emb_size * (i + 1) for i in range(n_regs)]
        subprompt_ends = [emb_size * (i + 2) for i in range(n_regs)]
        
        # Handle different cases based on the number of regions
        if n_regs == 1:
            # For one region
            sp_start = subprompt_starts[0]
            sp_end = subprompt_ends[0]
            
            # Block attention between main prompt and subprompt
            cross_mask[sp_start:sp_end, main_prompt_start:main_prompt_end] = 1
            cross_mask[main_prompt_start:main_prompt_end, sp_start:sp_end] = 1
            
            # Scale based on region size
            scale = lin_masks[0].sum() / Nx
            if scale > 1e-5:
                q_scale[sp_start:sp_end] = 1 / scale
                k_scale[sp_start:sp_end] = 1 / scale
            
            # Create mask including tokens and positions
            m_with_tokens = torch.cat([torch.ones(emb_len), lin_masks[0]])
            mb = m_with_tokens > 0.5
            
            # Block attention between positions not in mask and subprompt
            cross_mask[~mb, sp_start:sp_end] = 1
            cross_mask[sp_start:sp_end, ~mb] = 1

        elif n_regs == 2:
            # For two regions
            for i in range(2):
                sp_start = subprompt_starts[i]
                sp_end = subprompt_ends[i]
                
                # Block attention between main prompt and subprompts
                cross_mask[sp_start:sp_end, main_prompt_start:main_prompt_end] = 1
                cross_mask[main_prompt_start:main_prompt_end, sp_start:sp_end] = 1
                
                # Block attention between subprompts
                other_sp_start = subprompt_starts[1 - i]
                other_sp_end = subprompt_ends[1 - i]
                cross_mask[sp_start:sp_end, other_sp_start:other_sp_end] = 1
                cross_mask[other_sp_start:other_sp_end, sp_start:sp_end] = 1
                
                # Scale based on region size
                scale = lin_masks[i].sum() / Nx
                if scale > 1e-5:
                    q_scale[sp_start:sp_end] = 1 / scale
                    k_scale[sp_start:sp_end] = 1 / scale
                
                # Create mask including tokens and positions
                m_with_tokens = torch.cat([torch.ones(emb_len), lin_masks[i]])
                mb = m_with_tokens > 0.5
                
                # Block attention between positions not in mask and subprompt
                cross_mask[~mb, sp_start:sp_end] = 1
                cross_mask[sp_start:sp_end, ~mb] = 1
            
            # Handle intersection between regions
            intersect_idx = torch.logical_and(lin_masks[0] > 0.5, lin_masks[1] > 0.5)
            if intersect_idx.any():
                idx = intersect_idx.nonzero(as_tuple=True)[0] + emb_len
                cross_mask[idx[:, None], idx[None, :]] = 1

        elif n_regs == 3:
            # For three regions
            for i in range(3):
                sp_start = subprompt_starts[i]
                sp_end = subprompt_ends[i]
                
                # Block attention between main prompt and subprompts
                cross_mask[sp_start:sp_end, main_prompt_start:main_prompt_end] = 1
                cross_mask[main_prompt_start:main_prompt_end, sp_start:sp_end] = 1
                
                # Block attention between subprompts
                for j in range(3):
                    if i != j:
                        other_sp_start = subprompt_starts[j]
                        other_sp_end = subprompt_ends[j]
                        cross_mask[sp_start:sp_end, other_sp_start:other_sp_end] = 1
                        cross_mask[other_sp_start:other_sp_end, sp_start:sp_end] = 1
                
                # Scale based on region size
                scale = lin_masks[i].sum() / Nx
                if scale > 1e-5:
                    q_scale[sp_start:sp_end] = 1 / scale
                    k_scale[sp_start:sp_end] = 1 / scale
                
                # Create mask including tokens and positions
                m_with_tokens = torch.cat([torch.ones(emb_len), lin_masks[i]])
                mb = m_with_tokens > 0.5
                
                # Block attention between positions not in mask and subprompt
                cross_mask[~mb, sp_start:sp_end] = 1
                cross_mask[sp_start:sp_end, ~mb] = 1
            
            # Handle intersections between regions
            for i in range(3):
                for j in range(i + 1, 3):
                    intersect_idx = torch.logical_and(lin_masks[i] > 0.5, lin_masks[j] > 0.5)
                    if intersect_idx.any():
                        idx = intersect_idx.nonzero(as_tuple=True)[0] + emb_len
                        cross_mask[idx[:, None], idx[None, :]] = 1
        else:
            raise ValueError("Number of regions must be 1, 2, or 3.")
        
        # Ensure self-attention is allowed
        cross_mask.fill_diagonal_(0)
        
        # Prepare scales for GPU
        q_scale = q_scale.reshape(1, 1, -1, 1).cuda()
        k_scale = k_scale.reshape(1, 1, -1, 1).cuda()
        
        return cross_mask, q_scale, k_scale


    def xformers_attention(self, q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
                           attn_mask: Optional[Tensor] = None) -> Tensor:
        q, k = flux_math.apply_rope(q, k, pe)
        q = rearrange(q, "B H L D -> B L H D")
        k = rearrange(k, "B H L D -> B L H D")
        v = rearrange(v, "B H L D -> B L H D")
        if attn_mask is not None:
            x = xattention(q, k, v, attn_bias=attn_mask)
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
                                region2: Optional[Dict] = None,
                                region3: Optional[Dict] = None,
                                region4: Optional[Dict] = None):
        if not enabled:
            # Restore original attention functions
            flux_math.attention = self.original_flux_attention
            flux_layers.attention = self.original_flux_layers_attention
            print("Regional control disabled. Restored original attention functions.")
            return (model, condition)

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
        
        # Collect regions
        regions = [region1, region2, region3, region4]
        for idx, region in enumerate(regions[:number_of_regions]):
            if region is not None and region.get('conditioning') is not None:
                subprompts_embeds.append(region['conditioning'][0][0])
                masks.append(self.generate_region_mask(region, iW, iH))
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
            lin_masks, subprompts_embeds, Nx, emb_size, emb_len)

        # Format for xFormers
        device = torch.device('cuda')
        attn_dtype = torch.bfloat16 if model_management.should_use_bf16(device=device) else torch.float16
        
        if attn_mask is not None:
            print(f'Applying attention masks: {attn_mask.shape}')
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
