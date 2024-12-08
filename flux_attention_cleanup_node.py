import torch
from comfy.ldm.modules import attention as comfy_attention
from comfy.ldm.flux import math as flux_math
from comfy.ldm.flux import layers as flux_layers

class AnyType(str):
    """A special class that is always equal in not equal comparisons"""
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class FluxAttentionCleanup:
    def __init__(self):
        self.original_attention = comfy_attention.optimized_attention
        self.original_flux_attention = flux_math.attention
        self.original_flux_layers_attention = flux_layers.attention
        self.current_attn_mask = None
        print("FluxAttentionCleanup initialized")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": (any_type, {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "cleanup_attention"
    CATEGORY = "ControlAltAI Nodes/Flux Region"

    def cleanup_attention(self, any_input):
        """Skip cleanup during normal operation, but clean on workflow switch"""
        message = "Attention preserved for current workflow. Will clean on workflow switch."
        print("\n" + message)
        return (message,)

    def __del__(self):
        """Clean up attention when switching workflows"""
        try:
            print("\nStarting attention cleanup for workflow switch...")
            
            # Reset attention functions to original state
            flux_math.attention = self.original_flux_attention
            flux_layers.attention = self.original_flux_layers_attention
            
            # Clear attention mask
            if hasattr(flux_math.attention, 'keywords'):
                if 'attn_mask' in flux_math.attention.keywords:
                    flux_math.attention.keywords['attn_mask'] = None
            
            # Clear stored mask
            if self.current_attn_mask is not None:
                del self.current_attn_mask
                self.current_attn_mask = None

            # Force CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("Workflow switch: Region Attention Cleanup Successful")
        except:
            pass

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FluxAttentionCleanup": FluxAttentionCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAttentionCleanup": "Flux Attention Cleanup"
}