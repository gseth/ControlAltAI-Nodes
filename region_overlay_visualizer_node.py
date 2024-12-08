import torch
import numpy as np
from typing import Tuple

class RegionOverlayVisualizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region_preview": ("IMAGE",),
                "opacity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "Overlay Opacity"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_regions"
    CATEGORY = "ControlAltAI Nodes/Flux Region"

    def visualize_regions(
        self,
        image: torch.Tensor,
        region_preview: torch.Tensor,
        opacity: float,
    ) -> Tuple[torch.Tensor]:
        try:
            print("\n=== Starting Region Overlay Visualization ===")
            print(f"Initial shapes - Image: {image.shape}, Preview: {region_preview.shape}")

            # Ensure input tensors are in [B, H, W, C] format
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(region_preview.shape) == 3:
                region_preview = region_preview.unsqueeze(0)

            # Get working copies
            base_image = image.clone()
            preview = region_preview.clone()

            # Convert to numpy for mask creation (keeping batch and HWC format)
            preview_np = (preview * 255).byte().cpu().numpy()
            
            # Create mask based on preview content (operating on the last dimension - channels)
            color_sum = np.sum(preview_np, axis=-1)  # Sum across color channels
            max_channel = np.max(preview_np, axis=-1)
            min_channel = np.min(preview_np, axis=-1)
            
            # Create binary mask where content exists
            mask = (
                (color_sum > 50) & 
                (max_channel > 30) & 
                ((max_channel - min_channel) > 10)
            )
            
            # Expand mask to match input dimensions
            mask = mask[..., None]  # Add channel dimension back
            mask = torch.from_numpy(mask).to(image.device)

            print(f"Mask shape: {mask.shape}")
            print(f"Masked pixels: {mask.sum().item()}/{mask.numel()} ({mask.sum().item()/mask.numel()*100:.2f}%)")

            # Apply blending only where mask is True
            result = torch.where(
                mask.bool(),
                (1 - opacity) * base_image + opacity * preview,
                base_image
            )

            print(f"Final shape: {result.shape}")
            return (result,)

        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return (image,)

NODE_CLASS_MAPPINGS = {
    "RegionOverlayVisualizer": RegionOverlayVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionOverlayVisualizer": "Region Overlay Visualizer"
}