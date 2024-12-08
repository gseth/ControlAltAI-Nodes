import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RegionMaskGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "number_of_regions": ("INT", {"default": 1, "min": 1, "max": 3}),
                # Region 1
                "region1_x1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region1_y1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region1_x2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region1_y2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Region 2
                "region2_x1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region2_y1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region2_x2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region2_y2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Region 3
                "region3_x1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region3_y1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region3_x2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region3_y2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK", "INT", "BBOX", "BBOX", "BBOX")
    RETURN_NAMES = ("colored_regions_image", "bbox_preview", 
                   "mask1", "mask2", "mask3",
                   "number_of_regions",
                   "bbox1", "bbox2", "bbox3")
    FUNCTION = "generate_regions"
    CATEGORY = "ControlAltAI Nodes/Flux Region"

    def create_bbox(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        """Create bbox with debug output"""
        print(f"Creating BBOX: x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}")
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "active": True
        }

    def create_mask_from_bbox(self, bbox: Dict, width: int, height: int) -> torch.Tensor:
        """Create mask from bbox with debug output"""
        mask = torch.zeros((height, width), dtype=torch.float32)
        if bbox["active"]:
            x1 = int(bbox["x1"] * width)
            y1 = int(bbox["y1"] * height)
            x2 = int(bbox["x2"] * width)
            y2 = int(bbox["y2"] * height)
            print(f"Creating mask at pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            mask[y1:y2, x1:x2] = 1.0
        return mask

    def create_preview(self, masks: List[torch.Tensor], bboxes: List[Dict], 
                      number_of_regions: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create preview images with debug info"""
        if not masks:
            return torch.zeros((3, 64, 64), dtype=torch.float32), torch.zeros((3, 64, 64), dtype=torch.float32)

        height, width = masks[0].shape
        
        # Create both preview images
        region_preview = Image.new("RGB", (width, height), (0, 0, 0))
        bbox_preview = Image.new("RGB", (width, height), (0, 0, 0))
        region_draw = ImageDraw.Draw(region_preview)
        bbox_draw = ImageDraw.Draw(bbox_preview)

        colors = [
            (255, 0, 0),    # Red - Region 1
            (0, 255, 0),    # Green - Region 2
            (255, 255, 0),  # Yellow - Region 3
        ]

        # Store regions for ordered preview
        preview_regions = []
        for i in range(number_of_regions):
            if bboxes[i]["active"]:
                mask_np = masks[i].cpu().numpy() > 0.5
                if mask_np.any():
                    preview_regions.append((i, mask_np, bboxes[i]))

        # Draw regions in reverse order (Region 3 first, Region 1 last)
        for i, mask_np, bbox in sorted(preview_regions, reverse=True):
            # Draw on region preview
            color_array = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                color_array[mask_np, c] = colors[i][c]
            preview_np = np.array(region_preview)
            preview_np[mask_np] = color_array[mask_np]
            region_preview = Image.fromarray(preview_np)

            # Draw on bbox preview - maintaining original bbox drawing order
            x1 = int(bbox["x1"] * width)
            y1 = int(bbox["y1"] * height)
            x2 = int(bbox["x2"] * width)
            y2 = int(bbox["y2"] * height)
            print(f"Drawing preview for region {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            bbox_draw.rectangle([x1, y1, x2, y2], outline=colors[i], width=2)

        return pil2tensor(region_preview), pil2tensor(bbox_preview)

    def generate_regions(self,
                        width: int,
                        height: int,
                        number_of_regions: int,
                        **kwargs) -> Tuple:
        try:
            print(f"\nGenerating {number_of_regions} regions for {width}x{height} image")
            bboxes = []
            masks = []

            # Create regions
            for i in range(3):
                if i < number_of_regions:
                    print(f"\nProcessing region {i+1}:")
                    bbox = self.create_bbox(
                        kwargs[f"region{i+1}_x1"],
                        kwargs[f"region{i+1}_y1"],
                        kwargs[f"region{i+1}_x2"],
                        kwargs[f"region{i+1}_y2"]
                    )
                    mask = self.create_mask_from_bbox(bbox, width, height)
                    bboxes.append(bbox)
                    masks.append(mask)
                else:
                    print(f"Creating empty region {i+1}")
                    empty_bbox = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "active": False}
                    bboxes.append(empty_bbox)
                    masks.append(torch.zeros((height, width), dtype=torch.float32))

            # Create previews
            region_preview, bbox_preview = self.create_preview(masks, bboxes, number_of_regions)

            return (region_preview, bbox_preview, *masks, number_of_regions, *bboxes)

        except Exception as e:
            print(f"Error in generate_regions: {str(e)}")
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            empty_bbox = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "active": False}
            empty_preview = torch.zeros((3, height, width), dtype=torch.float32)
            return (empty_preview, empty_preview, 
                   empty_mask, empty_mask, empty_mask,
                   0, empty_bbox, empty_bbox, empty_bbox)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "RegionMaskGenerator": RegionMaskGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionMaskGenerator": "Region Mask Generator"
}