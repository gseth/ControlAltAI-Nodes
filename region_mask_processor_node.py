import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np
from PIL import Image, ImageDraw

def pil2tensor(image):
    """Convert a PIL image to a PyTorch tensor in the expected format."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RegionMaskProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "bbox1": ("BBOX",),
                "blur_radius": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "display": "Blur Radius"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "Mask Threshold"
                }),
                "feather_edges": ("BOOLEAN", {
                    "default": True,
                    "display": "Feather Edges"
                }),
                "number_of_regions": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 3,
                    "display": "Number of Regions"
                }),
            },
            "optional": {
                "mask2": ("MASK",),
                "bbox2": ("BBOX",),
                "mask3": ("MASK",),
                "bbox3": ("BBOX",),
            }
        }

    RETURN_TYPES = ("MASK", "BBOX", "MASK", "BBOX", "MASK", "BBOX", "IMAGE", "INT")
    RETURN_NAMES = ("processed_mask1", "processed_bbox1",
                   "processed_mask2", "processed_bbox2",
                   "processed_mask3", "processed_bbox3",
                   "preview_image", "region_count")
    FUNCTION = "process_regions"
    CATEGORY = "ControlAltAI Nodes/Flux Region"

    def apply_gaussian_blur(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Apply gaussian blur to mask edges"""
        if radius <= 0:
            return mask

        kernel_size = 2 * radius + 1
        sigma = radius / 3.0

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        kernel_1d = torch.exp(torch.linspace(-radius, radius, kernel_size).pow(2) / (-2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        padding = radius
        kernel_h = kernel_1d.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(mask.device)
        kernel_v = kernel_1d.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(mask.device)

        mask = F.pad(mask, (padding, padding, 0, 0), mode='reflect')
        mask = F.conv2d(mask, kernel_h)
        mask = F.pad(mask, (0, 0, padding, padding), mode='reflect')
        mask = F.conv2d(mask, kernel_v)

        return mask.squeeze()

    def apply_feathering(self, mask: torch.Tensor, bbox: Dict, radius: int) -> Tuple[torch.Tensor, Dict]:
        """Apply feathering to mask edges while preserving bbox boundaries"""
        if radius <= 0 or not bbox["active"]:
            return mask, bbox

        height, width = mask.shape
        x1 = int(bbox["x1"] * width)
        y1 = int(bbox["y1"] * height)
        x2 = int(bbox["x2"] * width)
        y2 = int(bbox["y2"] * height)

        inner_mask = torch.zeros_like(mask)
        inner_mask[y1+radius:y2-radius, x1+radius:x2-radius] = 1.0
        edge_mask = mask - inner_mask

        if edge_mask.any():
            blurred = self.apply_gaussian_blur(mask, radius)
            result = mask.clone()
            result[edge_mask > 0] = blurred[edge_mask > 0]
        else:
            result = mask

        return result, bbox

    def process_single_region(self,
                            mask: torch.Tensor,
                            bbox: Dict,
                            blur_radius: int,
                            threshold: float,
                            feather_edges: bool) -> Tuple[torch.Tensor, Dict]:
        """Process a single mask-bbox pair"""
        if mask is None or not bbox["active"]:
            return mask, bbox

        try:
            processed = (mask > threshold).float()

            if feather_edges and blur_radius > 0:
                processed, bbox = self.apply_feathering(processed, bbox, blur_radius)
            elif blur_radius > 0:
                processed = self.apply_gaussian_blur(processed, blur_radius)

            return processed, bbox

        except Exception as e:
            print(f"Error processing region: {str(e)}")
            return mask, bbox

    def create_preview(self, masks: List[torch.Tensor], bboxes: List[Dict], 
                      number_of_regions: int) -> torch.Tensor:
        """Create preview of processed regions with PIL for consistent coloring"""
        if not masks:
            return torch.zeros((3, 64, 64), dtype=torch.float32)

        height, width = masks[0].shape
        
        # Create PIL Image for preview
        preview = Image.new("RGB", (width, height), (0, 0, 0))

        colors = [
            (255, 0, 0),    # Red - Region 1
            (0, 255, 0),    # Green - Region 2
            (255, 255, 0),  # Yellow - Region 3
        ]

        # Store regions for ordered preview
        preview_regions = []
        for i in range(number_of_regions):
            if bboxes[i]["active"] and masks[i] is not None:
                mask_np = masks[i].cpu().numpy() > 0.5
                preview_regions.append((i, mask_np))

        # Draw regions in reverse order (Region 3 first, Region 1 last)
        for i, mask_np in sorted(preview_regions, reverse=True):
            color_array = np.zeros((height, width, 3), dtype=np.uint8)
            color_array[mask_np] = colors[i]
            
            # Convert to PIL and composite
            region_img = Image.fromarray(color_array, 'RGB')
            preview = Image.alpha_composite(
                preview.convert('RGBA'),
                Image.merge('RGBA', (*region_img.split(), Image.fromarray((mask_np * 255).astype(np.uint8))))
            )

        return pil2tensor(preview.convert('RGB'))

    def process_regions(self,
                       mask1: torch.Tensor,
                       bbox1: Dict,
                       blur_radius: int,
                       threshold: float,
                       feather_edges: bool,
                       number_of_regions: int,
                       mask2: Optional[torch.Tensor] = None,
                       bbox2: Optional[Dict] = None,
                       mask3: Optional[torch.Tensor] = None,
                       bbox3: Optional[Dict] = None) -> Tuple:
        try:
            # Process each mask-bbox pair
            mask_bbox_pairs = [
                (mask1, bbox1),
                (mask2, bbox2) if mask2 is not None else (None, None),
                (mask3, bbox3) if mask3 is not None else (None, None),
            ]

            processed_masks = []
            processed_bboxes = []
            active_count = 0

            for i, (mask, bbox) in enumerate(mask_bbox_pairs):
                if i < number_of_regions and mask is not None and bbox is not None:
                    proc_mask, proc_bbox = self.process_single_region(
                        mask, bbox, blur_radius, threshold, feather_edges
                    )
                    if proc_bbox["active"]:
                        active_count += 1
                    processed_masks.append(proc_mask)
                    processed_bboxes.append(proc_bbox)
                else:
                    empty_mask = torch.zeros_like(mask1)
                    empty_bbox = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "active": False}
                    processed_masks.append(empty_mask)
                    processed_bboxes.append(empty_bbox)

            # Create preview
            preview = self.create_preview(processed_masks, processed_bboxes, number_of_regions)

            return (*[item for pair in zip(processed_masks, processed_bboxes) for item in pair], 
                   preview, active_count)

        except Exception as e:
            print(f"Error processing regions: {str(e)}")
            empty_mask = torch.zeros_like(mask1)
            empty_bbox = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "active": False}
            empty_preview = torch.zeros((3, mask1.shape[0], mask1.shape[1]), dtype=torch.float32)
            return (empty_mask, empty_bbox, empty_mask, empty_bbox,
                   empty_mask, empty_bbox,
                   empty_preview, 0)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "RegionMaskProcessor": RegionMaskProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionMaskProcessor": "Region Mask Processor"
}