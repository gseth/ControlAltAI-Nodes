import torch
from typing import Tuple, Dict, Optional, List
import numpy as np
from PIL import Image, ImageDraw

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RegionMaskValidator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "bbox1": ("BBOX",),
                "number_of_regions": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 3,
                    "step": 1
                }),
                "min_region_size": ("INT", {
                    "default": 64,
                    "min": 32,
                    "max": 512,
                    "step": 32,
                    "display": "Minimum Region Size (px)"
                }),
                "max_overlap": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "Maximum Region Overlap"
                }),
            },
            "optional": {
                "mask2": ("MASK",),
                "bbox2": ("BBOX",),
                "mask3": ("MASK",),
                "bbox3": ("BBOX",),
            }
        }

    RETURN_TYPES = ("MASK", "BBOX", "MASK", "BBOX", "MASK", "BBOX", 
                    "INT", "BOOLEAN", "STRING", "IMAGE")
    RETURN_NAMES = ("valid_mask1", "valid_bbox1",
                   "valid_mask2", "valid_bbox2",
                   "valid_mask3", "valid_bbox3",
                   "valid_region_count", "is_valid", "validation_message",
                   "validation_preview")
    FUNCTION = "validate_regions"
    CATEGORY = "ControlAltAI Nodes/Flux Region"

    def get_region_dimensions(self, bbox: Dict, width: int, height: int) -> Tuple[int, int, Tuple[int, int]]:
        """Calculate region dimensions in pixels"""
        if not bbox["active"]:
            return 0, (0, 0)
            
        x1 = int(bbox["x1"] * width)
        y1 = int(bbox["y1"] * height)
        x2 = int(bbox["x2"] * width)
        y2 = int(bbox["y2"] * height)
        
        w = x2 - x1
        h = y2 - y1
        area = w * h
        print(f"Region dimensions: {w}x{h} pixels")
        return area, (w, h)

    def calculate_overlap(self, bbox1: Dict, bbox2: Dict, width: int, height: int) -> Tuple[Tuple[int, int], float]:
        """Calculate overlap dimensions and ratio"""
        if not (bbox1["active"] and bbox2["active"]):
            return (0, 0), 0.0

        # Convert to pixel coordinates
        x1_1 = int(bbox1["x1"] * width)
        y1_1 = int(bbox1["y1"] * height)
        x2_1 = int(bbox1["x2"] * width)
        y2_1 = int(bbox1["y2"] * height)
        
        x1_2 = int(bbox2["x1"] * width)
        y1_2 = int(bbox2["y1"] * height)
        x2_2 = int(bbox2["x2"] * width)
        y2_2 = int(bbox2["y2"] * height)

        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right > x_left and y_bottom > y_top:
            overlap_width = x_right - x_left
            overlap_height = y_bottom - y_top
            overlap_area = overlap_width * overlap_height
            
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            smaller_area = min(area1, area2)
            overlap_ratio = overlap_area / smaller_area
            
            print(f"Overlap dimensions: {overlap_width}x{overlap_height} pixels ({overlap_ratio:.1%})")
            return (overlap_width, overlap_height), overlap_ratio
        
        return (0, 0), 0.0

    def create_validation_preview(self, masks: List[torch.Tensor], bboxes: List[Dict], 
                                number_of_regions: int, is_valid: bool,
                                messages: List[str], img_width: int, img_height: int) -> torch.Tensor:
        """Create visual validation feedback with improved text rendering"""
        if not masks:
            return torch.zeros((3, 64, 64), dtype=torch.float32)

        preview = Image.new("RGB", (img_width, img_height), (0, 0, 0))
        draw = ImageDraw.Draw(preview)

        # Colors for valid/invalid regions
        colors = {
            'valid': [(0, 255, 0), (0, 200, 0), (0, 150, 0)],  # Green shades
            'invalid': [(255, 0, 0), (200, 0, 0), (150, 0, 0)]  # Red shades
        }

        # Draw regions with validation status and improved text
        for i, (mask, bbox) in enumerate(zip(masks[:number_of_regions], bboxes[:number_of_regions])):
            if bbox["active"]:
                x1 = int(bbox["x1"] * img_width)
                y1 = int(bbox["y1"] * img_height)
                x2 = int(bbox["x2"] * img_width)
                y2 = int(bbox["y2"] * img_height)

                w = x2 - x1
                h = y2 - y1
                color = colors['valid' if is_valid else 'invalid'][i]
                
                # Draw thicker rectangle outline
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                
                # Improved region label with dimensions
                label = f"R{i+1}: {w}x{h}"
                # Position text with offset from corner and draw twice for better visibility
                text_x = x1 + 10
                text_y = y1 + 10
                
                # Draw text shadow/outline for better contrast
                shadow_offset = 2
                shadow_color = (0, 0, 0)
                for dx in [-shadow_offset, shadow_offset]:
                    for dy in [-shadow_offset, shadow_offset]:
                        draw.text((text_x + dx, text_y + dy), label, fill=shadow_color, font=None, size=64)
                
                # Draw main text
                draw.text((text_x, text_y), label, fill=color, font=None, size=64)

                # If region is invalid, add error message below the label
                if not is_valid and i < len(messages):
                    error_y = text_y + 30  # Position error message below label
                    # Draw error message with shadow for contrast
                    for dx in [-shadow_offset, shadow_offset]:
                        for dy in [-shadow_offset, shadow_offset]:
                            draw.text((text_x + dx, error_y + dy), messages[i], fill=shadow_color, font=None, size=20)
                    draw.text((text_x, error_y), messages[i], fill=color, font=None, size=20)

        return pil2tensor(preview)

    def validate_regions(self,
                        mask1: torch.Tensor,
                        bbox1: Dict,
                        number_of_regions: int,
                        min_region_size: int,
                        max_overlap: float,
                        mask2: Optional[torch.Tensor] = None,
                        bbox2: Optional[Dict] = None,
                        mask3: Optional[torch.Tensor] = None,
                        bbox3: Optional[Dict] = None) -> Tuple:
        try:
            print(f"\nValidating {number_of_regions} regions:")
            messages = []
            is_valid = True
            height, width = mask1.shape
            print(f"Canvas size: {width}x{height} pixels")

            # Collect regions
            regions = [
                (mask1, bbox1),
                (mask2, bbox2) if mask2 is not None else (None, None),
                (mask3, bbox3) if mask3 is not None else (None, None),
            ]

            # Validate each region
            valid_regions = []
            valid_count = 0
            for i, (mask, bbox) in enumerate(regions):
                if i < number_of_regions and mask is not None and bbox is not None:
                    print(f"\nValidating Region {i+1}:")
                    # Check region size
                    _, (w, h) = self.get_region_dimensions(bbox, width, height)
                    
                    if w < min_region_size or h < min_region_size:
                        message = f"Region {i+1} too small: {w}x{h} pixels (minimum: {min_region_size}x{min_region_size})"
                        print(f"Failed: {message}")
                        messages.append(message)
                        is_valid = False
                        bbox = bbox.copy()
                        bbox["active"] = False
                    else:
                        print(f"Passed: Region {i+1} size check ({w}x{h} pixels)")
                        valid_count += 1

                    valid_regions.append((mask, bbox))
                else:
                    valid_regions.append((
                        torch.zeros_like(mask1),
                        {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "active": False}
                    ))

            # Check overlaps
            if valid_count > 1:
                print("\nChecking region overlaps:")
                for i in range(len(valid_regions)):
                    for j in range(i + 1, len(valid_regions)):
                        mask_i, bbox_i = valid_regions[i]
                        mask_j, bbox_j = valid_regions[j]
                        
                        if bbox_i["active"] and bbox_j["active"]:
                            print(f"Checking overlap between regions {i+1} and {j+1}:")
                            (ow, oh), overlap_ratio = self.calculate_overlap(bbox_i, bbox_j, width, height)
                            
                            if overlap_ratio > max_overlap:
                                message = f"Excessive overlap ({ow}x{oh} pixels, {overlap_ratio:.1%}) between regions {i+1} and {j+1}"
                                print(f"Failed: {message}")
                                messages.append(message)
                                is_valid = False

            # Create validation message
            validation_message = "All regions valid" if is_valid else "\n".join(messages)
            print(f"\nValidation {'passed' if is_valid else 'failed'}:")
            print(validation_message)

            # Create validation preview
            preview = self.create_validation_preview(
                [r[0] for r in valid_regions],
                [r[1] for r in valid_regions],
                number_of_regions,
                is_valid,
                messages,
                width,
                height
            )

            return (*[item for region in valid_regions for item in region],
                   valid_count, is_valid, validation_message, preview)

        except Exception as e:
            print(f"Validation error: {str(e)}")
            empty_mask = torch.zeros_like(mask1)
            empty_bbox = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "active": False}
            empty_preview = torch.zeros((3, height, width), dtype=torch.float32)
            return (empty_mask, empty_bbox, empty_mask, empty_bbox,
                   empty_mask, empty_bbox,
                   0, False, f"Validation error: {str(e)}", empty_preview)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "RegionMaskValidator": RegionMaskValidator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionMaskValidator": "Region Mask Validator"
}