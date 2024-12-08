import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RegionMaskConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "bbox1": ("BBOX",),
                "conditioning1": ("CONDITIONING",),
                "number_of_regions": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "display": "Number of Regions"
                }),
                "strength1": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "Strength for Region 1"
                }),
            },
            "optional": {
                "mask2": ("MASK",),
                "bbox2": ("BBOX",),
                "conditioning2": ("CONDITIONING",),
                "strength2": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "Strength for Region 2"
                }),
                "mask3": ("MASK",),
                "bbox3": ("BBOX",),
                "conditioning3": ("CONDITIONING",),
                "strength3": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "Strength for Region 3"
                }),
            }
        }

    RETURN_TYPES = ("REGION", "REGION", "REGION", "INT", "IMAGE")
    RETURN_NAMES = ("region1", "region2", "region3",
                   "region_count", "preview_image")
    FUNCTION = "create_conditioned_regions"
    CATEGORY = "ControlAltAI Nodes/Flux Region"

    def validate_bbox(self, bbox: Dict) -> bool:
        """Validate bbox coordinates and structure"""
        print(f"\n=== Validating BBox ===")
        print(f"Input bbox: {bbox}")
        
        if bbox is None or not isinstance(bbox, dict):
            print("Failed: Invalid bbox type")
            return False
            
        required_keys = ["x1", "y1", "x2", "y2"]
        if not all(k in bbox for k in required_keys):
            print(f"Failed: Missing keys. Required: {required_keys}")
            return False
            
        # Validate coordinate values
        if not all(isinstance(bbox[k], (int, float)) for k in required_keys):
            print("Failed: Invalid coordinate types")
            return False
            
        # Validate coordinate ranges
        if not all(0 <= bbox[k] <= 1.0 for k in required_keys):
            print("Failed: Coordinates out of range [0,1]")
            return False
            
        # Validate proper ordering
        if bbox["x1"] >= bbox["x2"] or bbox["y1"] >= bbox["y2"]:
            print("Failed: Invalid coordinate ordering")
            return False
            
        print("Passed: BBox validation successful")
        return True

    def scale_conditioning(self, conditioning: List, strength: float) -> List:
        """Scale conditioning tensors by strength"""
        print(f"\n=== Scaling Conditioning ===")
        print(f"Strength: {strength}")
        
        try:
            if not conditioning or not isinstance(conditioning, list):
                print("Failed: Invalid conditioning format")
                raise ValueError("Invalid conditioning format")

            # Get the conditioning tensors and dict
            cond_tensors = conditioning[0][0]
            cond_dict = conditioning[0][1]
            
            print(f"Input tensor shape: {cond_tensors.shape}")
            print(f"Conditioning keys: {list(cond_dict.keys())}")
            print(f"Input tensor stats: min={cond_tensors.min():.3f}, max={cond_tensors.max():.3f}, mean={cond_tensors.mean():.3f}")

            # Scale the tensors
            scaled_tensors = cond_tensors.clone() * strength
            print(f"Scaled tensor stats: min={scaled_tensors.min():.3f}, max={scaled_tensors.max():.3f}, mean={scaled_tensors.mean():.3f}")

            return [[scaled_tensors, cond_dict]]

        except Exception as e:
            print(f"Error in scale_conditioning: {str(e)}")
            import traceback
            traceback.print_exc()
            return conditioning

    def create_region(self, mask: Optional[torch.Tensor], bbox: Optional[Dict], 
                     conditioning: Optional[List], strength: float, region_idx: int) -> Dict:
        """Create a single region with its conditioning"""
        print(f"\n=== Creating Region {region_idx} ===")
        
        # Debug inputs
        print("Input validation:")
        print(f"- Mask: {type(mask)}, shape={mask.shape if mask is not None else None}")
        print(f"- BBox: {bbox}")
        print(f"- Conditioning type: {type(conditioning)}")
        print(f"- Strength: {strength}")
        
        # Default empty region
        empty_region = {
            "conditioning": None,
            "bbox": [0.0, 0.0, 0.0, 0.0],  # Array format for empty
            "is_active": False,
            "strength": 1.0
        }

        try:
            # Validate inputs
            if mask is None or bbox is None or conditioning is None:
                print(f"Region {region_idx}: Missing components")
                return empty_region
                
            if not self.validate_bbox(bbox):
                print(f"Region {region_idx}: Invalid bbox")
                return empty_region
            
            # Scale conditioning
            scaled_conditioning = self.scale_conditioning(conditioning, strength)
            
            # Create region output - bbox array, conditioning, and strength
            region = {
                "conditioning": scaled_conditioning,
                "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],  # Array format
                "is_active": True,
                "strength": strength
            }
            
            print(f"\nSuccessfully created region {region_idx}")
            return region
            
        except Exception as e:
            print(f"Error creating region {region_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            return empty_region

    def create_preview(self, masks: List[torch.Tensor], bboxes: List[Dict],
                      number_of_regions: int) -> torch.Tensor:
        """Create preview of conditioned regions"""
        print("\n=== Creating Preview ===")
        
        if not masks:
            print("No masks provided")
            return torch.zeros((3, 64, 64), dtype=torch.float32)

        height, width = masks[0].shape
        print(f"Preview dimensions: {width}x{height}")

        # Create PIL Image for preview
        preview = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(preview)

        # Define colors for 3 regions
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
        ]

        # Draw each region
        for i, (mask, bbox) in enumerate(zip(masks[:number_of_regions], bboxes[:number_of_regions])):
            validation_result = self.validate_bbox(bbox)
            if validation_result and mask is not None:
                print(f"\nDrawing region {i+1}:")
                # Get pixel coordinates
                x1 = int(bbox["x1"] * width)
                y1 = int(bbox["y1"] * height)
                x2 = int(bbox["x2"] * width)
                y2 = int(bbox["y2"] * height)
                
                print(f"Region {i+1} coordinates: ({x1},{y1}) to ({x2},{y2})")
                
                # Draw region outline
                draw.rectangle([x1, y1, x2, y2], outline=colors[i], width=4)

        return pil2tensor(preview)

    def create_conditioned_regions(self,
                                 mask1: torch.Tensor,
                                 bbox1: Dict,
                                 conditioning1: List,
                                 number_of_regions: int,
                                 strength1: float,
                                 mask2: Optional[torch.Tensor] = None,
                                 bbox2: Optional[Dict] = None,
                                 conditioning2: Optional[List] = None,
                                 strength2: Optional[float] = 1.0,
                                 mask3: Optional[torch.Tensor] = None,
                                 bbox3: Optional[Dict] = None,
                                 conditioning3: Optional[List] = None,
                                 strength3: Optional[float] = 1.0) -> Tuple:
        print("\n=== Creating Conditioned Regions ===")
        print(f"Number of regions: {number_of_regions}")

        try:
            # Create regions
            regions = []
            active_count = 0

            # Process required number of regions
            inputs = [
                (mask1, bbox1, conditioning1, strength1),
                (mask2, bbox2, conditioning2, strength2),
                (mask3, bbox3, conditioning3, strength3)
            ]

            # Store masks and bboxes for preview only
            preview_masks = []
            preview_bboxes = []

            for i, (mask, bbox, conditioning, strength) in enumerate(inputs[:number_of_regions]):
                # Create region with per-region strength
                region = self.create_region(mask, bbox, conditioning, strength, i+1)
                if region["is_active"]:
                    active_count += 1
                regions.append(region)
                print(f"Processed region {i+1}: active={region['is_active']}")

                # Store for preview
                preview_masks.append(mask)
                preview_bboxes.append(bbox)

            # Fill remaining slots with empty regions
            empty_region = {
                "conditioning": None,
                "bbox": [0.0, 0.0, 0.0, 0.0],  # Array format
                "is_active": False,
                "strength": 1.0
            }

            while len(regions) < 3:
                idx = len(regions) + 1
                print(f"Adding empty region {idx}")
                regions.append(empty_region)

            print(f"\nCreated {active_count} active regions out of {number_of_regions} requested")

            # Create preview using stored masks and bboxes
            preview = self.create_preview(preview_masks, preview_bboxes, number_of_regions)

            return (*regions, active_count, preview)

        except Exception as e:
            print(f"Error in create_conditioned_regions: {str(e)}")
            import traceback
            traceback.print_exc()

            empty_region = {
                "conditioning": None,
                "bbox": [0.0, 0.0, 0.0, 0.0],  # Array format
                "is_active": False,
                "strength": 1.0
            }
            empty_preview = torch.zeros((3, mask1.shape[0], mask1.shape[1]), dtype=torch.float32)
            return (empty_region, empty_region, empty_region, 0, empty_preview)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "RegionMaskConditioning": RegionMaskConditioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionMaskConditioning": "Region Mask Conditioning"
}