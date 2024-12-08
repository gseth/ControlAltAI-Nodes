print("\n\033[32mInitializing ControlAltAI Nodes\033[0m")  # Fixed green reset
try:
   import xformers
   print("\n\033[94mControlAltAI: xformers found!\033[0m")  # Light blue
except ImportError:
   print("\n" + "="*70)
   print("\033[94mCONTROLALTAI-NODES: XFORMERS REQUIRED\033[0m")  # Light blue
   print("\033[33mSome nodes in this custom node pack require xformers to function properly.")  # Start orange
   print("\nPlease check \"import_error_solution.txt\" in ComfyUI\\custom_nodes\\ControlAltAI-Nodes for how to install XFormers\033[0m")  # End orange
   print("="*70 + "\n")
   raise ImportError("xformers is required for ControlAltAI Nodes")

from .flux_resolution_cal_node import FluxResolutionNode
from .flux_sampler_node import FluxSampler
from .flux_union_controlnet_node import FluxUnionControlNetApply
from .boolean_basic_node import BooleanBasic
from .boolean_reverse_node import BooleanReverse
from .get_image_size_ratio_node import GetImageSizeRatio
from .noise_plus_blend_node import NoisePlusBlend
from .integer_settings_node import IntegerSettings
from .choose_upscale_model_node import ChooseUpscaleModel
from .region_mask_generator_node import RegionMaskGenerator
from .region_mask_validator_node import RegionMaskValidator
from .region_mask_processor_node import RegionMaskProcessor
from .region_mask_conditioning_node import RegionMaskConditioning
from .flux_attention_control_node import FluxAttentionControl
from .region_overlay_visualizer_node import RegionOverlayVisualizer
from .flux_attention_cleanup_node import FluxAttentionCleanup

NODE_CLASS_MAPPINGS = {
    "FluxResolutionNode": FluxResolutionNode,
    "FluxSampler": FluxSampler,
    "FluxUnionControlNetApply": FluxUnionControlNetApply,
    "BooleanBasic": BooleanBasic,
    "BooleanReverse": BooleanReverse,
    "GetImageSizeRatio": GetImageSizeRatio,
    "NoisePlusBlend": NoisePlusBlend,
    "IntegerSettings": IntegerSettings,
    "ChooseUpscaleModel": ChooseUpscaleModel,
    "RegionMaskGenerator": RegionMaskGenerator,
    "RegionMaskValidator": RegionMaskValidator,
    "RegionMaskProcessor": RegionMaskProcessor,
    "RegionMaskConditioning": RegionMaskConditioning,
    "FluxAttentionControl": FluxAttentionControl,
    "RegionOverlayVisualizer": RegionOverlayVisualizer,
    "FluxAttentionCleanup": FluxAttentionCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxResolutionNode": "Flux Resolution Calc",
    "FluxSampler": "Flux Sampler",
    "FluxUnionControlNetApply": "Flux Union ControlNet Apply",
    "BooleanBasic": "Boolean Basic",
    "BooleanReverse": "Boolean Reverse",
    "GetImageSizeRatio": "Get Image Size Ratio",
    "NoisePlusBlend": "Noise Plus Blend",
    "IntegerSettings": "Integer Settings",
    "ChooseUpscaleModel": "Choose Upscale Model",
    "RegionMaskGenerator": "Region Mask Generator",
    "RegionMaskValidator": "Region Mask Validator",
    "RegionMaskProcessor": "Region Mask Processor",
    "RegionMaskConditioning": "Region Mask Conditioning",
    "FluxAttentionControl": "Flux Attention Control",
    "RegionOverlayVisualizer": "Region Overlay Visualizer",
    "FluxAttentionCleanup": "Flux Attention Cleanup",
}