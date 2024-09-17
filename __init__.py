from .flux_resolution_cal_node import FluxResolutionNode
from .flux_sampler_node import FluxSampler
from .flux_controlnet_node import FluxControlNetApply
from .boolean_basic_node import BooleanBasic
from .boolean_reverse_node import BooleanReverse
from .get_image_size_ratio_node import GetImageSizeRatio
from .noise_plus_blend_node import NoisePlusBlend
from .integer_settings_node import IntegerSettings
from .choose_upscale_model_node import ChooseUpscaleModel  # Import the new node

print("Initializing ControlAltAI Nodes")

NODE_CLASS_MAPPINGS = {
    "FluxResolutionNode": FluxResolutionNode,
    "FluxSampler": FluxSampler,
    "FluxControlNetApply": FluxControlNetApply,
    "BooleanBasic": BooleanBasic,
    "BooleanReverse": BooleanReverse,
    "GetImageSizeRatio": GetImageSizeRatio,
    "NoisePlusBlend": NoisePlusBlend,
    "IntegerSettings": IntegerSettings,
    "ChooseUpscaleModel": ChooseUpscaleModel,  # Add new node to class mappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxResolutionNode": "Flux Resolution Calculator",
    "FluxSampler": "Flux Sampler",
    "FluxControlNetApply": "Flux ControlNet",
    "BooleanBasic": "Boolean Basic",
    "BooleanReverse": "Boolean Reverse",
    "GetImageSizeRatio": "Get Image Size & Ratio",
    "NoisePlusBlend": "Noise Plus Blend",
    "IntegerSettings": "Integer Settings",
    "ChooseUpscaleModel": "Choose Upscale Model",  # Add display name mapping for new node
}
