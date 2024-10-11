from .flux_resolution_cal_node import FluxResolutionNode
from .flux_sampler_node import FluxSampler
from .flux_union_controlnet_node import FluxUnionControlNetApply
from .boolean_basic_node import BooleanBasic
from .boolean_reverse_node import BooleanReverse
from .get_image_size_ratio_node import GetImageSizeRatio
from .noise_plus_blend_node import NoisePlusBlend
from .integer_settings_node import IntegerSettings
from .choose_upscale_model_node import ChooseUpscaleModel

print("Initializing ControlAltAI Nodes")

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
}
