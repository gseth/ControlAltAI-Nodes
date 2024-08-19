from .flux_resolution_cal_node import FluxResolutionNode
from .flux_sampler_node import FluxSampler
from .flux_controlnet_node import FluxControlNetApply
from .boolean_basic_node import BooleanBasic
from .boolean_reverse_node import BooleanReverse

print("Initializing ControlAltAI Nodes")

NODE_CLASS_MAPPINGS = {
    "FluxResolutionNode": FluxResolutionNode,
    "FluxSampler": FluxSampler,
    "FluxControlNetApply": FluxControlNetApply,
    "BooleanBasic": BooleanBasic,
    "BooleanReverse": BooleanReverse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxResolutionNode": "Flux Resolution Calculator",
    "FluxSampler": "Flux Sampler",
    "FluxControlNetApply": "Flux ControlNet",
    "BooleanBasic": "Boolean Basic",
    "BooleanReverse": "Boolean Reverse",
}