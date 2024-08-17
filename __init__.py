from .flux_resolution_cal_node import FluxResolutionNode
from .flux_sampler_node import FluxSampler
from .flux_controlnet_node import FluxControlNetApply

print("Initializing Flux Resolution Node, Flux Sampler Node, and Flux ControlNet Apply Node")

NODE_CLASS_MAPPINGS = {
    "FluxResolutionNode": FluxResolutionNode,
    "FluxSampler": FluxSampler,
    "FluxControlNetApply": FluxControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxResolutionNode": "Flux Resolution Calculator",
    "FluxSampler": "Flux Sampler",
    "FluxControlNetApply": "Flux ControlNet",
}