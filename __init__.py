from .flux_resolution_cal_node import FluxResolutionNode
from .flux_sampler_node import FluxSampler

print("Initializing Flux Resolution Node and Flux Sampler Node...")

# Registering the nodes
NODE_CLASS_MAPPINGS = {
    "FluxResolutionNode": FluxResolutionNode,
    "FluxSampler": FluxSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxResolutionNode": "Flux Resolution Calculator",
    "FluxSampler": "Flux Sampler",
}