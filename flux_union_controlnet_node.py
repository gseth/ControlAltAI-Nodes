import torch
import comfy
import folder_paths

class FluxUnionControlNetApply:
    # Correct UNION_CONTROLNET_TYPES mapping
    UNION_CONTROLNET_TYPES = {
        "canny": 0,
        "tile": 1,
        "depth": 2,
        "blur": 3,
        "pose": 4,
        "gray": 5,
        "low quality": 6,
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "union_controlnet_type": (list(s.UNION_CONTROLNET_TYPES.keys()), ),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001
                }),
                "vae": ("VAE", ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "VAE")
    FUNCTION = "apply_flux_union_controlnet"
    CATEGORY = "ControlAltAI Nodes/Flux"

    def apply_flux_union_controlnet(self, conditioning, control_net, image, union_controlnet_type, strength, start_percent, end_percent, vae):
        if strength == 0:
            return (conditioning, vae)

        # Map the 'union_controlnet_type' to 'control_type'
        control_type = self.UNION_CONTROLNET_TYPES[union_controlnet_type]
        control_type_list = [control_type]

        # Set the 'control_type' using 'set_extra_arg'
        control_net = control_net.copy()
        control_net.set_extra_arg("control_type", control_type_list)

        # Process the image to get 'control_hint'
        control_hint = image.movedim(-1, 1)  # Assuming the image is in HWC format

        # Apply the ControlNet to the positive conditioning
        cnets = {}
        c = []
        for t in conditioning:
            d = t[1].copy()
            prev_cnet = d.get('control', None)

            # Create a unique key for caching
            cache_key = (prev_cnet, tuple(control_net.extra_args.get('control_type', [])))

            if cache_key in cnets:
                c_net_instance = cnets[cache_key]
            else:
                # Create a copy of the 'control_net' and set the conditional hint
                c_net_instance = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae)
                c_net_instance.set_previous_controlnet(prev_cnet)
                cnets[cache_key] = c_net_instance

            d['control'] = c_net_instance
            d['control_apply_to_uncond'] = False

            n = [t[0], d]
            c.append(n)

        return (c, vae)

NODE_CLASS_MAPPINGS = {
    "FluxUnionControlNetApply": FluxUnionControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxUnionControlNetApply": "Flux Union ControlNet",
}
