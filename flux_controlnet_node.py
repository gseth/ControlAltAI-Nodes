class FluxControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "flux_controlnet"
    CATEGORY = "ControlAltAI Nodes/Flux"

    def flux_controlnet(self, conditioning, control_net, image, strength):
        if strength == 0:
            return (conditioning,)

        c = []
        control_hint = image.movedim(-1, 1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = False  # This ensures it's only applied to positive
            c.append(n)
        
        return (c,)

NODE_CLASS_MAPPINGS = {
    "FluxControlNetApply": FluxControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxControlNetApply": "Flux ControlNet",
}