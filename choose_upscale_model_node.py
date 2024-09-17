class ChooseUpscaleModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model_1": ("UPSCALE_MODEL",),
                "upscale_model_2": ("UPSCALE_MODEL",),
                "use_model_1": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    RETURN_NAMES = ("upscale_model",)
    FUNCTION = "choose_upscale_model"

    CATEGORY = "ControlAltAI Nodes/Logic"

    def choose_upscale_model(self, upscale_model_1, upscale_model_2, use_model_1):
        if use_model_1:
            return (upscale_model_1,)
        else:
            return (upscale_model_2,)

NODE_CLASS_MAPPINGS = {
    "ChooseUpscaleModel": ChooseUpscaleModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChooseUpscaleModel": "Choose Upscale Model",
}
