class BooleanBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "process_boolean"
    CATEGORY = "ControlAltAI Nodes/Logic"

    def process_boolean(self, boolean):
        return (boolean,)

NODE_CLASS_MAPPINGS = {
    "BooleanBasic": BooleanBasic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BooleanBasic": "Boolean Basic",
}