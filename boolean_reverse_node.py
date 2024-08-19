class BooleanReverse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "reverse_boolean"
    CATEGORY = "ControlAltAI Nodes/Logic"

    def reverse_boolean(self, boolean):
        return (not boolean,)

NODE_CLASS_MAPPINGS = {
    "BooleanReverse": BooleanReverse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BooleanReverse": "Boolean Reverse",
}