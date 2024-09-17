class IntegerSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "setting": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("setting_value",)
    FUNCTION = "integer_settings"

    CATEGORY = "ControlAltAI Nodes/Logic"

    def integer_settings(self, setting):
        # Handle the single boolean setting
        status = 2 if setting else 1
        return (status,)


NODE_CLASS_MAPPINGS = {
    "IntegerSettings": IntegerSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntegerSettings": "Integer Settings",
}
