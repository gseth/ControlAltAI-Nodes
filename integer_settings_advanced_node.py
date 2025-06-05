class IntegerSettingsAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "setting_1": ("BOOLEAN", {"default": True, "label_on": "Enable", "label_off": "Disable"}),
                "setting_2": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
                "setting_3": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("setting_value",)
    FUNCTION = "integer_settings_advanced"

    CATEGORY = "ControlAltAI Nodes/Logic"

    def integer_settings_advanced(self, setting_1, setting_2, setting_3):
        """
        Returns integer based on which setting is enabled.
        Due to mutual exclusion (handled by JS), only one should be True.
        Priority order: setting_3 > setting_2 > setting_1
        """
        if setting_3:
            return (3,)
        elif setting_2:
            return (2,)
        else:
            # Default to 1 (setting_1 or fallback)
            return (1,)

NODE_CLASS_MAPPINGS = {
    "IntegerSettingsAdvanced": IntegerSettingsAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntegerSettingsAdvanced": "Integer Settings Advanced",
}