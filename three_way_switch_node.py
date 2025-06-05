class AnyType(str):
    """A special string subclass that equals any other type for ComfyUI type checking."""
    def __ne__(self, __value: object) -> bool:
        return False

# Create an instance to use as the any type
any_type = AnyType("*")

class ThreeWaySwitch:
    """Three-way switch that selects between three inputs based on selection setting."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selection_setting": ("INT", {"default": 1, "min": 1, "max": 3}),
            },
            "optional": {
                "input_1": (any_type,),
                "input_2": (any_type,),
                "input_3": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch_inputs"
    
    CATEGORY = "ControlAltAI Nodes/Logic"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Allow any input types."""
        return True

    def switch_inputs(self, selection_setting=1, input_1=None, input_2=None, input_3=None):
        """
        Three-way switch that selects between three inputs based on the selection_setting.
        Compatible with IntegerSettingsAdvanced node:
        - selection_setting = 1: selects input_1
        - selection_setting = 2: selects input_2
        - selection_setting = 3: selects input_3
        """
        if selection_setting == 2:
            # Second option - select input_2, fallback to input_1, then input_3
            selected_output = input_2 if input_2 is not None else (input_1 if input_1 is not None else input_3)
        elif selection_setting == 3:
            # Third option - select input_3, fallback to input_1, then input_2
            selected_output = input_3 if input_3 is not None else (input_1 if input_1 is not None else input_2)
        else:
            # Default/First option (1) - select input_1, fallback to input_2, then input_3
            selected_output = input_1 if input_1 is not None else (input_2 if input_2 is not None else input_3)
        
        return (selected_output,)

NODE_CLASS_MAPPINGS = {
    "ThreeWaySwitch": ThreeWaySwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThreeWaySwitch": "Switch (Three Way)",
}