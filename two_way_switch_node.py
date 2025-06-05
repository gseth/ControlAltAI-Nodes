class AnyType(str):
    """A special string subclass that equals any other type for ComfyUI type checking."""
    def __ne__(self, __value: object) -> bool:
        return False

# Create an instance to use as the any type
any_type = AnyType("*")

class TwoWaySwitch:
    """Two-way switch that selects between two inputs based on selection setting."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selection_setting": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "input_1": (any_type,),
                "input_2": (any_type,),
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

    def switch_inputs(self, selection_setting=1, input_1=None, input_2=None):
        """
        Two-way switch that selects between two inputs based on the selection_setting.
        Compatible with IntegerSettings node:
        - selection_setting = 1 (Disable): selects input_1
        - selection_setting = 2 (Enable): selects input_2
        """
        if selection_setting == 2:
            # Enable state - select second input
            selected_output = input_2 if input_2 is not None else input_1
        else:
            # Disable state (1) or any other value - select first input
            selected_output = input_1 if input_1 is not None else input_2
        
        return (selected_output,)

NODE_CLASS_MAPPINGS = {
    "TwoWaySwitch": TwoWaySwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TwoWaySwitch": "Switch (Two Way)",
}