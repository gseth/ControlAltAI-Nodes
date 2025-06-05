class TextBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "passthrough_text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "bridge_text"
    
    CATEGORY = "ControlAltAI Nodes/Utility"

    def bridge_text(self, text_input="", passthrough_text=""):
        """
        Bridge function that allows editing of input text and passes it through as output.
        If passthrough_text is connected, it uses that as the base text.
        The text_input field allows manual editing/override.
        """
        # If passthrough_text is provided and text_input is empty or default, use passthrough
        if passthrough_text and (not text_input or text_input == ""):
            output_text = passthrough_text
        else:
            # Use the manually entered/edited text
            output_text = text_input
        
        return (output_text,)

NODE_CLASS_MAPPINGS = {
    "TextBridge": TextBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextBridge": "Text Bridge",
}