class GetImageSizeRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }
        
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "ratio")
    FUNCTION = "get_image_size_ratio"
    
    CATEGORY = "ControlAltAI Nodes/Image"
    
    def get_image_size_ratio(self, image):
        _, height, width, _ = image.shape
        
        gcd = self.greatest_common_divisor(width, height)
        ratio_width = width // gcd
        ratio_height = height // gcd
        
        ratio = f"{ratio_width}:{ratio_height}"
        
        return width, height, ratio
    
    def greatest_common_divisor(self, a, b):
        while b != 0:
            a, b = b, a % b
        return a

NODE_CLASS_MAPPINGS = {
    "GetImageSizeRatio": GetImageSizeRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageSizeRatio": "Get Image Size & Ratio",
}