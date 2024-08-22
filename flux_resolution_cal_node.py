class FluxResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixel": (["0.1", "0.5", "1.0", "1.5", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5"], {"default": "1.0"}),
                "aspect_ratio": ([
                    "1:1 (Perfect Square)",
                    "2:3 (Classic Portrait)", "3:4 (Golden Ratio)", "3:5 (Elegant Vertical)", "4:5 (Artistic Frame)", "5:7 (Balanced Portrait)", "5:8 (Tall Portrait)",
                    "7:9 (Modern Portrait)", "9:16 (Slim Vertical)", "9:19 (Tall Slim)", "9:21 (Ultra Tall)", "9:32 (Skyline)",
                    "3:2 (Golden Landscape)", "4:3 (Classic Landscape)", "5:3 (Wide Horizon)", "5:4 (Balanced Frame)", "7:5 (Elegant Landscape)", "8:5 (Cinematic View)",
                    "9:7 (Artful Horizon)", "16:9 (Panorama)", "19:9 (Cinematic Ultrawide)", "21:9 (Epic Ultrawide)", "32:9 (Extreme Ultrawide)"
                ], {"default": "1:1 (Perfect Square)"}),
                "custom_ratio": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
            },
            "optional": {
                "custom_aspect_ratio": ("STRING", {"default": "1:1"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "resolution")
    FUNCTION = "calculate_dimensions"
    CATEGORY = "ControlAltAI Nodes/Flux"
    OUTPUT_NODE = True

    def calculate_dimensions(self, megapixel, aspect_ratio, custom_ratio, custom_aspect_ratio=None):
        megapixel = float(megapixel)
        
        if custom_ratio and custom_aspect_ratio:
            numeric_ratio = custom_aspect_ratio
        else:
            numeric_ratio = aspect_ratio.split(' ')[0]
        
        width_ratio, height_ratio = map(int, numeric_ratio.split(':'))
        
        total_pixels = megapixel * 1_000_000
        dimension = (total_pixels / (width_ratio * height_ratio)) ** 0.5
        width = int(dimension * width_ratio)
        height = int(dimension * height_ratio)

        # Apply rounding logic based on megapixel value
        if megapixel in [0.1, 0.5]:
            round_to = 8
        elif megapixel in [1.0, 1.5]:
            round_to = 64
        else:  # 2.0 and above
            round_to = 32

        width = round(width / round_to) * round_to
        height = round(height / round_to) * round_to

        resolution = f"{width} x {height}"
        
        return width, height, resolution

NODE_CLASS_MAPPINGS = {
    "FluxResolutionNode": FluxResolutionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxResolutionNode": "Flux Resolution Calculator",
}