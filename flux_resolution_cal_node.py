class FluxResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixel": (["0.1", "0.5", "1.0", "1.5", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5"], {"default": "1.0"}),
                "aspect_ratio": ([
                    "1:1 (Square)",
                    "2:3 (Portrait)", "4:5 (Portrait)", "5:8 (Portrait)",
                    "9:16 (Portrait)", "9:19 (Tall)", "9:21 (Tall)", "9:32 (Tall)",
                    "3:2 (Landscape)", "5:4 (Landscape)", "8:5 (Landscape)",
                    "16:9 (Landscape)", "19:9 (Ultrawide)", "21:9 (Ultrawide)", "32:9 (Ultrawide)"
                ], {"default": "1:1 (Square)"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "resolution")
    FUNCTION = "calculate_dimensions"
    CATEGORY = "ControlAltAI Nodes/Flux"
    OUTPUT_NODE = True

    def calculate_dimensions(self, megapixel, aspect_ratio):
        megapixel = float(megapixel)
        # Extract the numeric ratio
        numeric_ratio = aspect_ratio.split(' ')[0]
        width_ratio, height_ratio = map(int, numeric_ratio.split(':'))
        total_pixels = megapixel * 1_000_000
        dimension = (total_pixels / (width_ratio * height_ratio)) ** 0.5
        width = int(dimension * width_ratio)
        height = int(dimension * height_ratio)

        # Determine rounding factor based on megapixel value
        if megapixel in [0.1, 0.5, 1.0, 1.5]:
            round_to = 8
        if megapixel in [1.0, 1.5]:
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
