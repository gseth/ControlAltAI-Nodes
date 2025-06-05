from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

def pil2tensor(image):
    """Convert PIL image to tensor in the correct format"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class HiDreamResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": ([
                    "1:1 (Perfect Square)",
                    "3:4 (Standard Portrait)",
                    "2:3 (Classic Portrait)",
                    "9:16 (Widescreen Portrait)",
                    "4:3 (Standard Landscape)",
                    "3:2 (Classic Landscape)",
                    "16:9 (Widescreen Landscape)",
                ], {"default": "1:1 (Perfect Square)"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("width", "height", "resolution", "preview")
    FUNCTION = "get_dimensions"
    CATEGORY = "ControlAltAI Nodes/HiDream"
    OUTPUT_NODE = True

    def create_preview_image(self, width, height, resolution):
        # 1024x1024 preview size
        preview_size = (1024, 1024)
        image = Image.new('RGB', preview_size, (0, 0, 0))  # Black background
        draw = ImageDraw.Draw(image)

        # Draw grid with grey lines
        grid_color = '#333333'  # Dark grey for grid
        grid_spacing = 50  # Adjusted grid spacing
        for x in range(0, preview_size[0], grid_spacing):
            draw.line([(x, 0), (x, preview_size[1])], fill=grid_color)
        for y in range(0, preview_size[1], grid_spacing):
            draw.line([(0, y), (preview_size[0], y)], fill=grid_color)

        # Calculate preview box dimensions
        preview_width = 800  # Increased size
        preview_height = int(preview_width * (height / width))
        
        # Adjust if height is too tall
        if preview_height > 800:  # Adjusted for larger preview
            preview_height = 800
            preview_width = int(preview_height * (width / height))

        # Calculate center position
        x_offset = (preview_size[0] - preview_width) // 2
        y_offset = (preview_size[1] - preview_height) // 2

        # Draw the aspect ratio box with thicker outline
        draw.rectangle(
            [(x_offset, y_offset), (x_offset + preview_width, y_offset + preview_height)],
            outline='red',
            width=4  # Thicker outline
        )

        # Add text with larger font sizes
        try:
            # Draw text (centered)
            text_y = y_offset + preview_height//2
            
            # Resolution text in red
            draw.text((preview_size[0]//2, text_y), 
                     f"{width}x{height}", 
                     fill='red', 
                     anchor="mm",
                     font=ImageFont.truetype("arial.ttf", 48))
            
        except:
            # Fallback if font loading fails
            draw.text((preview_size[0]//2, text_y), f"{width}x{height}", fill='red', anchor="mm")

        # Convert to tensor using the helper function
        return pil2tensor(image)

    def get_dimensions(self, resolution):
        # Map from aspect ratio to actual dimensions
        resolution_map = {
            "1:1 (Perfect Square)": (1024, 1024),
            "3:4 (Standard Portrait)": (880, 1168),
            "2:3 (Classic Portrait)": (832, 1248),
            "9:16 (Widescreen Portrait)": (768, 1360),
            "4:3 (Standard Landscape)": (1168, 880),
            "3:2 (Classic Landscape)": (1248, 832),
            "16:9 (Widescreen Landscape)": (1360, 768)
        }
        
        # Get dimensions from the map
        width, height = resolution_map[resolution]
        
        # Resolution as string
        resolution_str = f"{width} x {height}"
        
        # Generate preview image
        preview = self.create_preview_image(width, height, resolution_str)
        
        return width, height, resolution_str, preview

def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

NODE_CLASS_MAPPINGS = {
    "HiDreamResolutionNode": HiDreamResolutionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamResolutionNode": "HiDream Resolution",
}