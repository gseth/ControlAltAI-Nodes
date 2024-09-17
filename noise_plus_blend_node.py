import numpy as np
from PIL import Image, ImageChops
import torch

class NoisePlusBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_scale": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 100.00, "step": 0.01}),
                "blend_opacity": ("INT", {"default": 20, "min": 0, "max": 100}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("blended_image_output", "noise_output")
    FUNCTION = "noise_plus_blend"
    
    CATEGORY = "ControlAltAI Nodes/Image"

    def tensor_to_pil(self, tensor_image):
        """Converts tensor to a PIL Image"""
        tensor_image = tensor_image.squeeze(0)  # Remove batch dimension if it exists
        pil_image = Image.fromarray((tensor_image.cpu().numpy() * 255).astype(np.uint8))
        return pil_image

    def pil_to_tensor(self, pil_image):
        """Converts a PIL image back to a tensor"""
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255).unsqueeze(0)

    def generate_gaussian_noise(self, width, height, noise_scale=0.05):
        """Generates Gaussian noise with a given scale."""
        noise = np.random.normal(128, 128 * noise_scale, (height, width, 3)).astype(np.uint8)
        return Image.fromarray(noise)

    def soft_light_blend(self, base_image, noise_image, mask=None, opacity=15):
        """Blends noise over the base image using soft light, applying mask if present."""
        # Resize noise to match base image size
        noise_image = noise_image.resize(base_image.size)

        base_image = base_image.convert('RGB')
        noise_image = noise_image.convert('RGB')

        noise_blended = ImageChops.soft_light(base_image, noise_image)
        blended_image = Image.blend(base_image, noise_blended, opacity / 100)

        # Apply mask only if it's provided, valid, and contains more than a single value
        if mask is not None:
            mask_pil = self.tensor_to_pil(mask).convert('L')
            mask_resized = mask_pil.resize(base_image.size)

            # Invert the mask by subtracting from 255
            inverted_mask = ImageChops.invert(mask_resized)

            # Apply the inverted mask to the composite blending
            blended_image = Image.composite(base_image, blended_image, inverted_mask)

        return blended_image

    def noise_plus_blend(self, image, noise_scale=0.05, blend_opacity=15, mask=None):
        """Main function to generate noise, blend, and return results."""
        # Convert Tensor image to PIL
        base_image = self.tensor_to_pil(image)
        image_size = base_image.size

        # Generate Gaussian noise with the size of the input image
        noise_image = self.generate_gaussian_noise(image_size[0], image_size[1], noise_scale)

        # Blend the noise with the base image using soft light
        blended_image = self.soft_light_blend(base_image, noise_image, mask, blend_opacity)

        # Convert the final blended image back to tensor
        noise_tensor = self.pil_to_tensor(noise_image)
        blended_tensor = self.pil_to_tensor(blended_image)

        # Return both the noise and blended image as tensors
        return blended_tensor, noise_tensor

NODE_CLASS_MAPPINGS = {
    "NoisePlusBlend": NoisePlusBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoisePlusBlend": "Noise Plus Blend",
}
