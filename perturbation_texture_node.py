import numpy as np
from PIL import Image, ImageChops
import torch

class PerturbationTexture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_scale": ("FLOAT", {"default": 0.5, "min": 0.00, "max": 1.00, "step": 0.01}),
                "texture_strength": ("INT", {"default": 50, "min": 0, "max": 100}),
                "texture_type": (["Film Grain", "Skin Pore", "Natural", "Fine Detail"], {"default": "Skin Pore"}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1}),
                "perturbation_factor": ("FLOAT", {"default": 0.30, "min": 0.01, "max": 0.5, "step": 0.01}),
                "use_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("textured_image_output", "texture_layer")
    FUNCTION = "apply_perturbation_texture"
    
    CATEGORY = "ControlAltAI Nodes/Image"

    def tensor_to_pil(self, tensor_image):
        """Converts tensor to a PIL Image"""
        tensor_image = tensor_image.squeeze(0)  # Remove batch dimension if it exists
        pil_image = Image.fromarray((tensor_image.cpu().numpy() * 255).astype(np.uint8))
        return pil_image

    def pil_to_tensor(self, pil_image):
        """Converts a PIL image back to a tensor"""
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255).unsqueeze(0)

    def generate_adaptive_texture(self, base_image, noise_scale, texture_type, frequency, perturbation_factor, texture_strength, seed=None):
        """Generate texture with adaptive color matching."""
        width, height = base_image.size
        
        # Set seed for reproducibility if provided
        if seed is not None and seed >= 0:
            np.random.seed(seed)
        
        # Convert base image to numpy array
        base_np = np.array(base_image).astype(np.float32) / 255.0
        
        # Generate noise patterns based on texture type
        noise_patterns = self.generate_noise_patterns(width, height, noise_scale, texture_type, frequency)
        
        # Convert noise to -1 to 1 range for proper mixing
        noise_normalized = (noise_patterns.astype(np.float32) - 128.0) / 128.0
        
        # Apply perturbation with texture_strength controlling the final intensity
        effective_perturbation = perturbation_factor * (texture_strength / 100.0)
        
        # Apply noise as color-matched variations around the base color
        result = base_np + (noise_normalized * effective_perturbation)
        
        # Clamp to valid range
        result = np.clip(result, 0, 1)
        
        # Create a more visible texture layer for preview/debugging
        texture_layer = base_np + (noise_normalized * perturbation_factor * 2.0)  
        texture_layer = np.clip(texture_layer, 0, 1)
        
        final_image = Image.fromarray((result * 255).astype(np.uint8))
        texture_image = Image.fromarray((texture_layer * 255).astype(np.uint8))
        
        return final_image, texture_image

    def generate_noise_patterns(self, width, height, noise_scale, texture_type, frequency):
        """Generates noise patterns optimized for each texture type."""
        
        # Safe resize function for noise scaling
        def safe_resize(arr, target_height, target_width):
            from PIL import Image
            if arr.ndim == 2:
                img = Image.fromarray((arr * 255 / arr.max()).astype(np.uint8))
            else:
                img = Image.fromarray(arr.astype(np.uint8))
            img = img.resize((target_width, target_height), Image.LANCZOS)
            return np.array(img).astype(np.float32) / 255.0 * arr.max()
        
        if texture_type == "Film Grain":
            # Film grain - larger, more irregular pattern with RGB variation
            base_noise_r = np.random.normal(128, 64 * noise_scale, (height, width))
            base_noise_g = np.random.normal(128, 64 * noise_scale, (height, width))
            base_noise_b = np.random.normal(128, 64 * noise_scale, (height, width))
            
            # Add larger scale variation for film-like clustering
            large_scale_h = max(4, int(height/(4*frequency)))
            large_scale_w = max(4, int(width/(4*frequency)))
            large_scale_r = np.random.normal(0, 30 * noise_scale, (large_scale_h, large_scale_w))
            large_scale_g = np.random.normal(0, 30 * noise_scale, (large_scale_h, large_scale_w))
            large_scale_b = np.random.normal(0, 30 * noise_scale, (large_scale_h, large_scale_w))
            
            large_scale_r = safe_resize(large_scale_r, height, width)
            large_scale_g = safe_resize(large_scale_g, height, width)
            large_scale_b = safe_resize(large_scale_b, height, width)
            
            combined_r = np.clip(base_noise_r * 0.7 + large_scale_r * 0.3, 0, 255)
            combined_g = np.clip(base_noise_g * 0.7 + large_scale_g * 0.3, 0, 255)
            combined_b = np.clip(base_noise_b * 0.7 + large_scale_b * 0.3, 0, 255)
            
        elif texture_type == "Skin Pore":
            # Fine, subtle texture optimized for skin with reduced intensity
            base_scale = noise_scale * 0.6  # More subtle for natural skin texture
            
            # Create subtle RGB variations for realistic skin texture
            base_noise_r = np.random.normal(128, 32 * base_scale, (height, width))
            base_noise_g = np.random.normal(128, 28 * base_scale, (height, width))
            base_noise_b = np.random.normal(128, 24 * base_scale, (height, width))
            
            # Fine pore-like details at higher frequency
            fine_h = max(4, int(height*frequency*1.5))
            fine_w = max(4, int(width*frequency*1.5))
            fine_noise_r = np.random.normal(0, 20 * base_scale, (fine_h, fine_w))
            fine_noise_g = np.random.normal(0, 18 * base_scale, (fine_h, fine_w))
            fine_noise_b = np.random.normal(0, 16 * base_scale, (fine_h, fine_w))
            
            fine_noise_r = safe_resize(fine_noise_r, height, width)
            fine_noise_g = safe_resize(fine_noise_g, height, width)
            fine_noise_b = safe_resize(fine_noise_b, height, width)
            
            combined_r = np.clip(base_noise_r + fine_noise_r * 0.8, 0, 255)
            combined_g = np.clip(base_noise_g + fine_noise_g * 0.8, 0, 255)
            combined_b = np.clip(base_noise_b + fine_noise_b * 0.8, 0, 255)
            
        elif texture_type == "Natural":
            # Multi-layered natural texture with organic frequency distribution
            base_noise_r = np.random.normal(128, 48 * noise_scale, (height, width))
            base_noise_g = np.random.normal(128, 44 * noise_scale, (height, width))
            base_noise_b = np.random.normal(128, 40 * noise_scale, (height, width))
            
            # Multiple frequency layers for natural complexity
            frequencies = [frequency*2, frequency, frequency/3]
            weights = [0.5, 0.3, 0.2]
            
            combined_r = base_noise_r.copy()
            combined_g = base_noise_g.copy()
            combined_b = base_noise_b.copy()
            
            for freq, weight in zip(frequencies, weights):
                f_h = max(4, int(height*freq))
                f_w = max(4, int(width*freq))
                
                layer_r = np.random.normal(0, 30 * noise_scale * weight, (f_h, f_w))
                layer_g = np.random.normal(0, 28 * noise_scale * weight, (f_h, f_w))
                layer_b = np.random.normal(0, 26 * noise_scale * weight, (f_h, f_w))
                
                layer_r = safe_resize(layer_r, height, width)
                layer_g = safe_resize(layer_g, height, width)
                layer_b = safe_resize(layer_b, height, width)
                
                combined_r += layer_r * weight
                combined_g += layer_g * weight
                combined_b += layer_b * weight
            
            combined_r = np.clip(combined_r, 0, 255)
            combined_g = np.clip(combined_g, 0, 255)
            combined_b = np.clip(combined_b, 0, 255)
            
        else:  # Fine Detail
            # High-frequency detailed texture for micro-details
            high_freq = frequency * 2.5
            
            base_noise_r = np.random.normal(128, 40 * noise_scale, (height, width))
            base_noise_g = np.random.normal(128, 38 * noise_scale, (height, width))
            base_noise_b = np.random.normal(128, 36 * noise_scale, (height, width))
            
            # High-frequency fine details
            fine_h = max(4, int(height*high_freq))
            fine_w = max(4, int(width*high_freq))
            fine_detail_r = np.random.normal(0, 25 * noise_scale, (fine_h, fine_w))
            fine_detail_g = np.random.normal(0, 23 * noise_scale, (fine_h, fine_w))
            fine_detail_b = np.random.normal(0, 21 * noise_scale, (fine_h, fine_w))
            
            fine_detail_r = safe_resize(fine_detail_r, height, width)
            fine_detail_g = safe_resize(fine_detail_g, height, width)
            fine_detail_b = safe_resize(fine_detail_b, height, width)
            
            combined_r = np.clip(base_noise_r + fine_detail_r * 0.7, 0, 255)
            combined_g = np.clip(base_noise_g + fine_detail_g * 0.7, 0, 255)
            combined_b = np.clip(base_noise_b + fine_detail_b * 0.7, 0, 255)
        
        # Stack RGB channels into final noise pattern
        return np.stack([combined_r, combined_g, combined_b], axis=2)

    def apply_perturbation_texture(self, image, noise_scale=0.5, texture_strength=50, texture_type="Skin Pore", 
                                  frequency=1.0, perturbation_factor=0.15, use_mask=False, mask=None, seed=-1):
        """Main function to apply adaptive color-matched texture."""
        # Convert tensor image to PIL
        base_image = self.tensor_to_pil(image)
        
        # Use provided seed or generate random if -1
        seed_value = seed if seed >= 0 else None
        
        # Generate adaptive texture
        textured_image, texture_layer = self.generate_adaptive_texture(
            base_image, noise_scale, texture_type, frequency, 
            perturbation_factor, texture_strength, seed_value
        )
        
        # Apply mask if specified
        if use_mask and mask is not None:
            mask_pil = self.tensor_to_pil(mask).convert('L')
            mask_resized = mask_pil.resize(base_image.size)
            # Invert mask so white areas get texture, black areas are protected
            inverted_mask = ImageChops.invert(mask_resized)
            # Composite: base where mask is black, textured where mask is white
            textured_image = Image.composite(base_image, textured_image, inverted_mask)
        
        # Convert results back to tensors
        texture_tensor = self.pil_to_tensor(texture_layer)
        textured_tensor = self.pil_to_tensor(textured_image)
        
        return textured_tensor, texture_tensor

NODE_CLASS_MAPPINGS = {
    "PerturbationTexture": PerturbationTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbationTexture": "Perturbation Texture",
}