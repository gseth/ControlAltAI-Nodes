import comfy.samplers
import torch
import comfy.sample
import latent_preview

FLUX_SAMPLER_NAMES = [
    "euler", "heun", "heunpp2", "dpm_2", "lms", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2m", 
    "ipndm", "ipndm_v", "deis", "ddim", "uni_pc", "uni_pc_bh2"
]

FLUX_SCHEDULER_NAMES = ["simple", "normal", "sgm_uniform", "ddim_uniform", "beta"]

class FluxSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "sampler_name": (FLUX_SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (FLUX_SCHEDULER_NAMES, {"default": "beta"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_seed": ("INT", {"default": 143220275975594, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "ControlAltAI Nodes/Flux"

    def sample(self, model, conditioning, latent_image, sampler_name, scheduler, steps, denoise, noise_seed):
        device = comfy.model_management.get_torch_device()
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise)

        latent = latent_image.copy()
        latent_image = latent["samples"]

        # Handle noise_mask if present
        noise_mask = latent.get("noise_mask", None)

        noise = comfy.sample.prepare_noise(latent_image, noise_seed)
        
        positive = conditioning
        negative = []  # Empty list for negative conditioning

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = sampler.sample(noise, positive, negative, cfg=1.0, latent_image=latent_image,
                                 force_full_denoise=True, denoise_mask=noise_mask, callback=callback,
                                 disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        return (out,)

NODE_CLASS_MAPPINGS = {
    "FluxSampler": FluxSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxSampler": "Flux Sampler"
}