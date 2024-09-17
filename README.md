# ComfyUI ControlAltAI Nodes

This repository contains custom nodes designed for the ComfyUI framework, focusing on quality-of-life improvements. These nodes aim to make tasks easier and more efficient. Two Flux nodes are available to enhance functionality and streamline workflows within ComfyUI.

## Nodes

### List of Nodes:
- Flux
  - Flux Resolution Calculator
  - Flux Sampler
  - Flux ControlNet (work in progress)
- Logic
  - Boolean Basic
  - Boolean Reverse
- Image
  - Get Image Size & Ratio

### Flux Resolution Calculator

The Flux Resolution Calculator is designed to determine the optimal image resolution for outputs generated using the Flux model, which is notably more oriented towards megapixels. Unlike traditional methods that rely on standard SDXL resolutions, this calculator operates based on user-specified megapixel inputs. Users can select their desired megapixel count, ranging from 0.1 to 2.0 megapixels, and aspect ratio. The calculator then provides the exact image dimensions necessary for optimal performance with the Flux model. This approach ensures that the generated images meet specific quality and size requirements tailored to the user's needs. Additionally, while the official limit is 2.0 megapixels, during testing, I have successfully generated images at higher resolutions, indicating the model's flexibility in accommodating various image dimensions without compromising quality.

- **Supported Megapixels:** 0.1 MP, 1.0 MP, 2.0 MP, 2.1 MP, 2.2 MP, 2.3 MP, 2.4MP, 2.5MP
- **Note:** Generations above 1 MP may appear slightly blurry, but resolutions of 3k+ have been successfully tested on the Flux1.Dev model.
- **Custom Ratio:** Custom Ratio is now supported. Enable or Disable Custom Ratio and input any ratio. (Example: 4:9)

### Flux Sampler

The Flux Sampler node combines the functionality of the CustomSamplerAdvance node and input nodes into a single, streamlined node.

- **CFG Setting:** The CFG is fixed at 1.
- **Conditioning Input:** Only positive conditioning is supported.
- **Compatibility:** Only the samplers and schedulers compatible with the Flux model are included.
- **Latent Compatibility:** Use SD3 Empty Latent Image only. The normal empty latent image node is not compatible.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-3957.png)

### Get Image Size & Ratio
This node is designed to get the image resolution in width, height, and ratio. The node can be further connected to the Flux Resolution Calculator. To do so, follow the following steps:
- Right-click on the Flux Resolution Calculator -- > Convert widget to input -- > Convert custom_aspect_ratio to input.
- Connect Ratio output to custom_aspect_ratio input.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-3959.png)

### Integer Setting
This node is designed to give output as a raw value of 1 or 2 integers. Enable = 2, Disable = 1.

Use case: This can be set up before a two-way switch, allowing workflow logical control to flow in one or the other direction. As of now, it only controls two logical flows. In the future, we will upgrade the node to support three or more logical switch flows.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4239.png)

### Choose Upscale Model
A simple node that can be connected with a boolean logic. A true response will use upscale model 1, and a false response will use upscale model 2.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4240.png)

### Noise Plus Blend
This node will generate a Gaussian blur noise based on the dimensions of the input image and will blend the noise into the entire image or only the mask region.

**Issue:** Generated faces/landscapes are realistic, but during upscale, the AI model smoothens the skin or texture, making it look plastic or adding smooth fine lines.

**Solution:** For upscaling, auto segment or manually mask the face or specified regions and add noise. Then, pass the blended image output to the K-Sampler and denoise at 0.20 - 0.50.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4241.png)

You can see the noise has been applied only to the face as per the mask. This will maintain the smooth bokeh and preserve the facial details during upscale.
![ComfyUI Screenshot](https://gseth.com/images/SNAG-4242.png)

Denoise the image using Flux or SDXL sampler. Recommended sampler denoise: 0.10 - 0.50
![ComfyUI Screenshot](https://gseth.com/images/SNAG-4243.png)

**Settings:**<br>
noise_scale: 0.30 - 0.50.<br>
blend_opacity: 10-25.

If you find too many artifacts on the skin or other textures, reduce both values. Increase the values if upscaling output results in plastic, velvet-like smooth lines.

**Best Setting for AI-generated Faces:**<br>
noise_scale: 0.40-0.50.<br>
blend_opacity: 15-25.

**Best Setting for AI-generated texture (landscapes):**<br>
noise_scale: 0.30.<br>
blend_opacity: 12-15.

Results:
**Example 1**<br>
Without Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/without_noise_blend_1.png)

With Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/with_noise_blend_1.png)

**Example 2**<br>
Without Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/without_noise_blend_2.png)

With Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/with_noise_blend_2.png)

**Example 3**<br>
Without Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/without_noise_blend_3.png)

With Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/with_noise_blend_3.png)

**Example 4**<br>
Without Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/without_noise_blend_4.png)

With Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/with_noise_blend_4.png)

## YouTube ComfyUI Tutorials

We are a team of two and create extensive tutorials for ComfyUI. Check out our YouTube channel:</br>
<a href="https://youtube.com/@controlaltai">ControlAltAI YouTube Tutorials</a>

## Black Forest Labs AI

Black Forest Labs, a pioneering AI research organization, has developed the Flux model series, which includes the Flux1.[dev] and Flux1.[schnell] models. These models are designed to push the boundaries of image generation through advanced deep-learning techniques.

For more details on these models, their capabilities, and licensing information, you can visit the <a href="https://blackforestlabs.ai/">Black Forest Labs website</a>


## Apply ControlNet Node

Work in Progress


## License

This project is licensed under the MIT License.
