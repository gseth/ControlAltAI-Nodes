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

![ComfyUI Screenshot](https://gseth.com/images/SNAG-3960.png)

### Get Image Size & Ratio
This node is designed to get the image resolution in width, height, and ratio. The node can be further connected to the Flux Resolution Calculator. To do so, follow the following steps:
- Right-click on the Flux Resolution Calculator -- > Convert widget to input -- > Convert custom_aspect_ratio to input.
- Connect Ratio output to custom_aspect_ratio input.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-3959.png)

## YouTube ComfyUI Tutorials

We are a team of two and create extensive tutorials for ComfyUI. Check out our YouTube channel:</br>
<a href="https://youtube.com/@controlaltai">ControlAltAI YouTube Tutorials</a>

## Black Forest Labs AI

Black Forest Labs, a pioneering AI research organization, has developed the Flux model series, which includes the Flux1.[dev] and Flux1.[schnell] models. These models are designed to push the boundaries of image generation through advanced deep-learning techniques.

For more details on these models, their capabilities, and licensing information, you can visit the <a href="https://blackforestlabs.ai/">Black Forest Labs website</a>


## Apply ControlNet Node

The Apply ControlNet Node is a placeholder node compatible with the Flux Sampler. ControlNet won't work at the moment. The ControlNets from XLabs AI are quite different, and their nodes use a transformer-based architecture. https://github.com/XLabs-AI/x-flux-comfyui


## License

This project is licensed under the MIT License.
