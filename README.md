# ComfyUI ControlAltAI Nodes

This repository contains custom nodes designed for the ComfyUI framework, focusing on quality-of-life improvements. These nodes aim to make tasks easier and more efficient. Currently, two Flux nodes are available that enhance functionality and streamline workflows within ComfyUI.

## Nodes

![ComfyUI Screenshot](https://gseth.com/images/SNAG-3894.png)

### Flux Resolution Calculator

The Flux Resolution Calculator is designed to work with the Flux model, which operates based on megapixels rather than standard SDXL resolutions. This node calculates the resolution according to the defined megapixels and the selected aspect ratio.

- **Supported Megapixels:** 0.1 MP, 1.0 MP, 2.0 MP, 2.1 MP, 2.2 MP, 2.3 MP, 2.4MP, 2.5MP
- **Note:** Generations above 1 MP may appear slightly blurry, but resolutions of 3k+ have been successfully tested on the Flux1.Dev model.

### Flux Sampler

The Flux Sampler node combines the functionality of the CustomSamplerAdvance node and input nodes into a single, streamlined node.

- **CFG Setting:** The CFG is fixed at 1.
- **Conditioning Input:** Only positive conditioning is supported.
- **Compatibility:** Only the samplers and schedulers compatible with the Flux model are included.

## YouTube ComfyUI Tutorials

We are a team of two and create extensive tutorials for ComfyUI. Check out our YouTube channel:

<a href="https://youtube.com/@controlaltai" style="display: inline-block; vertical-align: middle;">
  <img src="https://gseth.com/images/Youtube.svg" alt="YouTube Logo" width="50">
</a> [**ControlAltAI**](https://youtube.com/@controlaltai)

## License

This project is licensed under the MIT License.
