# ControlAltAI Nodes
Overview
This repository contains custom nodes designed for the ComfyUI framework, focusing on quality-of-life improvements. These nodes aim to make tasks easier and more efficient. Currently, two Flux nodes are available that enhance functionality and streamline workflows within ComfyUI.
Nodes
Flux Resolution Calculator
The Flux Resolution Calculator works with the Flux model, which operates based on megapixels rather than standard SDXL resolutions. This node calculates the resolution according to the defined megapixels and the selected aspect ratio.
Supported Megapixels: 0.1 MP, 1.0 MP, 2.0 MP
Note: Generations above 1 MP may appear slightly blurry, but resolutions of 3k+ have been successfully tested on the Flux1.Dev model.
Flux Sampler
The Flux Sampler node merges the functionality of the CustomSamplerAdvance node and input nodes into a single, streamlined node.
CFG Setting: The CFG is fixed at a value of 1.
Conditioning Input: Only positive conditioning is supported.
Compatibility: Includes only the samplers and schedulers compatible with the Flux model.
Usage
To use these nodes, integrate them into your ComfyUI setup and configure them according to your project requirements. Ensure that your environment is compatible with the Flux model specifications.
License
This project is licensed under the MIT License. Feel free to adjust any sections to better fit your project's details or requirements.
