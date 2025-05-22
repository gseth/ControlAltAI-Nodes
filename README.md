### Requirements Update: 8 Dec 2024: Flux Attention Control Node requires XFormers. Check your version of PyTorch and install a compatible version of XFormers. Please follow the instructions here: <a href="https://github.com/gseth/ControlAltAI-Nodes/blob/master/xformers_instructions.txt">xformers_instructions</a>

# ComfyUI ControlAltAI Nodes

This repository contains custom nodes designed for the ComfyUI framework, focusing on quality-of-life improvements. These nodes aim to make tasks easier and more efficient. Two Flux nodes are available to enhance functionality and streamline workflows within ComfyUI.

## Nodes

### List of Nodes:
- Flux
  - Flux Resolution Calculator (Updated May 2025)
  - Flux Sampler
  - Flux Union ControlNet Apply
- Logic
  - Boolean Basic
  - Boolean Reverse
  - Integer Settings
  - Choose Upscale Model
- Image
  - Get Image Size & Ratio
  - Noise Plus Blend
- Flux Region
  - Region Mask Generator
  - Region Mask Processor
  - Region Mask Validator
  - Region Mask Conditioning
  - Flux Attention Control
  - Region Overlay Visualizer
  - Flux Attention Cleanup

### Flux Resolution Calculator

The Flux Resolution Calculator is designed to determine the optimal image resolution for outputs generated using the Flux model, which is notably more oriented towards megapixels. Unlike traditional methods that rely on standard SDXL resolutions, this calculator operates based on user-specified megapixel inputs. Users can select their desired megapixel count, ranging from 0.1 to 2.0 megapixels, and aspect ratio. The calculator then provides the exact image dimensions necessary for optimal performance with the Flux model. This approach ensures that the generated images meet specific quality and size requirements tailored to the user's needs. Additionally, while the official limit is 2.0 megapixels, during testing, I have successfully generated images at higher resolutions, indicating the model's flexibility in accommodating various image dimensions without compromising quality.

- **Supported Megapixels:** 0.1 MP - 2.5 MP (change stepping to 0.1 for fine-tuned selection)
- **Note:** Generations above 1 MP may appear slightly blurry, but resolutions of 3k+ have been successfully tested on the Flux1.Dev model.
- **Custom Ratio:** Custom Ratio is now supported. Enable or Disable the Custom Ratio and input any ratio. (Example: 4:9).
- **Preview:** The preview node is just a visual representation of the ratio.
- **Divisible By:** You can now choose the divisibility by 8/16/32/64. By default, it is 64. To get fine-tuned results, choose divisibility by 8. Divisibility by 32/64 is recommended for Flux Dev 1.

### Flux Sampler

The Flux Sampler node combines the functionality of the CustomSamplerAdvance node and input nodes into a single, streamlined node.

- **CFG Setting:** The CFG is fixed at 1.
- **Conditioning Input:** Only positive conditioning is supported.
- **Compatibility:** Only the samplers and schedulers compatible with the Flux model are included.
- **Latent Compatibility:** Use SD3 Empty Latent Image only. The normal empty latent image node is not compatible.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-7361.png)
![ComfyUI Screenshot](https://gseth.com/images/SNAG-7362.png)
![ComfyUI Screenshot](https://gseth.com/images/SNAG-7363.png)

### Flux Union ControlNet Apply

The Flux Union ControlNet Apply node is an all-in-one node compatible with InstanX Union Pro ControlNet. It has been tested extensively with the union controlnet type and works as intended. You can combine two ControlNet Union units and get good results. Not recommended to combine more than two. The ControlNet is tested only on the Flux 1.Dev Model.

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4402.png)

**Recommended Settings:**<br>
strength: 0.15-0.65.<br>
end percentage: 0.200 - 0.900.

**Recommended PreProcessors:**<br>
Canny: Canny Edge (ControlNet Aux).<br>
Tile: Tile (ControlNet Aux).<br>
Depth: Depth Anything V2 Relative (ControlNet Aux).<br>
Blue: Direct Input (Blurry Image) or Tile (ControlNet Aux).<br>
Pose: DWPose Estimator (ControlNet Aux).<br>
Gray: Image Desaturate (Comfy Essentials Custom Node).<br>
Low Quality: Direct Input.

Results: (Canny and Depth Examples not included. They are straightforward.)<br><br>
**Pixel Low Resolution to High Resolution**<br><br>
![ComfyUI Screenshot](https://gseth.com/images/SNAG-4386.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4343.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4387.png)

**Photo Restoration**<br><br>
![ComfyUI Screenshot](https://gseth.com/images/SNAG-4375.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4376.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4381.png)

**Game Asset Low Resolution Upscale**<br><br>
![ComfyUI Screenshot](https://gseth.com/images/SNAG-4389.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4340.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4341a.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4342.png)

**Blur to UnBlur**<br><br>
![ComfyUI Screenshot](https://gseth.com/images/SNAG-4364.png)

**Re-Color**<br><br>
![ComfyUI Screenshot](https://gseth.com/images/SNAG-4390.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4392.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4394.png)

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4395.png)

**YouTube tutorial Union ControlNet Usage: <a href="https://www.youtube.com/watch?v=4_1A5pQkJkg">Video Tutorial</a>**

**Shakker Labs & InstantX Flux ControlNet Union Pro Model Download:** <a href="https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro">Hugging Face Link</a>

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
![ComfyUI Screenshot](https://gseth.com/images/without_noise_blend_3a.png)

With Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/with_noise_blend_3.png)

**Example 4**<br>
Without Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/without_noise_blend_4.png)

With Noise Blend:
![ComfyUI Screenshot](https://gseth.com/images/with_noise_blend_4.png)

### Flux Region (Spatial Control)

The node pipeline is as follows: Region Mask Generator --> Region Mask Processor --> Region Mask Validator --> Flux Region Conditioning --> Flux Attention Control --> Flux Overlay Visualizer (optional) --> Flux Attention Cleanup. </br>
*Note: Watching the video tutorial is a must. The learning curve is a bit high to use Flux Region Spatial Control.*

**Region Mask Generator:** This node generates the regions in mask and bbox format. This information is then passed on to the Mask Processor.</br>

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4945.png)</br>

**Region Mask Processor:** This node processes the generated mask and applies Gaussian Blur and feathering. This pre-processor node preprocesses the mask and sends the preprocessed information in the pipeline.</br>

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4947.png)</br>

**Region Mask Validator:** This node calculates the validity of the regions. The "is valid" message will be true if there are no overlaps. The validation message would show you detailed information on the overlapping regions and the overlap percentage. Although the methodology used requires zero overlaps, the issue is resolved in the flux attention control with feathering. Overlapping will only be an issue if it is excessive, beyond 40-50%.</br>

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4946.png)</br>

**Region Mask Conditioning:** Up to three separate conditioning can be connected. The node will process based on the number of regions defined rather than the actual conditioning connections. The strength values are independent for each region. Strength 1 for Region 1, Strength 2 for Region 2, and Strength 3 for Region 3. The strength value range is from 0 to 10 with an increment/decrement step of 0.1. At Value 1, the region strength will match the base conditioning strength, which is always set at 1 as a global value. Strength Values are not only relative to the base conditioning value but are also relative to each other. They are also affected by the Region % area in the canvas and the feathering value in the attention control. Please note. Only use the dual clip and flux conditioning in comfy. The base + region flux guidance should be set to 1.</br>

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4950.png)</br>

**Flux Attention Control:** The node takes the region conditioning + base conditioning + the feathering strengths and all the previous information in the pipeline and overrides the Flux Attention. When disabled, it only passes through the base conditioning to the Sampler.</br>

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4951.png)</br>

**Region Overlay Visualizer:** This node overlays the region on the final output for visual purposes only.</br>

**Flux Attention Cleanup:** Since the attention is overridden in the model, a tensor mismatch error will occur when you switch the workflow. We also do not want the attention to be cleaned up in the existing workflow. This node automatically will preserve attention during re-runs in the existing workflow, but when switching workflow will do a fresh clean up and restore flux original attention. This process is achieved without a model unload or manual cache cleanup, as they will not work.</br>

![ComfyUI Screenshot](https://gseth.com/images/SNAG-4957.png)</br>

**Xformers & Token/Attention Limits:** The pipeline uses an advanced attention mechanism that combines text tokens from your prompts with spatial information from defined regions. As you increase prompt length or add multiple, complex regions, you create larger attention matrices. While xFormers helps optimize memory usage, there is still a practical limit on how many tokens and spatial positions the model can handle without causing dimension or shape alignment errors.

Example Error: 'Invalid shape for attention bias: torch.Size([1, 24, 5264, 5264]) (expected (1, 24, 5118, 5118))'

This limit isn’t about a fixed “5,000 x 5,000” size or a strict VRAM cap. Instead, it’s determined by the model’s architecture and how tokens are combined with spatial positions. Extremely long prompts or too many intricate regions can produce attention shapes that the model’s code cannot process, resulting in shape mismatch errors rather than running out of memory. If you encounter these errors, try shortening your prompt or reducing the complexity of your regional conditioning. There isn’t a simple formula linking VRAM size directly to token count. Instead, it’s about balancing your prompt length and region definitions to keep the attention mechanism within workable limits. Testing with the Flux model and T5-XXL in FP16 on a 4090 shows that keeping prompts relatively short (each clip under 80 tokens) and regions manageable helps avoid such issues.

**GGUF & CivitAI fine-tune models:** The Flux Region Pipeline was tested with GGUF models without issues. Third-party CivitAI Copax Timeless XPlus 3 Flux models also worked without problems. 

**LoRA Support:** LoRA is supported and will apply to all attention. At this stage, using different LoRA for different Regions is not possible. Research work is still ongoing.

**ControlNet Support:** Currently not tested. Research work is still ongoing.

Results:
**Example 1**<br>
3 Region Split Blend using Advance LLM: Base Conditioning (ignored) + 3 Regions
![ComfyUI Screenshot](https://gseth.com/images/region_control_11.png)
![ComfyUI Screenshot](https://gseth.com/images/region_control_12.png)

**Example 2**<br>
Style manipulation: Base Conditioning + 1 Region
![ComfyUI Screenshot](https://gseth.com/images/region_control_1.png)
![ComfyUI Screenshot](https://gseth.com/images/region_control_2.png)

**Example 3**<br>
Simple Splitting Contrast: Base Conditioning (ignored) + 2 Regions
![ComfyUI Screenshot](https://gseth.com/images/region_control_3.png)
![ComfyUI Screenshot](https://gseth.com/images/region_control_4.png)

**Example 4**<br>
Simple Splitting Blend: Base Conditioning + 1 Region
![ComfyUI Screenshot](https://gseth.com/images/region_control_5.png)
![ComfyUI Screenshot](https://gseth.com/images/region_control_6.png)

**Example 5**<br>
3 Region Split Blend: Base Conditioning (ignored) + 3 Regions
![ComfyUI Screenshot](https://gseth.com/images/region_control_7.png)
![ComfyUI Screenshot](https://gseth.com/images/region_control_8.png)

**Example 6**<br>
3 Region Split Blend using Advance LLM: Base Conditioning (ignored) + 3 Regions
![ComfyUI Screenshot](https://gseth.com/images/region_control_13.png)
![ComfyUI Screenshot](https://gseth.com/images/region_control_14.png)

**Example 7**<br>
Color Manipulation: Base Conditioning (ignored) + 2 Regions
![ComfyUI Screenshot](https://gseth.com/images/region_control_15.png)
![ComfyUI Screenshot](https://gseth.com/images/region_control_16.png)

**YouTube tutorial Flux Region Usage: <a href="https://youtu.be/kNwz6kJRDc0">Flux Region Spatial Control Tutorial</a>**

### YouTube ComfyUI Tutorials

We are a team of two and create extensive tutorials for ComfyUI. Check out our YouTube channel:</br>
<a href="https://youtube.com/@controlaltai">ControlAltAI YouTube Tutorials</a>

### Black Forest Labs AI

Black Forest Labs, a pioneering AI research organization, has developed the Flux model series, which includes the Flux1.[dev] and Flux1.[schnell] models. These models are designed to push the boundaries of image generation through advanced deep-learning techniques.

For more details on these models, their capabilities, and licensing information, you can visit the <a href="https://blackforestlabs.ai/">Black Forest Labs website</a>

### Flux Regional Spatial Control Acknowledgment

Inspired from: <a href="https://github.com/attashe/ComfyUI-FluxRegionAttention">Flux Region Attention by Attashe</a>

### License

This project is licensed under the MIT License.
