# Forge HiDiffusion

## Overview
Forge HiDiffusion is a port of [Panchovix's JankHidiffusion](https://github.com/Panchovix/reforge_jankhidiffusion) from [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) to [lllyasviel's Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge). HiDiffusion is a technique designed to improve the ability to quickly generate images at higher resolutions without structural collapse.

## Features
- **High Resolution Stability**: Ensures that images generated at higher resolutions maintain structural integrity.
- **Compatibility with Forge**: Seamless integration with lllyasviel's Forge WebUI for enhanced user experience.
- **Advanced Controls**: Offers advanced settings for users who require finer control over the image generation process.

## Installation
To use Forge HiDiffusion, follow these steps:

1. **Install the Extension**:
   - Download and install the Forge HiDiffusion extension in your WebUI.
   
2. **Accessing the Extension**:
   - Navigate to either the `txt2img` or `img2img` tab in the WebUI.
   - Locate the "Forge HiDiffusion" section.

## Usage Instructions
1. **Adjust Image Dimensions**:
   - Set the desired image width and height. Ensure that the resolutions are divisible by 64 for optimal results.
   
2. **Select Resolution Mode**:
   - Choose a resolution mode that closely matches your desired resolution from the RAUNet tab.
   - Note: This option is disabled if `Advanced` mode is checked.

3. **Enable Advanced Mode** (Optional):
   - For finer control, you can enable advanced mode in both the RAUNet and MSW-MSA tabs.

4. **Apply during Hires Fix** (Optional):
   - Turn on the top-level checkbox to apply HiDiffusion to Hires Fix's high-resolution second pass and Forge quick upscale. By default, regular Hires Fix applies HiDiffusion only to the first pass.
   - RAUNet's early time range may have little effect during a low-denoise second pass; MSW-MSA can still affect later timesteps. Results and memory use depend on the final resolution.

### Advanced block settings

Block numbers refer to Forge's U-Net, and each control targets a different layer type. These are the default pairs for base SD models:

| Model | RAUNet downsample → upsample | Additional feature pooling → restoration | MSW-MSA attention blocks |
| --- | --- | --- | --- |
| SD 1.5/2.1 | Input 3 → output 8 | Input 1 → output 11 | Inputs 1,2; outputs 9,10,11 |
| SDXL | Input 3 → output 5 | Input 4 → output 5 | Inputs 4,5; outputs 3,4,5 |

SDXL also has a deeper RAUNet pair, input 6 → output 2. When choosing custom RAUNet blocks, select each matching input and output pair; invalid or mismatched selections now report an error. MSW-MSA blocks must contain attention layers.

Advanced RAUNet uses the matching model pooling preset unless **Use custom pooling settings** is enabled under Cross-Attention Settings. Turn it on to use the CA block and time fields. To disable pooling while using custom settings, clear both CA block fields.

The time controls accept percent (0–1), training timestep (0–999), or sigma values. Percent is converted through the model's sigma schedule and does not necessarily correspond to the same fraction of sampling steps.

## Troubleshooting
- **Common Issues**:
  - If images are not generating as expected, ensure that the dimensions are set correctly and are divisible by 64.
  - Make sure to select the appropriate resolution mode to match your desired output.
  
- **Support**:
  - For additional help, visit our [GitHub Issues page](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues).

## Contributing
We welcome contributions to Forge HiDiffusion. To contribute, follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## Credits
- **reForge Implementation**: [Panchovix's JankHidiffusion](https://github.com/Panchovix/reforge_jankhidiffusion)
- **ComfyUI Implementation**: [blepping's comfyui_jankhidiffusion](https://github.com/blepping/comfyui_jankhidiffusion)
- **Original Implementation**: [Megvii Research's HiDiffusion](https://github.com/megvii-research/HiDiffusion)
