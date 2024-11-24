# Forge HiDiffusion
A port of [Panchovix's JankHidiffusion](https://github.com/Panchovix/reforge_jankhidiffusion) from reForge to [lllyasviel's Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)

HiDiffusion is a technique that improves the ability to quickly generate at higher resolutions without structural collapse.

## How to Use
1. Install extension in the WebUI
2. In the `txt2img` or `img2img` tab, find the "HiDiffusion" section
3. Adjust image width/height as desired (maintain resolutions divisble by 64)
4. In Simple mode, select a resolution mode that is close to your chosen resolution in the RAUNet tab
5. Advanced mode can be enabled in both the RAUNet and MSW-MSA tabs instead for finer control

Note: Higher scale factors may increase generation time but can result in sharper details.

## Requirements
- Stable Diffusion WebUI Forge
- Sufficient VRAM (recommended 6GB+ for SD 1.5, 12GB+ for SDXL at FP16)

# Credits
- reForge Implementation https://github.com/Panchovix/reforge_jankhidiffusion
- ComfyUI Implementation: https://github.com/blepping/comfyui_jankhidiffusion
- Original Implementation: https://github.com/megvii-research/HiDiffusion
