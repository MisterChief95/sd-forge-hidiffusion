import os
from typing import Any

import torch.nn.functional as F

import backend.nn.unet as unet

from .utils import *
from .logger import logger


class HDConfigClass:
    """A configuration class for high-definition image processing.
    This class manages settings and parameters for the HD upscale of the generation.
    Attributes:
        enabled (bool): Flag to enable/disable HD processing. Defaults to False.
        start_sigma (float, optional): Starting sigma value for processing range.
        end_sigma (float, optional): Ending sigma value for processing range.
        use_blocks (list, optional): List of valid processing blocks.
        two_stage_upscale (bool): Whether to use two-stage upscaling. Defaults to True.
        upscale_mode (str): Upscaling algorithm to use. Defaults to "bislerp".
    Methods:
        check(topts): Validates processing options against configuration settings.
            Args:
                topts (dict): Dictionary containing processing options to validate.
            Returns:
                bool: True if options are valid according to configuration, False otherwise.
    """

    enabled: bool = False
    start_sigma: float | None = None
    end_sigma: float | None = None
    use_blocks: float | None = None
    two_stage_upscale: bool = True
    upscale_mode: str = UPSCALE_METHODS[0]

    def check(self, topts: dict[str, torch.Tensor]) -> bool:
        if not self.enabled or not isinstance(topts, dict):
            return False
        if topts.get("block") not in self.use_blocks:
            return False
        return check_time(topts, self.start_sigma, self.end_sigma)


HDCONFIG = HDConfigClass()

CONTROLNET_SCALE_ARGS: dict[str, Any] = {"mode": "bilinear", "align_corners": False}
NO_CONTROLNET_WORKAROUND: bool = os.environ.get("JANKHIDIFFUSION_NO_CONTROLNET_WORKAROUND") is not None
ORIG_APPLY_CONTROL = unet.apply_control
ORIG_FORWARD_TIMESTEP_EMBED = unet.TimestepEmbedSequential.forward
PATCHED_FREEU: bool = False
ORIG_UPSAMPLE = unet.Upsample 
ORIG_DOWNSAMPLE = unet.Downsample


class HDUpsample(ORIG_UPSAMPLE):
    def forward(self, x, output_shape=None, transformer_options=None):
        if self.dims == 3 or not self.use_conv or not HDCONFIG.check(transformer_options):
            return super().forward(x, output_shape=output_shape)
        shape = output_shape[2:4] if output_shape is not None else (x.shape[2] * 4, x.shape[3] * 4)
        if HDCONFIG.two_stage_upscale:
            x = F.interpolate(x, size=(shape[0] // 2, shape[1] // 2), mode="nearest")
        x = scale_samples(
            x,
            shape[1],
            shape[0],
            mode=HDCONFIG.upscale_mode,
        )
        return self.conv(x)


class HDDownsample(ORIG_DOWNSAMPLE):
    COPY_OP_KEYS = (
        "parameters_manual_cast",
        "weight_function",
        "bias_function",
        "weight",
        "bias",
    )

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)

    def forward(self, x, transformer_options=None):
        if self.dims == 3 or not self.use_conv or not HDCONFIG.check(transformer_options):
            return super().forward(x)
        tempop = unet.conv_nd(
            self.dims,
            self.channels,
            self.out_channels,
            3,  # kernel size
            stride=(4, 4),
            padding=(2, 2),
            dilation=(2, 2),
        )
        for k in self.COPY_OP_KEYS:
            if hasattr(self.op, k):
                setattr(tempop, k, getattr(self.op, k))
        return tempop(x)


# Create proxy classes that inherit from original UNet classes
class ProxyUpsample(HDUpsample):
    """Proxy class that can switch between HD and original upsampling implementations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_instance = ORIG_UPSAMPLE(*args, **kwargs)
        # Transfer weights and parameters
        self.orig_instance.conv = self.conv
        
    def forward(self, *args, **kwargs):
        if HDCONFIG.enabled:
            return super().forward(*args, **kwargs)
        return self.orig_instance.forward(*args, **kwargs)
    

class ProxyDownsample(HDDownsample):
    """Proxy class that can switch between HD and original downsampling implementations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_instance = ORIG_DOWNSAMPLE(*args, **kwargs)
        # Transfer weights and parameters
        self.orig_instance.op = self.op
        
    def forward(self, *args, **kwargs):
        if HDCONFIG.enabled:
            return super().forward(*args, **kwargs)
        return self.orig_instance.forward(*args, **kwargs)
        

# Replace original classes with proxy classes
unet.Upsample = ProxyUpsample
unet.Downsample = ProxyDownsample

logger.info("\x1b[32m[HiDiffusion]\x1b[0m Proxied UNet Upsample and Downsample classes")


# TODO: Implement Forge FreeU compatibility
# Try to be compatible with FreeU Advanced.
# def try_patch_freeu_advanced():
#     global PATCHED_FREEU  # noqa: PLW0603
#     if PATCHED_FREEU:
#         return
#     # We only try one time.
#     PATCHED_FREEU = True
#     fua_nodes = sys.modules.get("FreeU_Advanced.nodes")
#     if not fua_nodes:
#         return

#     def fu_forward_timestep_embed(*args: list, **kwargs: dict):
#         fun = hd_forward_timestep_embed if HDCONFIG.enabled else ORIG_FORWARD_TIMESTEP_EMBED
#         return fun(*args, **kwargs)

#     def fu_apply_control(*args: list, **kwargs: dict):
#         fun = hd_apply_control if HDCONFIG.enabled else ORIG_APPLY_CONTROL
#         return fun(*args, **kwargs)

#     fua_nodes.forward_timestep_embed = fu_forward_timestep_embed
#     if not NO_CONTROLNET_WORKAROUND:
#         fua_nodes.unet.apply_control = fu_apply_control
#     print("** jankhidiffusion: Patched FreeU_Advanced")


def hd_apply_control(h, control, name):
    ctrls = control.get(name) if control is not None else None
    if ctrls is None or len(ctrls) == 0:
        return h
    ctrl = ctrls.pop()
    if ctrl is None:
        return h
    if ctrl.shape[-2:] != h.shape[-2:]:
        logger.info(f"Scaling controlnet conditioning: {ctrl.shape[-2:]} -> {h.shape[-2:]}")
        ctrl = F.interpolate(ctrl, size=h.shape[-2:], **CONTROLNET_SCALE_ARGS)
    h += ctrl
    return h


class NotFound:
    pass


def hd_forward_timestep_embed(ts, x, emb, *args: list, **kwargs: dict):
    transformer_options = kwargs.get("transformer_options", NotFound)
    output_shape = kwargs.get("output_shape", NotFound)
    transformer_options = args[1] if transformer_options is NotFound and len(args) > 1 else {}
    output_shape = args[2] if output_shape is NotFound and len(args) > 2 else None
    for layer in ts:
        if isinstance(layer, HDUpsample):
            x = layer.forward(
                x,
                output_shape=output_shape,
                transformer_options=transformer_options,
            )
        elif isinstance(layer, HDDownsample):
            x = layer.forward(x, transformer_options=transformer_options)
        else:
            x = ORIG_FORWARD_TIMESTEP_EMBED((layer,), x, emb, *args, **kwargs)
    return x


def apply_unet_patches():
    """
    Apply patches to modify UNet behavior for HiDiffusion functionality.
    This function applies patches to the UNet model by:
    1. Enabling the HDCONFIG flag
    2. Overriding TimestepEmbedSequential's forward method with HiDiffusion implementation
    3. Overriding UNet's apply_control method with HiDiffusion implementation
    The patches allow the UNet to work with the HiDiffusion architecture and processing.
    Note:
        This is a side-effect function that modifies global state.
    """

    HDCONFIG.enabled = True
    unet.TimestepEmbedSequential.forward = hd_forward_timestep_embed
    unet.apply_control = hd_apply_control
    logger.info("Applied UNet patches")


def remove_unet_patches():
    """
    Removes patches applied to the UNet model by disabling HiDiffusion configuration and restoring original
    forward methods.
    This function removes patches to the UNet model by:
    1. Disabling the HDCONFIG flag
    2. Restores original forward method for TimestepEmbedSequential
    3. Restores original apply_control method
    This function should be called to restore UNet to its original state after using HiDiffusion.
    Returns:
        None
    """

    HDCONFIG.enabled = False
    unet.TimestepEmbedSequential.forward = ORIG_FORWARD_TIMESTEP_EMBED
    unet.apply_control = ORIG_APPLY_CONTROL
    logger.info("Removed UNet patches")


def apply_rau_net(
    enabled,
    unet_patcher,
    input_blocks,
    output_blocks,
    time_mode,
    start_time,
    end_time,
    skip_two_stage_upscale,
    upscale_mode,
    ca_start_time,
    ca_end_time,
    ca_input_blocks,
    ca_output_blocks,
    ca_upscale_mode,
):
    global ORIG_FORWARD_TIMESTEP_EMBED
    use_blocks = parse_blocks("input", input_blocks)
    use_blocks |= parse_blocks("output", output_blocks)
    ca_use_blocks = parse_blocks("input", ca_input_blocks)
    ca_use_blocks |= parse_blocks("output", ca_output_blocks)

    unet_patcher = unet_patcher.clone()
    if not enabled:
        HDCONFIG.enabled = False
        if ORIG_FORWARD_TIMESTEP_EMBED is not None:
            unet.TimestepEmbedSequential.forward = ORIG_FORWARD_TIMESTEP_EMBED
        if unet.apply_control is not ORIG_APPLY_CONTROL and not NO_CONTROLNET_WORKAROUND:
            unet.apply_control = ORIG_APPLY_CONTROL
        return (unet_patcher,)

    # Access model_sampling through the actual model object
    ms = unet_patcher.model.predictor

    HDCONFIG.start_sigma, HDCONFIG.end_sigma = convert_time(
        ms,
        time_mode,
        start_time,
        end_time,
    )
    ca_start_sigma, ca_end_sigma = convert_time(
        ms,
        time_mode,
        ca_start_time,
        ca_end_time,
    )

    def input_block_patch(h, extra_options):
        if extra_options.get("block") not in ca_use_blocks or not check_time(
            extra_options,
            ca_start_sigma,
            ca_end_sigma,
        ):
            return h
        return F.avg_pool2d(h, kernel_size=(2, 2))

    def output_block_patch(h, hsp, extra_options):
        if extra_options.get("block") not in ca_use_blocks or not check_time(
            extra_options,
            ca_start_sigma,
            ca_end_sigma,
        ):
            return h, hsp
        sigma = get_sigma(extra_options)
        block = extra_options.get("block", ("", 0))[1]
        if sigma is not None and (block < 3 or block > 6):
            sigma /= 16
        return (
            scale_samples(
                h,
                hsp.shape[3],
                hsp.shape[2],
                mode=ca_upscale_mode,
            ),
            hsp,
        )

    unet_patcher.set_model_input_block_patch(input_block_patch)
    unet_patcher.set_model_output_block_patch(output_block_patch)
    
    HDCONFIG.use_blocks = use_blocks
    HDCONFIG.two_stage_upscale = not skip_two_stage_upscale
    HDCONFIG.upscale_mode = upscale_mode

    return (unet_patcher,)


def configure_blocks(
    model_type: str, res: str
) -> tuple[bool, tuple[str, str], tuple[str, str], tuple[float, float], tuple[float, float]]:
    enabled = True

    model_configs = {
        "SD15": {
            "blocks": ("3", "8"),
            "ca_blocks": ("1", "11"),
            "modes": {
                "low": (True, ("", ""), (0.0, 0.4), (1.0, 0.0)),
                "high": (True, ("1", "11"), (0.0, 0.5), (0.0, 0.35)),
                "ultra": (True, ("1", "11"), (0.0, 0.6), (0.0, 0.45))
            }
        },
        "SDXL": {
            "blocks": ("3", "5"),
            "ca_blocks": ("4", "5"),
            "modes": {
                "low": (False, None, None, None),
                "high": (True, ("4", "5"), (0.0, 0.5), (1.0, 0.0)),
                "ultra": (True, ("4", "5"), (0.0, 0.6), (0.0, 0.45))
            }
        }
    }

    if model_type not in model_configs:
        raise ValueError("Unknown model_type")

    config = model_configs[model_type]
    if res not in config["modes"]:
        raise ValueError("Unknown res_mode")

    mode_config = config["modes"][res]
    enabled, ca_blocks, time_range, ca_time_range = mode_config
    blocks = config["blocks"] if enabled else None

    return enabled, blocks, ca_blocks, time_range, ca_time_range


def apply_rau_net_simple(enabled, model_type, res_mode, upscale_mode, ca_upscale_mode, model):
    upscale_mode = "bicubic" if upscale_mode == "default" else upscale_mode
    ca_upscale_mode = "bicubic" if ca_upscale_mode == "default" else ca_upscale_mode
    res = res_mode.split(" ", 1)[0]

    enabled, blocks, ca_blocks, time_range, ca_time_range = configure_blocks(model_type, res)

    if not enabled:
        logging.debug("** ApplyRAUNetSimple: Disabled")
        return (model.clone(),)

    prettyblocks = " / ".join(b if b else "none" for b in blocks)
    prettycablocks = " / ".join(b if b else "none" for b in ca_blocks)

    logging.debug(
        f"""** ApplyRAUNetSimple: Using preset {model_type} {res}:
        upscale: {upscale_mode}
        in/out blocks: [{prettyblocks}]
        start/end percent: {time_range[0]:.2}/{time_range[1]:.2}
        CA upscale: {ca_upscale_mode}
        CA in/out blocks: [{prettycablocks}]
        CA start/end percent: {ca_time_range[0]:.2}/{ca_time_range[1]:.2}"""
    )

    return apply_rau_net(
        True,  # noqa: FBT003
        model,
        *blocks,
        "percent",
        *time_range,
        False,  # noqa: FBT003
        upscale_mode,
        *ca_time_range,
        *ca_blocks,
        ca_upscale_mode,
    )
