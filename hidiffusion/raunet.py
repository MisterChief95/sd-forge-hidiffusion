import torch.nn.functional as F

from backend.modules.k_model import KModel
from backend.modules.k_prediction import Prediction
from backend.patcher.unet import UnetPatcher
import backend.nn.unet as unet

from .logger import logger
from .utils import (
    check_time,
    convert_time,
    get_sigma,
    parse_blocks,
    scale_samples,
)
from .types import (
    CONTROLNET_SCALE_ARGS,
    HD_CONFIG,
    HDDownsample,
    HDUpsample,
    ORIG_APPLY_CONTROL,
    ORIG_FORWARD_TIMESTEP_EMBED,
)


logger.info("Proxied UNet Upsample and Downsample classes")


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
    1. Enabling the HD_CONFIG flag
    2. Overriding TimestepEmbedSequential's forward method with HiDiffusion implementation
    3. Overriding UNet's apply_control method with HiDiffusion implementation
    The patches allow the UNet to work with the HiDiffusion architecture and processing.
    Note:
        This is a side-effect function that modifies global state.
    """

    HD_CONFIG.enabled = True
    unet.TimestepEmbedSequential.forward = hd_forward_timestep_embed
    unet.apply_control = hd_apply_control
    logger.info("Applied UNet patches")


def remove_unet_patches():
    """
    Removes patches applied to the UNet model by disabling HiDiffusion configuration and restoring original
    forward methods.
    This function removes patches to the UNet model by:
    1. Disabling the HD_CONFIG flag
    2. Restores original forward method for TimestepEmbedSequential
    3. Restores original apply_control method
    This function should be called to restore UNet to its original state after using HiDiffusion.
    Returns:
        None
    """

    HD_CONFIG.enabled = False
    unet.TimestepEmbedSequential.forward = ORIG_FORWARD_TIMESTEP_EMBED
    unet.apply_control = ORIG_APPLY_CONTROL
    logger.info("Removed UNet patches")


def apply_rau_net(
    unet_patcher: UnetPatcher,
    input_blocks: str,
    output_blocks: str,
    time_mode: str,
    start_time: float,
    end_time: float,
    skip_two_stage_upscale: bool,
    upscale_mode: str,
    ca_start_time: float,
    ca_end_time: float,
    ca_input_blocks: str,
    ca_output_blocks: str,
    ca_upscale_mode: str,
) -> UnetPatcher:
    global ORIG_FORWARD_TIMESTEP_EMBED
    use_blocks = parse_blocks("input", input_blocks)
    use_blocks |= parse_blocks("output", output_blocks)
    ca_use_blocks = parse_blocks("input", ca_input_blocks)
    ca_use_blocks |= parse_blocks("output", ca_output_blocks)

    unet_patcher: UnetPatcher = unet_patcher.clone()

    kmodel: KModel = unet_patcher.model
    predictor: Prediction = kmodel.predictor

    HD_CONFIG.start_sigma, HD_CONFIG.end_sigma = convert_time(
        predictor,
        time_mode,
        start_time,
        end_time,
    )
    ca_start_sigma, ca_end_sigma = convert_time(
        predictor,
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

    HD_CONFIG.use_blocks = use_blocks
    HD_CONFIG.two_stage_upscale = not skip_two_stage_upscale
    HD_CONFIG.upscale_mode = upscale_mode

    return unet_patcher


def configure_blocks(
    model_type: str, res: str
) -> tuple[bool, tuple[str, str], tuple[str, str], tuple[float, float], tuple[float, float]]:
    enabled = True

    model_configs = {
        "SD 1.5/2.1": {
            "blocks": ("3", "8"),
            "ca_blocks": ("1", "11"),
            "modes": {
                "low": (True, ("", ""), (0.0, 0.4), (1.0, 0.0)),
                "high": (True, ("1", "11"), (0.0, 0.5), (0.0, 0.35)),
                "ultra": (True, ("1", "11"), (0.0, 0.6), (0.0, 0.45)),
            },
        },
        "SDXL": {
            "blocks": ("4", "5"),
            "ca_blocks": ("4", "5"),
            "modes": {
                "low": (False, None, None, None),
                "high": (True, ("4", "5"), (0.0, 0.5), (1.0, 0.0)),
                "ultra": (True, ("4", "5"), (0.0, 0.6), (0.0, 0.45)),
            },
        },
    }

    if model_type not in model_configs:
        raise ValueError("Unknown model_type", model_type)

    config = model_configs[model_type]
    if res not in config["modes"]:
        raise ValueError("Unknown resolution mode", res)

    mode_config = config["modes"][res]
    enabled, ca_blocks, time_range, ca_time_range = mode_config
    blocks = config["blocks"] if enabled else None

    return enabled, blocks, ca_blocks, time_range, ca_time_range


def apply_rau_net_simple(
    model_type: str, res_mode: str, upscale_mode: str, ca_upscale_mode: str, unet_patcher: UnetPatcher
) -> UnetPatcher:
    res = res_mode.split(" ", 1)[0]

    enabled, blocks, ca_blocks, time_range, ca_time_range = configure_blocks(model_type, res)

    if not enabled:
        logger.debug("Disabled RAUNet due to low resolution mode")
        return (unet_patcher.clone(),)

    prettyblocks = " / ".join(b if b else "none" for b in blocks)
    prettycablocks = " / ".join(b if b else "none" for b in ca_blocks)

    logger.debug(
        f"""Applying RAUNet using preset: {model_type} - {res}:
        upscale: {upscale_mode}
        in/out blocks: [{prettyblocks}]
        start/end percent: {time_range[0]:.2}/{time_range[1]:.2}
        CA upscale: {ca_upscale_mode}
        CA in/out blocks: [{prettycablocks}]
        CA start/end percent: {ca_time_range[0]:.2}/{ca_time_range[1]:.2}"""
    )

    return apply_rau_net(
        unet_patcher,
        *blocks,
        "percent",
        *time_range,
        False,  # noqa: FBT003
        upscale_mode,
        *ca_time_range,
        *ca_blocks,
        ca_upscale_mode,
    )
