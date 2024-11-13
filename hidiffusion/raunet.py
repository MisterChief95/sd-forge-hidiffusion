import logging
import os
import sys
from typing import Any

from backend.modules.k_model import KModel
from backend.patcher.unet import UnetPatcher
import torch.nn.functional as F

import backend.nn.unet as unet

from .utils import *


class HDConfigClass:
    """A configuration class for high-definition image processing.
    This class manages settings and parameters for the HD upscale of the generation.
    Attributes:
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

    start_sigma: float | None = None
    end_sigma: float | None = None
    use_blocks: float | None = None
    two_stage_upscale: bool = True
    upscale_mode: UpscaleMethod = UpscaleMethod.BISLERP

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
ORIG_FORWARD_TIMESTEP_EMBED = unet.TimestepEmbedSequential().forward
PATCHED_FREEU: bool = False

OrigUpsample, OrigDownsample = unet.Upsample, unet.Downsample


class HDUpsample(OrigUpsample):
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
            mode=HDCONFIG.upscale_mode.value,
        )
        return self.conv(x)


class HDDownsample(OrigDownsample):
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


# Try to be compatible with FreeU Advanced.
def try_patch_freeu_advanced():
    global PATCHED_FREEU  # noqa: PLW0603
    if PATCHED_FREEU:
        return
    # We only try one time.
    PATCHED_FREEU = True
    fua_nodes = sys.modules.get("FreeU_Advanced.nodes")
    if not fua_nodes:
        return

    def fu_forward_timestep_embed(*args: list, **kwargs: dict):
        fun = hd_forward_timestep_embed if HDCONFIG.enabled else ORIG_FORWARD_TIMESTEP_EMBED
        return fun(*args, **kwargs)

    def fu_apply_control(*args: list, **kwargs: dict):
        fun = hd_apply_control if HDCONFIG.enabled else ORIG_APPLY_CONTROL
        return fun(*args, **kwargs)

    fua_nodes.forward_timestep_embed = fu_forward_timestep_embed
    if not NO_CONTROLNET_WORKAROUND:
        fua_nodes.unet.apply_control = fu_apply_control
    print("** jankhidiffusion: Patched FreeU_Advanced")


def hd_apply_control(h, control, name):
    ctrls = control.get(name) if control is not None else None
    if ctrls is None or len(ctrls) == 0:
        return h
    ctrl = ctrls.pop()
    if ctrl is None:
        return h
    if ctrl.shape[-2:] != h.shape[-2:]:
        print(
            f"* jankhidiffusion: Scaling controlnet conditioning: {ctrl.shape[-2:]} -> {h.shape[-2:]}",
        )
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


def apply_monkey_patch():
    global ORIG_FORWARD_TIMESTEP_EMBED, NO_CONTROLNET_WORKAROUND

    unet.Upsample = HDUpsample
    unet.Downsample = HDDownsample

    if unet.TimestepEmbedSequential.forward is not hd_forward_timestep_embed:
        try_patch_freeu_advanced()
        ORIG_FORWARD_TIMESTEP_EMBED = unet.TimestepEmbedSequential.forward
        unet.TimestepEmbedSequential.forward = hd_forward_timestep_embed

    if unet.apply_control is hd_apply_control or NO_CONTROLNET_WORKAROUND:
        return
    unet.apply_control = hd_apply_control


def remove_monkey_patch():
    global NO_CONTROLNET_WORKAROUND, ORIG_APPLY_CONTROL, ORIG_FORWARD_TIMESTEP_EMBED

    unet.Upsample = OrigUpsample
    unet.Downsample = OrigDownsample

    if ORIG_FORWARD_TIMESTEP_EMBED is not None:
        unet.TimestepEmbedSequential.forward = ORIG_FORWARD_TIMESTEP_EMBED
    if unet.apply_control is not ORIG_APPLY_CONTROL and not NO_CONTROLNET_WORKAROUND:
        unet.apply_control = ORIG_APPLY_CONTROL


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
) -> tuple[UnetPatcher]:
    """
    Apply the RAU (Recurrent Attention Unit) network to the given UNet model.
    Parameters:
    - enabled (bool): Flag to enable or disable the RAU network.
    - unet_patcher (UnetPatcher): The UNet patcher object to modify the UNet model.
    - input_blocks (str): Blocks to be used as input.
    - output_blocks (str): Blocks to be used as output.
    - time_mode (str): Mode for time conversion.
    - start_time (float): Start time for the RAU network.
    - end_time (float): End time for the RAU network.
    - skip_two_stage_upscale (bool): Flag to skip two-stage upscaling.
    - upscale_mode (str): Mode for upscaling.
    - ca_start_time (float): Start time for the conditional attention.
    - ca_end_time (float): End time for the conditional attention.
    - ca_input_blocks (str): Blocks to be used as input for conditional attention.
    - ca_output_blocks (str): Blocks to be used as output for conditional attention.
    - ca_upscale_mode (str): Mode for upscaling in conditional attention.
    Returns:
    - tuple[UnetPatcher]: A tuple containing the modified UNet patcher object.
    """
    global ORIG_FORWARD_TIMESTEP_EMBED

    upscale_mode = UpscaleMethod.from_str(upscale_mode) if isinstance(upscale_mode, str) else upscale_mode

    use_blocks = parse_blocks("input", input_blocks)
    use_blocks |= parse_blocks("output", output_blocks)
    ca_use_blocks = parse_blocks("input", ca_input_blocks)
    ca_use_blocks |= parse_blocks("output", ca_output_blocks)

    unet_patcher = unet_patcher.clone()

    # Access model_sampling through the actual model object
    kmodel: KModel = unet_patcher.model
    predictor: Prediction = kmodel.predictor

    HDCONFIG.start_sigma, HDCONFIG.end_sigma = convert_time(
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

    def input_block_patch(h: torch.Tensor, extra_options: dict) -> torch.Tensor:
        if extra_options.get("block") not in ca_use_blocks or not check_time(
            extra_options,
            ca_start_sigma,
            ca_end_sigma,
        ):
            return h
        return F.avg_pool2d(h, kernel_size=(2, 2))

    def output_block_patch(h: torch.Tensor, hsp: torch.Tensor, extra_options: dict) -> tuple[torch.Tensor, torch.Tensor]:
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
    model_type: ModelType, res: str
) -> tuple[tuple[str, str], tuple[str, str], tuple[float, float], tuple[float, float]]:
    """
    Configure the blocks, ca_blocks, time_range, and ca_time_range based on the model type and resolution mode.
    Args:
        model_type (ModelType): The type of the model, either ModelType.SD15 or ModelType.SDXL.
        res (str): The resolution mode, which can be "low", "high", or "ultra".
    Returns:
        tuple: A tuple containing:
            - blocks (tuple[str, str]): The blocks configuration.
            - ca_blocks (tuple[str, str]): The ca_blocks configuration.
            - time_range (tuple[float, float]): The time range configuration.
            - ca_time_range (tuple[float, float]): The ca_time range configuration.
    Raises:
        ValueError: If the model_type is unknown or if the res mode is unknown.
    """
    
    model_configs = {
        ModelType.SD15: {
            "blocks": ("3", "8"),
            "ca_blocks": ("1", "11"),
            "modes": {
                "low": (("", ""), (0.0, 0.4), (1.0, 0.0)),
                "high": (("1", "11"), (0.0, 0.5), (0.0, 0.35)),
                "ultra": (("1", "11"), (0.0, 0.6), (0.0, 0.45))
            }
        },
        ModelType.SDXL: {
            "blocks": ("3", "5"),
            "ca_blocks": ("4", "5"),
            "modes": {
                "low": (None, None, None),
                "high": (("4", "5"), (0.0, 0.5), (1.0, 0.0)),
                "ultra": (("4", "5"), (0.0, 0.6), (0.0, 0.45))
            }
        }
    }

    if model_type not in model_configs:
        raise ValueError("Unknown model_type")

    config = model_configs[model_type]
    if res not in config["modes"]:
        raise ValueError("Unknown res_mode")

    mode_config = config["modes"][res]
    ca_blocks, time_range, ca_time_range = mode_config
    blocks = config["blocks"] if model_type is not ModelType.SDXL and not all(ca_blocks, time_range, ca_time_range) else None

    return blocks, ca_blocks, time_range, ca_time_range


def apply_rau_net_simple(
    model_type: ModelType,
    res_mode: str,
    upscale_mode: str,
    ca_upscale_mode: str,
    model: KModel,
) -> tuple[KModel]:
    """
    Apply the RAU Net model with the specified configurations.
    Parameters:
        enabled (bool): Flag to enable or disable the RAU Net application.
        model_type (ModelType | str): The type of model to be used.
        res_mode (str): The resolution mode to be used.
        upscale_mode (str): The mode to be used for upscaling.
        ca_upscale_mode (str): The mode to be used for CA upscaling.
        model (KModel): The model to which RAU Net will be applied.
    Returns:
        tuple: A tuple containing the modified model.
    """

    res = res_mode.split(" ", 1)[0]
    enabled, blocks, ca_blocks, time_range, ca_time_range = configure_blocks(model_type, res)

    if not enabled:
        logging.debug("** ApplyRAUNetSimple: Disabled")
        return (model.clone(),)

    prettyblocks = " / ".join(b if b else "none" for b in blocks)
    prettycablocks = " / ".join(b if b else "none" for b in ca_blocks)

    logging.debug(
        """** ApplyRAUNetSimple: Using preset {} {}:
    \tupscale: {}
    \tin/out blocks: [{}]
    \tstart/end percent: {:.2}/{:.2}
    \tCA upscale: {}
    \tCA in/out blocks: [{}]
    \tCA start/end percent: {:.2}/{:.2}""".format(
            model_type,
            res,
            upscale_mode,
            prettyblocks,
            time_range[0],
            time_range[1],
            ca_upscale_mode,
            prettycablocks,
            ca_time_range[0],
            ca_time_range[1],
        )
    )

    return apply_rau_net(
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
