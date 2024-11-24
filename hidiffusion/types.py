from typing import Any

import torch
import torch.nn.functional as F

from backend.nn import unet
from .utils import check_time, scale_samples


UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "area")


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
        if not self.enabled or not isinstance(topts, dict) or topts.get("block") not in self.use_blocks:
            return False
        return check_time(topts, self.start_sigma, self.end_sigma)


HD_CONFIG = HDConfigClass()


CONTROLNET_SCALE_ARGS: dict[str, Any] = {"mode": "bilinear", "align_corners": False}
ORIG_APPLY_CONTROL = unet.apply_control
ORIG_FORWARD_TIMESTEP_EMBED = unet.TimestepEmbedSequential.forward
ORIG_UPSAMPLE = unet.Upsample
ORIG_DOWNSAMPLE = unet.Downsample


class HDUpsample(ORIG_UPSAMPLE):
    """
    A modified upsampling layer that extends ORIG_UPSAMPLE for high-definition image processing.
    This class implements custom upsampling behavior based on configuration settings,
    with options for two-stage upscaling and different upscaling modes.
    Parameters:
        Inherits all parameters from ORIG_UPSAMPLE parent class.
    Returns:
        torch.Tensor: The upsampled tensor.
    Methods:
        forward(x, output_shape=None, transformer_options=None):
            Performs the upsampling operation on the input tensor.
            Args:
                x (torch.Tensor): Input tensor to be upsampled
                output_shape (tuple, optional): Desired output shape. Defaults to None.
                transformer_options (dict, optional): Configuration options for transformation. Defaults to None.
            Returns:
                torch.Tensor: Upsampled tensor after processing through interpolation and convolution
    """

    def forward(self, x, output_shape=None, transformer_options=None):
        if self.dims == 3 or not self.use_conv or not HD_CONFIG.check(transformer_options):
            return super().forward(x, output_shape=output_shape)
        shape = output_shape[2:4] if output_shape is not None else (x.shape[2] * 4, x.shape[3] * 4)
        if HD_CONFIG.two_stage_upscale:
            x = F.interpolate(x, size=(shape[0] // 2, shape[1] // 2), mode="nearest")
        x = scale_samples(
            x,
            shape[1],
            shape[0],
            mode=HD_CONFIG.upscale_mode,
        )
        return self.conv(x)


class HDDownsample(ORIG_DOWNSAMPLE):
    """HDDownsample is a modified downsampling layer that extends ORIG_DOWNSAMPLE.
    This class implements specialized downsampling for images using dilated convolutions
    when specific conditions are met. Otherwise, it falls back to original downsampling behavior.
    Attributes:
        COPY_OP_KEYS (tuple): Keys of attributes to copy from original operation to temporary operation.
            Includes parameters_manual_cast, weight_function, bias_function, weight, and bias.
    Args:
        *args (list): Variable length argument list passed to parent class.
        **kwargs (dict): Arbitrary keyword arguments passed to parent class.
    Methods:
        forward(x, transformer_options=None): Performs the downsampling operation.
            Uses dilated convolution when dims==2, use_conv is True and HDCONFIG conditions are met.
            Otherwise falls back to original downsampling.
            Args:
                x: Input tensor to downsample
                transformer_options: Optional configuration for transformation
            Returns:
                Downsampled tensor using either dilated convolution or original method
    """

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
        if self.dims == 3 or not self.use_conv or not HD_CONFIG.check(transformer_options):
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
        if HD_CONFIG.enabled:
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
        if HD_CONFIG.enabled:
            return super().forward(*args, **kwargs)
        return self.orig_instance.forward(*args, **kwargs)


# Replace original classes with proxy classes
unet.Upsample = ProxyUpsample
unet.Downsample = ProxyDownsample
