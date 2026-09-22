import torch
import torch.nn.functional as F
import logging
import math

from .utils import (
    check_time,
    convert_time,
    get_sigma,
    parse_blocks,
)

from backend.patcher.unet import UnetPatcher
from backend.modules.k_model import KModel
from backend.modules.k_prediction import Prediction


def window_partition(
    x: torch.Tensor,
    window_size: tuple[int, int],
    shift_size: int | tuple[int, int],
    height: int,
    width: int,
) -> torch.Tensor:
    """Partitions spatial input tensor into windows.
    This function takes a tensor and divides it into windows according to specified window size,
    with an option to shift the partitioning grid.
    Args:
        x (torch.Tensor): Input tensor of shape (batch, height * width, channels)
        window_size (tuple): Tuple of (height, width) specifying window dimensions
        shift_size (int or tuple): Amount to shift windows. If int, same shift is applied to both dimensions
        height (int): Height of the spatial input
        width (int): Width of the spatial input
    Returns:
        torch.Tensor: Windowed tensor of shape (batch * num_windows, window_size[0] * window_size[1], channels)
            where num_windows = (height // window_size[0]) * (width // window_size[1])
    Example:
        >>> x = torch.randn(2, 64*64, 128)  # batch=2, spatial=64x64, channels=128
        >>> windows = window_partition(x, (8,8), 0, 64, 64)
        >>> windows.shape
        torch.Size([128, 64, 128])  # 128 windows of 8x8=64 pixels each
    """
    batch, features, channels = x.shape
    if height <= 0 or width <= 0 or features != height * width:
        raise ValueError(f"Cannot reshape {features} attention tokens as {height}x{width}")
    if window_size[0] <= 0 or window_size[1] <= 0:
        raise ValueError(f"Invalid attention window size: {window_size}")

    x = x.reshape(batch, height, width, channels)

    if not isinstance(shift_size, (list, tuple)):
        shift_size = (shift_size, shift_size)

    padded_height = math.ceil(height / window_size[0]) * window_size[0]
    padded_width = math.ceil(width / window_size[1]) * window_size[1]
    x = F.pad(x, (0, 0, 0, padded_width - width, 0, padded_height - height))

    if any(shift_size):
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    x = x.view(
        batch,
        padded_height // window_size[0],
        window_size[0],
        padded_width // window_size[1],
        window_size[1],
        channels,
    )

    windows: torch.Tensor = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], channels)

    return windows.reshape(-1, window_size[0] * window_size[1], channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int],
    shift_size: int | tuple[int, int],
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Reverses the window partitioning operation by reconstructing the original tensor from window segments.
    This function takes window segments and reconstructs them back into the original tensor format,
    with optional shifting support for overlapping windows.
    Args:
        windows (torch.Tensor): Input tensor of shape (batch * num_windows, window_size[0] * window_size[1], channels)
            containing the window segments.
        window_size (tuple): Size of each window as (height, width).
        shift_size (int or tuple): Amount to shift windows. If tuple, represents (height_shift, width_shift).
            Zero means no shifting.
        height (int): Original height of the input tensor.
        width (int): Original width of the input tensor.
    Returns:
        torch.Tensor: Reconstructed tensor of shape (batch, height * width, channels).
    Note:
        This operation is the inverse of window_partition. It reconstructs the original tensor
        by properly arranging and shifting (if specified) the window segments back to their
        original positions.
    """
    _, tokens_per_window, channels = windows.shape
    if window_size[0] <= 0 or window_size[1] <= 0 or tokens_per_window != window_size[0] * window_size[1]:
        raise ValueError(f"Invalid attention windows with shape {tuple(windows.shape)}")

    padded_height = math.ceil(height / window_size[0]) * window_size[0]
    padded_width = math.ceil(width / window_size[1]) * window_size[1]
    windows_per_batch = (padded_height // window_size[0]) * (padded_width // window_size[1])
    if windows.shape[0] % windows_per_batch:
        raise ValueError("Attention window count does not match the feature shape")
    batch = windows.shape[0] // windows_per_batch
    windows = windows.reshape(-1, window_size[0], window_size[1], channels)

    x = windows.view(
        batch,
        padded_height // window_size[0],
        padded_width // window_size[1],
        window_size[0],
        window_size[1],
        -1,
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, padded_height, padded_width, -1)

    if not isinstance(shift_size, (list, tuple)):
        shift_size = (shift_size, shift_size)

    if any(shift_size):
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    return x[:, :height, :width, :].reshape(batch, height * width, channels)


def get_window_args(
    n: torch.Tensor, orig_shape: tuple[int, int], shift: int
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    """
    Calculate window arguments for shifted window attention.
    This function determines the window size, shift size, and dimensions based on input tensor
    and original shape parameters. Used for implementing shifted window based self-attention.
    Args:
        n (torch.Tensor): Input tensor with shape (batch_size, num_features, dimension)
        orig_shape (tuple): Original height and width dimensions (H, W)
        shift (int): Shift index determining the amount of window shift (0-3)
    Returns:
        tuple: Contains:
            - window_size (tuple): Size of each of four attention windows
            - shift_size (tuple): Amount to shift window (depends on shift parameter)
            - height (int): Downsampled height
            - width (int): Downsampled width
    """
    _, features, _ = n.shape
    orig_height, orig_width = orig_shape[-2:]
    if features <= 0 or orig_height <= 0 or orig_width <= 0:
        raise ValueError("Attention and original shapes must be positive")

    # The U-Net can round one spatial axis differently from the other after a
    # downsample. Find the factor pair with the closest original aspect ratio
    # instead of assuming that the scale factor is an integer square.
    target_ratio = orig_height / orig_width
    candidates = tuple(
        shape
        for factor in range(1, math.isqrt(features) + 1)
        if features % factor == 0
        for shape in {(factor, features // factor), (features // factor, factor)}
    )
    height, width = min(candidates, key=lambda shape: abs(math.log((shape[0] / shape[1]) / target_ratio)))
    window_size = (max(1, math.ceil(height / 2)), max(1, math.ceil(width / 2)))

    match shift:
        case 0:
            shift_size = (0, 0)
        case 1:
            shift_size = (window_size[0] // 4, window_size[1] // 4)
        case 2:
            shift_size = (window_size[0] // 4 * 2, window_size[1] // 4 * 2)
        case _:
            shift_size = (window_size[0] // 4 * 3, window_size[1] // 4 * 3)

    return (window_size, shift_size, height, width)


def apply_mswmsaa_attention(
    unet_patcher: UnetPatcher,
    input_blocks: str,
    middle_blocks: str,
    output_blocks: str,
    time_mode: str,
    start_time: float,
    end_time: float,
) -> UnetPatcher:
    """Applies Multi-Scale Window Masked Self-Attention (MSW-MSA) to specific UNet blocks.
    This function implements MSW-MSA attention mechanism by patching the attention layers
    in specified UNet blocks. It enables shifted window-based self-attention for better
    feature learning at multiple scales.
    Args:
        self: The instance of the class containing this method.
        unet_patcher: Patcher object for modifying UNet model behavior.
        input_blocks (str): Specification of input blocks to apply attention to.
        middle_blocks (str): Specification of middle blocks to apply attention to.
        output_blocks (str): Specification of output blocks to apply attention to.
        time_mode (str): Mode for time/step calculation ('steps' or 'sigma').
        start_time (float): Starting time/step for applying attention.
        end_time (float): Ending time/step for applying attention.
    Returns:
        UnetPatcher: modified unet.
    Raises:
        RuntimeError: If window partitioning fails due to incompatible model patches
                     or incorrect input resolution. Resolution should be multiples of
                     32 or 64.
    Note:
        The function implements random shift patterns for window partitioning to avoid
        boundary artifacts. It uses a modulo-4 shift pattern that avoids consecutive
        identical shifts.
    """
    use_blocks = parse_blocks("input", input_blocks)
    use_blocks |= parse_blocks("middle", middle_blocks)
    use_blocks |= parse_blocks("output", output_blocks)

    # A patcher can be reused for many samples. Keep only the pending layout for
    # the current block, then consume it in the matching output hook.
    pending_window_args: dict[tuple | str | None, tuple] = {}

    unet_patcher = unet_patcher.clone()
    kmodel: KModel = unet_patcher.model
    predictor: Prediction = kmodel.predictor

    start_sigma, end_sigma = convert_time(predictor, time_mode, start_time, end_time)

    def attn1_patch(
        q: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        extra_options,
    ) -> tuple[torch.Tensor | None, ...]:
        """
        Applies Multiscale Window Multi-head Self-Attention (MSW-MSA) partitioning to query, key and value tensors.
        This function implements the shifting window mechanism for self-attention, where windows are randomly shifted
        to enable cross-window connections while maintaining efficiency.
        Args:
            q (torch.Tensor, optional): Query tensor. Can be None.
            k (torch.Tensor, optional): Key tensor. Can be None.
            v (torch.Tensor, optional): Value tensor. Can be None.
            extra_options (dict): Dictionary containing additional parameters:
                - block (str): Current processing block identifier
                - original_shape (tuple): Original shape of the input tensor
        Returns:
            tuple: Tuple of transformed (q, k, v) tensors after window partitioning.
                    If input tensor is None, corresponding output will be None.
        Raises:
            RuntimeError: If window partitioning fails due to incompatible model patches
                         or inappropriate input resolution. Resolution should be multiple of 32 or 64.
        Notes:
            - The shift varies by sigma and transformer block without changing Forge's RNG state.
            - Handles cases where q, k, v are the same tensor for efficiency.
        """

        block = extra_options.get("block")
        pending_window_args.pop(block, None)
        if block not in use_blocks or not check_time(
            extra_options,
            start_sigma,
            end_sigma,
        ):
            return q, k, v
        orig_shape = extra_options.get("original_shape")
        if orig_shape is None:
            logging.warning("MSW-MSA skipped %s because Forge did not provide original_shape", block)
            return q, k, v

        # MSW-MSA
        # Vary the shift across denoising steps and transformer blocks without
        # drawing from an RNG outside Forge's selectable RNG implementations.
        shift = (
            round(get_sigma(extra_options) * 10000)
            + int(extra_options.get("transformer_index", 0))
            + int(extra_options.get("block_index", 0))
        ) % 4
        try:
            window_args = tuple(get_window_args(x, orig_shape, shift) if x is not None else None for x in (q, k, v))
            if q is not None and q is k and q is v:
                partitioned = window_partition(q, *window_args[0])
                pending_window_args[block] = window_args
                return (partitioned,) * 3
            partitioned = tuple(
                window_partition(x, *window_args[idx]) if x is not None else None for idx, x in enumerate((q, k, v))
            )
            pending_window_args[block] = window_args
            return partitioned
        except (RuntimeError, ValueError, TypeError) as exc:
            logging.warning("MSW-MSA skipped %s because its attention shape is incompatible: %s", block, exc)
            return q, k, v

    def attn1_output_patch(n: torch.Tensor, extra_options: dict[str, str]) -> torch.Tensor:
        """
        Patches the output of attention layer 1 by reversing windowing if window arguments are available.
        Args:
            n: The input tensor to be processed
            extra_options (dict): Dictionary containing extra options, including the 'block' key
        Returns:
            tensor: Either the original input tensor if no window args are available,
                   or the window-reversed tensor using stored window arguments
        Note:
            This function uses nonlocal variables `window_args` and `last_block` which must be
            defined in the outer scope. The `window_reverse` function must also be available.
        """

        args = pending_window_args.pop(extra_options.get("block"), None)
        if args is None or args[0] is None:
            return n
        try:
            return window_reverse(n, *args[0])
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"MSW-MSA could not restore block {extra_options.get('block')} "
                f"from attention output shape {tuple(n.shape)}"
            ) from exc

    unet_patcher.set_model_attn1_patch(attn1_patch)
    unet_patcher.set_model_attn1_output_patch(attn1_output_patch)

    return unet_patcher


def apply_mswmsaa_attention_simple(model_type: str, model: UnetPatcher) -> UnetPatcher:
    """
    Applies Multi-Scale Window Multi-head Self Attention (MSWMSA) to a given model using predefined settings based on model type.
    Args:
        model_type (str): The type of model. Must be either "SD15" or "SDXL".
        model: The model to apply attention to.
    Returns:
        The model with MSWMSA attention applied according to the specified parameters.
    Raises:
        ValueError: If the model_type is neither "SD15" nor "SDXL".
    Notes:
        - For SD15, applies attention to blocks (1,2), none, and (11,10,9)
        - For SDXL, applies attention to blocks (4,5), none, and (5,4)
        - Uses time range of 0.2 to 1.0 for both model types
    """

    time_range: tuple[float] = (0.2, 1.0)

    if model_type == "SD 1.5/2.1":
        blocks: tuple[str] = ("1,2", "", "11,10,9")
    elif model_type == "SDXL":
        blocks: tuple[str] = ("4,5", "", "3,4,5")
    else:
        raise ValueError("Unknown model type")

    prettyblocks = " / ".join(b if b else "none" for b in blocks)

    logging.debug(
        f"** ApplyMSWMSAAttentionSimple: Using preset {model_type}: in/mid/out blocks [{prettyblocks}], start/end percent {time_range[0]:.2}/{time_range[1]:.2}",
    )

    return apply_mswmsaa_attention(model, *blocks, "percent", *time_range)
