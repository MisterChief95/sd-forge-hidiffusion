import torch
import logging

from .utils import (
    check_time,
    convert_time,
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
    # int, discard, int
    batch, _, channels = x.shape

    x = x.view(batch, height, width, channels)

    if not isinstance(shift_size, (list, tuple)):
        shift_size = (shift_size, shift_size)

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    x = x.view(
        batch,
        height // window_size[0],
        window_size[0],
        width // window_size[1],
        window_size[1],
        channels,
    )

    windows: torch.Tensor = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], channels)
    )

    return windows.view(-1, window_size[0] * window_size[1], channels)


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
    # int, discard, int
    batch, _, channels = windows.shape
    windows: torch.Tensor = windows.view(-1, window_size[0], window_size[1], channels)

    batch = int(
        windows.shape[0] / (height * width / window_size[0] / window_size[1]),
    )

    x = windows.view(
        batch,
        height // window_size[0],
        width // window_size[1],
        window_size[0],
        window_size[1],
        -1,
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)

    if not isinstance(shift_size, (list, tuple)):
        shift_size = (shift_size, shift_size)

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    return x.view(batch, height * width, channels)


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
            - window_size (tuple): Size of attention window (height//2, width//2)
            - shift_size (tuple): Amount to shift window (depends on shift parameter)
            - height (int): Downsampled height
            - width (int): Downsampled width
    """
    # discard, int, discard
    _, features, _ = n.shape
    orig_height, orig_width = orig_shape[-2:]

    downsample_ratio = int(
        ((orig_height * orig_width) // features) ** 0.5,
    )
    height, width = (
        orig_height // downsample_ratio,
        orig_width // downsample_ratio,
    )
    window_size = (height // 2, width // 2)

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
) -> tuple[UnetPatcher]:
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
        tuple: Contains the modified unet_patcher object.
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

    window_args = last_block = last_shift = None

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
            - Function uses random shift values (0-3) for window positioning
            - Maintains shift history to avoid consecutive same shifts
            - Handles cases where q, k, v are the same tensor for efficiency
        """

        nonlocal window_args, last_shift, last_block
        window_args = None
        last_block = extra_options.get("block")
        if last_block not in use_blocks or not check_time(
            extra_options,
            start_sigma,
            end_sigma,
        ):
            return q, k, v
        orig_shape = extra_options["original_shape"]

        # MSW-MSA
        shift = int(torch.rand(1, device="cpu").item() * 4)

        if shift == last_shift:
            shift = (shift + 1) % 4
        last_shift = shift
        window_args = tuple(
            get_window_args(x, orig_shape, shift) if x is not None else None
            for x in (q, k, v)
        )
        try:
            if q is not None and q is k and q is v:
                return (
                    window_partition(
                        q,
                        *window_args[0],
                    ),
                ) * 3
            return tuple(
                window_partition(x, *window_args[idx]) if x is not None else None
                for idx, x in enumerate((q, k, v))
            )
        except RuntimeError as exc:
            errstr = f"\x1b[31mMSW-MSA attention error: Incompatible model patches or bad resolution. Try using resolutions that are multiples of 32 or 64. Original exception: {exc}\x1b[0m"
            raise RuntimeError(errstr) from exc

    def attn1_output_patch(
        n: torch.Tensor, extra_options: dict[str, str]
    ) -> torch.Tensor:
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

        nonlocal window_args
        if window_args is None or last_block != extra_options.get("block"):
            window_args = None
            return n
        args, window_args = window_args[0], None
        return window_reverse(n, *args)

    unet_patcher.set_model_attn1_patch(attn1_patch)
    unet_patcher.set_model_attn1_output_patch(attn1_output_patch)
    return (unet_patcher,)


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

    if model_type == "SD15":
        blocks: tuple[str] = ("1,2", "", "11,10,9")
    elif model_type == "SDXL":
        blocks: tuple[str] = ("4,5", "", "5,4")
    else:
        raise ValueError("Unknown model type")

    prettyblocks = " / ".join(b if b else "none" for b in blocks)

    logging.debug(
        f"** ApplyMSWMSAAttentionSimple: Using preset {model_type}: in/mid/out blocks [{prettyblocks}], start/end percent {time_range[0]:.2}/{time_range[1]:.2}",
    )

    return apply_mswmsaa_attention(model, *blocks, "percent", *time_range)
