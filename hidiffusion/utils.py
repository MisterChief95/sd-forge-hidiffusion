import torch
import torch.nn.functional as torchf

from backend.modules.k_prediction import Prediction

from packages_3rdparty.comfyui_lora_collection.utils import bislerp


def parse_blocks(name: str, s: str) -> set[tuple[str, int]]:
    """Parse a comma-separated string into a set of (name, int) tuples.
    This function takes a name and a comma-separated string of integers, and returns
    a set of tuples where each tuple contains the name and one of the parsed integers.
    Args:
        name (str): The name to be paired with each parsed integer value.
        s (str): A comma-separated string of integer values.
    Returns:
        set[tuple[str, int]]: A set of tuples, where each tuple contains:
            - The input name (str)
            - A parsed integer value (int)
    Example:
        >>> parse_blocks("block", "1, 2, 3")
        {('block', 1), ('block', 2), ('block', 3)}
    """

    vals = (rawval.strip() for rawval in s.split(","))
    return {(name, int(val.strip())) for val in vals if val}


def convert_time(predictor: Prediction, time_mode: str, start_time: float, end_time: float) -> tuple[float, float]:
    """Convert time values according to specified time mode.
    This function converts time values from various modes (sigma, percent, timestep) to sigma values
    used in the diffusion process.
    Args:
        predictor (Prediction): Prediction object containing percent_to_sigma conversion method
        time_mode (str): Mode for time interpretation. One of "sigma", "percent", or "timestep"
        start_time (float): Start time value in specified mode
        end_time (float): End time value in specified mode
    Returns:
        tuple[float, float]: A tuple containing (start_sigma, end_sigma) values
    Raises:
        ValueError: If time_mode is invalid or if percent values are out of range [0,1]
    """

    if time_mode == "sigma":
        return (start_time, end_time)

    if time_mode not in ("percent", "timestep"):
        raise ValueError("invalid time mode")

    if time_mode == "timestep":
        start_time = 1.0 - (start_time / 999.0)
        end_time = 1.0 - (end_time / 999.0)

    elif time_mode == "percent":
        if not (0.0 <= start_time <= 1.0):
            raise ValueError("start percent must be between 0 and 1")
        if not (0.0 <= end_time <= 1.0):
            raise ValueError("end percent must be between 0 and 1")

    return (
        predictor.percent_to_sigma(start_time),
        predictor.percent_to_sigma(end_time),
    )


def get_sigma(options: dict[str, torch.Tensor], key="sigmas") -> float | None:
    """
    Retrieves the maximum sigma value from a tensor stored in a dictionary.
    Args:
        options (dict[str, torch.Tensor]): Dictionary containing tensor values
        key (str, optional): Key to lookup in options dictionary. Defaults to "sigmas"
    Returns:
        Optional[float]: Maximum sigma value if found, None otherwise
    Example:
        >>> options = {"sigmas": torch.tensor([1.0, 2.0, 3.0])}
        >>> get_sigma(options)
        3.0
        >>> get_sigma({})
        None
    """

    if not isinstance(options, dict):
        return None

    sigmas = options.get(key)
    return sigmas.detach().cpu().max().item() if sigmas is not None else None


def check_time(options: dict[str, torch.Tensor], start_sigma: float, end_sigma: float) -> bool:
    """
    Check if the current sigma value falls within a specified range.
    Args:
        options (dict[str, torch.Tensor]): Dictionary containing model options and tensors
        start_sigma (float): Upper bound of the sigma range
        end_sigma (float): Lower bound of the sigma range
    Returns:
        bool: True if sigma exists and falls within [end_sigma, start_sigma], False otherwise
    Note:
        The function gets the sigma value from options using get_sigma() helper
        and checks if it exists and falls within the specified inclusive range.
    """

    sigma = get_sigma(options)

    return sigma is not None and sigma <= start_sigma and sigma >= end_sigma


def scale_samples(
    samples: torch.Tensor,
    width: int,
    height: int,
    mode: str | None = "bicubic",
) -> torch.Tensor:
    """
    Scale image samples to a target width and height using specified interpolation mode.
    Args:
        samples (torch.Tensor): Input tensor of image samples to be scaled
        width (int): Target width to scale to
        height (int): Target height to scale to
        mode (str, optional): Interpolation mode to use. Can be "bicubic" or "bislerp". Defaults to "bicubic"
    Returns:
        torch.Tensor: Scaled image samples tensor with shape matching target dimensions
    Example:
        >>> samples = torch.randn(1, 3, 64, 64)
        >>> scaled = scale_samples(samples, width=128, height=128)
        >>> scaled.shape
        torch.Size([1, 3, 128, 128])
    """

    if mode == "bislerp":
        return bislerp(samples, width, height)

    return torchf.interpolate(samples, size=(height, width), mode=mode)
