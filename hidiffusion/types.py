from dataclasses import dataclass
from typing import TypeAlias

import torch

from .utils import check_time


UPSCALE_METHODS = ["bicubic", "bislerp", "bilinear", "nearest-exact", "area"]

MODEL_TYPES = ["SD 1.5", "SDXL"]

MODE_LEVELS = ["Simple", "Advanced"]

RESOLUTION_MODES = {
    "low (1024 or lower)": "low",
    "high (1536-2048)": "high",
    "ultra (over 2048)": "ultra",
}


Blocks: TypeAlias = tuple[str, ...]
FloatRange: TypeAlias = tuple[float, float]


class HDConfigClass:
    """A configuration class for high-definition image processing.
    This class manages settings and parameters for the HD upscale of the generation.
    Attributes:
        start_sigma (float): Starting sigma value for processing range.
        end_sigma (float): Ending sigma value for processing range.
        use_blocks (set): List of valid processing blocks.
        two_stage_upscale (bool): Whether to use two-stage upscaling. Defaults to True.
        upscale_mode (str): Upscaling algorithm to use. Defaults to "bislerp".
    Methods:
        check(topts): Validates processing options against configuration settings.
            Args:
                topts (dict): Dictionary containing processing options to validate.
            Returns:
                bool: True if options are valid according to configuration, False otherwise.
    """

    start_sigma: float = 0.0
    end_sigma: float = 1.0
    use_blocks: set[tuple[str, int]] = []
    two_stage_upscale: bool = True
    upscale_mode: str = UPSCALE_METHODS[0]

    def check(self, topts: dict[str, torch.Tensor]) -> bool:
        if not isinstance(topts, dict):
            return False
        if topts.get("block") not in self.use_blocks:
            return False
        return check_time(topts, self.start_sigma, self.end_sigma)


@dataclass
class ResolutionConfig:
    ca_blocks: Blocks
    time_range: FloatRange
    ca_time_range: FloatRange

    def __iter__(self):
        return iter((self.ca_blocks, self.time_range, self.ca_time_range))


@dataclass
class ModelConfig:
    blocks: Blocks
    ca_blocks: Blocks
    modes: dict[str, ResolutionConfig]

    def __iter__(self):
        return iter((self.blocks, self.ca_blocks, self.modes))
