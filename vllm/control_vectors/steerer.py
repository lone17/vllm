from typing import Optional

import torch


class SteererWeights:

    def __init__(
        self,
        first_direction: torch.Tensor,
        second_direction: Optional[torch.Tensor] = None,
        scale_factor: float = 1.0,
        target_degree: float = 0.0,
        keep_norm: bool = False,
        adaptive_mode: int = 0,
    ):
        self.first_direction = first_direction
        self.second_direction = second_direction
        self.scale_factor = scale_factor
        self.target_degree = target_degree
        self.keep_norm = keep_norm
        self.adaptive_mode = adaptive_mode
