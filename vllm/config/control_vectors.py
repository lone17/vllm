import torch
from dataclasses import dataclass
from typing import Optional

from vllm.config.utils import config


@config
@dataclass
class ControlVectorConfig:
    max_control_vectors: int
    adapter_dtype: Optional[torch.dtype] = torch.float16
    normalize: bool = False

    def __post_init__(self):
        if self.max_control_vectors < 1:
            raise ValueError("max_control_vectors must be >= 1")

