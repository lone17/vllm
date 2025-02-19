import gc
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn

from vllm.control_vectors.steerer import SteererWeights


@dataclass
class ControlVectorMapping:
    layer_mapping: Dict[int, torch.Tensor]


class BaseLayerWithControlVector(nn.Module):
    pass


class MLPWithControlVector(BaseLayerWithControlVector):

    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.control_vectors: Dict[int, torch.Tensor] = {}
        self.keep_norm = False
        self.active_index: int = None

    # def set_normalization(self, normalize: bool) -> None:
    #     self.keep_norm = normalize

    # def set_layer_id(self, layer_id: int) -> None:
    #     """assign the layer id of this MLP layer"""
    #     self.layer_id = layer_id

    def set_control_vector(self, index, steer_weights: SteererWeights) -> None:
        """Set a control vector at a specific index."""
        self.reset_control_vector(index)
        self.control_vectors[index] = (
            steer_weights.first_direction * steer_weights.scale_factor
        )
        self.keep_norm = steer_weights.keep_norm

    # def get_control_vector(self, index: int) -> Optional[torch.Tensor]:
    #     """Get a control vector by index."""
    #     return self.control_vectors.get(index)

    def reset_control_vector(self, index: int):
        """Reset a control vector to zero at a specific index."""
        if index in self.control_vectors:
            del self.control_vectors[index]

    def set_active_tensor(self, index: int):
        """Sets the active vector"""
        if index is not None and index in self.control_vectors:
            self.active_index = index
        else:
            self.active_index = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional application of control vectors."""
        print(hidden_states.shape)
        hidden_states = self.base_layer(hidden_states)
        print("  ", hidden_states.shape)
        cv = self.control_vectors.get(self.active_index)

        if cv is not None and cv.numel() > 0:
            norm_pre = torch.norm(hidden_states, dim=-1, keepdim=True)
            hidden_states += cv
            if self.keep_norm:
                norm_post = torch.norm(hidden_states, dim=-1, keepdim=True)
                hidden_states = hidden_states * norm_pre / norm_post

        return hidden_states


class LayerNormWithSteering(nn.Module):
    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.proj_matrices: dict[int, torch.Tensor] = {}  # u@u^T + v@v^T
        self.rotated_components: dict[int, torch.Tensor] = (
            {}
        )  # [u v] @ R_theta @ [1 0]^T
        self.active_index: float = None

    def set_control_vector(self, index, steer_weights: SteererWeights) -> None:
        """Set a control vector at a specific index."""
        self.reset_control_vector(index)

        first_direction = steer_weights.first_direction
        second_direction = steer_weights.second_direction
        target_degree = steer_weights.target_degree

        # ensure bases are orthonormal
        u = first_direction / first_direction.norm()
        v = second_direction - (second_direction @ u) * u
        v /= v.norm()

        theta = np.deg2rad(target_degree)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        proj_matrix = torch.outer(u, u) + torch.outer(v, v)

        uv = torch.column_stack([u, v])

        # rotate counter-clockwise
        R_theta = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            device=uv.device,
            dtype=uv.dtype,
        )

        rotated_component = (
            uv @ R_theta @ torch.tensor([1, 0], device=uv.device, dtype=uv.dtype)
        )

        self.proj_matrices[index] = proj_matrix
        self.rotated_components[index] = rotated_component

    def reset_control_vector(self, index: int):
        """Reset a control vector to zero at a specific index."""
        if index in self.proj_matrices:
            del self.proj_matrices[index]
            del self.rotated_components[index]

    def set_active_tensor(self, index: int):
        """Sets the active vector"""
        if index is not None and index in self.proj_matrices:
            self.active_index = index
        else:
            self.active_index = None

    def forward(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional application of control vectors."""
        if residual is None:
            hidden_states = self.base_layer(hidden_states)
        else:
            hidden_states, residual = self.base_layer(hidden_states, residual)

        if self.active_index is not None:
            device = hidden_states.device
            dtype = hidden_states.dtype
            proj_matrix = self.proj_matrices[self.active_index].to(device, dtype=dtype)
            rotated_component = self.rotated_components[self.active_index].to(
                device, dtype=dtype
            )

            Px = hidden_states @ proj_matrix
            scale = Px.norm(dim=-1, keepdim=True)

            hidden_states = hidden_states - Px + scale * rotated_component

        if residual is None:
            return hidden_states
        return hidden_states, residual
