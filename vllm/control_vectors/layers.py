import gc
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from sympy import print_jscode
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
        self.active_index: int | None = None

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
        # print(hidden_states.shape)
        hidden_states = self.base_layer(hidden_states)
        # print("  ", hidden_states.shape)
        if self.active_index is not None:
            cv = self.control_vectors.get(self.active_index)
        else:
            cv = None

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

        # u (normalized)
        self.first_directions_collection: dict[int, torch.Tensor] = {}

        # v (normalzied)
        self.second_directions_collection: dict[int, torch.Tensor | None] = {}  # v

        # 0: not adaptive
        # 1: adaptive to 1st direction on span(1st dir, 2nd dir)
        # 2: adaptive to 2nd direction on span(1st dir, 2nd dir)
        # 3: adaptive to 1st direction on span(1st dir, hidden_states)
        # 4: non-adaptive on span(1st dir, hidden_states)
        # 5: activation addition
        # 6: directional ablation
        self.adaptive_mode_collection: dict[int, int] = {}

        self.target_degree_collection: dict[int, float] = {}

        self.scale_factor_collection: dict[int, float] = {}

        # u@u^T + v@v^T
        self.proj_matrices: dict[int, torch.Tensor | None] = {}

        # [u v] @ R_theta @ [1 0]^T
        self.rotated_components: dict[int, torch.Tensor | None] = {}

        self.active_index: int | None = None

    def set_control_vector(self, index, steer_weights: SteererWeights) -> None:
        """Set a control vector at a specific index."""
        self.reset_control_vector(index)

        first_direction = steer_weights.first_direction
        second_direction = steer_weights.second_direction
        target_degree = steer_weights.target_degree

        self.first_directions_collection[index] = (
            first_direction / first_direction.norm()
        )
        if second_direction is not None:
            self.second_directions_collection[index] = (
                second_direction / second_direction.norm()
            )
        else:
            self.second_directions_collection[index] = None
        self.adaptive_mode_collection[index] = steer_weights.adaptive_mode
        self.target_degree_collection[index] = target_degree
        self.scale_factor_collection[index] = steer_weights.scale_factor

        proj_matrix, rotated_component = self._get_rotation_args(
            self.first_directions_collection[index],
            self.second_directions_collection[index],
            self.target_degree_collection[index],
        )

        self.proj_matrices[index] = proj_matrix
        self.rotated_components[index] = rotated_component

    def _get_rotation_args(
        self,
        first_directions: torch.Tensor,
        second_directions: Optional[torch.Tensor],
        target_degree: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the rotated component with respect to a 2D subspace and an rotation
        angle."""

        if second_directions is None:
            return None, None

        # first_direction: (batch) x hidden_dim
        # second_directions: (batch) x hidden_dim

        # ensure bases are orthonormal
        b1 = first_directions / first_directions.norm(dim=-1, keepdim=True)
        b2 = (
            second_directions
            - torch.sum(second_directions * b1, dim=-1, keepdim=True) * b1
        )
        b2 /= b2.norm(dim=-1, keepdim=True)

        theta = np.deg2rad(target_degree)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        proj_matrix = torch.einsum("...i, ...j -> ...ij", b1, b1) + torch.einsum(
            "...i, ...j -> ...ij", b2, b2
        )

        uv = torch.stack([b1.expand_as(b2), b2], dim=-1)  # shape (..., 2)

        # rotate counter-clockwise
        R_theta = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            device=uv.device,
            dtype=uv.dtype,
        )

        rotated_component = (
            uv @ R_theta @ torch.tensor([1, 0], device=uv.device, dtype=uv.dtype)
        )

        return proj_matrix, rotated_component

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional application of control vectors."""
        if residual is None:
            hidden_states = self.base_layer(hidden_states)
        else:
            hidden_states, residual = self.base_layer(hidden_states, residual)

        if self.active_index is not None:
            device = hidden_states.device
            dtype = hidden_states.dtype

            adaptive_mode = self.adaptive_mode_collection[self.active_index]

            rotated_component = None
            proj_matrix = None
            if adaptive_mode in {0, 1, 2}:
                proj_matrix = self.proj_matrices[self.active_index].to(
                    device, dtype=dtype
                )
                rotated_component = self.rotated_components[self.active_index].to(
                    device, dtype=dtype
                )
            elif adaptive_mode in {3, 4}:
                proj_matrix, rotated_component = self._get_rotation_args(
                    self.first_directions_collection[self.active_index],
                    hidden_states,
                    # self.second_directions_collection[self.active_index].expand_as(
                    #     hidden_states
                    # ),
                    self.target_degree_collection[self.active_index],
                )
                if proj_matrix is None or rotated_component is None:
                    if residual is None:
                        return hidden_states
                    else:
                        return hidden_states, residual

                proj_matrix = proj_matrix.to(device, dtype=dtype)
                rotated_component = rotated_component.to(device, dtype=dtype)

            Px = None
            scale = None
            if adaptive_mode not in {5, 6}:
                # hidden_states: batch x hidden_dim
                # proj_matrix: (batch) x hidden_dim x hidden_dim
                # Px: batch x hidden_dim
                # scale: batch x 1
                Px = torch.einsum("...i, ...ij -> ...j", hidden_states, proj_matrix)
                scale = Px.norm(dim=-1, keepdim=True)

            if adaptive_mode in {5}:
                feature_direction = self.first_directions_collection[
                    self.active_index
                ].to(device, dtype=dtype)
                scale_factor = self.scale_factor_collection[self.active_index]
                hidden_states += scale_factor * feature_direction
            elif adaptive_mode in {6}:
                feature_direction = self.first_directions_collection[
                    self.active_index
                ].to(device, dtype=dtype)
                proj_to_feature_direction = hidden_states @ feature_direction
                hidden_states -= (
                    proj_to_feature_direction.unsqueeze(1) * feature_direction
                )
            elif adaptive_mode in {0, 4}:
                hidden_states += -Px + scale * rotated_component
            else:
                if adaptive_mode in {1, 3, 5}:
                    feature_direction = self.first_directions_collection[
                        self.active_index
                    ]
                elif adaptive_mode == 2:
                    feature_direction = self.second_directions_collection[
                        self.active_index
                    ]
                else:
                    raise ValueError(f"Invalid adaptive mode: {adaptive_mode}")

                feature_direction = feature_direction.to(device, dtype=dtype)

                proj_to_feature_direction = hidden_states @ feature_direction
                mask = proj_to_feature_direction > 0

                # hidden_states: batch x hidden_dim
                # feature_direction: hidden_dim
                # proj_to_feature_direction: batch
                # mask: batch
                # scale: batch
                # rotated_component: (batch) x hidden_dim
                # Px: batch x hidden_dim

                hidden_states += mask.unsqueeze(1) * (scale * rotated_component - Px)

        if residual is None:
            return hidden_states
        return hidden_states, residual


if __name__ == "__main__":
    m = LayerNormWithSteering(None)
    b1 = torch.rand(10)
    b2 = torch.rand(10)
    bb2 = torch.rand(4, 10)

    proj_matrix1, rotated_component1 = m._get_rotation_args(b1, b2, 60)
    proj_matrix2, rotated_component2 = m._get_rotation_args(b1, b2.expand_as(bb2), 60)

    assert proj_matrix1.shape == (10, 10)
    assert rotated_component1.shape == (10,)
    assert proj_matrix2.shape == (4, 10, 10)
    assert rotated_component2.shape == (4, 10)
    assert torch.allclose(proj_matrix1, proj_matrix2[0])
    assert torch.allclose(rotated_component1, rotated_component2[0])

    proj_matrix1, rotated_component1 = m._get_rotation_args(b1, bb2[0], 60)
    proj_matrix2, rotated_component2 = m._get_rotation_args(b1, bb2, 60)
    assert torch.allclose(proj_matrix1, proj_matrix2[0])
    assert torch.allclose(rotated_component1, rotated_component2[0])
