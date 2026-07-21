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
        self.first_directions_collection: dict[int, torch.Tensor | None] = {}

        # v (normalzied)
        self.second_directions_collection: dict[int, torch.Tensor | None] = {}  # v

        # 0: not adaptive
        # 1: adaptive to 1st direction on span(1st dir, 2nd dir)
        # 2: adaptive to 2nd direction on span(1st dir, 2nd dir)
        # 3: adaptive to 1st direction on span(1st dir, hidden_states)
        # 4: non-adaptive on span(1st dir, hidden_states)
        # 5: activation addition
        # 6: directional ablation
        # 7: Non Parametric Steering
        # 8: Cluster-PCA Steering
        # 9: Non Parametric Steering (Directional Ablation)
        # 10: Cluster-PCA Steering (Directional Ablation)
        # 11: Corrected-PCA Steering
        # 12: Corrected-PCA Steering (Directional Ablation)

        # Rebuttal 
        # 13: CHaRS with diagonal Affine Maps
        # 14: CHaRS with low-rank Affine Maps
        self.adaptive_mode_collection: dict[int, int] = {}

        self.target_degree_collection: dict[int, float] = {}

        self.scale_factor_collection: dict[int, float] = {}

        # u@u^T + v@v^T
        self.proj_matrices: dict[int, torch.Tensor | None] = {}

        # [u v] @ R_theta @ [1 0]^T
        self.rotated_components: dict[int, torch.Tensor | None] = {}

        # Adaptive 7 & 8 Paramaters

        # src clusters
        self.source_acts_normed_clusters_collection: dict[int, torch.Tensor | None] = {}

        # tgt clusters
        self.target_acts_normed_clusters_collection: dict[int, torch.Tensor | None] = {}

        # transport plans
        self.transport_plan_collection: dict[int, torch.Tensor | None] = {}

        # similarity kernel
        self.similarity_kernel_collection: dict[int, str] = {} 

        # cluster steering vectors
        self.cluster_steering_vectors: dict[int, torch.Tensor | None] = {}

        # Bandaids for adaptive mode 5-8
        self.new_adaptive: dict[int, bool] = {}
        self.steering_vec_reversed: dict[int, bool] = {}
        self.normed_ot_weighted_mean: dict[int, torch.Tensor | None] = {}

        self.active_index: int | None = None

        self.total_count = torch.zeros((), device="cuda", dtype=torch.int32)
        self.count = torch.zeros((), device="cuda", dtype=torch.int32)

        # Corrected PC Steering (Mode 11, 12)
        self.v_bar_collection: dict[int, torch.Tensor | None] = {}
        self.pc_scores_collection: dict[int, torch.Tensor | None] = {} # k, i, j index
        self.top_K_pc_collection: dict[int, torch.Tensor | None] = {}
        self.no_of_pc_collection: dict[int, int] = {}

        # Diagonal Affine Map Matrices
        self.diag_affine_map_collection: dict[int, torch.Tensor | None] = {}
        self.low_rank_components_collection: dict[int, torch.Tensor | None] = {}

        # self.total_count = 0
        # self.count = 0

    def set_control_vector(self, index, steer_weights: SteererWeights) -> None:
        """Set a control vector at a specific index."""
        self.reset_control_vector(index)

        first_direction = steer_weights.first_direction
        second_direction = steer_weights.second_direction
        target_degree = steer_weights.target_degree

        if first_direction is not None:
            self.first_directions_collection[index] = ( 
                first_direction / first_direction.norm()
            )
        else:
            self.first_directions_collection[index] = None
        if second_direction is not None:
            self.second_directions_collection[index] = (
                second_direction / second_direction.norm()
            )
        else:
            self.second_directions_collection[index] = None
        self.adaptive_mode_collection[index] = steer_weights.adaptive_mode
        self.target_degree_collection[index] = target_degree
        self.scale_factor_collection[index] = steer_weights.scale_factor

        if steer_weights.adaptive_mode == 5:
            self.first_directions_collection[index] = first_direction

        proj_matrix, rotated_component = self._get_rotation_args(
            self.first_directions_collection[index],
            self.second_directions_collection[index],
            self.target_degree_collection[index],
        )

        # Indicator that Active Index can be called
        self.proj_matrices[index] = proj_matrix
        self.rotated_components[index] = rotated_component

        # Adaptive Mode 7, 8 Params
        src_clusters = steer_weights.source_acts_normed_clusters
        tgt_clusters = steer_weights.target_acts_normed_clusters
        transport_plan = steer_weights.transport_plan
        cluster_steering_vectors = steer_weights.cluster_steering_vectors

        self.source_acts_normed_clusters_collection[index] = src_clusters
        self.target_acts_normed_clusters_collection[index] = tgt_clusters
        self.transport_plan_collection[index] = transport_plan
        self.similarity_kernel_collection[index] = steer_weights.similarity_kernel
        self.cluster_steering_vectors[index] = cluster_steering_vectors

        new_adaptive = steer_weights.new_adaptive
        steering_vec_reversed = steer_weights.steering_vec_reversed
        self.new_adaptive[index] = new_adaptive
        self.steering_vec_reversed[index] = steering_vec_reversed

        if steer_weights.adaptive_mode == 7:
            steering_vectors = tgt_clusters[None, :, :] - src_clusters[:, None, :]
            scaled_steering_vectors = steering_vectors * transport_plan[:, :, None]
            steering_vector_bar = scaled_steering_vectors.sum(dim = (0, 1))
            normed_steering_vector_bar = steering_vector_bar / steering_vector_bar.norm()
            self.normed_ot_weighted_mean[index] = normed_steering_vector_bar
        elif steer_weights.adaptive_mode == 8:
            scaled_steering_vectors = cluster_steering_vectors * transport_plan[:, :, None]
            steering_vector_bar = scaled_steering_vectors.sum(dim = (0, 1))
            normed_steering_vector_bar = steering_vector_bar / steering_vector_bar.norm()
            self.normed_ot_weighted_mean[index] = normed_steering_vector_bar
        else:
            self.normed_ot_weighted_mean[index] = None

        # Adaptive Modes 11 and 12 (Corrected PC Steering)
        v_bar = steer_weights.v_bar # This is D
        pc_scores = steer_weights.pc_scores # This is (K, i, j)
        top_K_pc = steer_weights.top_K_pc # This is (K, D)
        no_of_pc = steer_weights.no_of_pc 

        # Assertion check of no of pc against top_K_pc
        if top_K_pc is not None:
            max_no_of_pc = top_K_pc.shape[0]
            assert 0 <= no_of_pc <= max_no_of_pc, f"Invalid Choice of PC! Chosen: {no_of_pc}; Maximum: {max_no_of_pc}"

        self.v_bar_collection[index] = v_bar
        self.pc_scores_collection[index] = pc_scores
        self.top_K_pc_collection[index] = top_K_pc
        self.no_of_pc_collection[index] = no_of_pc

        # (Rebuttal) Adaptive mode 13
        self.diag_affine_map_collection[index] = steer_weights.diag_affine_map

        # (Rebuttal) Adaptive mode 14
        self.low_rank_components_collection[index] = steer_weights.low_rank_components
            
    def _get_rotation_args(
        self,
        first_directions: Optional[torch.Tensor],
        second_directions: Optional[torch.Tensor],
        target_degree: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the rotated component with respect to a 2D subspace and an rotation
        angle."""

        if (first_directions is None) or (second_directions is None):
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
        # This is sufficient as in set_active_tensor, removing this will force the active index to be None
        # This allows CV for layer 1 only, then CV for layer 2 next
        # Index will always be 0 for current implementation
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

        self.total_count.add_(1)
        # self.total_count += 1
        # print("Total Call Count:", self.total_count, "Active Index", self.active_index)

        if self.active_index is not None:
            self.count.add_(1)
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
            if adaptive_mode not in {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
                # hidden_states: batch x hidden_dim
                # proj_matrix: (batch) x hidden_dim x hidden_dim
                # Px: batch x hidden_dim
                # scale: batch x 1
                Px = torch.einsum("...i, ...ij -> ...j", hidden_states, proj_matrix)
                scale = Px.norm(dim=-1, keepdim=True)

            if adaptive_mode in {5}:
                # print("Call Count:", self.count, "Total Call Count:", self.total_count)
                feature_direction = self.first_directions_collection[
                    self.active_index
                ].to(device, dtype=dtype)
                new_adaptive = self.new_adaptive[self.active_index]
                steering_vec_reversed = self.steering_vec_reversed[self.active_index]
                scale_factor = self.scale_factor_collection[self.active_index]
                if new_adaptive:
                    if not steering_vec_reversed:
                        refusal_dir = - feature_direction
                        raise NotImplementedError
                    else:
                        refusal_dir = feature_direction
                    proj_to_feature_direction = (hidden_states @ refusal_dir)
                    mask = proj_to_feature_direction > 0
                    hidden_states += mask.unsqueeze(1) * (scale_factor * feature_direction)
                else:
                    hidden_states += scale_factor * feature_direction
            elif adaptive_mode in {6}:
                feature_direction = self.first_directions_collection[
                    self.active_index
                ].to(device, dtype=dtype)
                new_adaptive = self.new_adaptive[self.active_index]
                steering_vec_reversed = self.steering_vec_reversed[self.active_index]
                proj_to_feature_direction = hidden_states @ feature_direction
                if new_adaptive:
                    if not steering_vec_reversed:
                        mask = (- proj_to_feature_direction) > 0
                        raise NotImplementedError
                    else:
                        mask = proj_to_feature_direction > 0
                    hidden_states += mask.unsqueeze(1) * (-(proj_to_feature_direction.unsqueeze(1) * feature_direction))
                
                else:
                    hidden_states -= (
                        proj_to_feature_direction.unsqueeze(1) * feature_direction
                    )
            elif adaptive_mode in {7, 8, 9, 10, 11, 12, 13, 14}:
                
                # self.count += 1 
                new_adaptive = self.new_adaptive[self.active_index]
                steering_vec_reversed = self.steering_vec_reversed[self.active_index]
                
                assert 1 < len(hidden_states.shape) <= 3, f"Need to Revisit and Account for this, current shape is {hidden_states.shape}"
                if len(hidden_states.shape) == 3:
                    B, L, D = hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
                    BL = B*L
                else:
                    B, L = None, None
                    BL, D = hidden_states.shape[0], hidden_states.shape[1]

                # Load Vectors
                src_clusters = self.source_acts_normed_clusters_collection[self.active_index].to(device, dtype=dtype)
                transport_plan = self.transport_plan_collection[self.active_index].to(device, dtype=dtype)
                scale_factor = self.scale_factor_collection[self.active_index]
                similarity_kernel = self.similarity_kernel_collection[self.active_index]

                # Similarity Kernel
                if similarity_kernel == "gaussian":
                    similarity_hs_src = torch.exp(- (torch.linalg.norm(hidden_states.reshape(BL, D)[:,None,:].to(dtype = torch.float32)-src_clusters[None,:,:].to(dtype =torch.float32), dim=-1) ** 2) / (D ** 0.5) )
                    similarity_hs_src = similarity_hs_src / torch.sum(similarity_hs_src, dim = -1, keepdim = True)
                    similarity_hs_src = similarity_hs_src.to(dtype=dtype)
                elif similarity_kernel == "adaptive_gaussian":
                    similarity_hs_src = (torch.linalg.norm(hidden_states.reshape(BL, D)[:,None,:].to(dtype = torch.float32)-src_clusters[None,:,:].to(dtype =torch.float32), dim=-1) ** 2)
                    similarity_hs_src = torch.exp(- (similarity_hs_src / torch.median(similarity_hs_src, dim = -1, keepdim=True).values))
                    similarity_hs_src = similarity_hs_src / torch.sum(similarity_hs_src, dim = -1, keepdim = True)
                    similarity_hs_src = similarity_hs_src.to(dtype=dtype)
                else:
                    raise NotImplementedError

                if adaptive_mode in {7, 8, 9, 10, 11, 12}:
                    if adaptive_mode in {7, 8, 9, 10}:
                        if adaptive_mode in {7, 9}:
                            tgt_clusters = self.target_acts_normed_clusters_collection[self.active_index].to(device, dtype=dtype)
                            # Steering vectors is (i, j) indexed as j belongs to tgt and i belongs to src
                            steering_vectors = tgt_clusters[None, :, :] - src_clusters[:, None, :]
                        else:
                            steering_vectors = self.cluster_steering_vectors[self.active_index].to(device, dtype=dtype)
                        
                        # Compute the steering direction: P_ij * K_i where transport plan and sim scores are (i,j) and (L,i) indexed respectively
                        num = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                        # num = num[:, :, :, None] * (tgt_clusters[None, :, None, :] - src_clusters[None, None, :, :])
                        num = num[:, :, :, None] * steering_vectors[None, :, :, :]
                        num = num.sum(axis = -3).sum(axis = -2)
                        # denom = transport_plan * similarity_hs_src[:, None]
                        denom = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                        denom = denom.sum(axis = -2).sum(axis = -1)
                        feature_direction = num / denom[:, None]
                        
                        
                    elif adaptive_mode in {11, 12}:
                        no_of_pc = self.no_of_pc_collection[self.active_index]
                        v_bar = self.v_bar_collection[self.active_index]
                        pc_scores = self.pc_scores_collection[self.active_index]
                        top_K_pc = self.top_K_pc_collection[self.active_index]

                        # Pruning Based on Number of PC
                        pc_scores = pc_scores[:no_of_pc] # k. i, j
                        top_K_pc = top_K_pc[:no_of_pc] # k, D

                        # Compute the steering direction: P_ij * K_i where transport plan and sim scores are (i,j) and (L,i) indexed respectively
                        num = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                        # L tokens, multiply with pc scores: "L i j, k i j -> L k i j"
                        num = num[:, None, :, :] * pc_scores[None, :, :, :]
                        num = num.sum(axis = -2).sum(axis = -1) # L k i j -> L k
                        denom = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                        denom = denom.sum(axis = -2).sum(axis = -1) # L,
                        pc_coeff = num / denom[:, None] # L k, L -> L, k
                        scaled_pc = pc_coeff[:, :, None] * top_K_pc[None, :, :] # L k, k D -> L, k, D
                        scaled_pc = scaled_pc.sum(axis = -2) # L, k, D -> L, D
                        feature_direction = v_bar[None, :] + scaled_pc

                    if B is not None:
                        feature_direction = feature_direction.reshape(B, L, D)
                        raise NotImplementedError


                    # print(hidden_states.shape, hidden_states)
                    # torch.save(hidden_states, '/data/ryanltz/20250811_Steering/lda_steering/temp_states_3/vllm_states.pt')

                    # print(feature_direction)

                    # ActAdd Routine
                    if adaptive_mode in {7, 8, 11}:
                        if new_adaptive:
                            if adaptive_mode == 11:
                                raise NotImplementedError
                            normed_steering_vector_bar = self.normed_ot_weighted_mean[self.active_index]
                            if not steering_vec_reversed:
                                refusal_dir = - normed_steering_vector_bar
                            else:
                                refusal_dir = normed_steering_vector_bar
                                raise NotImplementedError
                            refusal_dir = refusal_dir.to(device, dtype=dtype)
                            proj_to_feature_direction = (hidden_states @ refusal_dir)
                            mask = proj_to_feature_direction > 0
                            hidden_states += mask.unsqueeze(1) * (scale_factor * feature_direction)
                            
                        else:
                            # += here casts the dtype back to hidden_states.dtype
                            hidden_states += scale_factor * feature_direction
                    else:
                        if new_adaptive:
                            raise NotImplementedError
                        else:
                            # Normalize Feature Vector
                            normed_feature_direction = feature_direction / feature_direction.norm(dim = -1, keepdim = True)
                            projection_coeff = (hidden_states * normed_feature_direction).sum(dim = -1)
                            hidden_states -= (
                                projection_coeff.unsqueeze(1) * normed_feature_direction
                            )
                
                # (Rebuttal) Adaptive Mode 13 with diagonal affine maps
                elif adaptive_mode in {13}:
                    tgt_clusters = self.target_acts_normed_clusters_collection[self.active_index].to(device, dtype=dtype)
                    diag_affine_map = self.diag_affine_map_collection[self.active_index].to(device, dtype=dtype)

                    # src_clusters: (I, D)
                    # tgt_clusters: (J, D)
                    # diag_affine_map: (I, J, D)
                    # hs: (BL, D)
                    # sm_hs: (BL, I)

                    # Estimate Maps T_ij
                    # hs - m_1 -> BL D, I D -> BL, I, D
                    # A(hs - m_1) -> I J D, BL I D -> BL, I, J, D
                    # m_2 + A(hs - m_1) -> J D, BL I J D ->BL, I, J, D
                    if scale_factor <= 1:
                        transported_hs = hidden_states[:, None, :] - src_clusters[None, :, :]
                        transported_hs = diag_affine_map[None, :, :, :] * transported_hs[:, :, None, :]
                        transported_hs = tgt_clusters[None, None, :, :] + transported_hs[:, :, :, :]
                        
                        # Interpolation: BL D, BL I J D -> BL I J D
                        transported_hs = ((1 - scale_factor) * hidden_states)[:, None, None, :] + scale_factor * transported_hs
                    else:
                        # Am_1: I J D, I D -> I J D
                        # m_2 - Am_1: J D, I J D -> I J D 
                        difference_component = tgt_clusters[None, :, :] - diag_affine_map[:, :, :] * src_clusters[:, None, :]
                        transported_hs = (diag_affine_map[None, :, :, :] * hidden_states[:, None, None, :]) + (scale_factor * difference_component)[None, :, :, :]

                    # Numerator
                    num = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                    # B I J, B I J D -> B, I, J, D
                    transported_hs = num[:, :, :, None] * transported_hs[:, :, :, :]
                    # B I J D -> B D
                    transported_hs = transported_hs.sum(axis = -3).sum(axis = -2) 
                    # Denominator
                    denom = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                    denom = denom.sum(axis = -2).sum(axis = -1)
                    hidden_states = transported_hs / denom[:, None]

                # (Rebuttal) Adaptive Mode 14 with low rank affine maps
                else:
                    tgt_clusters = self.target_acts_normed_clusters_collection[self.active_index].to(device, dtype=dtype)
                    low_rank_components = self.low_rank_components_collection[self.active_index].to(device, dtype=dtype)

                    # src_clusters: (I, D)
                    # tgt_clusters: (J, D)
                    # low_rank_components: (I, J, D, r)
                    # hs: (BL, D)
                    # sm_hs: (BL, I)

                    # Estimate Maps T_ij
                    # hs - m_1 -> BL D, I D -> BL, I, D
                    # temp = C^T(hs - m_1) -> I J r D, BL I D -> BL, I, J, r
                    # hs = C temp -> I J D r, BL I J r -> BL, I, J, D
                    # m_2 + A(hs - m_1) -> J D, BL I J D ->BL, I, J, D
                    if scale_factor <= 1:
                        transported_hs = hidden_states[:, None, :] - src_clusters[None, :, :]
                        transported_hs = torch.matmul(transported_hs[:, :, None, None, :], low_rank_components[None, :, :, :, :]).squeeze(-2)
                        transported_hs = torch.matmul(transported_hs[:, :, :, None, :], low_rank_components[None, :, :, :, :].transpose(-1, -2)).squeeze(-2)
                        transported_hs = tgt_clusters[None, None, :, :] + transported_hs[:, :, :, :]
                        
                        # Interpolation: BL D, BL I J D -> BL I J D
                        transported_hs = ((1 - scale_factor) * hidden_states)[:, None, None, :] + scale_factor * transported_hs
                    else:
                        # C^T a -> I J r D, I D-> I J r (by I None None D, I J D r -> I J r)
                        # C (C^T a) -> I J D r, I J r -> I J D
                        # m_2 - Am_1: J D, I J D -> I J D 
                        difference_component = torch.matmul(src_clusters[:, None, None, :], low_rank_components[:, :, :, :]).squeeze(-2)
                        difference_component = torch.matmul(difference_component[:, :, None, :], low_rank_components[:, :, :, :].transpose(-1, -2)).squeeze(-2)
                        difference_component = tgt_clusters[None, :, :] - difference_component[:, :, :]

                        # C^T x -> I J r D, BL D-> BL I J r (by BL None None None D, None I J D r -> BL I J r)
                        # C (C^T x) -> I J D r, BL I J r -> BL I J D (by BL I J None r, None I J r D -> BL I J D)
                        # m_2 - Am_1: J D, I J D -> I J D 
                        transported_hs = torch.matmul(hidden_states[:, None, None, None, :], low_rank_components[None, :, :, :, :]).squeeze(-2)
                        transported_hs = torch.matmul(transported_hs[:, :, :, None, :], low_rank_components[None, :, :, :, :].transpose(-1, -2)).squeeze(-2)
                        transported_hs = transported_hs + (scale_factor * difference_component)[None, :, :, :]

                    # Numerator
                    num = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                    # B I J, B I J D -> B, I, J, D
                    transported_hs = num[:, :, :, None] * transported_hs[:, :, :, :]
                    # B I J D -> B D
                    transported_hs = transported_hs.sum(axis = -3).sum(axis = -2) 
                    # Denominator
                    denom = transport_plan[None, :, :] * similarity_hs_src[:, :, None]
                    denom = denom.sum(axis = -2).sum(axis = -1)
                    hidden_states = transported_hs / denom[:, None]


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
