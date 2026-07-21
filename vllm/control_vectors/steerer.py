from typing import Optional

import torch


class SteererWeights:

    def __init__(
        self,
        first_direction: Optional[torch.Tensor] = None,
        second_direction: Optional[torch.Tensor] = None,
        scale_factor: float = 1.0,
        target_degree: float = 0.0,
        keep_norm: bool = False,
        adaptive_mode: int = 0,
        source_acts_normed_clusters: Optional[torch.Tensor] = None,
        target_acts_normed_clusters: Optional[torch.Tensor] = None,
        transport_plan: Optional[torch.Tensor] = None,
        similarity_kernel: str = "gaussian",
        cluster_steering_vectors: Optional[torch.Tensor] = None,
        new_adaptive: bool = False,
        steering_vec_reversed: bool = False,
        v_bar: Optional[torch.Tensor] = None,
        pc_scores: Optional[torch.Tensor] = None, # alpha scores
        top_K_pc: Optional[torch.Tensor] = None, # Let's suppose we store up to 20?
        no_of_pc: int = 10,
        diag_affine_map: Optional[torch.Tensor] = None, # Rebuttal: diag affine maps
        low_rank_components: Optional[torch.Tensor] = None,
    ):
        self.first_direction = first_direction
        self.second_direction = second_direction
        self.scale_factor = scale_factor
        self.target_degree = target_degree
        self.keep_norm = keep_norm
        self.adaptive_mode = adaptive_mode
        self.source_acts_normed_clusters = source_acts_normed_clusters
        self.target_acts_normed_clusters = target_acts_normed_clusters
        self.transport_plan = transport_plan
        self.similarity_kernel = similarity_kernel
        self.cluster_steering_vectors = cluster_steering_vectors
        self.new_adaptive = new_adaptive
        self.steering_vec_reversed = steering_vec_reversed
        self.v_bar = v_bar
        self.pc_scores = pc_scores
        self.top_K_pc = top_K_pc
        self.no_of_pc = no_of_pc
        self.diag_affine_map = diag_affine_map
        self.low_rank_components = low_rank_components

