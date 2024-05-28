import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse
from .tr_loss.mmd import MMD_loss


class GraphBasedDomainDiscrepancy(nn.Module):
    r"""
    Calculating discrepancy between each pair of domains.
    """

    def __init__(
        self,
        discrepancy: str = "mmd",
        keep_iterations: int = 10,
        top_ratio: float = 0.5,
        decay_factor: float = 0.999,
        min_top_ratio: float = 0.4,
        random_factor: float = 0.8,
    ):
        r"""
        Args:
            discrepancy: discrepancy type
            keep_iterations: number of iterations to re-compute the graph
            top_ratio: ratio of top edges to keep
            decay_factor: decay the top ratio
            min_top_ratio: minimum top ratio
            random_factor: adding a little randomness to domain_pairs
        """
        super().__init__()
        if discrepancy == "mmd":
            self.discrepancy = MMD_loss(kernel_type="linear")
        else:
            raise NotImplementedError
        self.keep_iterations = keep_iterations
        self.current_iteration = 0
        self.domain_pairs = None
        self._edge_index, self._edge_weight = None, None
        self.num_domains = None
        self.top_ratio = top_ratio
        self.decay_ratio = decay_factor
        self.min_top_ratio = min_top_ratio
        self.random_factor = random_factor

    def forward(self, feature: list):
        device = feature[0].device
        if self.num_domains is None:
            self.num_domains = len(feature)
        if len(feature) != self.num_domains:
            raise ValueError("Number of domains should be fixed.")
        if self.domain_pairs is None or self.current_iteration == self.keep_iterations:
            adj = torch.zeros(self.num_domains, self.num_domains, device=device)
            for i in range(self.num_domains):
                for j in range(i + 1, self.num_domains):
                    adj[i, j] = self.discrepancy(feature[i], feature[j])
            self._edge_index, self._edge_weight = dense_to_sparse(adj)
            self.current_iteration = 0
            self.top_ratio *= self.decay_ratio
            self.top_ratio = max(self.top_ratio, self.min_top_ratio)
        else:
            self.current_iteration += 1

        self.domain_pairs = self.get_domain_pairs(self.random_factor)
        return self.domain_pairs

    def get_domain_pairs(self, factor=0.4):
        r"""
        Adding a little randomness to domain_pairs.

        Args:
            ratio: value + rand * value.mean() * factor
        """
        return self._edge_index[
            :,
            torch.topk(
                self._edge_weight
                + torch.randn(
                    self._edge_weight.shape[0], device=self._edge_weight.device
                )
                * self._edge_weight.mean()
                * factor,
                max(int(self.top_ratio * self._edge_weight.shape[0]), 1),
            )[1],
        ]
