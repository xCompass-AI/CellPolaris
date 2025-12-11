# -*- coding: UTF-8 -*-

import torch
import scipy.stats as stats


def z_score(x: torch.Tensor):
    return (x - x.mean()) / (x.std() + 1e-8)


def min_max_scale(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


class RNASeqStatisticsFeature:
    def compute(
        self,
        rna_seq: torch.Tensor,
        mask: torch.BoolTensor = None,
        is_bulk: bool = False,
        is_sc: bool = False,
        is_mouse: bool = False,
        is_human: bool = False,
    ):
        r"""
        Args:
            rna_seq: (num_genes, 1)
            mask: (num_genes,), indicates which genes are used
        """
        if mask is None:
            mask = torch.ones_like(rna_seq, dtype=torch.bool)
        normalized_rna_seq = self.normalize_rna_seq(rna_seq, mask)
        rank = self.rank(rna_seq, mask)

        domain_features = self.domain_features(
            rna_seq, mask, is_bulk, is_sc, is_mouse, is_human
        )
        cat_domain_features = torch.zeros((rna_seq.shape[0], domain_features.shape[0]))
        cat_domain_features[mask] = domain_features.repeat(mask.sum(), 1)

        return torch.hstack(
            [
                normalized_rna_seq.unsqueeze(1),
                rank,
                cat_domain_features,
            ],
        )

    def __call__(
        self,
        rna_seq: torch.Tensor,
        mask: torch.BoolTensor = None,
        is_bulk: bool = False,
        is_sc: bool = False,
        is_mouse: bool = False,
        is_human: bool = False,
    ):
        return self.compute(rna_seq, mask, is_bulk, is_sc, is_mouse, is_human)

    def normalize_rna_seq(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        rna_seq[mask] = min_max_scale((rna_seq[mask] + 1).log())
        return rna_seq


    def rank(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        rank = torch.zeros_like(rna_seq)
        rank_indices = rna_seq[mask].argsort(descending=True)
        rank_mask = rank[mask]
        rank_mask[rank_indices] = torch.arange(mask.sum()).float()
        rank[mask] = rank_mask

        rank = torch.stack(
            [
                rank / rank.max(),
                self.is_first_quarter(rank, mask),
                self.is_first_half(rank, mask),
                self.is_first_three_quarter(rank, mask),
            ],
            dim=1,
        )
        return rank

    def is_first_quarter(self, rank: torch.Tensor, mask: torch.BoolTensor):
        return rank < mask.sum() / 4

    def is_first_half(self, rank: torch.Tensor, mask: torch.BoolTensor):
        return rank < mask.sum() / 2

    def is_first_three_quarter(self, rank: torch.Tensor, mask: torch.BoolTensor):
        return rank < mask.sum() / 4 * 3

    def domain_features(
        self,
        rna_seq: torch.Tensor,
        mask: torch.BoolTensor,
        is_bulk: bool,
        is_sc: bool,
        is_mouse: bool,
        is_human: bool,
    ):
        cell_type = torch.zeros(4, dtype=torch.bool)
        if is_bulk:
            cell_type[0] = True
        if is_sc:
            cell_type[1] = True
        if is_mouse:
            cell_type[2] = True
        if is_human:
            cell_type[3] = True
        return torch.hstack(
            [
                cell_type.float(),
            ],
        )

    def mean(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return rna_seq[mask].mean()

    def std(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return rna_seq[mask].std()

    def max(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return rna_seq[mask].max()

    def min(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return rna_seq[mask].min()

    def median(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return rna_seq[mask].median()

    def first_quarter(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return rna_seq[mask].quantile(0.25)

    def third_quarter(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return rna_seq[mask].quantile(0.75)

    def iqr(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return self.third_quarter(rna_seq, mask) - self.first_quarter(rna_seq, mask)

    def skew(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return torch.FloatTensor([stats.skew(rna_seq[mask].numpy())]).squeeze()

    def kurtosis(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return torch.FloatTensor([stats.kurtosis(rna_seq[mask].numpy())]).squeeze()

    def min_max_range(self, rna_seq: torch.Tensor, mask: torch.BoolTensor):
        return self.max(rna_seq, mask) - self.min(rna_seq, mask)


if __name__ == "__main__":
    rna_seq = torch.rand(1000)
    mask = torch.rand(1000) > 0.5
    feature = RNASeqStatisticsFeature()
    out = feature(rna_seq, mask)
    ...
