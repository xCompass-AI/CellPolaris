from itertools import permutations
from typing import List

import torch
from torch import Tensor, nn

from .graph_discrepancy import GraphBasedDomainDiscrepancy
from .transfer_loss import TransferLoss


class TRModel(nn.Module):
    r"""Domain generalization model. DGModel."""

    def __init__(
        self,
        link_prediction_model: nn.Module,
        loss_type: str = "adv_multi",
        num_sourcedomains=1,
        mixup_alpha: float = None,
        ignore_regression: bool = False,
        top_ratio: float = 1,
    ):
        r"""
        Args:
            ignore_regression (bool): whether to ignore regression loss
                This argument is for graph-based methods. graph_mixup will ignore regression loss,
                while graph_mmd, graph_coral will not, this will influence whether to freeze bn layers.
        """
        super(TRModel, self).__init__()
        assert loss_type in [
            "adv_multi",
            "random_mixup",
            "graph_mixup",
            "graph_mmd",
            None,
        ]
        self.link_prediction_model = link_prediction_model
        self.input_dim = self.hidden_dim = link_prediction_model.out_feature_dim
        if loss_type is not None:
            self.transfer_loss = TransferLoss(
                loss_type=loss_type,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=num_sourcedomains,
                mixup_alpha=mixup_alpha,
            )
        self.transfer_loss_type = loss_type
        if loss_type is not None and loss_type.startswith("graph"):
            self.graph_discrepancy = GraphBasedDomainDiscrepancy(top_ratio=top_ratio)
        self.ignore_regression = ignore_regression

    def forward(
        self,
        x0: List[Tensor],
        x1: List[Tensor],
        edges: List[Tensor],
        tissue_index: List[Tensor],
        y: List[Tensor] = None,
    ):
        if self.transfer_loss_type is None:
            x0 = torch.cat(x0, dim=0)
            x1 = torch.cat(x1, dim=0)
            edges = torch.cat(edges, dim=0).t()
            y = torch.cat(y, dim=0)
            out, _ = self.link_prediction_model(x0, x1, edges)
            loss = None
        elif self.transfer_loss_type == "adv_multi":
            r"""
            For this method, the inputs are:
                x0: List of Tensor (batch_size, num_features)
                x1: List of Tensor (batch_size, num_features)
                edges: List of Tensor (2, batch_size)
                tissue_index: List of Tensor (batch_size)
                y: None
            """
            x0 = torch.cat(x0, dim=0)
            x1 = torch.cat(x1, dim=0)
            edges = torch.cat(edges, dim=0).t()
            y = torch.cat(y, dim=0)
            tissue_index = torch.cat(tissue_index, dim=0)
            out, embed = self.link_prediction_model(x0, x1, edges)
            loss = self.transfer_loss.compute(embed, tissue_index)
        elif self.transfer_loss_type == "random_mixup":
            r"""
            For this method, the inputs are:
                x0: List of Tensor (batch_size, num_features)
                x1: List of Tensor (batch_size, num_features)
                edges: List of Tensor (2, batch_size)
                tissue_index: neglected
                y: (batch_size, 1)
            """

            batch_sizes = [x.shape[0] for x in x0]
            cat_x0 = torch.cat(x0, dim=0)
            cat_x1 = torch.cat(x1, dim=0)
            edges = torch.cat(edges, dim=0).t()
            embed0, embed1 = self.link_prediction_model.get_gene_embedding(edges)
            out, _ = self.link_prediction_model(cat_x0, cat_x1, (embed0, embed1))

            embed0 = torch.split(embed0, batch_sizes)
            embed1 = torch.split(embed1, batch_sizes)


            loss = self.transfer_loss.compute(
                x0,
                x1,
                link_prediction_model=self.link_prediction_model,
                embed0=embed0,
                embed1=embed1,
                y=y,
            )
        elif self.transfer_loss_type.startswith("graph"):
            r"""
            Supported models:
                graph_mmd, graph_mixup, graph_coral, ...
            Considering each domain as a vertex, and the similarity between two domains
                as the edge weight. Then we can calculate the discrepancy between each pair of domains.
            For this method, the inputs are:
                x0: List of Tensor (batch_size, num_features)
                x1: List of Tensor (batch_size, num_features)
                edges: List of Tensor (2, batch_size)
                tissue_index: neglected
                y: None
            """
            mode = self.transfer_loss_type.split("_")[1]

            batch_sizes = [x.shape[0] for x in x0]
            cat_x0 = torch.cat(x0, dim=0)
            cat_x1 = torch.cat(x1, dim=0)
            edges = torch.cat(edges, dim=0).t()

            if self.training and self.ignore_regression:
                freeze_bn_layer(self.link_prediction_model)
            embed0, embed1 = self.link_prediction_model.get_gene_embedding(edges)
            out, feature = self.link_prediction_model(cat_x0, cat_x1, (embed0, embed1))
            if self.training and self.ignore_regression:
                unfreeze_bn_layer(self.link_prediction_model)
            feature = torch.split(feature, batch_sizes)
            domain_pairs = self.graph_discrepancy(feature)

            if mode in ["mmd"]:  # graph_mmd
                loss = self.transfer_loss.compute(
                    x0=None,
                    x1=None,
                    feature=feature,
                    domain_pairs=domain_pairs,
                )
            elif mode == "mixup":  # graph_mixup
                embed0 = torch.split(embed0, batch_sizes)
                embed1 = torch.split(embed1, batch_sizes)
                loss = self.transfer_loss.compute(
                    x0=x0,
                    x1=x1,
                    link_prediction_model=self.link_prediction_model,
                    embed0=embed0,
                    embed1=embed1,
                    domain_pairs=domain_pairs,
                    y=y,
                )

        return out, loss


def freeze_bn_layer(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()


def unfreeze_bn_layer(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
