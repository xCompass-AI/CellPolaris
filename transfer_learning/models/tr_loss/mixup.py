import torch
import torch.nn as nn
from torch.distributions.beta import Beta


class Mixup(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        assert 0 <= alpha
        self.alpha = alpha
        self.criterion = nn.MSELoss(reduction="none")

    def forward(
        self, link_prediction_model, x0, x1, embed0, embed1, y, domain_pairs=None
    ):
        r"""
        Args:
            link_prediction_model: nn.Module
            x0: (batch_size, num_features)
            x1: (batch_size, num_features)
            embed0: (batch_size, embed_dim)
            embed1: (batch_size, embed_dim)
            y: (batch_size, 1)
            domain_pairs (2, num_domain_pairs): Indicate which domains are paired.
                If set to None, applying random pairing.
        """
        if domain_pairs is None:

            num_domains = len(x0)
            domain_index = torch.randperm(num_domains)
            pair0 = domain_index[: num_domains // 2]
            pair1 = domain_index[num_domains // 2 :]

            x0_pair0 = torch.cat([x0[i] for i in pair0], dim=0)
            x0_pair1 = torch.cat([x1[i] for i in pair1], dim=0)
            num_pair0 = x0_pair0.shape[0]
            num_pair1 = x0_pair1.shape[0]

            if num_pair1 < num_pair0:
                pair0, pair1 = pair1, pair0
                x0_pair0, x0_pair1 = x0_pair1, x0_pair0
                num_pair0, num_pair1 = num_pair1, num_pair0
            perm_pair1 = torch.randperm(num_pair1)
            x0_pair1 = x0_pair1[perm_pair1][:num_pair0]

            x1_pair0 = torch.cat([x1[i] for i in pair0], dim=0)
            x1_pair1 = torch.cat([x1[i] for i in pair1], dim=0)[perm_pair1][:num_pair0]
            embed0_pair0 = torch.cat([embed0[i] for i in pair0], dim=0)
            embed0_pair1 = torch.cat([embed0[i] for i in pair1], dim=0)[perm_pair1][:num_pair0]
            embed1_pair0 = torch.cat([embed1[i] for i in pair0], dim=0)
            embed1_pair1 = torch.cat([embed1[i] for i in pair1], dim=0)[perm_pair1][:num_pair0]
            y_pair0 = torch.cat([y[i] for i in pair0], dim=0).unsqueeze(1)
            y_pair1 = torch.cat([y[i] for i in pair1], dim=0)[perm_pair1][:num_pair0].unsqueeze(1)


            out, mixup_lambda = self.mixup_forward(
                link_prediction_model,
                x0_pair0,
                x0_pair1,
                x1_pair0,
                x1_pair1,
                embed0_pair0,
                embed0_pair1,
                embed1_pair0,
                embed1_pair1,
            )

        else:
            domain_pairs = self.drop_unbalance_domain_pairs(x0, domain_pairs)
            domain_pairs = domain_pairs[:, torch.randperm(domain_pairs.shape[1])]
            domain_pairs = torch.split(domain_pairs, 100, dim=1)
            out, mixup_lambda, y_pair0, y_pair1 = [], [], [], []
            for pairs in domain_pairs:
                x0_pair0 = torch.cat([x0[i] for i in pairs[0]], dim=0)
                x0_pair1 = torch.cat([x0[i] for i in pairs[1]], dim=0)
                x1_pair0 = torch.cat([x1[i] for i in pairs[0]], dim=0)
                x1_pair1 = torch.cat([x1[i] for i in pairs[1]], dim=0)
                embed0_pair0 = torch.cat([embed0[i] for i in pairs[0]], dim=0)
                embed0_pair1 = torch.cat([embed0[i] for i in pairs[1]], dim=0)
                embed1_pair0 = torch.cat([embed1[i] for i in pairs[0]], dim=0)
                embed1_pair1 = torch.cat([embed1[i] for i in pairs[1]], dim=0)
                y_pair0.append(torch.cat([y[i] for i in pairs[0]], dim=0).unsqueeze(1))
                y_pair1.append(torch.cat([y[i] for i in pairs[1]], dim=0).unsqueeze(1))
                out_, mixup_lambda_ = self.mixup_forward(
                    link_prediction_model,
                    x0_pair0,
                    x0_pair1,
                    x1_pair0,
                    x1_pair1,
                    embed0_pair0,
                    embed0_pair1,
                    embed1_pair0,
                    embed1_pair1,
                )
                out.append(out_)
                mixup_lambda.append(mixup_lambda_)
            out = torch.cat(out, dim=0)
            mixup_lambda = torch.cat(mixup_lambda, dim=0)
            y_pair0 = torch.cat(y_pair0, dim=0)
            y_pair1 = torch.cat(y_pair1, dim=0)

        loss0 = self.criterion(out, y_pair0.squeeze())
        loss1 = self.criterion(out, y_pair1.squeeze())
        loss = mixup_lambda.squeeze() * loss0 + (1 - mixup_lambda).squeeze() * loss1
        loss = loss.mean()

        return loss

    def drop_unbalance_domain_pairs(self, x, domain_pairs):
        r"""
        Some domains may have few samples, which cannot be paired.
            This function drops these unbalanced domains.

        Args:
            x: (batch_size, num_features)
            domain_pairs (2, num_domain_pairs): Indicate which domains are paired.

        Return:
            domain_pairs (2, num_domain_pairs): Drop unbalanced domain pairs.
        """
        balance_domain_pairs = []
        for pair in domain_pairs.t():
            if len(x[pair[0]]) == len(x[pair[1]]):
                balance_domain_pairs.append(pair)
        return torch.stack(balance_domain_pairs, dim=1)

    def mixup_forward(
        self,
        link_prediction_model,
        x0_pair0,
        x0_pair1,
        x1_pair0,
        x1_pair1,
        embed0_pair0,
        embed0_pair1,
        embed1_pair0,
        embed1_pair1,
    ):

        mixup_lambda = Beta(self.alpha, self.alpha).sample((x0_pair0.shape[0], 1))
        mixup_lambda = mixup_lambda.to(x0_pair0.device)

        x0 = mixup_lambda * x0_pair0 + (1 - mixup_lambda) * x0_pair1
        x1 = mixup_lambda * x1_pair0 + (1 - mixup_lambda) * x1_pair1
        embed0 = mixup_lambda * embed0_pair0 + (1 - mixup_lambda) * embed0_pair1
        embed1 = mixup_lambda * embed1_pair0 + (1 - mixup_lambda) * embed1_pair1
        out, _ = link_prediction_model(x0, x1, (embed0, embed1))
        return out, mixup_lambda
