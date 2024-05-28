from torch import nn
from .tr_loss.adv_loss import Multi_adv
from .tr_loss.mixup import Mixup
from .tr_loss.mmd import MMD_loss


class TransferLoss(nn.Module):
    def __init__(
        self,
        loss_type="adv_multi",
        input_dim=5120,
        hidden_dim=192,
        output_dim=None,
        mixup_alpha: float = None,
    ):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, adv
        """
        super().__init__()
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if self.loss_type in ["adv_multi"]:
            self.multi_advloss = Multi_adv(
                self.input_dim, self.hidden_dim, self.output_dim
            )
        elif self.loss_type in ["random_mixup", "graph_mixup"]:
            self.mixup_alpha = mixup_alpha
            self.mixup = Mixup(alpha=self.mixup_alpha)

    def compute(self, x0, x1, **kwargs):
        """Compute transfer loss

        Arguments:
            x0 {tensor} -- source matrix
            x1 {tensor} -- target matrix
            kwargs {dict} -- other arguments, containing:
                For random_mixup and graph_mixup:
                    y {tensor} -- edge weight (label)
                    link_prediction_model {nn.Module} -- link prediction model
                    embed0 {tensor} -- embedding matrix of tfs
                    embed1 {tensor} -- embedding matrix of tgs
                For graph based methods (graph_mmd, graph_mixup):
                    domain_pairs {tensor} -- pairs of domain index
                For graph_mmd:
                    feature {List} -- List of feature matrix
                For graph_mixup:
                    y {tensor} -- edge weight (label)
                    link_prediction_model {nn.Module} -- link prediction model
                    embed0 {tensor} -- embedding matrix of tfs
                    embed1 {tensor} -- embedding matrix of tgs

        Note:
            for adv_multi, x0 is the embedding matrix, and x1 is the domain label.
        Returns:
            [tensor] -- transfer loss
        """
        if self.loss_type == "adv_multi":
            loss = self.multi_advloss(x0, x1)
        elif self.loss_type == "random_mixup":
            loss = self.mixup(
                link_prediction_model=kwargs["link_prediction_model"],
                x0=x0,
                x1=x1,
                embed0=kwargs["embed0"],
                embed1=kwargs["embed1"],
                y=kwargs["y"],
            )
        elif self.loss_type in ["graph_mmd"]:
            mode = self.loss_type.split("_")[1]
            if mode == "mmd":
                loss_function = MMD_loss(kernel_type="linear")
            else:
                raise NotImplementedError
            pairs = kwargs["domain_pairs"]
            feature = kwargs["feature"]

            loss = 0
            for pair in pairs.t():
                loss += loss_function(feature[pair[0]], feature[pair[1]])
            loss /= pairs.shape[1]
        elif self.loss_type == "graph_mixup":
            loss = self.mixup(
                link_prediction_model=kwargs["link_prediction_model"],
                x0=x0,
                x1=x1,
                embed0=kwargs["embed0"],
                embed1=kwargs["embed1"],
                y=kwargs["y"],
                domain_pairs=kwargs["domain_pairs"],
            )
        return loss


if __name__ == "__main__":
    import torch

    trans_loss = TransferLoss("adv")
    a = (torch.randn(5, 512) * 10).cuda()
    b = torch.randint(0, 5, (5,)).cuda()
    print(trans_loss.compute(a, b))
