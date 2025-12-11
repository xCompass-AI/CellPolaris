import torch
import torch.nn as nn
from torch.distributions.beta import Beta


class Extra_mixup(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        assert 0 <= alpha
        self.alpha = alpha
        self.criterion = nn.MSELoss(reduction="none")


    def forward(
        self, link_prediction_model, x0, x1, embed0, embed1, y, strong_related_source, tissue_index_map
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
            
            change domain_pairs
            target: human heart
            stage-1: human-other e.g. kidney, liver mixup with mouse heart
            stage-2: mouse-other e.g. kidney, liver extra-mixup(lambda>1) the generated samples from stage-1

            tissue_index_map: the first species is the target species
        """

        if tissue_index_map is not None:
            first_keys = next(iter(tissue_index_map.values())).keys()
            if list(first_keys) == ["human", "mouse"]:  # if in human,mouse order; human is target species
                paired_index_list = [
                    [idxs["human"], idxs["mouse"]]
                    for idxs in tissue_index_map.values()
                    if idxs["human"] is not None and idxs["mouse"] is not None
                ]
            else:
                paired_index_list = [
                    [idxs["mouse"], idxs["human"]]
                    for idxs in tissue_index_map.values()
                    if idxs["mouse"] is not None and idxs["human"] is not None
                ]

            total_loss = []
            for pairs in paired_index_list: 
                # stage-1

                pair0 = strong_related_source 
                pair1 = pairs[0]

                x0_pair0 = x0[pair0]
                x0_pair1 = x0[pair1]
                x1_pair0 = x1[pair0]
                x1_pair1 = x1[pair1]
                embed0_pair0 = embed0[pair0]
                embed0_pair1 = embed0[pair1]
                embed1_pair0 = embed1[pair0]
                embed1_pair1 = embed1[pair1]
                y_pair0 = y[pair0]
                y_pair1 = y[pair1]


                out, mixup_lambda, x0_mix, x1_mix, embed0_mix, embed1_mix = self.mixup_forward_1(
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

                loss0 = self.criterion(out, y_pair0)
                loss1 = self.criterion(out, y_pair1)
                loss_stage1 = mixup_lambda.squeeze() * loss0 + (1 - mixup_lambda).squeeze() * loss1
                loss_stage1 = loss_stage1.mean()

                y_mix = mixup_lambda.squeeze() * y_pair0 + (1 - mixup_lambda).squeeze() * y_pair1
                
                #stage-2

                pair2 = pairs[1]
                x0_pair2 = x0[pair2]
                x1_pair2 = x1[pair2]
                embed0_pair2 = embed0[pair2]
                embed1_pair2 = embed1[pair2]
                y_pair2 = y[pair2]

                out_2, mixup_lambda_2 = self.mixup_forward_2(
                    link_prediction_model,
                    x0_mix,
                    x0_pair2,
                    x1_mix,
                    x1_pair2,
                    embed0_mix,
                    embed0_pair2,
                    embed1_mix,
                    embed1_pair2,
                )

                y_mix2 = y_pair2 + mixup_lambda_2.squeeze() * (y_mix - y_pair2)
                loss_stage2 = self.criterion(out_2, y_mix2)
                loss_stage2 = loss_stage2.mean()

                loss = (loss_stage1 + loss_stage2)/2
                total_loss.append(loss)
        
        return torch.stack(total_loss).mean()
        

    def mixup_forward_1(
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
        return out, mixup_lambda, x0, x1, embed0, embed1

    def mixup_forward_2(
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
        mixup_lambda = Beta(self.alpha, self.alpha).sample((x0_pair0.shape[0], 1)) + 1
        mixup_lambda = mixup_lambda.to(x0_pair0.device)

        x0 = x0_pair1 + mixup_lambda * (x0_pair0 - x0_pair1)
        x1 = x1_pair1 + mixup_lambda * (x1_pair0 - x1_pair1)
        embed0 = embed0_pair1 + mixup_lambda * (embed0_pair0-embed0_pair1)
        embed1 = embed1_pair1 + mixup_lambda * (embed1_pair0-embed1_pair1)

        out, _ = link_prediction_model(x0, x1, (embed0, embed1))
        return out, mixup_lambda