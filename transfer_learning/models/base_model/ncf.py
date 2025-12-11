import torch
from torch import nn


class NCF(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_feature_dim = 2 * (hidden_dim // 2)
        self.mf_embed = nn.Embedding(num_nodes, self.hidden_dim)
        self.mlp_embed = nn.Embedding(num_nodes, self.hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear((self.hidden_dim + 9) * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
        )
        self.mf_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 9, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(2 * (self.hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * (hidden_dim // 2), self.hidden_dim // 4),
            nn.BatchNorm1d(self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x0, x1, edges):
        r"""
        Args:
            x0: (batch_size, hidden_dim)
            x1: (batch_size, hidden_dim)
            edges: (2, batch_size)
                or embeddings (for mixup based methods): Tuple[Tensor(hidden_dim, batch_size)]
        """
        if not isinstance(edges, tuple) and not isinstance(edges, list):
            tf_embed, tg_embed = self.get_gene_embedding(edges)
        else:
            tf_embed, tg_embed = edges[0], edges[1]


        tf_embed_mlp = tf_embed[:, : self.hidden_dim]
        tg_embed_mlp = tg_embed[:, : self.hidden_dim]
        tf_embed_mf = tf_embed[:, self.hidden_dim :]
        tg_embed_mf = tg_embed[:, self.hidden_dim :]


        tf_embed_mf = torch.cat([tf_embed_mf, x0], dim=1)
        tg_embed_mf = torch.cat([tg_embed_mf, x1], dim=1)
        mf_feature = torch.mul(tf_embed_mf, tg_embed_mf)
        mf_feature = self.mf_mlp(mf_feature)

        edge_embed = torch.cat(
            [x0, tf_embed_mlp, x1 - x0, tg_embed_mlp - tf_embed_mlp], dim=1
        )
        mlp_feature = self.mlp(edge_embed)
        feature = torch.cat([mf_feature, mlp_feature], dim=1)

        out = self.regressor(feature)
        return out.squeeze(), feature

    def get_gene_embedding(self, edges):
        tf_embed_mlp = self.mlp_embed(edges[0])
        tg_embed_mlp = self.mlp_embed(edges[1])
        tf_embed_mf = self.mf_embed(edges[0])
        tg_embed_mf = self.mf_embed(edges[1])
        tf_embed = torch.cat([tf_embed_mlp, tf_embed_mf], dim=1)
        tg_embed = torch.cat([tg_embed_mlp, tg_embed_mf], dim=1)
        return tf_embed, tg_embed
