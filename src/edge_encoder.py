import torch
from torch import nn


class LearnedEdgeEncoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=16,
        dropout=0.10,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, source, target):
        pair = torch.cat(
            [source, target],
            dim=-1,
        )

        edge = self.network(pair)

        return self.norm(edge)


class AdaptiveEdgeEncoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=16,
        dropout=0.10,
    ):
        super().__init__()

        pair_dim = 4 * node_dim

        self.value = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, source, target):

        difference = torch.abs(
            source - target
        )

        interaction = (
            source * target
        )

        pair = torch.cat(
            [
                source,
                target,
                difference,
                interaction,
            ],
            dim=-1,
        )

        value = self.value(pair)

        gate = torch.sigmoid(
            self.gate(pair)
        )

        edge = value * gate

        return self.norm(edge)


def static_edge_attributes(
    node_features,
    edge_index,
    edge_dim,
):
    source = node_features[
        edge_index[0]
    ]

    target = node_features[
        edge_index[1]
    ]

    # Simple symmetric edge representation
    edge = 0.5 * (
        source + target
    )

    current_dim = edge.size(-1)

    if current_dim > edge_dim:

        edge = edge[:, :edge_dim]

    elif current_dim < edge_dim:

        padding = torch.zeros(
            edge.size(0),
            edge_dim - current_dim,
            device=edge.device,
            dtype=edge.dtype,
        )

        edge = torch.cat(
            [edge, padding],
            dim=-1,
        )

    return edge