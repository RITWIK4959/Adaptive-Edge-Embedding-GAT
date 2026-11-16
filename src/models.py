import torch
import torch.nn.functional as F

from torch import nn

from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
)

from .edge_encoder import (
    LearnedEdgeEncoder,
    AdaptiveEdgeEncoder,
)


class BaseGATModel(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=8,
        heads=8,
        attention_dropout=0.6,
        classifier_dropout=0.6,
    ):
        super().__init__()

        self.feature_dropout = nn.Dropout(
            p=0.0
        )

        self.attention_dropout = (
            attention_dropout
        )

        self.classifier_dropout = (
            classifier_dropout
        )

        self.hidden_channels = (
            hidden_channels
        )

        self.heads = heads

        self.input_projection = nn.Linear(
            in_channels,
            hidden_channels * heads,
            bias=False,
        )

        self.classifier = nn.Linear(
            hidden_channels * heads,
            num_classes,
        )

    def classify(self, x):

        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        return self.classifier(x)


class GATModel(BaseGATModel):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=8,
        heads=8,
        attention_dropout=0.6,
        classifier_dropout=0.6,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            num_classes,
            hidden_channels,
            heads,
            attention_dropout,
            classifier_dropout,
        )

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            add_self_loops=True,
        )

        self.conv2 = GATConv(
            hidden_channels * heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            add_self_loops=True,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
    ):
        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        return x


class GATv2Model(BaseGATModel):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=8,
        heads=8,
        attention_dropout=0.6,
        classifier_dropout=0.6,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            num_classes,
            hidden_channels,
            heads,
            attention_dropout,
            classifier_dropout,
        )

        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            add_self_loops=True,
        )

        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            add_self_loops=True,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
    ):
        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        return x


class GATv2StaticEdge(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=8,
        heads=8,
        edge_dim=8,
        attention_dropout=0.6,
        classifier_dropout=0.6,
        **kwargs,
    ):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )

        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )

        self.classifier_dropout = (
            classifier_dropout
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
    ):
        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        x = self.conv1(
            x,
            edge_index,
            edge_attr=edge_attr,
        )

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
            edge_attr=edge_attr,
        )

        return x


class DynamicEdgeGATv2(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        edge_encoder,
        hidden_channels=8,
        heads=8,
        edge_dim=8,
        attention_dropout=0.6,
        classifier_dropout=0.6,
    ):
        super().__init__()

        self.edge_encoder = edge_encoder

        self.input_projection = nn.Linear(
            in_channels,
            hidden_channels * heads,
        )

        self.conv1 = GATv2Conv(
            hidden_channels * heads,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )

        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )

        self.classifier_dropout = (
            classifier_dropout
        )

        self.edge_dim = edge_dim

    def make_edge_attr(
        self,
        x,
        edge_index,
    ):
        source = x[
            edge_index[0]
        ]

        target = x[
            edge_index[1]
        ]

        edge_attr = self.edge_encoder(
            source,
            target,
        )

        if self.training:
            edge_attr = F.dropout(
                edge_attr,
                p=0.10,
                training=True,
            )

        return edge_attr

    def forward(
        self,
        x,
        edge_index,
    ):
        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        x = self.input_projection(x)

        edge_attr = self.make_edge_attr(
            x,
            edge_index,
        )

        x = self.conv1(
            x,
            edge_index,
            edge_attr=edge_attr,
        )

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.classifier_dropout,
            training=self.training,
        )

        edge_attr = self.make_edge_attr(
            x,
            edge_index,
        )

        x = self.conv2(
            x,
            edge_index,
            edge_attr=edge_attr,
        )

        return x


class GATv2LearnedEdge(
    DynamicEdgeGATv2
):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=8,
        heads=8,
        edge_dim=8,
        edge_encoder_hidden=16,
        attention_dropout=0.6,
        classifier_dropout=0.6,
        **kwargs,
    ):

        encoder = LearnedEdgeEncoder(
            node_dim=hidden_channels * heads,
            edge_dim=edge_dim,
            hidden_dim=edge_encoder_hidden,
            dropout=0.10,
        )

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            edge_encoder=encoder,
            hidden_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
            attention_dropout=attention_dropout,
            classifier_dropout=classifier_dropout,
        )


class AEGAT(
    DynamicEdgeGATv2
):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=8,
        heads=8,
        edge_dim=8,
        edge_encoder_hidden=16,
        attention_dropout=0.6,
        classifier_dropout=0.6,
        **kwargs,
    ):

        node_dim = (
            hidden_channels * heads
        )

        encoder = AdaptiveEdgeEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=edge_encoder_hidden,
            dropout=0.10,
        )

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            edge_encoder=encoder,
            hidden_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
            attention_dropout=attention_dropout,
            classifier_dropout=classifier_dropout,
        )