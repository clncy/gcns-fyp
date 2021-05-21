from typing import Optional

import torch
from torch_scatter import scatter
from torch.nn import Linear, ModuleList, Parameter, ParameterList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SGConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
from ogb.graphproppred.mol_encoder import AtomEncoder


def global_min_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    minimum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{min}_{n=1}^{N_i} \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Derived from torch_geometric
    """
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='min')


class GAT(torch.nn.Module):
    name = "GAT"
    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([GATConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GATConv(hidden_channels, hidden_channels)]\
                    + [GATConv(hidden_channels, representation_size)])

        self.lin = Linear(representation_size, output_size)
        self.dropout_probability = dropout_probability

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class ConcatPool(torch.nn.Module):
    name = "ConcatPool"
    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([GATConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GATConv(hidden_channels, hidden_channels)]\
                    + [GATConv(hidden_channels, representation_size)])

        self.lin = Linear(3*representation_size, output_size)
        self.dropout_probability = dropout_probability

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        # 2. Readout layer: Concatenate multiple pooling operations
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat((x_mean, x_add, x_max), 1)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class ResConcatPool(torch.nn.Module):
    name = "ResConcatPool"

    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([GATConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GATConv(2*hidden_channels, 2*hidden_channels)]\
                    + [GATConv(2*hidden_channels, representation_size)])

        self.lin = Linear(3*representation_size, output_size)
        self.dropout_probability = dropout_probability

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for i, conv in enumerate(self.convs):

            if i == 0:
                x = conv(x, edge_index)
            # Residual connection
            else:
                x = torch.cat((conv(x, edge_index), x), 1)
            x = x.relu()

        # 2. Readout layer: Concatenate multiple pooling operations
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat((x_mean, x_add, x_max), 1)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class MultiGCN(torch.nn.Module):
    name = "MultiGCN"
    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.gcn_convs = ModuleList([GCNConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GCNConv(hidden_channels, hidden_channels)]\
                    + [GCNConv(hidden_channels, representation_size)])

        self.gat_convs = ModuleList([GATConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GATConv(hidden_channels, hidden_channels)]\
                    + [GATConv(hidden_channels, representation_size)])

        self.thetas = ParameterList([Parameter(torch.randn(hidden_channels))] + \
                    hidden_layers * [Parameter(torch.randn(hidden_channels,))]\
                    + [Parameter(torch.randn(representation_size))])

        self.lin = Linear(representation_size, output_size)
        self.dropout_probability = dropout_probability

    @staticmethod
    def row_mult(t, vector):
        """ Performs a broadcasted matrix multiplication
        Adapted from https://stackoverflow.com/a/66636211
        """
        extra_dims = (1,) * (t.dim() - 1)
        return t * vector.view(*extra_dims, -1)

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for i in range(len(self.gcn_convs)):
            x_a = self.gcn_convs[i](x, edge_index)
            x_b = self.gat_convs[i](x, edge_index)

            theta = torch.clamp(self.thetas[i], min=0, max=1)
            one_minus_theta = (torch.ones_like(theta) - theta)
            x = self.row_mult(x_a, theta) + self.row_mult(x_b, one_minus_theta)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class GatMeanPool(torch.nn.Module):
    name = "GatMeanPool"

    def __init__(
        self,
        embedding_dimensions,
        hidden_channels,
        hidden_layers,
        representation_size,
        output_size,
        dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([GATConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GATConv(hidden_channels, hidden_channels)]\
                    + [GATConv(hidden_channels, representation_size)])

        self.lin = Linear(representation_size, output_size)
        self.dropout_probability = dropout_probability

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        # 2. Readout layer: Aggregate node vectors with mean pooling
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class GatStatPool(torch.nn.Module):
    name = "GatStatPool"
    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([GATConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GATConv(hidden_channels, hidden_channels)]\
                    + [GATConv(hidden_channels, representation_size)])

        self.lin = Linear(3*representation_size, output_size)
        self.dropout_probability = dropout_probability

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        # 2. Readout layer: Aggregate node vectors with mean pooling
        x_min = global_min_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat((x_min, x_mean, x_max), 1)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class GatJumpStatPool(torch.nn.Module):
    name = "GatJumpStatPool"
    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([GATConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [GATConv(hidden_channels, hidden_channels)]\
                    + [GATConv(hidden_channels, representation_size)])

        pooling_dims = 3*(embedding_dimensions + (len(self.convs)-1)*hidden_channels + representation_size)
        self.lin = Linear(
            pooling_dims,
            output_size
        )

        self.dropout_probability = dropout_probability

    def global_stat_pool(self, x, batch):
        x_min = global_min_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return torch.cat((x_min, x_mean, x_max), 1)

    def forward(self, x, edge_index, batch):
        jumping_knowledge = []
        x = self.emb(x)
        jumping_knowledge.append(self.global_stat_pool(x, batch))

        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
            jumping_knowledge.append(self.global_stat_pool(x, batch))

        # 2. Readout layer: Aggregate node vectors with jumping knowledge
        x = torch.cat(tuple(jumping_knowledge), 1)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x

class SgcMeanPool(torch.nn.Module):
    name = "SgcMeanPool"

    def __init__(
        self,
        embedding_dimensions,
        hidden_channels,
        hidden_layers,
        representation_size,
        output_size,
        dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([SGConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [SGConv(hidden_channels, hidden_channels)]\
                    + [SGConv(hidden_channels, representation_size)])

        self.lin = Linear(representation_size, output_size)
        self.dropout_probability = dropout_probability

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        # 2. Readout layer: Aggregate node vectors with mean pooling
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class SgcStatPool(torch.nn.Module):
    name = "SgcStatPool"
    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([SGConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [SGConv(hidden_channels, hidden_channels)]\
                    + [SGConv(hidden_channels, representation_size)])

        self.lin = Linear(3*representation_size, output_size)
        self.dropout_probability = dropout_probability

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        # 2. Readout layer: Aggregate node vectors with mean pooling
        x_min = global_min_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat((x_min, x_mean, x_max), 1)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x


class SgcJumpStatPool(torch.nn.Module):
    name = "SgcJumpStatPool"
    def __init__(
            self,
            embedding_dimensions,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimensions)
        self.convs = ModuleList([SGConv(embedding_dimensions, hidden_channels)] + \
                    hidden_layers * [SGConv(hidden_channels, hidden_channels)]\
                    + [SGConv(hidden_channels, representation_size)])

        pooling_dims = 3*(embedding_dimensions + (len(self.convs)-1)*hidden_channels + representation_size)
        self.lin = Linear(
            pooling_dims,
            output_size
        )

        self.dropout_probability = dropout_probability

    def global_stat_pool(self, x, batch):
        x_min = global_min_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return torch.cat((x_min, x_mean, x_max), 1)

    def forward(self, x, edge_index, batch):
        jumping_knowledge = []
        x = self.emb(x)
        jumping_knowledge.append(self.global_stat_pool(x, batch))

        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
            jumping_knowledge.append(self.global_stat_pool(x, batch))

        # 2. Readout layer: Aggregate node vectors with jumping knowledge
        x = torch.cat(tuple(jumping_knowledge), 1)
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin(x)

        return x
