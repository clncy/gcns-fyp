import torch
from torch.nn import Linear, ModuleList, Parameter, ParameterList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
from ogb.graphproppred.mol_encoder import AtomEncoder


class GAT(torch.nn.Module):
    def __init__(
            self,
            embedding_dimension,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimension)
        self.convs = ModuleList([GATConv(embedding_dimension, hidden_channels)] + \
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
    def __init__(
            self,
            embedding_dimension,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimension)
        self.convs = ModuleList([GATConv(embedding_dimension, hidden_channels)] + \
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
    def __init__(
            self,
            embedding_dimension,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimension)
        self.convs = ModuleList([GATConv(embedding_dimension, hidden_channels)] + \
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
    def __init__(
            self,
            embedding_dimension,
            hidden_channels,
            hidden_layers,
            representation_size,
            output_size,
            dropout_probability
    ):
        super().__init__()
        self.emb = AtomEncoder(embedding_dimension)
        self.gcn_convs = ModuleList([GCNConv(embedding_dimension, hidden_channels)] + \
                    hidden_layers * [GCNConv(hidden_channels, hidden_channels)]\
                    + [GCNConv(hidden_channels, representation_size)])

        self.gat_convs = ModuleList([GATConv(embedding_dimension, hidden_channels)] + \
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