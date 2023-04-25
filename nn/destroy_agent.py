import dgl
import torch
from torch import nn as nn

from nn.gnn import GNN


class MLP(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.pred = nn.Linear(55 + 3, 1)

        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.empty_tensor = torch.Tensor().to(self.device)
        self.to(self.device)

    def forward(self, input):
        x = self.layers(input).squeeze()
        x = self.pred(x)

        return x.squeeze()


class DestroyAgent(nn.Module):
    def __init__(self, embedding_dim=64, gnn_layers=3):
        super(DestroyAgent, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer = nn.Linear(2, embedding_dim)
        self.gnn = GNN(
            in_dim=embedding_dim,
            out_dim=embedding_dim,
            embedding_dim=embedding_dim,
            n_layers=gnn_layers,
            residual=True,
        )

        self.mlp = MLP()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.empty_tensor = torch.Tensor().to(self.device)
        self.to(self.device)

    def train(self, graphs: dgl.DGLHeteroGraph, destroys: list, graph_list: list, batch_config: list):
        data_size, batch_size, batch_num = batch_config

        g_node_feat = graphs.ndata['coord']
        g_node_feat_embed = self.layer(g_node_feat)
        g_embedding = self.gnn(graphs, g_node_feat_embed)
        g_embedding = g_embedding.reshape(batch_size, g_embedding.shape[0] // batch_size, 64)  # TODO: hard coded

        d_coord_tensor, costs = self.empty_tensor, self.empty_tensor
        for d, coord_graph in zip(destroys, graph_list):
            d_coords = self.empty_tensor
            for d_ids in d.keys():
                d_coord = self.empty_tensor
                for d_id in d_ids:
                    # bringing coordination of destroyed nodes
                    destroyed_idx = coord_graph.nodes()[coord_graph.ndata['idx'] == d_id].item()
                    d_coord = torch.cat([d_coord, coord_graph.nodes[destroyed_idx].data['coord']])
                d_coords = torch.cat([d_coords, d_coord.unsqueeze(0)])
            d_coord_tensor = torch.cat([d_coord_tensor, d_coords.unsqueeze(0)])
            costs = torch.cat([costs, torch.Tensor(list(d.values())).to(self.device).unsqueeze(0)])
        d_embedding = self.layer(d_coord_tensor)

        final = self.empty_tensor
        for xg, xd in zip(g_embedding, d_embedding):
            embed_per_map = self.empty_tensor
            for k_xd in xd:
                x = torch.cat([xg, k_xd])
                embed_per_map = torch.cat([embed_per_map, x.unsqueeze(0)])
            final = torch.cat([final, embed_per_map.unsqueeze(0)])

        x = self.mlp(final)  # batch_size x 100
        y = costs  # batch_size x 100

        loss = self.loss(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
