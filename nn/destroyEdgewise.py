import copy

import dgl
import torch
from torch import nn as nn

from nn.gnn import GNNEdgewise, GNNLayerEdgewise

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def destroyGraph(graph, destroy):
    discon, con = torch.LongTensor([0]).to(DEVICE), torch.LongTensor([1]).to(DEVICE)
    g = copy.deepcopy(graph)

    for d in destroy:
        n_id = g.nodes()[g.ndata['idx'] == d].item()
        prev_id, next_id = n_id - 1, n_id + 1
        if (next_id == g.number_of_nodes()) or (g.ndata['type'][next_id] == 1):
            g.edges[prev_id, n_id].data['connected'] = discon
            g.edges[n_id, prev_id].data['connected'] = discon
        else:
            if g.ndata['idx'][next_id].item() in destroy:
                pass
            else:
                g.edges[prev_id, next_id].data['connected'] = con
                g.edges[next_id, prev_id].data['connected'] = con
            g.edges[prev_id, n_id].data['connected'] = discon
            g.edges[n_id, prev_id].data['connected'] = discon
            g.edges[n_id, next_id].data['connected'] = discon
            g.edges[next_id, n_id].data['connected'] = discon

    return g


class MLP(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, edge_Size=94):
        super(MLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(edge_Size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.empty_tensor = torch.Tensor().to(DEVICE)
        self.to(DEVICE)

    def forward(self, input):
        x = self.linear(input)
        x = self.decoder(x.squeeze())

        return x.squeeze()


class DestroyEdgewise(nn.Module):
    def __init__(self, embedding_dim=64, gnn_layers=3):
        super(DestroyEdgewise, self).__init__()
        self.embedding_dim = embedding_dim
        self.node_layer = nn.Linear(2, embedding_dim)
        self.edge_layer = GNNLayerEdgewise(embedding_dim * 2, embedding_dim)

        self.gnn = GNNEdgewise(
            in_dim=embedding_dim,
            out_dim=embedding_dim,
            embedding_dim=embedding_dim,
            n_layers=gnn_layers,
            residual=True,
        )

        self.mlp = MLP()
        self.loss = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.empty_tensor = torch.Tensor().to(DEVICE)
        self.to(DEVICE)

    def learn(self, graphs: dgl.DGLHeteroGraph, destroys: list):
        nf = self.node_layer(graphs.ndata['coord'])
        next_nf = self.gnn(graphs, nf)
        ef = self.edge_layer(graphs, next_nf)

        destroyed_graphs = [destroyGraph(graphs, d) for d in destroys[0].keys()]

        des_src = [dg.edges()[0][dg.edata['connected'] == 1] for dg in destroyed_graphs]
        des_dst = [dg.edges()[1][dg.edata['connected'] == 1] for dg in destroyed_graphs]

        mask = [[graphs.edge_ids(_ds, _dd).item() for _ds, _dd in zip(ds, dd)] for ds, dd in zip(des_src, des_dst)]

        input_ef = self.empty_tensor
        for m in mask:
            input_ef = torch.cat([input_ef, ef[m].unsqueeze(0)])

        pred = self.mlp(input_ef)
        cost = torch.Tensor(list(map(lambda x: x / 64, list(destroys[0].values())))).to(DEVICE)

        loss = torch.sum(torch.abs(pred - cost))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, graph: dgl.DGLHeteroGraph, destroys: list):
        g_node_feat = graph.ndata['coord']
        g_node_feat_embed = self.node_layer(g_node_feat)
        g_embedding = self.gnn(graph, g_node_feat_embed)

        d_coords = self.empty_tensor
        for d in destroys:
            d_coord = self.empty_tensor
            for single_d in d:
                destroyed_idx = graph.nodes()[graph.ndata['idx'] == single_d].item()
                d_coord = torch.cat([d_coord, graph.nodes[destroyed_idx].data['coord']])
            d_coords = torch.cat([d_coords, d_coord.unsqueeze(0)])
        d_embedding = self.node_layer(d_coords)

        final = self.empty_tensor
        for d_embed in d_embedding:
            f = torch.cat([g_embedding, d_embed])
            final = torch.cat([final, f.unsqueeze(0)])

        cost = self.mlp(final)

        return cost.cpu().detach().numpy()
