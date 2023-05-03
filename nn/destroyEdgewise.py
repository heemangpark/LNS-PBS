import copy

import dgl
import torch
from torch import nn as nn
from torch.distributions.categorical import Categorical as C

from nn.gnn import GNNEdgewise, GNNLayerEdgewise

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def destroyGraph(graph, destroy):
    discon, con = torch.LongTensor([0]).to(DEVICE), torch.LongTensor([1]).to(DEVICE)
    g = copy.deepcopy(graph)

    to_con_f = []
    to_con_t = []
    to_discon_f = []
    to_discon_t = []

    for d in destroy:
        n_id = g.nodes()[g.ndata['idx'] == d].item()
        prev_id, next_id = n_id - 1, n_id + 1

        if (next_id == g.number_of_nodes()) or (g.ndata['type'][next_id] == 1):
            to_discon_f.extend([prev_id, n_id])
            to_discon_t.extend([n_id, prev_id])
        else:
            if (torch.Tensor(destroy) == g.ndata['idx'][next_id].item()).sum() > 0:
                pass
            else:
                to_con_f.extend([prev_id, next_id])
                to_con_t.extend([next_id, prev_id])
            to_discon_f.extend([prev_id, n_id, next_id, n_id])
            to_discon_t.extend([n_id, prev_id, n_id, next_id])

    g.edges[to_con_f, to_con_t].data['connected'] *= 0
    g.edges[to_con_f, to_con_t].data['connected'] += con

    g.edges[to_discon_f, to_discon_t].data['connected'] *= 0
    g.edges[to_discon_f, to_discon_t].data['connected'] += discon

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.empty_tensor = torch.Tensor().to(DEVICE)
        self.to(DEVICE)

        self.lossList = []

    def learn(self, graphs: dgl.DGLHeteroGraph, destroys: dict):
        """
        @param graphs: original graph (without destroy yet)
        @param destroys: destroyed node sets and each cost decrement
                        (cost decrement -> route length before destroy - route length after destroy)
        @return: L1loss value of model
        """
        nf = self.node_layer(graphs.ndata['coord'])
        next_nf = self.gnn(graphs, nf)
        ef = self.edge_layer(graphs, next_nf)

        destroyed_graphs = [destroyGraph(graphs, d) for d in destroys.keys()]

        des_src = [dg.edges()[0][dg.edata['connected'] == 1] for dg in destroyed_graphs]
        des_dst = [dg.edges()[1][dg.edata['connected'] == 1] for dg in destroyed_graphs]

        src_idx = torch.cat(des_src)
        dst_idx = torch.cat(des_dst)
        mask = graphs.edge_ids(src_idx, dst_idx)
        input_ef = ef[mask].reshape(len(des_src), -1, ef.shape[-1])

        pred = self.mlp(input_ef)

        " cost: original value - destroyed value (+ better, - worse)"
        cost = torch.Tensor(list(map(lambda x: x / 64, list(destroys.values())))).to(DEVICE)

        loss = torch.sum(torch.abs(pred - cost))
        self.lossList.append(loss)
        if len(self.lossList) == 10:
            loss = torch.cat(self.lossList).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lossList = []

        return loss.item()

    def act(self, graph: dgl.DGLHeteroGraph, destroyCand: list, evalMode: str):
        """
        @param graph: current graph status
        @param destroyCand: destroy node sets candidate to search for
        @param evalMode: greedy -> argmax model prediction, sample -> sample from softmax(prediction)
        @return: the best node set to destroy
        """
        nf = self.node_layer(graph.ndata['coord'])
        next_nf = self.gnn(graph, nf)
        ef = self.edge_layer(graph, next_nf)

        destroyed_graphs = [destroyGraph(graph, d) for d in destroyCand]

        des_src = [dg.edges()[0][dg.edata['connected'] == 1] for dg in destroyed_graphs]
        des_dst = [dg.edges()[1][dg.edata['connected'] == 1] for dg in destroyed_graphs]

        mask = [[graph.edge_ids(_ds, _dd).item() for _ds, _dd in zip(ds, dd)] for ds, dd in zip(des_src, des_dst)]

        input_ef = self.empty_tensor
        for m in mask:
            input_ef = torch.cat([input_ef, ef[m].unsqueeze(0)])

        pred = self.mlp(input_ef)

        if evalMode == 'greedy':
            return destroyCand[torch.argmax(pred).item()]
        else:
            softmax = nn.Softmax(0)
            probs = softmax(pred)
            m = C(probs=probs)
            sample = m.sample()

            return destroyCand[sample.item()]
