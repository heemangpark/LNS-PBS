import copy

import dgl
import torch
from lion_pytorch import Lion
from torch import nn as nn
from torch.distributions.categorical import Categorical as C

from nn.gnn import GNNEdgewise, GNNLayerEdgewise


def destroyGraph(graph, destroy, device):
    discon, con = torch.LongTensor([0]).to(device), torch.LongTensor([1]).to(device)
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


def destroyBatchGraph(graphs: list, destroy, device=None):
    discon, con = torch.LongTensor([0]).to(device), torch.LongTensor([1]).to(device)
    to_con_f = []
    to_con_t = []
    to_discon_f = []
    to_discon_t = []
    node_cnt = 0

    for _g, _destroy in zip(graphs, destroy):
        g = copy.deepcopy(_g)
        for d in _destroy:
            n_id = g.nodes()[g.ndata['idx'] == d].item()
            prev_id = n_id - 1
            next_id = n_id + 1

            n_id_b = n_id + node_cnt
            prev_id_b = prev_id + node_cnt
            next_id_b = next_id + node_cnt

            if (next_id == g.number_of_nodes()) or (g.ndata['type'][next_id] == 1):
                to_discon_f.extend([prev_id_b, n_id_b])
                to_discon_t.extend([n_id_b, prev_id_b])
            else:
                if (torch.Tensor(_destroy) == g.ndata['idx'][next_id].item()).sum() > 0:
                    pass
                else:
                    to_con_f.extend([prev_id_b, next_id_b])
                    to_con_t.extend([next_id_b, prev_id_b])
                to_discon_f.extend([prev_id_b, n_id_b, next_id_b, n_id_b])
                to_discon_t.extend([n_id_b, prev_id_b, n_id_b, next_id_b])

        node_cnt += g.number_of_nodes()

    batched_g = dgl.batch(graphs)

    batched_g.edges[to_con_f, to_con_t].data['connected'] *= 0
    batched_g.edges[to_con_f, to_con_t].data['connected'] += con

    batched_g.edges[to_discon_f, to_discon_t].data['connected'] *= 0
    batched_g.edges[to_discon_f, to_discon_t].data['connected'] += discon

    return batched_g


class MLP(nn.Module):
    def __init__(self, input_size=64, hidden_size=32):
        super(MLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.linear(input)
        x = x.squeeze()
        x = torch.sum(x, -1)
        x = self.softmax(x)

        return x


class DestroyEdgewise(nn.Module):
    def __init__(self, embedding_dim=64, gnn_layers=3, device=None):
        super(DestroyEdgewise, self).__init__()
        self.embedding_dim = embedding_dim
        self.nodeLayer = nn.Linear(2, embedding_dim)
        self.edgeLayer = GNNLayerEdgewise(embedding_dim * 2, embedding_dim)

        self.gnn = GNNEdgewise(
            in_dim=embedding_dim,
            out_dim=embedding_dim,
            embedding_dim=embedding_dim,
            n_layers=gnn_layers,
            residual=True,
        )

        self.mlp = MLP()
        self.optimizer = Lion(self.parameters(), lr=1e-4)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.to(device)

    def learn(self, graphs: dgl.DGLHeteroGraph, destroys: list, batchNum: int, device: str):
        """
        @param graphs: original graph (without destroy yet)
        @param destroys: destroyed node sets and each cost decrement
                        (cost decrement -> route length before destroy - route length after destroy)
        @param batchNum: number of batch data
        @param device: model device
        @return: loss
        """
        destroyNum = len(destroys[0])

        nf = self.nodeLayer(graphs.ndata['coord'])
        next_nf = self.gnn(graphs, nf)
        ef = self.edgeLayer(graphs, next_nf)

        # TODO: time
        unbatchGraphs = dgl.unbatch(graphs)
        gs_to_destroy = []
        destroy_lst = []
        for g, destroy in zip(unbatchGraphs, destroys):
            gs_to_destroy.extend([g] * len(destroy))
            destroy_lst.extend(list(destroys[0].keys()))
        destroyedGraph = destroyBatchGraph(gs_to_destroy, destroy_lst, device)

        # TODO: time
        desSrc, desDst = [], []
        node_cnt = 0
        for i, dg in enumerate(dgl.unbatch(destroyedGraph)):
            desSrc.extend([dg.edges()[0][dg.edata['connected'] == 1] + node_cnt])
            desDst.extend([dg.edges()[1][dg.edata['connected'] == 1] + node_cnt])
            if i % destroyNum == 0 and i > 0:
                node_cnt += dg.number_of_nodes()

        srcIdx = torch.cat(desSrc)
        dstIdx = torch.cat(desDst)
        mask = graphs.edge_ids(srcIdx, dstIdx)
        input_ef = ef[mask].reshape(batchNum, len(destroys[0]), -1, ef.shape[-1])

        pred = self.mlp(input_ef) + 1e-10

        " cost: original value - destroyed value (+ better, - worse)"
        # cost = torch.Tensor([list(map(lambda x: x / 64, list(d.values()))) for d in destroys]).to(device)
        cost = torch.Tensor([list(d.values()) for d in destroys]).to(device)
        baseline = torch.tile(torch.mean(cost, dim=-1).view(-1, 1), dims=(1, 10)).detach()
        # softmax = nn.Softmax(dim=-1)
        # cost = softmax(cost)

        # loss = self.loss(pred.log(), cost)
        loss = torch.mean(-(cost - baseline) * torch.log(pred))  # REINFORCE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, graph: dgl.DGLHeteroGraph, destroyCand: list, evalMode: str, device: str):
        """
        @param graph: current graph status
        @param destroyCand: destroy node sets candidate to search for
        @param evalMode: greedy -> argmax model prediction, sample -> sample from softmax(prediction)
        @param device: model device
        @return: the best node set to destroy
        """
        import time

        nf = self.nodeLayer(graph.ndata['coord'])
        next_nf = self.gnn(graph, nf)
        ef = self.edgeLayer(graph, next_nf)

        ' Before modification '
        destroyed_graphs = [destroyGraph(graph, d, device) for d in destroyCand]  # TODO: destroyBatch ?

        ' After modification '
        ###############################################################

        ###############################################################

        DG = dgl.batch(destroyed_graphs)
        SRC = DG.ndata['graph_id'][DG.edges()[0][DG.edata['connected'] == 1]]
        DST = DG.ndata['graph_id'][DG.edges()[1][DG.edata['connected'] == 1]]
        mask = graph.edge_ids(SRC, DST)
        input_ef = ef[mask]
        input_ef = input_ef.reshape(len(destroyCand), -1, input_ef.shape[-1])  # Batch X 94(=100-6) X emb
        pred = self.mlp(input_ef)

        if evalMode == 'greedy':
            act = torch.argmax(pred).item()
        else:
            m = C(probs=pred)
            act = m.sample()

        return destroyCand[act]
