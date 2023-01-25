import dgl
import torch
import torch.nn as nn


class Policy_gnn(nn.Module):
    def __init__(self):
        super(Policy_gnn, self).__init__()
        self.message_linear = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

    def actor(self, g, sample=True):
        """
        ntype (type, )
        etype (co)
        """
        nf = generate_feature(g)
        prob = self.get_edge_prob(g, nf)
        return prob

    def get_edge_prob(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf
        edges = g.filter_edges(filter_connected_edge)

        g.apply_edges(func=self.message_func, edges=edges)
        edge_score = g.edata['edge_score'][edges]
        prob = torch.softmax(edge_score, 0)

        del g.ndata['nf']

        return prob

    def message_func(self, edges):
        src_data = edges.src['nf']
        dst_data = edges.dst['nf']
        ef_input = torch.cat([src_data, dst_data], -1)
        ef_output = self.message_linear(ef_input)
        return {'edge_score': ef_output}


def generate_feature(g):
    # nf = g.ndata['type'].reshape(-1, 1)
    nf = g.ndata['loc']
    # nf = torch.cat([g.ndata['type'].reshape(-1, 1), g.ndata['loc']], -1)
    return nf


def filter_connected_edge(edges):
    return edges.data['co'] == 1
