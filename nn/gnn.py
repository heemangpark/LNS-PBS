import torch
import torch.nn as nn
import dgl


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim, n_layers):
        super(GNN, self).__init__()
        _ins = [in_dim] + [embedding_dim] * (n_layers - 1)
        _outs = [embedding_dim] * (n_layers - 1) + [out_dim]

        layers = []
        for _i, _o in zip(_ins, _outs):
            layers.append(GNNLayer(_i, _o))
        self.layers = nn.ModuleList(layers)

    def forward(self, g, nf=None):
        if nf is None:
            nf = torch.eye(3)[g.ndata['type']]

        for l in self.layers:
            nf = l(g, nf)
        return nf


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.node_embedding = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim),
                                            nn.LeakyReLU())
        self.edge_embedding = nn.Sequential(nn.Linear(in_dim * 2 + 1, out_dim),
                                            nn.LeakyReLU())

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf
        g.update_all(message_func=self.message_func,
                     reduce_func=self.reduce_func,
                     apply_node_func=self.apply_node_func)

        out_nf = g.ndata.pop('out_nf')
        g.ndata.pop('nf')
        return out_nf

    def message_func(self, edges):
        ef = torch.concat([edges.src['nf'], edges.dst['nf'], edges.data['traj'].view(-1, 1)], -1)
        msg = self.edge_embedding(ef)
        return {'msg': msg}

    def reduce_func(self, nodes):
        msg = nodes.mailbox['msg'].sum(1)
        return {'red_msg': msg}

    def apply_node_func(self, nodes):
        in_feat = torch.concat([nodes.data['nf'], nodes.data['red_msg']], -1)
        out_feat = self.node_embedding(in_feat)
        return {'out_nf': out_feat}
