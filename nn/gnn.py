import dgl
import torch
import torch.nn as nn


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim, n_layers, residual=False):
        super(GNN, self).__init__()
        self.residual = residual
        _ins = [in_dim] + [embedding_dim] * (n_layers - 1)
        _outs = [embedding_dim] * (n_layers - 1) + [out_dim]

        layers = []
        for _i, _o in zip(_ins, _outs):
            layers.append(GNNLayer(_i, _o))

        self.layers = nn.ModuleList(layers)
        self.layer = nn.Linear(embedding_dim, 1)

    def forward(self, g, nf):
        nf_prev = nf
        for layer in self.layers:
            nf_aft = layer(g, nf_prev)
            if self.residual:
                nf_aft += nf_prev
            nf_prev = nf_aft

        return nf_aft


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.node_embedding = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim, bias=False), nn.LeakyReLU())
        self.edge_embedding = nn.Sequential(nn.Linear(in_dim * 2 + 1, out_dim, bias=False), nn.LeakyReLU())

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf
        g.update_all(message_func=self.message_func,
                     reduce_func=self.reduce_func,
                     apply_node_func=self.apply_node_func)

        out_nf = g.ndata.pop('out_nf')
        g.ndata.pop('nf')
        return out_nf

    def message_func(self, edges):
        # init_feat = torch.concat([edges.data['a_dist'].view(-1, 1),
        #                           edges.data['dist'].view(-1, 1),
        #                           edges.data['obs_proxy'].view(-1, 1)], -1)
        init_feat = edges.data['dist'].view(-1, 1)
        feature = torch.concat([init_feat, edges.src['nf'], edges.dst['nf']], -1)

        msg = self.edge_embedding(feature)
        return {'msg': msg}

    def reduce_func(self, nodes):
        msg = nodes.mailbox['msg'].min(1).values
        return {'red_msg': msg}

    def apply_node_func(self, nodes):
        in_feat = torch.concat([nodes.data['nf'], nodes.data['red_msg']], -1)
        out_feat = self.node_embedding(in_feat)
        return {'out_nf': out_feat}


class GNNLayer_edge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer_edge, self).__init__()
        self.edge_embedding = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False),
                                            nn.LeakyReLU())

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf
        c_edges = g.filter_edges(_connected_edges)
        g.apply_edges(self.message_func, edges=c_edges)

        msg = g.edata.pop('msg')
        g.ndata.pop('nf')
        return msg[c_edges]

    def message_func(self, edges):
        feature = torch.concat([edges.src['nf'], edges.dst['nf']], -1)
        msg = self.edge_embedding(feature)
        return {'msg': msg}


def _connected_edges(edges):
    return (edges.data['connected'] == 1) * (edges.src['id'] < edges.dst['id'])
