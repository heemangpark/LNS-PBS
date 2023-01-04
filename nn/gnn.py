import torch
import torch.nn as nn
import dgl
import dgl.function as fn

AG_type = 1
TASK_type = 2


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim, n_layers, residual=False):
        super(GNN, self).__init__()
        _ins = [in_dim] + [embedding_dim] * (n_layers - 1)
        _outs = [embedding_dim] * (n_layers - 1) + [out_dim]

        layers = []
        for _i, _o in zip(_ins, _outs):
            layers.append(GNNLayer(_i, _o))
        self.layers = nn.ModuleList(layers)

        self.residual = residual

    def forward(self, g, nf):
        nf_prev = nf
        for layer in self.layers:
            nf = layer(g, nf_prev)
            if self.residual:
                nf_prev = nf + nf_prev
            else:
                nf_prev = nf
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


class Bipartite:
    def get_policy(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf

        # pull from task node idx to agent node idx
        ag_node_indices = g.filter_nodes(ag_node_func)
        task_node_indices = g.filter_nodes(task_node_func)

        ag_nfs = g.nodes[ag_node_indices].data['nf']
        task_nfs = g.nodes[task_node_indices].data['nf']
        ###### WIP

    # def


def ag_node_func(nodes):
    return (nodes.data['type'] == AG_type)  # .squeeze(1)


def task_node_func(nodes):
    return (nodes.data['type'] == TASK_type)  # .squeeze(1)
