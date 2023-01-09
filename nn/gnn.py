import dgl
import torch
import torch.nn as nn

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
        self.node_embedding = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim, bias=False),
                                            nn.LeakyReLU())
        self.edge_embedding = nn.Sequential(nn.Linear(in_dim * 2 + 1, out_dim, bias=False),
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


class Bipartite(nn.Module):
    def __init__(self, embedding_dim):
        super(Bipartite, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_fc = nn.Sequential(nn.Linear(2 * embedding_dim, 1, bias=False),
                                          nn.LeakyReLU()
                                          )

        self.ag_score = nn.Sequential(nn.Linear(embedding_dim, 1, bias=False), nn.LeakyReLU())

        # Todo:transformer
        # self.K = nn.Linear(embedding_dim, embedding_dim)
        # self.Q = nn.Linear(embedding_dim, embedding_dim)
        # self.V = nn.Linear(embedding_dim, embedding_dim)

    """
    Assume ag_size and task_size does not vary within batch
    """

    def forward(self, g: dgl.DGLGraph, bipartite_g: dgl.DGLGraph, nf, ag_node_indices, task_node_indices, task_finished):
        g.ndata['nf'] = nf

        ag_nfs = g.nodes[ag_node_indices].data['nf']
        task_nfs = g.nodes[task_node_indices].data['nf'][~task_finished]

        # pull from task node idx to agent node idx
        ag_node_indices = bipartite_g.filter_nodes(ag_node_func)
        task_node_indices = bipartite_g.filter_nodes(task_node_func)

        bipartite_g.nodes[ag_node_indices].data['nf'] = ag_nfs
        bipartite_g.nodes[task_node_indices].data['nf'] = task_nfs

        bipartite_g.update_all(message_func=self.message, reduce_func=self.reduce,
                               apply_node_func=self.apply_node)

        policy = bipartite_g.ndata.pop('policy')[ag_node_indices]

        ag_score = self.ag_score(ag_nfs).squeeze()
        ag_policy = torch.softmax(ag_score, -1)

        return policy, ag_policy

    def message(self, edges):
        src = edges.src['nf']
        dst = edges.dst['nf']
        m = torch.cat([src, dst], dim=1)
        score = self.attention_fc(m)

        # Todo:self-attention
        # K = self.K(m)
        # Q = self.Q(nf)
        #
        # score = (K * Q).sum(-1) / self.embedding_dim  # shape = (ag, task)
        # policy = torch.softmax(A, -1)

        return {'score': score}

    def reduce(self, nodes):
        score = nodes.mailbox['score']
        policy = torch.softmax(score, 1).squeeze()
        return {'policy': policy}

    def apply_node(self, nodes):
        return {'policy': nodes.data['policy']}


def ag_node_func(nodes):
    return nodes.data['type'] == AG_type  # .squeeze(1)


def task_node_func(nodes):
    return nodes.data['type'] == TASK_type  # .squeeze(1)
