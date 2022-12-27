import torch
import torch.nn as nn
import dgl


class GNN(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(GNN, self).__init__()
        self.node_embedding = nn.Linear(embedding_dim + node_dim, embedding_dim)
        self.edge_embedding = nn.Linear(node_dim * 2 + 1, embedding_dim)

    def forward(self, g: dgl.DGLGraph):
        nf = torch.eye(3)[g.ndata['type']]
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
