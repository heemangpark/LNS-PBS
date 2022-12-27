import torch.nn as nn
from nn.gnn import GNN


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.gnn = GNN(node_dim=3, embedding_dim=128)

    def forward(self, di_dgl_g, agent_pos):
        out_nf = self.gnn(di_dgl_g)

        return di_dgl_g

    # def gen_bipartite_graph(self, map, ag_loc, task_loc):
    #     self.gnn(ag_loc, task_loc)

    def fit(self):
        pass
