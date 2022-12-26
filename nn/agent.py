import torch.nn as nn
import dgl
from nn.ag_util import process_graph


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.embedding = None

    def forward(self, nx_graph, ag_loc, task_loc):
        g = process_graph(nx_graph)

        return task_loc

    def gen_bipartite_graph(self, map, ag_loc, task_loc):
        self.gnn(ag_loc, task_loc)

    def fit(self):
        pass
