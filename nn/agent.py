import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.embedding = None

    def forward(self, map, ag_loc, task_loc):
        return task_loc

    def gen_bipartite_graph(self, map, ag_loc, task_loc):
        self.gnn(ag_loc, task_loc)

    def fit(self):
        pass
