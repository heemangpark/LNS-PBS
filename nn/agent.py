import math
import torch
import torch.nn as nn
from nn.gnn import GNN, Bipartite
from nn.memory import ReplayMemory
from math import inf


class Agent(nn.Module):
    def __init__(self, embedding_dim=128, memory_size=50000, batch_size=100, gnn_layers=3):
        super(Agent, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Linear(3, embedding_dim)
        self.gnn = GNN(in_dim=embedding_dim, out_dim=embedding_dim, embedding_dim=embedding_dim, n_layers=gnn_layers,
                       residual=True)
        self.bipartite_policy = Bipartite()

        self.replay_memory = ReplayMemory(capacity=memory_size, batch_size=batch_size)

    def forward(self, g, bipartite_g, ag_node_idx, task_node_indices, finished_task):
        feature = self.generate_feature(g)
        nf = self.embedding(feature)
        out_nf = self.gnn(g, nf)
        policy = self.bipartite_policy.get_policy(g, bipartite_g, out_nf)  ###### WIP

        ag_nf = out_nf[ag_node_idx]
        ag_nfs = ag_nf.repeat(len(task_node_indices), 1)
        target_nf = out_nf[task_node_indices]

        similarity = target_nf * ag_nfs
        score = similarity.sum(-1, keepdims=True) / math.sqrt(self.embedding_dim)

        # mask out finished task
        score[finished_task] = -inf

        pi = torch.softmax(score, 0)
        action = torch.distributions.Categorical(pi.reshape(-1)).sample().item()

        return action

    def generate_feature(self, g):
        feature = torch.eye(3)[g.ndata['type']]
        return feature

    def fit(self):
        pass

    def push(self, *args):
        self.replay_memory.push([*args])
