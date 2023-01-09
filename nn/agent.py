import torch
import dgl
import torch.nn as nn

from nn.gnn import GNN, Bipartite
from nn.memory import ReplayMemory


class Agent(nn.Module):
    def __init__(self, embedding_dim=128, memory_size=50000, batch_size=100, gnn_layers=3):
        super(Agent, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Linear(3, embedding_dim)
        self.gnn = GNN(in_dim=embedding_dim, out_dim=embedding_dim, embedding_dim=embedding_dim, n_layers=gnn_layers,
                       residual=True)
        self.bipartite_policy = Bipartite(embedding_dim)

        self.replay_memory = ReplayMemory(capacity=memory_size, batch_size=batch_size)

    def forward(self, g, bipartite_g, task_finished, ag_node_indices, task_node_indices):
        feature = self.generate_feature(g)  # one-hot encoded feature 'type'
        nf = self.embedding(feature)
        out_nf = self.gnn(g, nf)

        joint_policy, ag_policy = self.bipartite_policy(g, bipartite_g, out_nf, ag_node_indices, task_node_indices, task_finished)
        joint_policy[:, task_finished] = -0

        joint_policy_temp = joint_policy.clone()
        ag_policy_temp = ag_policy.clone()

        selected_ag = []
        out_action = []

        n_ag = joint_policy.shape[-2]
        n_task = sum(~task_finished)
        for itr in range(min(n_ag, n_task)):
            agent_idx = torch.distributions.Categorical(ag_policy_temp).sample()
            action = torch.distributions.Categorical(joint_policy_temp[agent_idx]).sample()

            selected_ag.append(agent_idx.item())
            out_action.append(action.item())

            ag_policy_temp[agent_idx] = 0
            joint_policy_temp[:, out_action] = 0

        return selected_ag, out_action

    def generate_feature(self, g):
        feature = torch.eye(3)[g.ndata['type']]
        return feature

    def fit(self):
        if len(self.replay_memory) < self.replay_memory.batch_size:
            return {}

        di_dgl_g, bipartite_g, ag_node_indices, task_node_indices, next_t, terminated = self.replay_memory.sample()

        di_dgl_g = dgl.batch(di_dgl_g)
        bipartite_g = dgl.batch(bipartite_g)

        next_t = torch.Tensor(next_t)
        terminated = torch.Tensor(terminated)


    def push(self, *args):
        self.replay_memory.push([*args])
