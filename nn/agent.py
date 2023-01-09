import itertools

import dgl
import torch
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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, g, bipartite_g, task_finished, ag_node_indices, task_node_indices, sample=True):
        feature = self.generate_feature(g)  # one-hot encoded feature 'type'
        nf = self.embedding(feature)
        out_nf = self.gnn(g, nf)

        joint_policy, ag_policy = self.bipartite_policy(g, bipartite_g, out_nf, ag_node_indices, task_node_indices,
                                                        task_finished)
        joint_policy[:, task_finished] = -0
        # joint_policy[:, :, task_finished] = -0

        joint_policy_temp = joint_policy.clone()
        ag_policy_temp = ag_policy.clone()

        selected_ag = []
        out_action = []

        n_ag = joint_policy.shape[-2]
        n_task = sum(~task_finished)
        for itr in range(min(n_ag, n_task)):
            if sample:
                agent_idx = torch.distributions.Categorical(ag_policy_temp).sample()
                action = torch.distributions.Categorical(joint_policy_temp[agent_idx]).sample()
            else:
                agent_idx = ag_policy_temp.argmax()
                action = joint_policy_temp[agent_idx].argmax()
            # action = torch.distributions.Categorical(joint_policy_temp[:, agent_idx]).sample()

            selected_ag.append(agent_idx.item())
            out_action.append(action.item())

            # ag_policy_temp[:, agent_idx] = 0
            # joint_policy_temp[:, :, out_action] = 0
            ag_policy_temp[:, agent_idx] = 0
            joint_policy_temp[:, out_action] = 0

        return selected_ag, out_action

    def get_policy(self, g, bipartite_g, task_finished, ag_node_indices, task_node_indices):
        feature = self.generate_feature(g)  # one-hot encoded feature 'type'
        nf = self.embedding(feature)
        out_nf = self.gnn(g, nf)
        joint_policy, ag_policy = self.bipartite_policy(g, bipartite_g, out_nf, ag_node_indices, task_node_indices,
                                                        task_finished)

        return joint_policy, ag_policy

    def generate_feature(self, g):
        feature = torch.eye(3)[g.ndata['type']]
        return feature

    def fit(self, baseline=0):
        # if len(self.replay_memory) < self.replay_memory.batch_size:
        #     return {'loss': 0}

        di_dgl_g, bipartite_g, ag_node_indices, task_node_indices, selected_ag_idx, joint_action, task_finished, next_t, terminated = self.replay_memory.episode_sample()

        g_size = torch.tensor([g.number_of_nodes() for g in di_dgl_g])
        cumsum_g_size = torch.cumsum(g_size, 0) - g_size[0]

        # graph structured transition
        di_dgl_g = dgl.batch(di_dgl_g)
        bipartite_g = dgl.batch(bipartite_g)

        ag_node_indices = torch.tensor(ag_node_indices)
        task_node_indices = torch.tensor(task_node_indices)

        # get batched node idx
        ag_node_indices = ag_node_indices + cumsum_g_size.view(-1, 1)
        ag_node_indices = ag_node_indices.reshape(-1)
        task_node_indices = task_node_indices + cumsum_g_size.view(-1, 1)
        task_node_indices = task_node_indices.reshape(-1)

        task_finished = torch.tensor(task_finished).reshape(-1)

        next_t = torch.tensor(next_t)
        terminated = torch.Tensor(terminated)

        joint_policy, ag_policy = self.get_policy(di_dgl_g, bipartite_g, task_finished, ag_node_indices,
                                                  task_node_indices)

        n_ag = ag_policy.shape[-1]
        selected_ag_idx = [torch.tensor(s) + i * n_ag for i, s in enumerate(selected_ag_idx)]
        selected_ag_idx = torch.cat(selected_ag_idx)
        joint_action = list(itertools.chain.from_iterable(joint_action))
        joint_action = torch.tensor(joint_action).reshape(-1, 1)

        selected_ag_policy = joint_policy[selected_ag_idx]
        selected_ag_logit = selected_ag_policy.gather(0, joint_action)

        logit_sum = - (selected_ag_logit + 1e-5).log().mean()
        cost = next_t.sum()
        b_val = 0
        loss = (cost - baseline) * logit_sum
        # TODO better loss design

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)
        self.optimizer.step()

        self.replay_memory.memory = []  # on-policy

        return {'loss': loss.item()}

    def push(self, *args):
        self.replay_memory.push([*args])
