import itertools
import dgl
import torch
import torch.nn as nn

from nn.gnn import GNN, Bipartite
from nn.memory import ReplayMemory


class Agent(nn.Module):
    def __init__(self, embedding_dim=128, memory_size=50000, batch_size=100, gnn_layers=1):
        super(Agent, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Linear(2, embedding_dim)
        self.gnn = GNN(in_dim=embedding_dim, out_dim=embedding_dim, embedding_dim=embedding_dim, n_layers=gnn_layers,
                       residual=True)
        self.bipartite_policy = Bipartite(embedding_dim)

        self.replay_memory = ReplayMemory(capacity=memory_size, batch_size=batch_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, g, ag_order, continuing_ag, joint_action_prev, sample=True):
        bs = g.batch_size
        n_ag = ag_order.shape[-1]
        feature = self.generate_feature(g)  # one-hot encoded feature 'type'

        embeddings = self.embedding(feature)
        out_nf = self.gnn(g, embeddings)
        policy = self.bipartite_policy(g, out_nf)

        policy_temp = policy.clone().reshape(bs, n_ag, -1)
        selected_ag = []
        out_action = []
        for itr in range(n_ag):
            policy_temp[:, :, -1] = 1e-5  # dummy node
            agent_idx = ag_order[:, itr]
            # TODO: normalize prob

            if sample:
                selected_ag_policy = policy_temp[torch.arange(bs), agent_idx]
                action = torch.distributions.Categorical(selected_ag_policy).sample()
            else:
                action = policy_temp[:, agent_idx].argmax(-1)

            if bs > 1:
                selected_ag.append(agent_idx.tolist())
                out_action.append(action.tolist())
            else:
                selected_ag.append(agent_idx.item())
                out_action.append(action.item())

            policy_temp[torch.arange(bs), :, action] = 0

        return out_action

    def get_policy(self, g):
        feature = self.generate_feature(g)  # one-hot encoded feature 'type'
        nf = self.embedding(feature)
        out_nf = self.gnn(g, nf)
        policy = self.bipartite_policy(g, out_nf)

        return policy

    def generate_feature(self, g):
        # feature = torch.eye(3)[g.ndata['type']]
        # feature = torch.cat([feature, g.ndata['loc']], -1)
        feature = g.ndata['loc']

        if 'dist' not in g.edata.keys():
            g.apply_edges(lambda edges: {'dist': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})

        return feature

    def fit(self, baseline=0):
        # if len(self.replay_memory) < self.replay_memory.batch_size:
        #     return {'loss': 0}
        gs, ag_node_indices, task_node_indices, ag_order, joint_action, task_finished, next_t, terminated = self.replay_memory.episode_sample()

        g_size = torch.tensor([g.number_of_nodes() for g in gs])
        cumsum_g_size = torch.cumsum(g_size, 0) - g_size[0]

        # graph structured transition
        gs = dgl.batch(gs)

        ag_node_indices = torch.tensor(ag_node_indices)
        task_node_indices = torch.tensor(task_node_indices)

        # get batched node idx
        ag_node_indices = ag_node_indices + cumsum_g_size.view(-1, 1)
        ag_node_indices = ag_node_indices.reshape(-1)
        task_node_indices = task_node_indices + cumsum_g_size.view(-1, 1)
        task_node_indices = task_node_indices.reshape(-1)

        task_finished = torch.tensor(task_finished).reshape(-1)

        next_t = torch.tensor(next_t)

        policy = self.get_policy(gs)

        # TODO:WIP

        n_ag = ag_policy.shape[-1]
        selected_ag_idx = [torch.tensor(s) + i * n_ag for i, s in enumerate(selected_ag_idx)]
        selected_ag_idx = torch.cat(selected_ag_idx)
        joint_action = list(itertools.chain.from_iterable(joint_action))
        joint_action = torch.tensor(joint_action).reshape(-1, 1)

        selected_ag_policy = joint_policy[selected_ag_idx]
        selected_ag_logit = selected_ag_policy.gather(0, joint_action)

        high_level_logit = ag_policy.reshape(-1)[selected_ag_idx]

        high_logit_mean = (high_level_logit + 1e-5).log().mean()
        logit_sum = (selected_ag_logit + 1e-5).log().mean()
        cost = next_t.sum()
        b_val = 0
        loss = (cost - baseline) * (logit_sum + high_logit_mean)

        # TODO better loss design

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)
        self.optimizer.step()

        self.replay_memory.memory = []  # on-policy

        return {'loss': loss.item()}

    def push(self, *args):
        self.replay_memory.push([*args])
