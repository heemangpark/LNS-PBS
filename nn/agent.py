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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, g, ag_order, continuing_ag, joint_action_prev, sample=True):
        # bs = g.batch_size
        n_ag = len(ag_order)
        policy = self.get_policy(g)

        policy_temp = policy.clone().reshape(n_ag, -1)
        # selected_ag = []
        out_action = []
        for itr in range(n_ag):
            policy_temp[:, -1] = 1e-5  # dummy node
            agent_idx = ag_order[itr]
            # TODO: normalize prob?

            if sample:
                selected_ag_policy = policy_temp[agent_idx]
                action = torch.distributions.Categorical(selected_ag_policy).sample()
            else:
                action = policy_temp[agent_idx].argmax(-1)

            action[bool(continuing_ag[agent_idx])] = joint_action_prev[agent_idx].item()

            # if bs > 1:
            #     # selected_ag.append(agent_idx.tolist())
            #     out_action.append(action.tolist())
            # else:
            #     # selected_ag.append(agent_idx)
            out_action.append(action.item())

            policy_temp[:, action] = 0

        return out_action

    def forward_heuristic(self, g, ag_order, continuing_ag, joint_action_prev, **kwargs):
        n_ag = len(ag_order)
        dists = g.edata['dist'].reshape(-1, n_ag).T

        finished_type = g.ndata['type'][n_ag:] == 2
        dists[:, ~finished_type] = 999
        policy_temp = 1 / dists
        policy_temp = policy_temp / policy_temp.sum(-1, keepdims=True)

        out_action = []
        for itr in range(n_ag):
            policy_temp[:, -1] = 1e-5  # dummy node
            agent_idx = ag_order[itr]
            # TODO: normalize prob?
            action = policy_temp[agent_idx].argmax(-1)
            action[bool(continuing_ag[agent_idx])] = joint_action_prev[agent_idx].item()
            out_action.append(action.item())
            policy_temp[:, action] = 0

        return out_action

    def get_policy(self, g):
        feature = self.generate_feature(g)  # one-hot encoded feature 'type'
        embeddings = self.embedding(feature)
        out_nf = self.gnn(g, embeddings)
        policy = self.bipartite_policy(g, out_nf)
        policy[:, -1] = 1e-5

        return policy

    def generate_feature(self, g):
        # feature = torch.eye(3)[g.ndata['type']]
        # feature = torch.cat([feature, g.ndata['loc']], -1)
        feature = g.ndata['loc']

        if 'dist' not in g.edata.keys():
            g.apply_edges(lambda edges: {'dist': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})

        return feature

    def fit(self, baseline=0):
        gs, joint_action, ag_order, task_finished, next_t, terminated = self.replay_memory.episode_sample()
        bs = len(gs)
        gs = dgl.batch(gs)
        # ag_order = torch.Tensor(ag_order)

        joint_action = torch.tensor(joint_action)
        all_action = joint_action.reshape(-1, 1)
        # ag_order = torch.tensor(ag_order)

        next_t = torch.tensor(next_t)

        policy = self.get_policy(gs)  # shape = bs * M, N
        _pol = policy.gather(-1, all_action)
        _pol = _pol.log()
        _pol[all_action == 20] = 0
        _pol = _pol.reshape(bs, -1) + 1e-4

        _logit = ((next_t - baseline).unsqueeze(-1) * _pol).mean(-1)
        loss = _logit.mean()

        # behaved_agents = all_action < 20
        # selected_ag_pol = _pol[behaved_agents]

        # logit_sum = (selected_ag_pol + 1e-5).log().mean()
        # cost = next_t.sum()
        # loss = (cost - baseline) * (logit_sum)

        # TODO better loss design

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.replay_memory.memory = []  # on-policy

        return {'loss': loss.item()}

    def push(self, *args):
        self.replay_memory.push([*args])
