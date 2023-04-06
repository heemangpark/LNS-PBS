import copy

import dgl
import networkx as nx
import torch
import torch.nn as nn

from nn.gnn import GNN, GNNLayer_edge


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_size, output_size)
#         )
#         self.log_softmax = nn.LogSoftmax()
#
#     def forward(self, input):
#         batch = True if len(input.shape) == 3 else False
#         x = self.layers(input)
#         if batch:
#             log_probs = self.log_softmax(x)
#         else:
#             log_probs = self.log_softmax(x.unsqueeze(0))
#
#         return log_probs.squeeze()
#
#
# class SL_LNS(nn.Module):
#     def __init__(self, embedding_dim=64, gnn_layers=3, train=True):
#         super(SL_LNS, self).__init__()
#         self.device = 'cuda' if train else 'cpu'
#         self.embedding_dim = embedding_dim
#         self.embedding = nn.Linear(2, embedding_dim)
#         self.gnn = GNN(
#             in_dim=embedding_dim,
#             out_dim=embedding_dim,
#             embedding_dim=embedding_dim,
#             n_layers=gnn_layers,
#             residual=True,
#         )
#
#         self.log_softmax = nn.LogSoftmax(dim=1)
#         self.loss = torch.nn.KLDivLoss(reduction='batchmean')
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#
#     def train(self, task_idx=None, graphs=None, labels=None):
#         batch_size = 10
#         num_batches = len(task_idx) // batch_size
#
#         for b in range(num_batches):
#             t = task_idx[b * batch_size: (b + 1) * batch_size]
#             g = dgl.batch(graphs[b * batch_size: (b + 1) * batch_size]).to(self.device)
#             l = labels[b * batch_size: (b + 1) * batch_size]
#
#             n_ag = sum(g.ndata['type'] == 1).item() // batch_size
#             n_task = sum(g.ndata['type'] == 2).item() // batch_size
#
#             nf = g.ndata['coord'].to(self.embedding.weight.dtype)
#             n_embed = self.embedding(nf)
#             out_nf = self.gnn(g, n_embed)
#             out_nf = out_nf.reshape(batch_size, n_ag + n_task)
#
#             # log_probs = self.policy(out_nf[n_ag:])
#             log_probs = self.log_softmax(out_nf[:, n_ag:])
#             label_probs = torch.Tensor(l).to(self.device)
#
#             # label_idx_list = [[[i for i, j in enumerate(t_id) if j == k][0] for k in label]
#             #                   for t_id, label in zip(task_idx, labels)]
#             # label_probs = torch.Tensor(np.array([np.eye(n_task)[label_idx].mean(0)
#             #                                      for label_idx in label_idx_list])).to(self.device)
#
#             loss = self.loss(log_probs, label_probs)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#         return loss.item()
#
#     def eval(self, g=None, random_action=False, sample=True):
#         nf = g.ndata['coord'].to(self.embedding.weight.dtype)
#         n_embed = self.embedding(nf)
#         out_nf = self.gnn(g, n_embed)
#
#         n_ag = sum(g.ndata['type'] == 1).item()
#         n_task = sum(g.ndata['type'] == 2).item()
#
#         if random_action:
#             rand_probs = torch.rand(n_task)
#             log_probs = (rand_probs / sum(rand_probs)).log()
#         else:
#             log_probs = self.log_softmax(out_nf[n_ag:].unsqueeze(0)).squeeze()
#
#         if sample:
#             actions = torch.distributions.Categorical(logits=log_probs).sample((3,)).tolist()
#         else:
#             actions = torch.topk(log_probs, 3).indices.tolist()
#
#         return list(set(actions))


def tempDestroy(assign, graph, removal):
    partial_g = dgl.remove_nodes(graph, graph.nodes()[graph.ndata['idx'] == removal].item())
    for sch in assign.items():
        if removal in sch[1]:
            removed = sch[1].index(removal)
            ag, task = sch[0], sch[1]
    if removed == 0:
        src = graph.nodes()[graph.ndata['type'] == 1][ag].item()
        dst = src + 1
    elif removed == len(task) - 1:
        return partial_g
    else:
        src = graph.nodes()[graph.ndata['idx'] == task[removed - 1]].item()
        dst = src + 1
    partial_g.edges[src, dst].data['connected'] = torch.LongTensor([1]).to(partial_g.device)
    partial_g.edges[dst, src].data['connected'] = torch.LongTensor([1]).to(partial_g.device)

    return partial_g


def _connected_edges(edges):
    return (edges.data['connected'] == 1) * (edges.src['id'] < edges.dst['id'])


class Repair(nn.Module):
    def __init__(self, embedding_dim=64):
        super(Repair, self).__init__()
        self.init_node_embedding = nn.Linear(2, embedding_dim)
        self.gnn = GNN(
            in_dim=embedding_dim,
            out_dim=embedding_dim,
            embedding_dim=embedding_dim,
            n_layers=2,
            residual=True,
        )
        self.edge_layer = GNNLayer_edge(embedding_dim * 2, embedding_dim)
        self.score_layer = nn.Linear(embedding_dim * 2, 1)

    def forward(self, assign, decrement, graph, removal):
        g = copy.deepcopy(graph)
        if g.batch_size > 1:
            g = dgl.unbatch(g)
        else:
            g = [g]

        probs = []
        for _a, _g, _r in zip(assign, g, removal):
            if len(_r) == 0:
                continue
            prob = self.destroy_and_forward(_a, _g, _r)
            probs.append(prob)

        return torch.cat(probs, 0)

        # else:
        #     return self.destroy_and_forward(assign, g, removal)

    def destroy_and_forward(self, assign, graph, removal):

        prob_output = []

        for r in removal:
            partial_g = tempDestroy(assign, graph, r)
            c_edges = partial_g.filter_edges(_connected_edges)  # TODO: 98? 49?
            nf = self.init_node_embedding(partial_g.ndata['coord'])
            updated_nf = self.gnn(partial_g, nf)
            ef = self.edge_layer(partial_g, updated_nf)
            r_embed = self.init_node_embedding(graph.ndata['coord'][graph.ndata['idx'] == r])

            score_input = torch.cat([r_embed.repeat(len(c_edges), 1), ef], -1)
            score = self.score_layer(score_input)
            score_prob = torch.softmax(score, 0).squeeze()
            prob_output.append(score_prob)

        return torch.stack(prob_output)


class NeuroRepair(nn.Module):
    def __init__(self, embedding_dim=64, gnn_layers=3, train=True):
        super(NeuroRepair, self).__init__()
        self.device = 'cuda' if train else 'cpu'
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(2, embedding_dim)
        self.gnn = GNN(
            in_dim=embedding_dim,
            out_dim=embedding_dim,
            embedding_dim=embedding_dim,
            n_layers=gnn_layers,
            residual=True,
        )
        self.repair = Repair()
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.to(self.device)

    def train(self, assign, decrement, graph, removal):
        batch_graphs = []
        labels = []

        ############### Starts Graph Batching #################
        for a, g in zip(assign, graph):
            nf_idx = []
            for idx in a.values():
                nf_idx.append(-1), nf_idx.extend(idx)
            nx.set_node_attributes(g, dict(zip(g.nodes, nf_idx)), 'idx')
            dgl_g = dgl.from_networkx(g,
                                      node_attrs=['coord', 'type', 'idx'],
                                      edge_attrs=['a_dist', 'dist', 'obs_proxy', 'connected']
                                      )
            dgl_g.ndata['id'] = torch.arange(g.number_of_nodes())
            batch_graphs.append(dgl_g)

        dgl_batch = dgl.batch(batch_graphs).to(self.device)
        if dgl_batch.ndata['coord'].dtype == torch.int64:
            dgl_batch.ndata['coord'] = dgl_batch.ndata['coord'] / 32
            dgl_batch.edata['a_dist'] = dgl_batch.edata['a_dist'] / 32
            dgl_batch.edata['dist'] = dgl_batch.edata['dist'] / 32
            dgl_batch.edata['obs_proxy'] = dgl_batch.edata['obs_proxy'] / 32
        ############### Graph batching completed #################

        ############### Label Data Generating #################
        for a, r in zip(assign, removal):
            assign_schedule = []
            for v in a.values():
                if not len(v) == 0:
                    assign_schedule.append(-1), assign_schedule.extend(v)

            edge_src_indices = copy.copy(assign_schedule)
            removal_temp = copy.deepcopy(r)

            ##### Remove terminal task per agent to maintain only src indices in `edge_src_indices`
            for v in a.values():
                if len(v) == 0:
                    continue
                end_idx = 1
                end_task_idx = v[-end_idx]
                if end_task_idx in r:
                    if len(v) == 1:
                        where_removal_node = edge_src_indices.index(end_task_idx)
                        edge_src_indices.pop(where_removal_node - 1)
                        continue
                    while end_task_idx in r:
                        end_idx += 1
                        end_task_idx = v[-end_idx]
                edge_src_indices.remove(end_task_idx)

            ##### Generate label
            if len(edge_src_indices) != 50:
                A = 0
            for r_id in removal_temp:
                edge_src_indices_temp = copy.copy(edge_src_indices)
                edge_src_indices_temp.remove(r_id)

                schedule_idx = assign_schedule.index(r_id)
                assign_src, assign_dst = schedule_idx - 1, schedule_idx + 1

                # Exception: when assigned at last loc of last agent
                if assign_dst == len(assign_schedule):
                    r.remove(r_id)
                    continue

                # Exception: when assigned at last loc of non-last agent
                if assign_schedule[assign_dst] == -1:
                    r.remove(r_id)
                    continue

                assign_src_node = assign_schedule[assign_src]

                # Exception: when another removal node is assigned before current node
                if assign_src_node in removal_temp:
                    while assign_src_node in removal_temp:
                        assign_src -= 1
                        assign_src_node = assign_schedule[assign_src]

                # # Exception: when
                if assign_src_node not in edge_src_indices_temp:
                    assign_src_node = assign_schedule[assign_src - 1]

                target_label_idx = edge_src_indices_temp.index(assign_src_node)
                label = [0] * len(edge_src_indices_temp)
                label[target_label_idx] = 1
                labels.append(label)

        labels = torch.Tensor(labels).to(self.device)
        to_repair = self.repair(assign, decrement, dgl_batch, removal)

        loss = self.loss(torch.log(to_repair), labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # prev code start
        # for a, d, g, r in zip(assign, decrement, graph, removal):
        #     nf_idx = []
        #     for idx in a.values():
        #         nf_idx.append(-1), nf_idx.extend(idx)
        #     nx.set_node_attributes(g, dict(zip(g.nodes, nf_idx)), 'idx')
        #     dgl_g = dgl.from_networkx(g,
        #                               node_attrs=['coord', 'type', 'idx'],
        #                               edge_attrs=['a_dist', 'dist', 'obs_proxy', 'connected']
        #                               )
        #     dgl_g.ndata['id'] = torch.arange(g.number_of_nodes())
        #     if dgl_g.ndata['coord'].dtype == torch.int64:
        #         dgl_g.ndata['coord'] = dgl_g.ndata['coord'] / 32
        #         dgl_g.edata['a_dist'] = dgl_g.edata['a_dist'] / 32
        #         dgl_g.edata['dist'] = dgl_g.edata['dist'] / 32
        #         dgl_g.edata['obs_proxy'] = dgl_g.edata['obs_proxy'] / 32
        #
        #     dgl_g = dgl_g.to(self.device)
        #
        #     labels = []
        #     temp = []
        #     for v in assign[1].values():
        #         temp.append(-1), temp.extend(v)
        #
        #     label_edges = copy.deepcopy(temp)
        #     r_temp = copy.deepcopy(r)
        #
        #     for v in assign[1].values():
        #         end_task = v[-1]
        #         idx = 1
        #
        #         if end_task in r_temp:
        #             while end_task in r_temp:
        #                 idx += 1
        #                 end_task = v[-idx]
        #
        #         label_edges.remove(end_task)
        #
        #     for r_id in r_temp:
        #         idx = temp.index(r_id)
        #         label_edges_temp = copy.deepcopy(label_edges)
        #         label_edges_temp.remove(r_id)
        #         src, dst = idx - 1, idx + 1
        #         if idx + 1 == len(temp) or dst == -1:
        #             r.remove(r_id)
        #             continue
        #         src_idx = temp[src]
        #
        #         if src_idx in r_temp:
        #             while src_idx in r_temp:
        #                 src -= 1
        #                 src_idx = temp[src]
        #         if src_idx not in label_edges_temp:
        #             src_idx = temp[src - 1]
        #
        #         label_idx = label_edges_temp.index(src_idx)
        #         label = [0] * len(label_edges_temp)
        #         label[label_idx] = 1
        #         labels.append(label)
        #     labels = torch.Tensor(labels).to(self.device)
        #     to_repair = self.repair(a, d, dgl_g, r)
        #
        #     loss = self.loss(torch.log(to_repair), labels)
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #     total_loss += loss.item()

        return loss.item()

    def eval(self, g=None, random_action=False, sample=True):
        nf = g.ndata['coord'].to(self.embedding.weight.dtype)
        n_embed = self.embedding(nf)
        out_nf = self.gnn(g, n_embed)

        n_ag = sum(g.ndata['type'] == 1).item()
        n_task = sum(g.ndata['type'] == 2).item()

        if random_action:
            rand_probs = torch.rand(n_task)
            log_probs = (rand_probs / sum(rand_probs)).log()
        else:
            log_probs = self.log_softmax(out_nf[n_ag:].unsqueeze(0)).squeeze()

        if sample:
            actions = torch.distributions.Categorical(logits=log_probs).sample((3,)).tolist()
        else:
            actions = torch.topk(log_probs, 3).indices.tolist()

        return list(set(actions))

# class RolloutBuffer:
#     def __init__(self):
#         self.actions = []
#         self.states = []
#         self.log_probs = []
#         self.rewards = []
#         self.state_values = []
#         self.is_terminals = []
#
#     def clear(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.log_probs[:]
#         del self.rewards[:]
#         del self.state_values[:]
#         del self.is_terminals[:]
#
#     def push(self, state, action, reward, is_terminal):
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.is_terminals.append(is_terminal)
#
#
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, action_dim),
#             nn.Softmax(0)
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def act(self, state):
#         probs = self.actor(state)
#         distribution = torch.distributions.Categorical(probs.squeeze())
#         action = distribution.sample((3,))
#         log_probs = distribution.log_prob(action)
#         state_val = self.critic(state)
#
#         return action.detach(), log_probs.detach(), state_val.detach()
#
#     def evaluate(self, state, action):
#         probs = self.actor(state)
#         distribution = torch.distributions.Categorical(probs)
#         log_probs = distribution.log_prob(action)
#         dist_entropy = distribution.entropy()
#         state_values = self.critic(state)
#
#         return log_probs, state_values, dist_entropy
#
#
# class PPO_LNS(nn.Module):
#     """ PPO Implementation """
#
#     def __init__(self, state_dim, hidden_dim, action_dim, lr_actor, lr_critic, gamma, epochs, eps_clip):
#         super(PPO_LNS, self).__init__()
#         self.device = torch.device('cuda')
#
#         """PPO components"""
#         self.gamma = gamma
#         self.epochs = epochs
#         self.eps_clip = eps_clip
#         self.buffer = RolloutBuffer()
#
#         self.policy = ActorCritic(state_dim, hidden_dim, action_dim).to(self.device)
#         self.prev_policy = ActorCritic(state_dim, hidden_dim, action_dim).to(self.device)
#         self.prev_policy.load_state_dict(self.policy.state_dict())
#
#         """Graph embedding"""
#         self.embedding = nn.Linear(2, state_dim, device=self.device)
#         self.gnn = GNN(
#             in_dim=state_dim,
#             out_dim=state_dim,
#             embedding_dim=state_dim,
#             n_layers=3,
#             residual=True,
#             device=self.device
#         )
#
#         self.optimizer = torch.optim.Adam([
#             {'params': self.embedding.parameters(), 'lr': 1e-4},
#             {'params': self.gnn.parameters(), 'lr': 1e-4},
#             {'params': self.policy.actor.parameters(), 'lr': lr_actor},
#             {'params': self.policy.critic.parameters(), 'lr': lr_critic}
#         ])
#         self.loss = nn.MSELoss()
#
#     def select_action(self, state):
#         g = dgl.from_networkx(state, node_attrs=['coord', 'type'], edge_attrs=['a_dist', 'dist', 'obs_proxy']).to(
#             self.device)
#         n_ag, n_task = sum(g.ndata['type'] == 1).item(), sum(g.ndata['type'] == 2).item()
#         nf = g.ndata['coord'].to(self.embedding.weight.dtype)
#         n_embed = self.embedding(nf)
#         out_nf = self.gnn(g, n_embed)
#
#         with torch.no_grad():
#             action, log_probs, state_val = self.prev_policy.act(out_nf[n_ag:])
#
#         self.buffer.states.append(state)
#         self.buffer.log_probs.append(log_probs)
#         self.buffer.state_values.append(state_val)
#
#         return action.item()
#
#     def update(self):
#         rewards = []
#         discounted_reward = 0
#         for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#             if is_terminal:
#                 discounted_reward = 0
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
#
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
#
#         prev_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
#         prev_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
#         prev_log_probs = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(self.device)
#         prev_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
#
#         advantages = rewards.detach() - prev_state_values.detach()
#
#         for _ in range(self.epochs):
#             log_probs, state_values, dist_entropy = self.policy.evaluate(prev_states, prev_actions)
#
#             state_values = torch.squeeze(state_values)
#
#             ratios = torch.exp(log_probs - prev_log_probs.detach())
#
#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#
#             loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy
#
#             self.optimizer.zero_grad()
#             loss.mean().backward()
#             self.optimizer.step()
#
#         self.prev_policy.load_state_dict(self.policy.state_dict())
#         self.buffer.clear()
#
#     def save(self, checkpoint_path):
#         torch.save(self.policy_old.state_dict(), checkpoint_path)
#
#     def load(self, checkpoint_path):
#         self.prev_policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
#         self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
