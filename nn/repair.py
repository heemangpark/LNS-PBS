import copy

import dgl
import networkx as nx
import torch
from torch import nn as nn

from nn.gnn import GNN, GNNLayerEdgewise


def tempDestroy(assign, graph, removal):
    partial_g = dgl.remove_nodes(graph, graph.nodes()[graph.ndata['idx'] == removal].item())
    for sch in assign.items():
        if removal in sch[1]:
            removed = sch[1].index(removal)
            ag, task = sch[0], sch[1]
            continue
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
        self.edge_layer = GNNLayerEdgewise(embedding_dim * 2, embedding_dim)
        self.score_layer = nn.Linear(embedding_dim * 2, 1)

    def forward(self, assign, graph, removal):
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
            c_edges = partial_g.filter_edges(_connected_edges)
            nf = self.init_node_embedding(partial_g.ndata['coord'])
            updated_nf = self.gnn(partial_g, nf)
            ef = self.edge_layer(partial_g, updated_nf)
            r_embed = self.init_node_embedding(graph.ndata['coord'][graph.ndata['idx'] == r])

            score_input = torch.cat([r_embed.repeat(len(c_edges), 1), ef], -1)
            score = self.score_layer(score_input)
            score_prob = torch.softmax(score, 0).squeeze()
            prob_output.append(score_prob)

        return torch.stack(prob_output)

    def destroy_and_repair(self, assign, graph, removal):

        prob_output = []

        for r in removal:
            partial_g = tempDestroy(assign, graph, r)
            c_edges = partial_g.filter_edges(_connected_edges)
            nf = self.init_node_embedding(partial_g.ndata['coord'].float())
            updated_nf = self.gnn(partial_g, nf)
            ef = self.edge_layer(partial_g, updated_nf)
            r_embed = self.init_node_embedding(graph.ndata['coord'][graph.ndata['idx'] == r].float())

            score_input = torch.cat([r_embed.repeat(len(c_edges), 1), ef], -1)
            score = self.score_layer(score_input)
            score_prob = torch.softmax(score, 0).squeeze()
            prob_output.append(score_prob)

        return [torch.argmax(po).item() for po in prob_output]


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
        to_repair = self.repair(assign, dgl_batch, removal)

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

    def eval(self, assign=None, graph=None, removal=None):
        nf_idx = []
        for idx in assign.values():
            nf_idx.extend([-1] + idx)
        nx.set_node_attributes(graph, dict(zip(graph.nodes, nf_idx)), 'idx')
        dgl_graph = dgl.from_networkx(
            graph,
            node_attrs=['coord', 'type', 'idx'],
            edge_attrs=['a_dist', 'dist', 'obs_proxy', 'connected']
        )
        dgl_graph.ndata['id'] = torch.arange(graph.number_of_nodes())
        re_insert_position = self.repair.destroy_and_repair(assign, dgl_graph, removal)

        abs_pos = []
        for r in removal:
            for a, t in assign.items():
                if r in t:
                    abs_pos.append([a, t.index(r)])

        loc = []
        for r, rip in zip(removal, re_insert_position):
            temp_g = tempDestroy(assign, dgl_graph, r)
            c_edges = temp_g.filter_edges(_connected_edges)
            src = temp_g.edges()[0][c_edges[rip]]
            into = temp_g.ndata['idx'][src].item()

            if into == -1:
                dst = temp_g.edges()[1][c_edges[rip]]
                into = temp_g.ndata['idx'][dst].item()
                for a, t in assign.items():
                    if into in t:
                        loc.append([a, t.index(into)])
            else:
                for a, t in assign.items():
                    if into in t:
                        loc.append([a, t.index(into) + 1])

        for a, l, r in zip(abs_pos, loc, removal):
            assign[a[0]].remove(r)
            assign[l[0]].insert(l[1], r)

        return assign