import numpy as np
from copy import deepcopy
# from functools import partial
from math import inf
# from utils.astar import graph_astar




def manhattan(coord_1, coord_2, *args):
    x = abs(list(coord_1)[0] - list(coord_2)[0])
    y = abs(list(coord_1)[1] - list(coord_2)[1])
    return x + y


def f_ijk(current_tasks, agent_pos, removal_indices, total_tasks, graph, metric='man'):
    assert metric == 'man', "else than man not implemented yet. " \
                            "Check commit 0f816c10240a59fa8f2edef3f925206aead4c7bd to revert "
    n_ag = len(agent_pos)
    total_tasks = np.array(total_tasks)
    before_all_locs = []

    # before location
    for ag_idx in range(n_ag):
        schedule = current_tasks[ag_idx]
        task_locs = total_tasks[schedule]
        all_locs = np.concatenate([agent_pos[ag_idx].reshape(1, -1), task_locs])
        before_all_locs.append(all_locs)

    before_all_locs = np.concatenate(before_all_locs)
    before_edge_cost = manhattan_edgewise(before_all_locs)

    # insertion
    ret_f = dict()
    for removal_idx in removal_indices:
        removal_pos = np.array(total_tasks[removal_idx])
        insert_src = before_all_locs[..., :-1, :]
        insert_dst = before_all_locs[..., 1:, :]
        removal_pos_r = np.tile(removal_pos, (*insert_src.shape[:-1], 1))

        # inserted_edges = (s, removal_node, d)
        inserted_edges = np.stack([insert_src, removal_pos_r, insert_dst], axis=-2)  # n_nodes, 3, 2
        inserted_edge_cost = manhattan_edgewise(inserted_edges).sum(-1)

        # to mask out schedule-to-schedule position, identify whether the destination location is agent position or not
        redundant_edge = insert_dst.reshape(-1, 1, 2) == agent_pos.reshape(1, -1, 2)
        agent_indicator = redundant_edge.prod(-1)  # (x1 == x2) and (y1 == y2)
        is_redundant = agent_indicator.sum(-1).astype(bool)  # 1 if the redundant edge

        edge_cost_gap = inserted_edge_cost - before_edge_cost
        edge_cost_gap = edge_cost_gap.astype(float)
        edge_cost_gap[is_redundant] = inf

        ret_f[removal_idx] = edge_cost_gap

    return ret_f


def get_regret(f_values):
    regret = dict()
    for removal_idx, vs in f_values.items():
        v = list()
        for ag_idx in range(len(vs)):
            v += vs[ag_idx]
        temp = np.argmin(v) + 1
        for a, v_ in enumerate(vs):
            temp -= len(v_)
            if temp <= 0:
                break
        j = np.argmin(vs[a])
        v = sorted(v)
        regret[removal_idx] = [v[1] - v[0], a, j]

    return regret


def manhattan_edgewise(positions):
    if type(positions) == list:
        positions = np.array(positions)

    src_pos = positions[..., :-1, :]
    dst_pos = positions[..., 1:, :]
    return np.abs(src_pos - dst_pos).sum(-1)
