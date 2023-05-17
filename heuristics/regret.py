from copy import deepcopy
from functools import partial

import numpy as np

from utils.astar import graph_astar


def manhattan(coord_1, coord_2, *args):
    x = abs(list(coord_1)[0] - list(coord_2)[0])
    y = abs(list(coord_1)[1] - list(coord_2)[1])
    return x + y


def f_ijk(current_tasks, agent_pos, removal_idx, total_tasks, graph, metric='man'):
    assert metric == 'man', "else than man not implemented yet. " \
                            "Check commit 0f816c10240a59fa8f2edef3f925206aead4c7bd to revert "
    n_ag = len(agent_pos)
    before_cost = []
    before_edge_cost = [[] for _ in range(n_ag)]

    for ag_idx in range(n_ag):
        schedule = current_tasks[ag_idx]
        if len(schedule) == 0:
            before_edge_cost[ag_idx].append(0)
            before_cost.append(0)
            continue
        task_locs = np.array(total_tasks)[schedule]
        all_locs = np.concatenate([agent_pos[ag_idx].reshape(1, -1), task_locs])
        all_edge_costs = np.abs(all_locs[:-1] - all_locs[1:]).sum(-1)

        before_edge_cost[ag_idx].append(all_edge_costs)
        before_cost.append(all_edge_costs.sum())

    f = dict(zip(removal_idx, [list() for _ in range(len(removal_idx))]))
    for i in removal_idx:
        rt = total_tasks[i]
        for k in range(n_ag):
            ats = current_tasks[k]
            f_list = list()
            for j in range(len(ats) + 1):
                temp = deepcopy(ats)
                path = list()
                for a in temp:
                    for task_pos in a.values():
                        path += task_pos

                if j == 0:
                    f_value = sum(before_edge_cost[k][1:])
                    f_value += method(agent_pos[k], rt[0], graph)
                    f_value += method(rt[0], path[0], graph)

                elif j == len(ats):
                    f_value = sum(before_edge_cost[k])
                    f_value += method(path[-1], rt[0], graph)

                else:
                    f_value = sum(before_edge_cost[k]) - before_edge_cost[k][j]
                    f_value += method(path[j - 1], rt[0], graph)
                    f_value += method(rt[0], path[j], graph)

                f_list.append(f_value)
            f[i].append(f_list)

    return f


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
