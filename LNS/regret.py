from copy import deepcopy

import numpy as np

from utils.astar import graph_astar


def f_ijk(current_tasks, agent_pos, removal_idx, total_tasks, graph):
    before_cost = list()
    for b in current_tasks.keys():
        b_path = list()
        for _a in current_tasks[b]:
            for _b in _a.values():
                b_path += _b
        b_cost = 0 if len(b_path) == 0 else graph_astar(graph, agent_pos[b], b_path[0])[1]
        for _s, _g in zip(b_path[:-1], b_path[1:]):
            b_cost += graph_astar(graph, _s, _g)[1]
        before_cost.append(b_cost)

    f = dict(zip(removal_idx, [list() for _ in range(len(removal_idx))]))
    for i in removal_idx:
        rt = total_tasks[i]
        for k in range(len(agent_pos)):
            ats = current_tasks[k]
            f_list = list()
            for j in range(len(ats) + 1):
                temp = deepcopy(ats)
                temp.insert(j, {i: rt})
                path = list()
                for a in temp:
                    for b in a.values():
                        path += b
                f_value = graph_astar(graph, agent_pos[k], path[0])[1]
                for s, g in zip(path[:-1], path[1:]):
                    f_value += graph_astar(graph, s, g)[1]
                for o in current_tasks.keys() - [k]:
                    f_value += before_cost[o]
                f_list.append(f_value)
            f[i].append(f_list)

    return f


def get_regret(f_values):
    regret = dict()
    for k, vs in f_values.items():
        v = list()
        for v_id in range(len(vs)):
            v += vs[v_id]
        temp = np.argmin(v) + 1
        for a, v_ in enumerate(vs):
            temp -= len(v_)
            if temp <= 0:
                break
        j = np.argmin(vs[a])
        v = sorted(v)
        regret[k] = [v[1] - v[0], a, j]

    return regret
