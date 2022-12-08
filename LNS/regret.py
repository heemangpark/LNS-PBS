from copy import deepcopy

import numpy as np

from utils.astar import graph_astar


def f_ijk(removal_idx, current_tasks, total_tasks, graph):
    before_cost = list()
    for b in current_tasks.keys():
        b_path = list()
        b_cost = 0
        for _a in current_tasks[b]:
            for _b in _a.values():
                b_path += _b
        for _s, _g in zip(b_path[:-1], b_path[1:]):
            b_cost += graph_astar(graph, _s, _g)[1]
        before_cost.append(b_cost)

    f = dict(zip(removal_idx, [list() for _ in range(len(removal_idx))]))
    for i in removal_idx:
        rt = total_tasks[i]
        for k in range(len(current_tasks)):
            ats = current_tasks[k]
            f_list = list()
            for j in range(len(ats) + 1):
                f_value = 0
                temp = deepcopy(ats)
                temp.insert(j, {i: rt})

                # cost via A* after re-insertion i, j, k
                path = list()
                for a in temp:
                    for b in a.values():
                        path += b
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
