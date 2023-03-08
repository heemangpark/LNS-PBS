from copy import deepcopy

import numpy as np

from utils.astar import graph_astar


def f_ijk(current_tasks, agent_pos, removal_idx, total_tasks, graph):
    n_ag = len(agent_pos)

    before_cost = []
    before_edge_cost = [[] for _ in range(n_ag)]
    for ag_idx in current_tasks.keys():
        b_path = list()
        for _a in current_tasks[ag_idx]:
            for _b in _a.values():
                b_path += _b

        #   Initial segment
        edge_cost = 0 if len(b_path) == 0 else graph_astar(graph, agent_pos[ag_idx], b_path[0])[1]
        before_edge_cost[ag_idx].append(edge_cost)

        # Edge cost
        for _s, _g in zip(b_path[:-1], b_path[1:]):
            edge_cost = graph_astar(graph, _s, _g)[1]
            before_edge_cost[ag_idx].append(edge_cost)

        path_cost = sum(before_edge_cost[ag_idx])
        before_cost.append(path_cost)

    f = dict(zip(removal_idx, [list() for _ in range(len(removal_idx))]))
    for i in removal_idx:
        rt = total_tasks[i]
        for k in range(n_ag):
            ats = current_tasks[k]
            f_list = list()
            for j in range(len(ats) + 1):
                temp = deepcopy(ats)
                # temp.insert(j, {i: rt})
                path = list()
                for a in temp:
                    for task_pos in a.values():
                        path += task_pos

                if j == 0:
                    f_value = sum(before_edge_cost[k][1:])
                    f_value += graph_astar(graph, agent_pos[k], rt[0])[1]
                    f_value += graph_astar(graph, rt[0], path[0])[1]
                elif j == len(ats):
                    f_value = sum(before_edge_cost[k])
                    f_value += graph_astar(graph, path[-1], rt[0])[1]
                else:
                    f_value = sum(before_edge_cost[k]) - before_edge_cost[k][j]
                    f_value += graph_astar(graph, path[j - 1], rt[0])[1]
                    f_value += graph_astar(graph, rt[0], path[j])[1]

                # f_value = graph_astar(graph, agent_pos[k], path[0])[1]
                # for s, g in zip(path[:-1], path[1:]):
                #     f_value += graph_astar(graph, s, g)[1]
                # for o in current_tasks.keys() - [k]:
                #     f_value += before_cost[o]
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
