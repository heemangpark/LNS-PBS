import random

import numpy as np

from utils.astar import graph_astar


def removal(tasks_idx, task_pos, graph, N=2, time_log=None, metric='man'):
    if time_log == 'stop':
        return 'stop'

    t_idx = sum(tasks_idx, [])
    chosen = random.choice(t_idx)

    if metric == 'man':
        chosen_pos = task_pos[chosen]
        curr_task_pos = np.array(task_pos)[t_idx]
        rel_d = np.abs(curr_task_pos - np.array(chosen_pos)).sum(-1)
    else:
        raise NotImplementedError("only manhattan distance option is implemented")

    if time_log is None:
        rel_t = 0
    else:
        raise NotImplementedError

    rel = rel_t + rel_d
    mink_indices = sorted(range(len(rel)), key=lambda i: rel[i])[:N + 1]  # the first index is as same as chosen
    removal_idx = np.array(t_idx)[mink_indices].tolist()

    return removal_idx


def relatedness(graph, ti, tj, time_log=None, metric='man'):
    if metric == 'man':
        d_si_sj = manhattan(ti, tj)
    else:
        d_si_sj = graph_astar(graph, ti[0], tj[0])[1]

    if time_log is None:
        timestep_i = 0
        timestep_j = 0
    elif time_log == 'error':
        return 'stop'
    else:
        timestep_i = time_log[tuple(ti[0])]
        timestep_j = time_log[tuple(tj[0])]

    return d_si_sj + abs(timestep_i - timestep_j)


def manhattan(coord_1, coord_2):
    x = abs(list(coord_1)[0] - list(coord_2)[0])
    y = abs(list(coord_1)[1] - list(coord_2)[1])
    return x + y
