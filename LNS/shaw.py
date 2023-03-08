import random

import numpy as np

from utils.astar import graph_astar


def removal(tasks_idx, task_pos, graph, N=2):
    t_idx = list()
    for v in tasks_idx.values():
        t_idx.extend(v)
    chosen = random.choice(t_idx)
    t_idx.remove(chosen)

    rs = dict()
    for t in t_idx:
        rs[t] = relatedness(graph, task_pos[chosen], task_pos[t])
    sorted_r = dict(sorted(rs.items(), key=lambda x: x[1], reverse=True))
    removal_idx = [chosen] + [list(sorted_r.keys())[s] for s in range(N)]

    return removal_idx


def relatedness(graph, ti, tj, w1=9, w2=3):
    _, d_si_sj = graph_astar(graph, ti[0], tj[0])
    _, d_gi_gj = graph_astar(graph, ti[-1], tj[-1])
    return w1 * (d_si_sj + d_gi_gj) + w2 * (len(ti) + len(tj))
