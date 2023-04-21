import random

from utils.astar_utils import graph_astar


def removal(tasks_idx, task_pos, graph, N=2, time_log=None, neighbors='related', metric='man'):
    t_idx = list()
    for v in tasks_idx.values():
        t_idx.extend(v)
    chosen = random.choice(t_idx)
    t_idx.remove(chosen)

    rs = dict()
    for t in t_idx:
        rela_ = relatedness(graph, task_pos[chosen], task_pos[t], time_log=time_log, metric=metric)
        if rela_ == 'stop':
            return 'stop'
        else:
            rs[t] = rela_
    sorted_r = dict(sorted(rs.items(), key=lambda x: x[1], reverse=True))
    removal_idx = [chosen] + [list(sorted_r.keys())[s] for s in range(N)]

    if neighbors == 'related':
        return removal_idx
    else:
        return [chosen] + list(random.sample(list(sorted_r.keys()), k=2))


def relatedness(graph, ti, tj, w1=9, w2=3, time_log=None, metric='man'):
    if metric == 'man':
        d_si_sj = manhattan(ti[0], tj[0])
        # d_gi_gj = manhattan(ti[-1], tj[-1])
    else:
        d_si_sj = graph_astar(graph, ti[0], tj[0])[1]
        # d_gi_gj = graph_astar(graph, ti[-1], tj[-1])[1]

    if time_log is None:
        timestep_i = 0
        timestep_j = 0
    elif time_log == 'error':
        return 'stop'
    else:
        timestep_i = time_log[tuple(ti[0])]
        timestep_j = time_log[tuple(tj[0])]

    return w1 * d_si_sj + w2 * abs(timestep_i - timestep_j)


def manhattan(coord_1, coord_2):
    x = abs(list(coord_1)[0] - list(coord_2)[0])
    y = abs(list(coord_1)[1] - list(coord_2)[1])
    return x + y
