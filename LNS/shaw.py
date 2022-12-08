import random

from utils.astar import graph_astar


def removal(tasks_idx, tasks, graph, N=2):
    t_idx = [v[i] for v in tasks_idx.values() for i in range(len(v))]
    chosen = random.choice(t_idx)
    t_idx.remove(chosen)

    rs = dict()
    for r in t_idx:
        rs[r] = relatedness(graph, tasks[chosen], tasks[r])
    sorted_r = dict(sorted(rs.items(), key=lambda x: x[1], reverse=True))
    removal_idx = [chosen] + [list(sorted_r.keys())[s] for s in range(N)]

    # ind = list()
    # for ri in removal_idx:
    #     for i, a in enumerate(tasks_idx.values()):
    #         if np.argwhere(np.array(a) == ri).shape == (1, 1):
    #             ind.append([i, np.argwhere(np.array(a) == ri).item()])

    return removal_idx


def relatedness(graph, ti, tj, w1=9, w2=3):
    _, d_si_sj = graph_astar(graph, ti[0], tj[0])
    _, d_gi_gj = graph_astar(graph, ti[-1], tj[-1])
    return w1 * (d_si_sj + d_gi_gj) + w2 * (len(ti) + len(tj))
