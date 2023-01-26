import csv
from collections import Counter

import numpy as np

from LNS.hungarian import HA
from main_ce import CE_collision
from main_lns import LNS
from utils.astar import graph_astar
from utils.convert import to_solver
from utils.generate_scenarios import load_scenarios

np.random.seed(42)
report = [[], []]
collide = False
for itr in range(1, 10001):
    scenario = load_scenarios('202020_1_10_50_0/scenario_{}.pkl'.format(itr))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
    ti, assign = HA(info['graph'], info['agents'], info['tasks'])
    i_a = to_solver(info['tasks'], assign)
    info['assign'] = (ti, assign, i_a)
    as_paths = [[tuple(info['agents'][a])] for a in range(len(info['agents']))]
    for p in range(len(info['agents'])):
        as_paths[p] += graph_astar(info['graph'], info['agents'][p], i_a[p][0])[0][1:-1]
        for i, j in zip(i_a[p][:-1], i_a[p][1:]):
            as_paths[p] += graph_astar(info['graph'], i, j)[0][:-1]
        as_paths[p] += [tuple(i_a[p][-1])]
    n_ag = len(as_paths)
    max_len = max([len(p) for p in as_paths])
    for p in range(len(as_paths)):
        for p_l in range(len(as_paths[p])):
            as_paths[p][p_l] = tuple(as_paths[p][p_l])
    for t in range(len(as_paths)):
        curr_path_l = len(as_paths[t])
        for _ in range(max_len - curr_path_l):
            as_paths[t].append(-t)
    as_paths = np.array(as_paths, dtype=np.ndarray)
    collision_agent = list()
    for t in range(max_len):
        if len(set(as_paths[:, t])) < n_ag:
            collide = True
            c1 = Counter(as_paths[:, t])
            c2 = Counter(set(as_paths[:, t]))
            collision_node_idx = c1 - c2
            collision_node = list(collision_node_idx.elements())[0]
            collision_agent.append(tuple([i for i, n in enumerate(as_paths[:, t]) if n == collision_node]))
            info['path'] = (as_paths, collision_agent)

    if collide:
        lns, ce = LNS(info), CE_collision(info)
        collide = False
        if lns == 'NaN' or ce == 'NaN':
            pass
        else:
            print(lns, ce)
            report[0].append(lns), report[1].append(ce)
            if len(report[0]) == 10:
                break
    else:
        pass

with open('exp_lns_collide.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(report)
