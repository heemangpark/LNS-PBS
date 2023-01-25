import csv

import numpy as np

from LNS.hungarian import HA
from main_ce import CE_collision
from main_lns import LNS
from utils.astar import graph_astar
from utils.convert import to_solver
from utils.generate_scenarios import load_scenarios

report = [[], []]

itr = 1
collide = False
for itr in range(1, 1001):
    scenario = load_scenarios('101020_1_10_30_0/scenario_{}.pkl'.format(itr))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
    _, assign = HA(info['graph'], info['agents'], info['tasks'])
    i_a = to_solver(info['tasks'], assign)
    as_paths = [[tuple(i_a[p][0])] for p in range(len(i_a))]
    for p in range(len(as_paths)):
        for i, j in zip(i_a[p][:-1], i_a[p][1:]):
            as_paths[p] += graph_astar(info['graph'], i, j)[0][1:]
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
    for t in range(max_len):
        if len(set(as_paths[:, t])) < n_ag:
            collide = True

    if collide:
        report[0].append(LNS(info)), report[1].append(CE_collision(info))
        collide = False
        print(itr, len(report[0]))
        if len(report[0]) == 100:
            break
    else:
        pass

with open('exp_collide.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(report)
