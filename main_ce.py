import copy
import time
from collections import Counter
from itertools import permutations

import numpy as np

from LNS.hungarian import HA
from seq_solver import seq_solver
from utils.astar import graph_astar
from utils.convert import to_solver


# "initial locations"
# scenario = load_scenarios('101020_1_10_30_0/scenario_1.pkl')
# grid, graph, agents, tasks = scenario[0], scenario[1], scenario[2], scenario[3]
#
# "Hungarian Assignment"
# _, assign = HA(graph, agents, tasks)
# i_a = to_solver(tasks, assign)
#
# as_paths = [[tuple(i_a[p][0])] for p in range(len(i_a))]
# for p in range(len(as_paths)):
#     for i, j in zip(i_a[p][:-1], i_a[p][1:]):
#         as_paths[p] += graph_astar(graph, i, j)[0][1:]
# cost, paths = seq_solver(grid, agents, i_a, {'time_limit': 1, 'sub_op': 1.1})
# print('HA || SOC: {:.4f}'.format(cost))
#
# n_ag = len(as_paths)
# max_len = max([len(p) for p in as_paths])
# for p in range(len(as_paths)):
#     for p_l in range(len(as_paths[p])):
#         as_paths[p][p_l] = tuple(as_paths[p][p_l])
# for t in range(len(as_paths)):
#     curr_path_l = len(as_paths[t])
#     for _ in range(max_len - curr_path_l):
#         as_paths[t].append(-t)
# as_paths = np.array(as_paths, dtype=np.ndarray)
# for t in range(max_len):
#     if len(set(as_paths[:, t])) < n_ag:
#         c1 = Counter(as_paths[:, t])
#         c2 = Counter(set(as_paths[:, t]))
#         collision_node_idx = c1 - c2
#         collision_node = list(collision_node_idx.elements())[0]
#         collision_agent = [i for i, n in enumerate(as_paths[:, t]) if n == collision_node]


# "Cross Exchange (Long-Short) | Sub-Optimal"
# max_t = time.time() + 10
# substring_size = 1
# itr = 0
#
# while True:
#     ce_time = time.time()
#
#     # Search and Swap
#     L, S = np.argmax([len(p) for p in paths]), np.argmin([len(p) for p in paths])
#     SL, SS = [l for l in permutations(range(len(i_a[L])), 2)], [s for s in permutations(range(len(i_a[S])), 2)]
#     # Sub-Optimal
#     limit_l = [cand for cand in SL if np.abs(cand[0] - cand[1]) <= substring_size]
#     limit_s = [cand for cand in SS if np.abs(cand[0] - cand[1]) <= substring_size]
#     SL, SS = limit_l, limit_s
#
#     np_i_a = np.array([np.array(i_a[i]) for i in range(len(i_a))], dtype=np.ndarray)
#     output = [[], [], []]
#     for sl in SL:
#         if sl[0] < sl[1]:
#             _sl = np.arange(sl[0], sl[1] + 1)
#         else:
#             _sl = np.arange(sl[0], sl[1] - 1, -1)
#         for ss in SS:
#             ids = copy.deepcopy(i_a)
#             if ss[0] < ss[1]:
#                 _ss = np.arange(ss[0], ss[1] + 1)
#             else:
#                 _ss = np.arange(ss[0], ss[1] - 1, -1)
#             l_h, l_s, l_t = np_i_a[L][:min(_sl)], np_i_a[L][_sl], np_i_a[L][max(_sl) + 1:]
#             s_h, s_s, s_t = np_i_a[S][:min(_ss)], np_i_a[S][_ss], np_i_a[S][max(_ss) + 1:]
#             ids[L], ids[S] = np.concatenate([l_h, s_s, l_t]).tolist(), np.concatenate([s_h, l_s, s_t]).tolist()
#             output[0].append([sl, ss]), output[1].append(ids)
#             output[2].append(seq_solver(grid, agents, ids, {'time_limit': 1, 'sub_op': 1.1})[0])
#     swap = output[1][np.argmin(output[2])]
#
#     ce_time = time.time() - ce_time
#     if time.time() > max_t:
#         break
#
#     itr += 1
#     i_a = swap
#     cost, paths = seq_solver(grid, agents, i_a, {'time_limit': 1, 'sub_op': 1.1})
#     print('{}_Solution || SOC: {:.4f} / TIMECOST: {:.4f}'.format(itr, cost, ce_time))


def CE_length(info):
    _, assign = HA(info['graph'], info['agents'], info['tasks'])
    i_a = to_solver(info['tasks'], assign)
    h_cost, paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})
    max_t = time.time() + 5
    substring_size = 1
    itr = 0
    while True:
        ce_time = time.time()
        L, S = np.argmax([len(p) for p in paths]), np.argmin([len(p) for p in paths])
        SL, SS = [l for l in permutations(range(len(i_a[L])), 2)], [s for s in permutations(range(len(i_a[S])), 2)]
        limit_l = [cand for cand in SL if np.abs(cand[0] - cand[1]) <= substring_size]
        limit_s = [cand for cand in SS if np.abs(cand[0] - cand[1]) <= substring_size]
        SL, SS = limit_l, limit_s
        np_i_a = np.array([np.array(i_a[i]) for i in range(len(i_a))], dtype=np.ndarray)
        output = [[], [], []]
        for sl in SL:
            if sl[0] < sl[1]:
                _sl = np.arange(sl[0], sl[1] + 1)
            else:
                _sl = np.arange(sl[0], sl[1] - 1, -1)
            for ss in SS:
                ids = copy.deepcopy(i_a)
                if ss[0] < ss[1]:
                    _ss = np.arange(ss[0], ss[1] + 1)
                else:
                    _ss = np.arange(ss[0], ss[1] - 1, -1)
                l_h, l_s, l_t = np_i_a[L][:min(_sl)], np_i_a[L][_sl], np_i_a[L][max(_sl) + 1:]
                s_h, s_s, s_t = np_i_a[S][:min(_ss)], np_i_a[S][_ss], np_i_a[S][max(_ss) + 1:]
                ids[L], ids[S] = np.concatenate([l_h, s_s, l_t]).tolist(), np.concatenate([s_h, l_s, s_t]).tolist()
                output[0].append([sl, ss]), output[1].append(ids)
                cost, _ = seq_solver(info['grid'], info['agents'], ids, {'time_limit': 1, 'sub_op': 1.1})
                if cost == 'shutdown_c':
                    return 'NaN'
                output[2].append(cost)

        swap = output[1][np.argmin(output[2])]
        ce_time = time.time() - ce_time
        if time.time() > max_t:
            break
        itr += 1
        i_a = swap
        cost, temp_paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})
        if cost < h_cost:
            paths = temp_paths
        else:
            pass
    return (h_cost - cost) / h_cost * 100


def CE_collision(info):
    _, assign = HA(info['graph'], info['agents'], info['tasks'])
    i_a = to_solver(info['tasks'], assign)
    h_cost, paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})

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
            c1 = Counter(as_paths[:, t])
            c2 = Counter(set(as_paths[:, t]))
            collision_node_idx = c1 - c2
            collision_node = list(collision_node_idx.elements())[0]
            collision_agent = [i for i, n in enumerate(as_paths[:, t]) if n == collision_node]

    max_t = time.time() + 5
    substring_size = 1
    itr = 0

    while True:
        ce_time = time.time()
        f, s = collision_agent[0], collision_agent[1]
        sub_f, sub_s = [l for l in permutations(range(len(i_a[f])), 2)], [s for s in
                                                                          permutations(range(len(i_a[s])), 2)]
        limit_f = [cand for cand in sub_f if np.abs(cand[0] - cand[1]) <= substring_size]
        limit_s = [cand for cand in sub_s if np.abs(cand[0] - cand[1]) <= substring_size]
        sub_f, sub_s = limit_f, limit_s
        np_i_a = np.array([np.array(i_a[i]) for i in range(len(i_a))], dtype=np.ndarray)
        output = [[], [], []]
        for sf in sub_f:
            if sf[0] < sf[1]:
                _sl = np.arange(sf[0], sf[1] + 1)
            else:
                _sl = np.arange(sf[0], sf[1] - 1, -1)
            for ss in sub_s:
                ids = copy.deepcopy(i_a)
                if ss[0] < ss[1]:
                    _ss = np.arange(ss[0], ss[1] + 1)
                else:
                    _ss = np.arange(ss[0], ss[1] - 1, -1)
                l_h, l_s, l_t = np_i_a[f][:min(_sl)], np_i_a[f][_sl], np_i_a[f][max(_sl) + 1:]
                s_h, s_s, s_t = np_i_a[s][:min(_ss)], np_i_a[s][_ss], np_i_a[s][max(_ss) + 1:]
                ids[f], ids[s] = np.concatenate([l_h, s_s, l_t]).tolist(), np.concatenate([s_h, l_s, s_t]).tolist()
                output[0].append([sf, ss]), output[1].append(ids)
                cost, _ = seq_solver(info['grid'], info['agents'], ids, {'time_limit': 1, 'sub_op': 1.1})
                if cost == 'shutdown_c':
                    return 'NaN'
                output[2].append(cost)

        swap = output[1][np.argmin(output[2])]
        ce_time = time.time() - ce_time
        if time.time() > max_t:
            break
        itr += 1
        i_a = swap
        cost, temp_paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})
        if cost < h_cost:
            paths = temp_paths
        else:
            pass
    return (h_cost - cost) / h_cost * 100


def CE_random(info):
    _, assign = HA(info['graph'], info['agents'], info['tasks'])
    i_a = to_solver(info['tasks'], assign)
    h_cost, paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})
    if h_cost == 'shutdown_c':
        return 'NaN'

    max_t = time.time() + 5
    substring_size = 1
    itr = 0

    while True:
        ce_time = time.time()
        rand_pool = np.random.choice(len(assign), 2, replace=False)
        f, s = rand_pool[0], rand_pool[1]
        sub_f, sub_s = [l for l in permutations(range(len(i_a[f])), 2)], [s for s in permutations(range(len(i_a[s])), 2)]
        limit_f = [cand for cand in sub_f if np.abs(cand[0] - cand[1]) <= substring_size]
        limit_s = [cand for cand in sub_s if np.abs(cand[0] - cand[1]) <= substring_size]
        sub_f, sub_s = limit_f, limit_s
        np_i_a = np.array([np.array(i_a[i]) for i in range(len(i_a))], dtype=np.ndarray)
        output = [[], [], []]
        for sf in sub_f:
            if sf[0] < sf[1]:
                _sl = np.arange(sf[0], sf[1] + 1)
            else:
                _sl = np.arange(sf[0], sf[1] - 1, -1)
            for ss in sub_s:
                ids = copy.deepcopy(i_a)
                if ss[0] < ss[1]:
                    _ss = np.arange(ss[0], ss[1] + 1)
                else:
                    _ss = np.arange(ss[0], ss[1] - 1, -1)
                l_h, l_s, l_t = np_i_a[f][:min(_sl)], np_i_a[f][_sl], np_i_a[f][max(_sl) + 1:]
                s_h, s_s, s_t = np_i_a[s][:min(_ss)], np_i_a[s][_ss], np_i_a[s][max(_ss) + 1:]
                ids[f], ids[s] = np.concatenate([l_h, s_s, l_t]).tolist(), np.concatenate([s_h, l_s, s_t]).tolist()
                output[0].append([sf, ss]), output[1].append(ids)
                cost, _ = seq_solver(info['grid'], info['agents'], ids, {'time_limit': 1, 'sub_op': 1.1})
                if cost == 'shutdown_c':
                    return 'NaN'
                output[2].append(cost)

        swap = output[1][np.argmin(output[2])]
        ce_time = time.time() - ce_time
        if time.time() > max_t:
            break
        itr += 1
        i_a = swap
        cost, temp_paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})
        if cost < h_cost:
            paths = temp_paths
        else:
            pass
    return (h_cost - cost) / h_cost * 100
