import copy
import time
from collections import Counter
from itertools import permutations

import numpy as np
import wandb

from seq_solver import seq_solver
from utils.astar import graph_astar
from utils.ce_tools import route_relatedness
from utils.solver_util import to_solver


def CE_random(info):
    assign = info['assign'][1]
    i_a = to_solver(info['tasks'], assign)
    h_cost, paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})
    if h_cost == 'error':
        return 'NaN'

    max_t = time.time() + 5
    substring_size = 1
    itr = 0

    while True:
        ce_time = time.time()
        rand_pool = np.random.choice(len(assign), 2, replace=False)
        f, s = rand_pool[0], rand_pool[1]
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
                if cost == 'error':
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


def CE_length(info):
    assign = info['assign'][1]
    i_a = to_solver(info['tasks'], assign)
    h_cost, paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.1})
    if h_cost == 'error':
        return 'NaN'
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
                if cost == 'error':
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
    i_a = info['assign'][2]
    h_cost, paths = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.2})
    # print(h_cost)
    before_cost = h_cost
    if h_cost == 'error':
        return 'NaN'

    max_itr = 5
    init = True

    for itr in range(max_itr):
        if init:
            as_paths, collision_agent = copy.deepcopy(info['path'][0]), copy.deepcopy(info['path'][1])
        else:
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
                    c1 = Counter(as_paths[:, t])
                    c2 = Counter(set(as_paths[:, t]))
                    collision_node_idx = c1 - c2
                    collision_node = list(collision_node_idx.elements())[0]
                    collision_agent.append(tuple([i for i, n in enumerate(as_paths[:, t]) if n == collision_node]))

        "CE한 경로에 더 이상 충돌이 존재 하지 않는 경우"
        if len(collision_agent) == 0:
            print("No collision after itr {}".format(itr))
            return (h_cost - cost) / h_cost * 100

        # swap_pool = sorted(Counter(collision_agent).items(), key=lambda x: x[1], reverse=True)
        init = False
        swap_pool = Counter(collision_agent).keys()
        substring_size = 1
        for swap in swap_pool:

            f, s = np.random.choice(swap, 2, replace=False)
            sub_f = [l for l in permutations(range(len(i_a[f])), 2)]
            sub_s = [s for s in permutations(range(len(i_a[s])), 2)]
            limit_f = [cand for cand in sub_f if np.abs(cand[0] - cand[1]) <= substring_size]
            limit_s = [cand for cand in sub_s if np.abs(cand[0] - cand[1]) <= substring_size]
            sub_f, sub_s = limit_f, limit_s
            np_i_a = np.array([np.array(i_a[i]) for i in range(len(i_a))], dtype=np.ndarray)

            output = [[], [], []]

            for sf in sub_f:
                _sl = np.arange(sf[0], sf[1] + 1) if sf[0] < sf[1] else np.arange(sf[0], sf[1] - 1, -1)

                for ss in sub_s:
                    ids = copy.deepcopy(i_a)
                    _ss = np.arange(ss[0], ss[1] + 1) if ss[0] < ss[1] else np.arange(ss[0], ss[1] - 1, -1)
                    l_h, l_s, l_t = np_i_a[f][:min(_sl)], np_i_a[f][_sl], np_i_a[f][max(_sl) + 1:]
                    s_h, s_s, s_t = np_i_a[s][:min(_ss)], np_i_a[s][_ss], np_i_a[s][max(_ss) + 1:]
                    ids[f], ids[s] = np.concatenate([l_h, s_s, l_t]).tolist(), np.concatenate([s_h, l_s, s_t]).tolist()
                    output[0].append([sf, ss]), output[1].append(ids)
                    cost, _ = seq_solver(info['grid'], info['agents'], ids, {'time_limit': 1, 'sub_op': 1.2})

                    if cost == 'error':
                        return 'NaN'

                    output[2].append(cost)

            if before_cost - min(output[2]) >= 0:
                i_a = output[1][np.argmin(output[2])]
                break

        cost = seq_solver(info['grid'], info['agents'], i_a, {'time_limit': 1, 'sub_op': 1.2})[0]
        before_cost = cost
        # print(before_cost)

    return (h_cost - cost) / h_cost * 100


def CE_relatedness(info):
    a = info['ce_assign']
    init_cost = info['init_cost']
    before_cost = init_cost
    select_other = False
    algo_optimality = [init_cost]

    wandb.init()
    for itr in range(100):
        wandb.log({'itr': itr})

        # swap agent index selection
        swap_a = copy.deepcopy(a)
        for i in range(len(swap_a)):
            swap_a[i].insert(0, list(info['agents'][i]))
        swap_1, swap_2 = route_relatedness(swap_a, select_other)[1]

        # substrings
        substring_length = 10000
        sub_1 = [l for l in permutations(range(len(a[swap_1])), 2)]
        sub_2 = [s for s in permutations(range(len(a[swap_1])), 2)]
        trunc_1 = [cand for cand in sub_1 if np.abs(cand[0] - cand[1]) <= substring_length]
        trunc_2 = [cand for cand in sub_2 if np.abs(cand[0] - cand[1]) <= substring_length]
        sub_1, sub_2 = trunc_1, trunc_2

        # cross exchange
        output = [[], []]
        for s1 in sub_1:
            _s1 = np.arange(s1[0], s1[1] + 1) if s1[0] < s1[1] else np.arange(s1[0], s1[1] - 1, -1)
            for s2 in sub_2:
                _s2 = np.arange(s2[0], s2[1] + 1) if s2[0] < s2[1] else np.arange(s2[0], s2[1] - 1, -1)

                if max(_s1) >= len(a[swap_1]) or max(_s2) >= len(a[swap_2]):
                    pass
                else:
                    l_h, l_s, l_t = np.array(a[swap_1])[:min(_s1)], np.array(a[swap_1])[_s1], np.array(a[swap_1])[
                                                                                              max(_s1) + 1:]
                    s_h, s_s, s_t = np.array(a[swap_2])[:min(_s2)], np.array(a[swap_2])[_s2], np.array(a[swap_2])[
                                                                                              max(_s2) + 1:]

                    test_a = copy.deepcopy(a)
                    test_a[swap_1] = np.concatenate([l_h, s_s, l_t]).tolist()
                    test_a[swap_2] = np.concatenate([s_h, l_s, s_t]).tolist()

                    cost, _ = seq_solver(info['grid'], info['agents'], test_a, {'time_limit': 1, 'sub_op': 1.2})
                    if cost == 'error':
                        pass
                    else:
                        output[0].append(cost), output[1].append(test_a)

        # if the swap operation reduced cost
        if before_cost > min(output[0]):
            before_cost = min(output[0])
            a = output[1][np.argmin(output[0])]
            select_other = False
            algo_optimality.append(before_cost)

        else:
            # print("{} <-> {} failed".format(swap_1, swap_2))
            select_other = True
            algo_optimality.append(before_cost)

    # return ((init_cost - before_cost) / init_cost * 100).item()
    return algo_optimality
