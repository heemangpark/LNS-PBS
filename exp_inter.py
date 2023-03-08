import copy
import os
import shutil
import subprocess
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import wandb

from LNS.hungarian import hungarian
from utils.generate_scenarios import load_scenarios


def seq_solver(instance, agents, tasks, solver_params):
    s_agents = copy.deepcopy(agents)
    todo = copy.deepcopy(tasks)
    seq_paths = [[list(agents[a])] for a in range(len(agents))]
    total_cost, itr = 0, 0

    while sum([len(t) for t in todo]) != 0:
        itr += 1
        s_tasks = list()
        for a, t in zip(s_agents, todo):
            if len(t) == 0:
                s_tasks.append([list(a)])
            else:
                s_tasks.append([t[0]])
        save_map(instance, exp_name)
        save_scenario(s_agents, s_tasks, exp_name, instance.shape[0], instance.shape[1])

        c = [solver_dir,
             "-m",
             save_dir + exp_name + '.map',
             "-a",
             save_dir + exp_name + '.scen',
             "-o",
             save_dir + exp_name + ".csv",
             "--outputPaths",
             save_dir + exp_name + "_paths_{}.txt".format(itr),
             "-k", "{}".format(len(s_agents)),
             "-t", "{}".format(solver_params['time_limit']),
             "--suboptimality={}".format(solver_params['sub_op'])]
        process_out = subprocess.run(c, capture_output=True)
        text_byte = process_out.stdout.decode('utf-8')
        if text_byte[37:44] != 'Succeed':
            return 'error', 'error'

        traj = read_trajectory(save_dir + exp_name + "_paths_{}.txt".format(itr))
        len_traj = [len(t) - 1 for t in traj]
        d_len_traj = [l for l in len_traj if l not in {0}]
        next_t = np.min(d_len_traj)

        fin_id = list()
        for e, t in enumerate(traj):
            if len(t) == 1:
                fin_id.append(False)
            else:
                fin_id.append(t[next_t] == s_tasks[e][0])
        fin_ag = np.array(range(len(s_agents)))[fin_id]

        for a_id in range(len(s_agents)):
            if a_id in fin_ag:
                if len(todo[a_id]) == 0:
                    pass
                else:
                    ag_to = todo[a_id].pop(0)
                    s_agents[a_id] = ag_to
            else:
                if len_traj[a_id] == 0:
                    pass
                else:
                    s_agents[a_id] = traj[a_id][next_t]

            seq_paths[a_id] += traj[a_id][1:next_t + 1]

        total_cost += next_t * len(d_len_traj)

    return total_cost, seq_paths


def save_map(grid, filename):
    f = open(save_dir + '{}.map'.format(filename), 'w')
    f.write('type four-directional\n')
    f.write('height {}\n'.format(grid.shape[0]))
    f.write('width {}\n'.format(grid.shape[1]))
    f.write('map\n')

    # creating map from grid
    map_dict = {0: '.', 1: '@'}
    for r in range(grid.shape[0]):
        line = grid[r]
        l = []
        for g in line:
            l.append(map_dict[g])
        f.write(''.join(l) + '\n')

    f.close()


def save_scenario(agent_pos, total_tasks, scenario_name, row, column):
    f = open(save_dir + '{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    for a, t in zip(agent_pos, total_tasks):
        task = t[0]
        dist = abs(np.array(a) - np.array(t)).sum()  # Manhattan dist
        line = '1 \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}'.format('{}.map'.format(scenario_name), row, column, a[1],
                                                                  a[0], task[1], task[0], dist)
        f.write(line + "\n")
    f.close()


def read_trajectory(path_file_dir):
    f = open(path_file_dir, 'r')
    lines = f.readlines()
    agent_traj = []

    for i, string in enumerate(lines):
        curr_agent_traj = []
        split_string = string.split('->')
        for itr, s in enumerate(split_string):
            if itr == len(split_string) - 1:
                continue
            if itr == 0:
                tup = s.split(' ')[-1]
            else:
                tup = s

            ag_loc = [int(i) for i in tup[1:-1].split(',')]
            curr_agent_traj.append(ag_loc)
        agent_traj.append(curr_agent_traj)

    f.close()

    return agent_traj


def to_solver(task_in_seq, assignment):
    s_in_tasks = [[] for _ in range(len(assignment))]
    for a, t in assignment.items():
        if len(t) == 0:
            pass
        else:
            __t = list()
            for _t in t:
                __t += task_in_seq[list(_t.keys())[0]]
            s_in_tasks[a] = __t
    return s_in_tasks


def route_relatedness(routes, explore):
    div = dict()
    R = 10000
    for i, j in combinations(range(len(routes)), 2):  # route must include agent & task sequence's total coords
        dist = 0
        for a in routes[i]:
            for b in routes[j]:
                dist += (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]))
        div[(i, j)] = dist / (len(routes[i]) * len(routes[j]))
    div = sorted(div.items(), key=lambda x: x[1])
    rd = np.random.choice([r for r in range(1, len(div))][:R])

    return (div, div[0][0]) if not explore else (div, div[rd][0])


def CE(routes, grid, agents):
    a = routes
    results = dict()

    for swap_1, swap_2 in combinations(range(5), 2):
        sub_1 = [l for l in permutations(range(len(a[swap_1])), 2)]
        sub_2 = [s for s in permutations(range(len(a[swap_1])), 2)]

        # cross exchange
        costs = []
        for s1 in sub_1:
            _s1 = np.arange(s1[0], s1[1] + 1) if s1[0] < s1[1] else np.arange(s1[0], s1[1] - 1, -1)
            for s2 in sub_2:
                _s2 = np.arange(s2[0], s2[1] + 1) if s2[0] < s2[1] else np.arange(s2[0], s2[1] - 1, -1)

                if max(_s1) >= len(a[swap_1]) or max(_s2) >= len(a[swap_2]):
                    pass

                else:
                    l_h, l_s, l_t = np.array(a[swap_1])[:min(_s1)], np.array(a[swap_1])[_s1], \
                                    np.array(a[swap_1])[max(_s1) + 1:]
                    s_h, s_s, s_t = np.array(a[swap_2])[:min(_s2)], np.array(a[swap_2])[_s2], \
                                    np.array(a[swap_2])[max(_s2) + 1:]

                    test_a = copy.deepcopy(a)
                    test_a[swap_1] = np.concatenate([l_h, s_s, l_t]).tolist()
                    test_a[swap_2] = np.concatenate([s_h, l_s, s_t]).tolist()

                    cost, _ = seq_solver(grid, agents, test_a, {'time_limit': 1, 'sub_op': 1.2})
                    if cost == 'error':
                        pass
                    else:
                        costs.append(cost)

        results[swap_1, swap_2] = min(costs)

    results = sorted(results.items(), key=lambda x: x[1])
    id_1, id_2 = results[0][0]
    min_cost = results[0][1]

    return id_1, id_2, min_cost, dict(results)


if __name__ == '__main__':
    wandb.init()
    curr_dir = os.path.realpath(__file__)
    solver_dir = os.path.join(Path(curr_dir).parent, 'EECBS/eecbs')
    save_dir = os.path.join(Path(curr_dir).parent, 'EECBS/exp_inter/')
    exp_name = 'inter'

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print("Error: Cannot create the directory.")

    for itr in range(1, 1001):
        scenario = load_scenarios('202020_5_15/scenario_{}.pkl'.format(itr))
        grid, graph, agents, tasks = scenario[0], scenario[1], scenario[2], scenario[3]
        assign_id, assign = hungarian(graph, agents, tasks)
        routes = to_solver(tasks, assign)

        tau_1, tau_2 = route_relatedness(routes, explore=False)[1]

        g_1, g_2, g_cost, ground = CE(routes, grid, agents)

        wandb.log({'gap': ground[tau_1, tau_2] - g_cost})

    try:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
    except OSError:
        print("Error: Cannot remove the directory.")
