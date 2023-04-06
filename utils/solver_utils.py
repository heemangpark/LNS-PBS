import copy
import subprocess

import numpy as np


def save_map(grid, filename, save_dir):
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


def save_scenario(agent_pos, total_tasks, scenario_name, row, column, save_dir):
    f = open(save_dir + '{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    for a, t in zip(agent_pos, total_tasks):
        task = t[0]  # TODO:add task seq
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
        splitted_string = string.split('->')
        for itr, s in enumerate(splitted_string):
            if itr == len(splitted_string) - 1:
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


def solver(instance, agents, tasks, ret_log=False, dirs=None):
    solver_dir, save_dir, exp_name = dirs[0], dirs[1], dirs[3]

    time_log = dict()
    s_agents = copy.deepcopy(agents)
    todo = copy.deepcopy(tasks)
    seq_paths = [[list(agents[a])] for a in range(len(agents))]
    total_cost, itr, T = 0, 0, 0

    while sum([len(t) for t in todo]) != 0:
        itr += 1
        s_tasks = list()
        for a, t in zip(s_agents, todo):
            if len(t) == 0:
                s_tasks.append([list(a)])
            else:
                s_tasks.append([t[0]])
        save_map(instance, exp_name, save_dir)
        save_scenario(s_agents, s_tasks, exp_name, instance.shape[0], instance.shape[1], save_dir)

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
             "-t", "{}".format(1),
             "--suboptimality={}".format(1.2)]
        process_out = subprocess.run(c, capture_output=True)
        text_byte = process_out.stdout.decode('utf-8')
        if (text_byte[37:44] != 'Succeed') & ret_log:
            return 'error', 'error', 'error'
        elif text_byte[37:44] != 'Succeed':
            return 'error', 'error'

        traj = read_trajectory(save_dir + exp_name + "_paths_{}.txt".format(itr))
        len_traj = [len(t) - 1 for t in traj]
        d_len_traj = [l for l in len_traj if l not in {0}]
        next_t = np.min(d_len_traj)
        T += next_t

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
                    time_log[tuple(ag_to)] = T
                    s_agents[a_id] = ag_to
            else:
                if len_traj[a_id] == 0:
                    pass
                else:
                    s_agents[a_id] = traj[a_id][next_t]

            seq_paths[a_id] += traj[a_id][1:next_t + 1]

        total_cost += next_t * len(d_len_traj)

    if ret_log:
        return total_cost, seq_paths, time_log
    else:
        return total_cost, seq_paths
