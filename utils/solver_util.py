import os
from pathlib import Path

import numpy as np

curr_path = os.path.realpath(__file__)
save_dir = os.path.join(Path(curr_path).parent.parent, 'EECBS/exp/')


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
