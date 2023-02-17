import subprocess
import numpy as np


def save_map(grid, filename):
    f = open('EECBS/{}.map'.format(filename), 'w')
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

    f = open('DecAstar/{}.map'.format(filename), 'w')
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
    f = open('EECBS/{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    for a, t in zip(agent_pos, total_tasks):
        task = t[0]  # TODO:add task seq
        dist = abs(np.array(a) - np.array(t)).sum()  # Manhatten dist
        line = '1 \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}'.format('{}.map'.format(scenario_name), row, column, a[1],
                                                                  a[0], task[1], task[0], dist)
        f.write(line + "\n")
    f.close()


def save_scenario_dec(agent_pos, total_tasks, scenario_name, row=32, column=32):
    f = open('DecAstar/{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    for a, t in zip(agent_pos, total_tasks):
        task = t[0]  # TODO:add task seq
        dist = abs(np.array(a) - np.array(t)).sum()  # Manhatten dist
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


def compute_astar(agent_pos, total_tasks, exp_name, task_finished_bef):
    dec_solver_path = "DecAstar/"
    n_ag = len(agent_pos)
    n_task = len(total_tasks)

    dists = []
    for i in range(n_task):
        if task_finished_bef[i]:
            ts = [0] * n_ag
        else:
            save_scenario_dec(agent_pos, [total_tasks[i] for _ in range(n_ag)], exp_name)
            dec_c = [dec_solver_path + "eecbs",
                     "-m",
                     dec_solver_path + exp_name + '.map',
                     "-a",
                     dec_solver_path + exp_name + '.scen',
                     "-o",
                     dec_solver_path + exp_name + ".csv",
                     "--outputPaths",
                     dec_solver_path + exp_name + "_paths.txt",
                     "-k", str(len(agent_pos)), "-t", "0.1", "--suboptimality=10"]
            subprocess.run(dec_c, capture_output=True)
            agent_traj = read_trajectory(dec_solver_path + exp_name + "_paths.txt")
            ts = [len(t) for t in agent_traj]
        dists.append(ts)

    return np.array(dists)

    # process_out_dec = subprocess.run(dec_c, capture_output=True)
    # text_byte_dec = process_out_dec.stdout.decode('utf-8')
    # return text_byte_dec
