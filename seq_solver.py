import copy
import os
import subprocess
from pathlib import Path

import numpy as np

from utils.solver_util import save_map, save_scenario, read_trajectory

curr_dir = os.path.realpath(__file__)
solver_dir = os.path.join(Path(curr_dir).parent, 'EECBS/eecbs')
file_dir = os.path.join(Path(curr_dir).parent, 'EECBS/exp/')
exp_name = 'opt'


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
             file_dir + exp_name + '.map',
             "-a",
             file_dir + exp_name + '.scen',
             "-o",
             file_dir + exp_name + ".csv",
             "--outputPaths",
             file_dir + exp_name + "_paths_{}.txt".format(itr),
             "-k", "{}".format(len(s_agents)),
             "-t", "{}".format(solver_params['time_limit']),
             "--suboptimality={}".format(solver_params['sub_op'])]
        process_out = subprocess.run(c, capture_output=True)
        text_byte = process_out.stdout.decode('utf-8')
        if text_byte[37:44] != 'Succeed':
            return 'error', 'error'

        traj = read_trajectory(file_dir + exp_name + "_paths_{}.txt".format(itr))
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

# if __name__ == '__main__':
#     from utils.generate_scenarios import load_scenarios
#
#     g = load_scenarios('323220_1_10_20_1/scenario_1.pkl')[0]
#     a = [[16, 10],
#          [12, 0],
#          [14, 28],
#          [13, 10],
#          [28, 22]]
#     t = [[[18, 10], [24, 27]],
#          [[14, 0], [24, 8]],
#          [[12, 25], [24, 31], [27, 31], [31, 30]],
#          [],
#          [[30, 17]]]
#     print(seq_solver(g, a, t, {'time_limit': 60, 'sub_op': 1.2}))
