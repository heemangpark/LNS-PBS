import subprocess
import time

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.env import save_env, load_env
from utils.soc_ms import cost
from utils.solver_util import save_map, save_scenario
from utils.vis_graph import vis_init_assign, vis_assign

"""
m: map size / o: obstacle ratio / tl: task length / na: the number of agents / nt: the number of tasks
lnt: the number of LNS iterations
t: time limit of EECBS / so: suboptimality of EECBS
"""

m = 32
o = 20
tl = 1
na = 10
nt = 10
lnt = 10
t = 60
so = 1.2

save_env(m, o, tl, na, nt)
environment = load_env('{}{}{}_{}_{}_{}/environment_4.pkl'.format(m, m, o, tl, na, nt))
grid, graph, agent_pos, total_tasks = environment[0], environment[1], environment[2], environment[3]
vis_init_assign(graph, agent_pos, total_tasks)


def LNS():
    # 1st step: Hungarian Assignment
    h_time = time.time()
    task_idx, tasks = hungarian(graph, agent_pos, total_tasks)
    print('INIT || SOC: {:.4f} / MAKESPAN: {:.4f} / TIMECOST: {:.4f}'
          .format(cost(tasks, graph)[0], cost(tasks, graph)[1], time.time() - h_time))
    vis_assign(graph, agent_pos, tasks, 'hungarian')

    # 2nd step: Large Neighborhood Search (iteratively)
    for itr in range(lnt):
        lns_time = time.time()

        # Destroy
        removal_idx = removal(task_idx, total_tasks, graph)
        for i, t in enumerate(tasks.values()):
            for r in removal_idx:
                if {r: total_tasks[r]} in t:
                    tasks[i].remove({r: total_tasks[r]})

        # Reconstruct
        while len(removal_idx) != 0:
            f = f_ijk(removal_idx, tasks, total_tasks, graph)
            regret = get_regret(f)
            regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
            re_ins = list(regret.keys())[0]
            re_a, re_j = regret[re_ins][1], regret[re_ins][2]
            removal_idx.remove(re_ins)
            to_insert = {re_ins: total_tasks[re_ins]}
            tasks[re_a].insert(re_j, to_insert)

        print('{}_Solution || SOC: {:.4f} / MAKESPAN: {:.4f} / TIMECOST: {:.4f}'
              .format(itr + 1, cost(tasks, graph)[0], cost(tasks, graph)[1], time.time() - lns_time))
        vis_assign(graph, agent_pos, tasks, itr + 1)


def EECBS():
    # processing environment into input of EECBS
    scenario_name = 'test'
    save_map(grid, scenario_name)
    save_scenario(agent_pos, total_tasks, scenario_name, grid.shape[0], grid.shape[1])

    solver_path = 'EECBS/'
    c = [solver_path + 'eecbs',
         "-m", solver_path + scenario_name + '.map',
         "-a", solver_path + scenario_name + '.scen',
         "-o", solver_path + scenario_name + '.csv',
         "--outputPaths = EECBS/paths.txt",
         "-k", "{}".format(na),
         "-t", "{}".format(t),
         "--suboptimality = {}".format(so)]

    subprocess.run(c)


if __name__ == "__main__":
    # LNS()
    EECBS()
