import random

import matplotlib.pyplot as plt

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.generate_scenarios import load_scenarios
from utils.soc_ms import costa


def test_lns(lns_step, exp_repeat, N):
    random.seed(3298)
    scenario = load_scenarios('323220_1_10_10/scenario_1.pkl')
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]

    for oi in range(exp_repeat):
        soc_list, ms_list = list(), list()
        task_idx, tasks = hungarian(graph, agent_pos, total_tasks)

        for ii in range(lns_step):
            removal_idx = removal(task_idx, total_tasks, graph, N=N)
            for i, t in enumerate(tasks.values()):
                for r in removal_idx:
                    if {r: total_tasks[r]} in t:
                        tasks[i].remove({r: total_tasks[r]})

            while len(removal_idx) != 0:
                f = f_ijk(tasks, agent_pos, removal_idx, total_tasks, graph)
                regret = get_regret(f)
                regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
                re_ins = list(regret.keys())[0]
                re_a, re_j = regret[re_ins][1], regret[re_ins][2]
                removal_idx.remove(re_ins)
                to_insert = {re_ins: total_tasks[re_ins]}
                tasks[re_a].insert(re_j, to_insert)

            soc, ms = cost(agent_pos, tasks, graph)
            soc_list.append(soc)
            ms_list.append(ms)

        plt.plot(list(range(lns_step)), soc_list)
    plt.savefig('fig/N{}.png'.format(N))
    plt.clf()


if __name__ == "__main__":
    for nn in range(10):
        print(nn)
        test_lns(in_itr=20, out_itr=10, N=nn)
