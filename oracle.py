import pickle

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.generate_scenarios import save_scenarios, load_scenarios

"""
323220 / C=1 / M=10 / N=10 setting
"""

save_scenarios(itr=1000)
for idx in list(range(1, 1001)):
    scenario = load_scenarios('323220_1_10_10/scenario_{}.pkl'.format(idx))
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
    task_idx, tasks = hungarian(graph, agent_pos, total_tasks)

    for _ in range(500):  # 10 10 setting 500 itrs might be enough / not sure :(
        removal_idx = removal(task_idx, total_tasks, graph)
        for i, t in enumerate(tasks.values()):
            for r in removal_idx:
                if {r: total_tasks[r]} in t:
                    tasks[i].remove({r: total_tasks[r]})
        while len(removal_idx) != 0:
            "time consuming"
            f = f_ijk(tasks, agent_pos, removal_idx, total_tasks, graph)
            regret = get_regret(f)
            regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
            re_ins = list(regret.keys())[0]
            re_a, re_j = regret[re_ins][1], regret[re_ins][2]
            removal_idx.remove(re_ins)
            to_insert = {re_ins: total_tasks[re_ins]}
            tasks[re_a].insert(re_j, to_insert)

    print(idx)
    with open('data/1010grid_{}'.format(idx), 'wb') as f:  # grid
        pickle.dump(grid, f)

    with open('data/1010graph_{}'.format(idx), 'wb') as f:  # graph
        pickle.dump(graph, f)

    with open('data/1010AP_{}'.format(idx), 'wb') as f:  # agent position
        pickle.dump(agent_pos, f)

    with open('data/1010TA_{}'.format(idx), 'wb') as f:  # tasks assignments
        pickle.dump(tasks, f)
