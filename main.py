import time

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.generate_scenarios import load_scenarios
from utils.soc_ms import cost
from utils.vis_graph import vis_dist, vis_ta

"""
Create random scenarios and load one of them
"""
# save_scenarios(C=1, M=20, N=50)
scenario = load_scenarios('323220_1_20_50/scenario_1.pkl')
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
vis_dist(graph, agent_pos, total_tasks)

"""
1st step: Hungarian Assignment
"""
h_time = time.time()
task_idx, tasks = hungarian(graph, agent_pos, total_tasks)
h_time = time.time() - h_time
soc, ms = cost(tasks, graph)
print('INIT || SOC: {:.4f} / MAKESPAN: {:.4f} / TIMECOST: {:.4f}'.format(soc, ms, h_time))
vis_ta(graph, agent_pos, tasks, 'HA', soc)

"""
2nd step: Large Neighborhood Search (iteratively)
"""
for itr in range(5):
    lns_time = time.time()

    # Destroy
    removal_idx = removal(task_idx, total_tasks, graph)
    for i, t in enumerate(tasks.values()):
        for r in removal_idx:
            if {r: total_tasks[r]} in t:
                tasks[i].remove({r: total_tasks[r]})

    # Reconstruct
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

    lns_time = time.time() - lns_time
    soc, ms = cost(tasks, graph)
    print('{}_Solution || SOC: {:.4f} / MAKESPAN: {:.4f} / TIMECOST: {:.4f}'.format(itr + 1, soc, ms, lns_time))
    vis_ta(graph, agent_pos, tasks, itr + 1, soc)
