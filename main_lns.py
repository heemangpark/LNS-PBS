import time

from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from seq_solver import seq_solver
from utils.convert import to_solver


# "Create random scenarios and load one of them"
# # save_scenarios(C=1, M=20, N=50)
# scenario = load_scenarios('101020_1_5_15_0/scenario_3.pkl')
# grid, graph, agents, tasks = scenario[0], scenario[1], scenario[2], scenario[3]
#
# "1st step: Hungarian Assignment"
# task_idx, assign = HA(graph, agents, tasks)
# cost, paths = seq_solver(grid, agents, to_solver(tasks, assign), {'time_limit': 1, 'sub_op': 1.1})
# print('HA || SOC: {:.4f}'.format(cost))
#
# "2nd step: Large Neighborhood Search (iteratively)"
# max_t = time.time() + 10
# itr = 0
#
# while True:
#     lns_time = time.time()
#
#     # Destroy
#     removal_idx = removal(task_idx, tasks, graph, N=2)
#     for i, t in enumerate(assign.values()):
#         for r in removal_idx:
#             if {r: tasks[r]} in t:
#                 assign[i].remove({r: tasks[r]})
#
#     # Repair
#     while len(removal_idx) != 0:
#         f = f_ijk(assign, agents, removal_idx, tasks, graph)
#         regret = get_regret(f)
#         regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
#         re_ins = list(regret.keys())[0]
#         re_a, re_j = regret[re_ins][1], regret[re_ins][2]
#         removal_idx.remove(re_ins)
#         to_insert = {re_ins: tasks[re_ins]}
#         assign[re_a].insert(re_j, to_insert)
#
#     lns_time = time.time() - lns_time
#     if time.time() > max_t:
#         break
#
#     itr += 1
#     cost, paths = seq_solver(grid, agents, to_solver(tasks, assign), {'time_limit': 1, 'sub_op': 1.2})
#     print('{}_Solution || SOC: {:.4f} / TIMECOST: {:.4f}'.format(itr, cost, lns_time))


def LNS(info):
    task_idx, assign = info['assign'][0], info['assign'][1]
    h_cost, paths = seq_solver(info['grid'], info['agents'], to_solver(info['tasks'], assign),
                               {'time_limit': 1, 'sub_op': 1.1})
    if h_cost == 'error':
        return 'NaN'
    # max_t = time.time() + 5
    itr, max_itr = 0, 10
    # while True:
    while itr < max_itr:
        # lns_time = time.time()
        removal_idx = removal(task_idx, info['tasks'], info['graph'], N=2)
        for i, t in enumerate(assign.values()):
            for r in removal_idx:
                if {r: info['tasks'][r]} in t:
                    assign[i].remove({r: info['tasks'][r]})
        while len(removal_idx) != 0:
            f = f_ijk(assign, info['agents'], removal_idx, info['tasks'], info['graph'])
            regret = get_regret(f)
            regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
            re_ins = list(regret.keys())[0]
            re_a, re_j = regret[re_ins][1], regret[re_ins][2]
            removal_idx.remove(re_ins)
            to_insert = {re_ins: info['tasks'][re_ins]}
            assign[re_a].insert(re_j, to_insert)
        # lns_time = time.time() - lns_time
        # if time.time() > max_t:
        #     break
        itr += 1
        cost, paths = seq_solver(info['grid'], info['agents'], to_solver(info['tasks'], assign),
                                 {'time_limit': 1, 'sub_op': 1.1})
        if cost == 'error':
            return 'NaN'

    return (h_cost - cost) / h_cost * 100
