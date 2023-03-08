import time

from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from exp_optimality import seq_solver
from utils.solver_util import to_solver


def LNS(info):
    task_idx, assign = info['lns_assign'][0], info['lns_assign'][1]
    init_cost = info['init_cost']
    algo_optimality = [init_cost]

    for itr in range(100):
        print('lns itr: {} || {}'.format(itr, time.strftime('%H:%M:%S')))

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

        cost, _ = seq_solver(info['grid'], info['agents'], to_solver(info['tasks'], assign),
                             {'time_limit': 1, 'sub_op': 1.2})
        if cost == 'error':
            pass
        else:
            algo_optimality.append(cost)

    # return (init_cost - cost) / h_cost * 100
    return algo_optimality
