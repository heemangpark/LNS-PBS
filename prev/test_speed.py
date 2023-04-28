import copy
import os
import shutil
import sys
import time
from multiprocessing import Process
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from utils.scenario import load_scenarios
from utils.solver import to_solver, solver


def LNS(info, dir):
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None

    for itr in range(100):
        s = time.time()
        temp_assign = copy.deepcopy(assign)
        removal_idx = removal(
            task_idx,
            info['tasks'],
            info['graph'],
            N=2,
            time_log=time_log,
            neighbors='related'
        )
        if removal_idx == 'stop':
            return 'stop'

        for i, t in enumerate(temp_assign.values()):
            for r in removal_idx:
                if {r: info['tasks'][r]} in t:
                    temp_assign[i].remove({r: info['tasks'][r]})

        while len(removal_idx) != 0:
            f_val = f_ijk(temp_assign, info['agents'], removal_idx, info['tasks'], info['graph'])
            regret = get_regret(f_val)
            regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
            re_ins = list(regret.keys())[0]
            re_a, re_j = regret[re_ins][1], regret[re_ins][2]
            removal_idx.remove(re_ins)
            to_insert = {re_ins: info['tasks'][re_ins]}
            temp_assign[re_a].insert(re_j, to_insert)

        cost, _, time_log = solver(
            info['grid'],
            info['agents'],
            to_solver(info['tasks'], temp_assign),
            ret_log=True,
            dir=dir
        )

        if cost == 'error':
            pass
        else:
            if cost < pre_cost:
                pre_cost = cost
                assign = temp_assign
                results.append(pre_cost)
            elif cost >= pre_cost:
                results.append(pre_cost)
        e = time.time()
        print(e - s)

    return results


def run(dir):
    try:
        for d in dir:
            if not os.path.exists(d[1]):
                os.makedirs(d[1])
    except OSError:
        print("Error: Cannot create the directory.")

    scenario = load_scenarios('646420_5_50_eval/scenario_{}.pkl'.format(dir[0][3]))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}

    assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
    routes = to_solver(info['tasks'], assign)

    info['lns'] = assign_id, assign
    info['init_cost'], info['init_routes'] = solver(info['grid'], info['agents'], routes, dir=dir[1])
    if info['init_cost'] == 'error':
        return 'abandon_seed'

    s = time.time()
    LNS(info, dir[0])
    e = time.time()
    print(e - s)

    try:
        for d in dir:
            if os.path.exists(d[1]):
                shutil.rmtree(d[1])
    except OSError:
        print("Error: Cannot remove the directory.")


if __name__ == '__main__':
    dirs_list = []
    for exp_num in range(1):
        solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/eecbs')
        save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/testSpeed_{}/'.format(exp_num)), \
                   os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/init_{}/'.format(exp_num))
        test_dir = [solver_dir, save_dir[0], 'testSpeed', exp_num]
        init_dir = [solver_dir, save_dir[1], 'init', exp_num]
        dirs = [test_dir, init_dir]
        dirs_list.append(dirs)

    run_list = [Process(target=run, args=[dirs]) for dirs in dirs_list]
    for r in run_list:
        r.start()

    # from matplotlib import pyplot as plt
    # import numpy as np
    #
    # data_dir = os.path.join(Path(os.path.realpath(__file__)).parent, 'testSpeed')
    # N_gap, heu_gap, rand_gap = 0, 0, 0
    # for pn in range(10):
    #     with open(data_dir + '{}.pkl'.format(pn), 'rb') as f:
    #         man, astar = pickle.load(f)
    #     plt.plot(np.arange(len(man)), man, label='manhattan')
    #     plt.plot(np.arange(len(astar)), astar, label='astar')
    #     plt.xlabel('iteration'), plt.ylabel('route length')
    #     plt.legend(loc='upper right')
    #     plt.savefig('fig_{}.png'.format(pn))
    #     plt.clf()
