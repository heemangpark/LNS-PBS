import copy
import os
import pickle
import shutil
import sys
from pathlib import Path

import dgl

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from utils.graph_utils import sch_to_nx
from utils.scenario_utils import load_scenarios
from utils.solver_utils import to_solver, solver


def LNS(info, exp_num, itrs, N, neigh='relative', dirs=None):
    task_idx, assign = info['assign']
    pre_cost = info['init_cost']
    time_log = None

    for itr in range(itrs):
        temp_assign = copy.copy(assign)
        removal_idx = removal(task_idx, info['tasks'], info['graph'], N=N, time_log=time_log, neighbors=neigh)
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

        cost, _, time_log = solver(info['grid'], info['agents'], to_solver(info['tasks'], temp_assign),
                                   ret_log=True, dir=dirs)
        if cost == 'error':
            pass

        else:
            if cost < pre_cost:
                pre_cost = cost
                assign = temp_assign

    keys, values = [], [[] for _ in range(len(info['agents']))]
    for ag_idx, tasks in enumerate(assign.values()):
        keys.append(ag_idx)
        for task in tasks:
            values[ag_idx].append(list(task.keys())[0])
    assign_id = dict(zip(keys, values))
    with open(dirs[2] + 'final_{}.pkl'.format(exp_num), 'wb') as a:
        pickle.dump(assign_id, a)


def collect(exp_num):
    curr_dir = os.path.realpath(__file__)
    solver_dir = os.path.join(str(Path(curr_dir).parent.parent), 'EECBS/eecbs')
    save_dir = os.path.join(str(Path(curr_dir).parent.parent), 'EECBS/exp_{}/'.format(exp_num))
    data_save_dir = os.path.join(str(Path(curr_dir).parent.parent), 'data/')
    exp_name = 'data'
    dirs = [solver_dir, save_dir, data_save_dir, exp_name]

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print("Error: Cannot create the directory.")

    scenario = load_scenarios('323220_5_50/scenario_{}.pkl'.format(exp_num))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
    assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
    routes = to_solver(info['tasks'], assign)

    temp_routes = copy.deepcopy(routes)
    schedules = info['agents'].tolist()
    for tr in temp_routes:
        schedules += tr
    n_agents, n_tasks = len(info['agents']), len(info['tasks'])
    sch_nx = sch_to_nx(schedules, info['graph'], n_agents, n_tasks)
    sch_space = dgl.from_networkx(sch_nx, node_attrs=['coord', 'type'], edge_attrs=['a_dist', 'dist', 'obs_proxy'])
    dgl.save_graphs(data_save_dir + 'sch_space_{}.pkl'.format(exp_num), sch_space)

    info['assign'] = (assign_id, assign)
    info['init_cost'], info['init_routes'] = solver(info['grid'], info['agents'], routes, dir=dirs)

    with open(data_save_dir + 'prev_{}.pkl'.format(exp_num), 'wb') as p:
        pickle.dump(assign_id, p)

    if info['init_cost'] == 'error':
        os.remove(data_save_dir + 'prev_{}.pkl'.format(exp_num))
        os.remove(data_save_dir + 'sch_space_{}.pkl'.format(exp_num))
    else:
        LNS(info, exp_num, itrs=100, N=2, dirs=dirs)

    try:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
    except OSError:
        print("Error: Cannot remove the directory.")


if __name__ == '__main__':
    # datasets = [Process(target=collect, args=[exp_num]) for exp_num in range(10000)]
    # for data in datasets:
    #     data.start()
    collect(1)
