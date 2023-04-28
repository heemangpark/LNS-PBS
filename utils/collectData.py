DEBUG = False

import copy
import os
import pickle
import shutil
import sys
from multiprocessing import Process
from pathlib import Path

from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from utils.graph import convert_to_nx
from utils.scenario import load_scenarios
from utils.solver import to_solver, solver, assignment_to_id


def LNS(info, exp_num, dirs=None):
    assign_id, assign = info['lns']
    pre_cost = info['init_cost']
    graph = info['init_graph']

    # trajectory data
    assign_id_list = []
    decrement_list = []
    graph_list = []
    removal_list = []

    time_log = None

    for itr in range(100):
        temp_assign = copy.deepcopy(assign)
        removal_idx = removal(assign_id, info['tasks'], info['graph'], N=2, time_log=time_log, neighbors='relative')
        r_idx = copy.deepcopy(removal_idx)
        removal_list.append(r_idx)
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
            return 'error'

        elif cost < pre_cost:  # nonzero decrement
            decrement_list.append(pre_cost - cost)

            pre_cost = cost
            assign = temp_assign

            assign_id = assignment_to_id(len(info['agents']), assign)
            assign_id_list.append(assign_id)

            coordination = [[a] for a in info['agents'].tolist()]
            for i, coords in enumerate(assign.values()):
                temp_schedule = [list(c.values())[0][0] for c in coords]
                coordination[i].extend(temp_schedule)

            graph = convert_to_nx(assign_id, coordination, info['graph'])
            graph_list.append(graph)

        else:
            removal_list.pop(-1)

    with open(dirs[2] + 'dataset_{}.pkl'.format(exp_num), 'wb') as f:
        pickle.dump([assign_id_list, decrement_list, graph_list, removal_list], f)


def collect(exp_num):
    curr_dir = os.path.realpath(__file__)
    solver_dir = os.path.join(str(Path(curr_dir).parent.parent), 'EECBS/eecbs')
    save_dir = os.path.join(str(Path(curr_dir).parent.parent), 'EECBS/dataset_{}/'.format(exp_num))
    data_save_dir = os.path.join(str(Path(curr_dir).parent.parent), 'dataset/')
    exp_name = 'dataset'
    dirs = [solver_dir, save_dir, data_save_dir, exp_name]

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print("Error: Cannot create the directory.")

    scenario = load_scenarios('323220_5_50/scenario_{}.pkl'.format(exp_num))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
    n_agents, n_tasks = len(info['agents']), len(info['tasks'])

    if DEBUG:
        assign_id = dict()
        assign = dict()
        for i in range(n_agents):
            assign_id[i] = list(range(i * 10, (i + 1) * 10))
            assign[i] = [{j: info['tasks'][j]} for j in assign_id[i]]

    else:
        assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])

    routes = to_solver(info['tasks'], assign)

    info['lns'] = assign_id, assign
    info['init_cost'], info['init_routes'] = solver(info['grid'], info['agents'], routes, dir=dirs)
    if info['init_cost'] == 'error':
        return 'abandon_seed'

    coordination = [[a] for a in info['agents'].tolist()]
    for i, coords in enumerate(assign.values()):
        temp_schedule = [list(c.values())[0][0] for c in coords]
        coordination[i].extend(temp_schedule)

    info['init_graph'] = convert_to_nx(coordination, info['graph'])

    LNS(info, exp_num, dirs=dirs)

    try:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
    except OSError:
        print("Error: Cannot remove the directory.")


if __name__ == '__main__':
    for collect_batch in trange(100):
        datasets = [Process(target=collect, args=[exp_num])
                    for exp_num in range(collect_batch * 100, (collect_batch + 1) * 100)]
        for data in datasets:
            data.start()
        while sum([not d.is_alive() for d in datasets]) != 100:
            pass
