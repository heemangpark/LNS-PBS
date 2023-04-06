import copy
import os
import pickle
import shutil
import sys
from multiprocessing import Process
from pathlib import Path

import dgl
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from utils.graph_utils import sch_to_nx
from nn.agent import SL_LNS
from utils.scenario_utils import load_scenarios
from utils.solver_utils import to_solver, solver
from utils.plot_utils import comparing_plot

model = SL_LNS(train=False)
model.load_state_dict(torch.load('SL_LNS_100.pt'))

rand_model = SL_LNS(train=False)


def SL_LNS(info, graph, dirs):
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]

    for itr in range(100):
        temp_assign = copy.deepcopy(assign)
        removal_idx = model.eval(graph)
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

        cost, _ = solver(info['grid'], info['agents'], to_solver(info['tasks'], temp_assign), dirs=dirs)
        if cost == 'error':
            pass

        else:
            if cost < pre_cost:
                pre_cost = cost
                assign = temp_assign
                routes = to_solver(info['tasks'], assign)
                schedules = info['agents'].tolist()
                for tr in routes:
                    schedules += tr
                sch_nx = sch_to_nx(schedules, info['graph'], len(info['agents']), len(info['tasks']))
                graph = dgl.from_networkx(sch_nx,
                                          node_attrs=['coord', 'type'],
                                          edge_attrs=['a_dist', 'dist', 'obs_proxy']
                                          )
                results.append(pre_cost)
            elif cost >= pre_cost:
                results.append(pre_cost)

    return results


def LNS(info, dirs):
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None

    for itr in range(100):
        temp_assign = copy.copy(assign)
        removal_idx = removal(task_idx, info['tasks'], info['graph'], N=2, time_log=time_log, neigh='relative')
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
                                   ret_log=True, dirs=dirs)
        if cost == 'error':
            pass

        else:
            if cost < pre_cost:
                pre_cost = cost
                assign = temp_assign
                results.append(pre_cost)
            elif cost >= pre_cost:
                results.append(pre_cost)

    return results


def rand_LNS(info, graph, dirs):
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]

    for itr in range(100):
        temp_assign = copy.deepcopy(assign)
        removal_idx = rand_model.eval(graph, random_action=True)
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

        cost, _ = solver(info['grid'], info['agents'], to_solver(info['tasks'], temp_assign), dirs=dirs)
        if cost == 'error':
            pass

        else:
            if cost < pre_cost:
                pre_cost = cost
                assign = temp_assign
                routes = to_solver(info['tasks'], assign)
                schedules = info['agents'].tolist()
                for tr in routes:
                    schedules += tr
                sch_nx = sch_to_nx(schedules, info['graph'], len(info['agents']), len(info['tasks']))
                graph = dgl.from_networkx(sch_nx,
                                          node_attrs=['coord', 'type'],
                                          edge_attrs=['a_dist', 'dist', 'obs_proxy']
                                          )
                results.append(pre_cost)
            elif cost >= pre_cost:
                results.append(pre_cost)

    return results


def run(dirs):
    SL_LNS_dirs = [dirs[0], dirs[1][0], dirs[2], dirs[3][0]]
    LNS_dirs = [dirs[0], dirs[1][1], dirs[2], dirs[3][1]]
    rand_dirs = [dirs[0], dirs[1][2], dirs[2], dirs[3][2]]
    init_dirs = [dirs[0], dirs[1][3], dirs[2], dirs[3][3]]

    try:
        for d in [SL_LNS_dirs[1], LNS_dirs[1], rand_dirs[1], init_dirs[1]]:
            if not os.path.exists(d):
                os.makedirs(d)
    except OSError:
        print("Error: Cannot create the directory.")

    scenario = load_scenarios('323220_5_50_eval/scenario_{}.pkl'.format(dirs[4]))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
    n_agents, n_tasks = len(info['agents']), len(info['tasks'])
    assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
    routes = to_solver(info['tasks'], assign)

    info['lns'] = assign_id, assign
    info['init_cost'], info['init_routes'] = solver(info['grid'], info['agents'], routes, dirs=init_dirs)

    temp_routes = copy.deepcopy(routes)
    schedules = info['agents'].tolist()
    for tr in temp_routes:
        schedules += tr
    sch_nx = sch_to_nx(schedules, info['graph'], n_agents, n_tasks)
    init_graph = dgl.from_networkx(
        sch_nx,
        node_attrs=['coord', 'type'],
        edge_attrs=['a_dist', 'dist', 'obs_proxy']
    )

    sl_lns = SL_LNS(info, init_graph, dirs=SL_LNS_dirs)
    # lns = heuristics(info, dirs=LNS_dirs)
    rand_lns = rand_LNS(info, init_graph, dirs=rand_dirs)

    with open('eval_{}.pkl'.format(dirs[4]), 'wb') as f:
        # pickle.dump([sl_lns, lns, rand_lns], f)
        pickle.dump([sl_lns, rand_lns], f)

    try:
        for d in [SL_LNS_dirs[1], LNS_dirs[1], rand_dirs[1], init_dirs[1]]:
            if os.path.exists(d):
                shutil.rmtree(d)
    except OSError:
        print("Error: Cannot remove the directory.")


if __name__ == '__main__':
    eval = False

    if eval:
        for total_exp in trange(10):
            dirs_list = []
            for exp_num in range(total_exp * 10, total_exp * 10 + 10):
                solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/eecbs')
                save_dir = os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/SL_LNS_{}/'.format(exp_num)), \
                           os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/LNS_{}/'.format(exp_num)), \
                           os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/rand_{}/'.format(exp_num)), \
                           os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/init_{}/'.format(exp_num))
                exp_name = 'SL_LNS', 'heuristics', 'rand', 'init'
                dirs = [solver_dir, save_dir, '', exp_name, exp_num]
                dirs_list.append(dirs)

            run_list = [Process(target=run, args=[dirs]) for dirs in dirs_list]
            for r in run_list:
                r.start()
            while sum([not r.is_alive() for r in run_list]) != 10:
                pass

    else:
        comparing_plot(100)
