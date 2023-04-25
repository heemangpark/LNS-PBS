import copy
import os
import pickle
import random
import shutil
import sys
from multiprocessing import Process
from pathlib import Path

import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from utils.graph_utils import convert_to_nx
from nn.repair_agent import NeuroRepair
from utils.scenario_utils import load_scenarios
from utils.solver_utils import to_solver, solver
from utils.solver_utils import assignment_to_id
from utils.solver_utils import id_to_assignment
from utils.plot_utils import comparing_plot

model = NeuroRepair(train=False)
model.load_state_dict(torch.load('repair.pt'))


def NLNS(info, graph, dir):
    task_idx, assign = copy.deepcopy(info['lns'])
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None

    for itr in range(100):
        removal_idx = removal(task_idx, info['tasks'], info['graph'], N=2, time_log=time_log, neighbors='relative')
        if removal_idx == 'stop':
            return 'stop'
        task_idx = model.eval(assignment_to_id(len(info['tasks']), assign), graph, removal_idx)
        temp_assign = id_to_assignment(task_idx, info['tasks'])
        cost, _ = solver(info['grid'], info['agents'], to_solver(info['tasks'], temp_assign), dir=dir)
        if cost == 'error':
            pass

        else:
            if cost < pre_cost:
                pre_cost = cost
                assign = temp_assign
                assign_id = assignment_to_id(len(info['agents']), assign)
                coordination = [[a] for a in info['agents'].tolist()]
                for i, coords in enumerate(assign.values()):
                    temp_schedule = [list(c.values())[0][0] for c in coords]
                    coordination[i].extend(temp_schedule)
                graph = convert_to_nx(assign_id, coordination, info['graph'])
                results.append(pre_cost)
            elif cost >= pre_cost:
                results.append(pre_cost)

    return results


def rLNS(info, dir):
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None

    for itr in range(100):
        temp_assign = copy.deepcopy(assign)
        removal_idx = removal(
            task_idx,
            info['tasks'],
            info['graph'],
            N=2,
            time_log=time_log,
            neighbors='random'
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

    return results


def bLNS(info, dir):
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None

    for itr in range(100):
        temp_assign = copy.deepcopy(assign)
        removal_idx = removal(
            task_idx,
            info['tasks'],
            info['graph'],
            N=9,
            time_log=time_log
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

    return results


def LNS(info, dir):
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None

    for itr in range(100):
        temp_assign = copy.deepcopy(assign)
        removal_idx = removal(
            task_idx,
            info['tasks'],
            info['graph'],
            N=2,
            time_log=time_log
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
    info['init_cost'], info['init_routes'] = solver(info['grid'], info['agents'], routes, dir=dir[3])
    if info['init_cost'] == 'error':
        return 'abandon_seed'

    # coordination = [[a] for a in info['agents'].tolist()]
    # for i, coords in enumerate(assign.values()):
    #     temp_schedule = [list(c.values())[0][0] for c in coords]
    #     coordination[i].extend(temp_schedule)
    # init_graph = sch_to_nx(assign_id, coordination, info['graph'])

    r_lns, b_lns, lns = rLNS(info, dir[0]), bLNS(info, dir[1]), LNS(info, dir[2])

    with open('eval_{}.pkl'.format(dir[0][3]), 'wb') as f:
        pickle.dump([r_lns, b_lns, lns], f)

    try:
        for d in dir:
            if os.path.exists(d[1]):
                shutil.rmtree(d[1])
    except OSError:
        print("Error: Cannot remove the directory.")


if __name__ == '__main__':
    random.seed(42)
    eval = False

    if eval:
        for total_exp in trange(3):
            dirs_list = []
            for exp_num in range(total_exp * 10, total_exp * 10 + 10):
                solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/eecbs')
                save_dir = os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/rLNS_{}/'.format(exp_num)), \
                           os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/bLNS_{}/'.format(exp_num)), \
                           os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/LNS_{}/'.format(exp_num)), \
                           os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/init_{}/'.format(exp_num))
                exp_name = 'random', 'big_neighbor', 'lns', 'init'

                r_dir = [solver_dir, save_dir[0], exp_name[0], exp_num]
                b_dir = [solver_dir, save_dir[1], exp_name[1], exp_num]
                l_dir = [solver_dir, save_dir[2], exp_name[2], exp_num]
                i_dir = [solver_dir, save_dir[3], exp_name[3], exp_num]
                dirs = [r_dir, b_dir, l_dir, i_dir]
                dirs_list.append(dirs)

            run_list = [Process(target=run, args=[dirs]) for dirs in dirs_list]
            for r in run_list:
                r.start()
            while sum([not r.is_alive() for r in run_list]) != 10:
                pass

    else:
        comparing_plot(30)
