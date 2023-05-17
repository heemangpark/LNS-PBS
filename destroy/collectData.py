import copy
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from utils.scenario import load_scenarios
from utils.solver import to_solver, solver
from utils.graph import convert_to_nx


def _collectEval(info, dir):
    random.seed(42)

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

    # task_idx, assign = info['lns']
    # cand_removal = [t for t in combinations(range(len(info['tasks'])), 3)]
    # output = dict()
    #
    # explore = list(random.sample(range(len(cand_removal)), 100))  # 50_C_3 = 19600
    # for c_r in np.array(cand_removal)[explore]:  # index of tasks
    #     removal_idx = list(c_r)
    #     temp_assign = copy.deepcopy(assign)
    #     for i, t in enumerate(temp_assign.values()):
    #         for r in removal_idx:
    #             if {r: info['tasks'][r]} in t:
    #                 temp_assign[i].remove({r: info['tasks'][r]})
    #     while len(removal_idx) != 0:
    #         f_val = f_ijk(temp_assign, info['agents'], removal_idx, info['tasks'], info['graph'])
    #         regret = get_regret(f_val)
    #         regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
    #         re_ins = list(regret.keys())[0]
    #         re_a, re_j = regret[re_ins][1], regret[re_ins][2]
    #         removal_idx.remove(re_ins)
    #         to_insert = {re_ins: info['tasks'][re_ins]}
    #         temp_assign[re_a].insert(re_j, to_insert)
    #
    #     cost, _, _ = solver(
    #         info['grid'],
    #         info['agents'],
    #         to_solver(info['tasks'], temp_assign),
    #         ret_log=True,
    #         dir=dir
    #     )
    #
    #     if cost == 'error':
    #         pass
    #     else:
    #         output[tuple(c_r)] = info['init_cost'] - cost

    return (results[0] - results[-1]) / results[0] * 100


def run(dir):
    random.seed(42)

    try:
        for d in dir:
            if not os.path.exists(d[1]):
                os.makedirs(d[1])
    except OSError:
        print("Error: Cannot create the directory.")

    scenario = load_scenarios('646420_5_50_eval/scenario_{}.pkl'.format(dir[0][3]))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}

    assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
    info['lns'] = assign_id, assign
    routes = to_solver(info['tasks'], assign)

    coordination = [[a] for a in info['agents'].tolist()]
    for i, coords in enumerate(assign.values()):
        temp_schedule = [list(c.values())[0][0] for c in coords]
        coordination[i].extend(temp_schedule)
    initGraph = convert_to_nx(assign_id, coordination, info['grid'].shape[0])

    info['init_cost'], _ = solver(info['grid'], info['agents'], routes, dir=dir[1])
    if info['init_cost'] == 'error':
        return 'abandon_seed'

    data = [info, initGraph]
    lnsResult = _collectEval(info, dir=dir[0])
    print(lnsResult)
    data.append(lnsResult)

    with open('evalData/550/evalData_{}.pkl'.format(dir[0][3]), 'wb') as f:
        pickle.dump(data, f)

    try:
        for d in dir:
            if os.path.exists(d[1]):
                shutil.rmtree(d[1])
    except OSError:
        print("Error: Cannot remove the directory.")


if __name__ == "__main__":
    from tqdm import trange
    from multiprocessing import Process

    for total_exp in trange(10):
        dirs_list = []
        for exp_num in range(total_exp * 10, total_exp * 10 + 10):
            solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/eecbs')
            save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/LNS_{}/'.format(exp_num)), \
                       os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/init_{}/'.format(exp_num))
            LNS_dirs = [solver_dir, save_dir[0], 'lns', exp_num]
            init_dirs = [solver_dir, save_dir[1], 'init', exp_num]
            dirs = [LNS_dirs, init_dirs]
            dirs_list.append(dirs)

        run_list = [Process(target=run, args=(dirs,)) for dirs in dirs_list]
        for r in run_list:
            r.start()
        while sum([not r.is_alive() for r in run_list]) != 10:
            pass
