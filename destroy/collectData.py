import copy
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from utils.scenario import load_scenarios
from utils.solver import to_solver, solver
from utils.graph import convert_to_nx


def _collectEval(info, solver_dir, save_dir, exp_name):
    random.seed(42)

    assign_idx, assign_pos = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None
    for itr in range(100):
        temp_assign_idx = copy.deepcopy(assign_idx)
        removal_idx = removal(
            assign_idx,
            info['tasks'],
            info['graph'],
            N=2,
            time_log=time_log
        )
        if removal_idx == 'stop':
            return 'stop'

        # remove 'removal_idx'
        removed = [False for _ in removal_idx]
        for schedule in temp_assign_idx:
            for i, r in enumerate(removal_idx):
                if removed[i]: continue
                if r in schedule:
                    schedule.remove(r)
                    removed[i] = True

        while len(removal_idx) != 0:
            f_val = f_ijk(temp_assign_idx, info['agents'], removal_idx, info['tasks'], info['graph'])

            # get min regret
            regrets = np.stack(list(f_val.values()))
            argmin_regret = np.argmin(regrets, axis=None)
            min_regret_idx = np.unravel_index(argmin_regret, regrets.shape)

            r_idx, insertion_edge_idx = min_regret_idx
            re_ins = removal_idx[r_idx]

            # get insertion agent index and location
            ag_idx = 0
            ins_pos = 0
            while True:
                ag_schedule = assign_idx[ag_idx]
                if insertion_edge_idx - (len(ag_schedule) + 1) < 0:
                    ins_pos = insertion_edge_idx
                    break
                else:
                    insertion_edge_idx -= (len(ag_schedule) + 1)
                    ag_idx += 1

            temp_assign_idx[ag_idx].insert(ins_pos, re_ins)
            removal_idx.remove(re_ins)

        assign_pos = [np.array(info['tasks'])[schedule].tolist() for schedule in temp_assign_idx]
        cost, _, time_log = solver(
            info['grid'],
            info['agents'],
            assign_pos,
            ret_log=True,
            solver_dir=solver_dir,
            save_dir=save_dir,
            exp_name=exp_name
        )

        if cost == 'error':
            pass
        else:
            if cost < pre_cost:
                pre_cost = cost
                assign_idx = temp_assign_idx
                results.append(pre_cost)
            elif cost >= pre_cost:
                results.append(pre_cost)

    return (results[0] - results[-1]) / results[0] * 100


def run(run_info, N, M):
    random.seed(42)

    exp_num = run_info['exp_num']
    solver_dir = run_info['solver_dir']
    LNS_save = run_info['LNS_save_dir']
    init_save = run_info['init_save_dir']

    scenario = load_scenarios('323220_{}_{}_eval/scenario_{}.pkl'.format(N, M, exp_num))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': [t[0] for t in scenario[3]]}

    assign_id, assign_pos = hungarian(info['graph'], info['agents'], info['tasks'])
    # assign_id, assign = hungarian_prev(info['graph'], info['agents'], scenario[3])
    info['lns'] = assign_id, assign_pos

    coordination = [[a.tolist()] + t for a, t in zip(info['agents'], assign_pos)]
    initGraph = convert_to_nx(assign_id, coordination, info['grid'].shape[0])
    info['init_cost'], _ = solver(info['grid'], info['agents'], assign_pos, solver_dir=solver_dir, save_dir=init_save,
                                  exp_name='init')
    if info['init_cost'] == 'error':
        return 'abandon_seed'

    data = [info, initGraph]
    lnsResult = _collectEval(info, solver_dir, LNS_save, 'lns')
    print(lnsResult)
    data.append(lnsResult)

    with open('evalData/{}{}/evalData_{}.pkl'.format(N, M, exp_num), 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    from tqdm import trange
    from multiprocessing import Process

    N, M = 5, 50
    n_process = 1
    n_data = 10
    solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/eecbs')
    temp_LNS_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/LNS')
    temp_init_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/init')

    # make directory per process
    for p in range(n_process):
        if not os.path.exists(temp_LNS_dir + str(p) + '/'):
            os.makedirs(temp_LNS_dir + str(p) + '/')
        if not os.path.exists(temp_init_dir + str(p) + '/'):
            os.makedirs(temp_init_dir + str(p) + '/')

    if not os.path.exists('evalData/{}{}/'.format(N, M)):
        os.makedirs('evalData/{}{}/'.format(N, M))

    for i in trange(n_data):
        run_infos = []
        for p, exp_num in enumerate(range(i * n_process, (i + 1) * n_process)):
            run_info = dict()
            run_info['solver_dir'] = solver_dir
            run_info['exp_num'] = exp_num
            run_info['LNS_save_dir'] = temp_LNS_dir + str(p) + '/'
            run_info['init_save_dir'] = temp_init_dir + str(p) + '/'

            run_infos.append(run_info)
            run(run_info, N, M)

        # run_list = [Process(target=run, args=(_info, N, M)) for _info in run_infos]

        # start process
        for r in run_list:
            r.start()
        while sum([not r.is_alive() for r in run_list]) != n_process:
            pass

    # remove temp directories
    for p in range(n_process):
        if os.path.exists(temp_LNS_dir + str(p) + '/'):
            shutil.rmtree(temp_LNS_dir + str(p) + '/')
        if os.path.exists(temp_init_dir + str(p) + '/'):
            shutil.rmtree(temp_init_dir + str(p) + '/')
