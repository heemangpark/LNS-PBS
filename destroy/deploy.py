import copy
import os
import pickle
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path

import dgl
import numpy as np
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.hungarian import hungarian
from heuristics.regret import f_ijk, get_regret
from heuristics.shaw import removal
from nn.destroyNaive import DestroyNaive
from utils.graph import convert_to_nx
from utils.plot import comparing_plot
from utils.scenario import load_scenarios
from utils.solver import to_solver, solver

model = DestroyNaive()
model.load_state_dict(torch.load('models/destroyAgent_0.pt'))
model.eval()
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def NLNS(info, graph, dir):
    task_idx, assign = copy.deepcopy(info['lns'])
    pre_cost = info['init_cost']
    results = [pre_cost]

    for itr in range(100):
        # graph preprocessing
        dgl_graph = dgl.from_networkx(
            graph,
            node_attrs=['coord', 'type', 'idx', 'graph_id'],
            edge_attrs=['dist', 'connected']
        ).to(device)

        if dgl_graph.edata['dist'].dtype == torch.int64:
            dgl_graph.edata['dist'] = dgl_graph.edata['dist'] / 64
        dgl_graph.edata['dist'] = dgl_graph.edata['dist'].to(torch.float32)

        if dgl_graph.ndata['coord'].dtype == torch.int64:
            dgl_graph.ndata['coord'] = dgl_graph.ndata['coord'] / 64

        temp_assign = copy.deepcopy(assign)
        num_tasks = len([i for i in dgl_graph.nodes() if dgl_graph.nodes[i]['type'] == 2])
        destroyCand = [c for c in combinations(range(num_tasks), 3)]
        toDestroy = random.sample(destroyCand, 1000)
        pred = model.act(dgl_graph, toDestroy)
        removal_idx = list(toDestroy[np.argmax(pred)])

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

                # generate graph
                coordination = [[a] for a in info['agents'].tolist()]
                for i, coords in enumerate(assign.values()):
                    temp_schedule = [list(c.values())[0][0] for c in coords]
                    coordination[i].extend(temp_schedule)

                graph = convert_to_nx(task_idx, coordination, info['graph'])
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


if __name__ == '__main__':
    random.seed(42)
    for e in trange(100):
        solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/eecbs')
        save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/LNS_{}/'.format(e)), \
                   os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/NLNS_{}/'.format(e)), \
                   os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/init_{}/'.format(e))

        l_dir = [solver_dir, save_dir[0], 'lns', e]
        n_dir = [solver_dir, save_dir[1], 'nlns', e]
        i_dir = [solver_dir, save_dir[2], 'init', e]
        dirs = [l_dir, n_dir, i_dir]

        try:
            for d in dirs:
                if not os.path.exists(d[1]):
                    os.makedirs(d[1])
        except OSError:
            print("Error: Cannot create the directory.")

        scenario = load_scenarios('646420_5_50_eval/scenario_{}.pkl'.format(e))
        info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}

        assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
        routes = to_solver(info['tasks'], assign)

        info['lns'] = assign_id, assign
        info['init_cost'], info['init_routes'] = solver(info['grid'], info['agents'], routes, dir=dirs[2])

        coordination = [[a] for a in info['agents'].tolist()]
        for i, coords in enumerate(assign.values()):
            temp_schedule = [list(c.values())[0][0] for c in coords]
            coordination[i].extend(temp_schedule)
        init_graph = convert_to_nx(assign_id, coordination, info['grid'].shape[0])

        # lns = LNS(info, dirs[0])
        # with open('evalData/evalData_{}.pkl'.format(e), 'wb') as f:
        #     pickle.dump([info, init_graph, (lns[0] - lns[-1]) / lns[0]], f)

        lns, nlns = LNS(info, dirs[0]), NLNS(info, init_graph, dirs[1])
        with open('eval_{}.pkl'.format(e), 'wb') as f:
            pickle.dump([lns, nlns], f)

        try:
            for d in dirs:
                if os.path.exists(d[1]):
                    shutil.rmtree(d[1])
        except OSError:
            print("Error: Cannot remove the directory.")

    comparing_plot(100)
