import copy
import os
import pickle
import random
import shutil
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import dgl
import torch
from tqdm import tqdm
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.shaw import removal
from heuristics.regret import f_ijk, get_regret
from nn.destroyEdgewise import DestroyEdgewise
from utils.graph import convert_to_nx
from utils.solver import solver, assignment_to_id, to_solver


def train(epochs=100, dataSize=10000, batchNum=100, method='topK', device='cuda:1', wandb=False, midEval=False):
    """
    @param epochs: exp epochs
    @param dataSize: exp file number
    @param batchNum: batch number
    @param method: label filtering method
    @param device: model device
    @param wandb: plot wandb or not
    @param midEval: conduct mid-evaluation or not
    @return: saved model pt
    """
    date = datetime.now().strftime("%m%d_%H%M%S")

    if wandb:
        import wandb
        wandb.init(project='NLNS-destroy', name=date, config={'score distribution': False,
                                                              'normalized distribution': False,
                                                              'method': method,
                                                              'loss type': 'REINFORCE'})

    model = DestroyEdgewise(device=device)
    model.load_state_dict(torch.load('models/0510_192046/destroyEdgewise_topK_60.pt'))
    batchSize = dataSize // batchNum
    data_idx = list(range(dataSize))

    for e in trange(epochs):
        random.shuffle(data_idx)
        epochLoss = 0

        for b in range(batchNum):
            batchGraph, batchDestroy = [], []

            for d_id in data_idx[b * batchSize: (b + 1) * batchSize]:
                with open('dataCuda/dataDestroy_{}.pkl'.format(d_id), 'rb') as f:
                    graph, destroy = pickle.load(f)
                    if method == 'topK':
                        destroy = dict(sorted(destroy.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
                    elif method == 'randomK':
                        randomKey = list(destroy.keys())
                        random.shuffle(randomKey)
                        randomKey = randomKey[:10]
                        destroy = dict(zip(randomKey, [destroy[k] for k in randomKey]))
                    batchGraph.append(graph)
                    batchDestroy.append(destroy)
            batchGraph = dgl.batch(batchGraph).to(device)

            batchLoss = model.learn(batchGraph, batchDestroy, batchNum, device=device)
            epochLoss += batchLoss
        epochLoss /= batchNum

        if wandb:
            wandb.log({'epochLoss': epochLoss})

        if (e + 1) % 10 == 0:
            dir = 'models/{}/'.format(date)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(model.state_dict(), dir + 'destroyEdgewise_{}_{}.pt'.format(method, e + 1))

        # if midEval & ((e + 1) % 10 == 0):
        #     evalNLNS, evalLNS = eval(
        #         dir='',
        #         evalNum=10,
        #         evalMode='greedy',
        #         dataDist='out',
        #         candSize=10,
        #         device=device
        #     )
        #     if wandb:
        #         wandb.log({'eval_NLNS': evalNLNS, 'eval_LNS': evalLNS})


def eval(dir=None, evalNum=10, dataDist='out', evalMode='greedy', candSize=10, device='cuda:1'):
    """
    @param dir: directory of the model to evaluate
    @param evalNum: number of maps to evaluate
    @param dataDist: in -> evaluate in train dataset, out -> evaluate in test dataset
    @param evalMode: greedy -> argmax model prediction, sample -> sample from softmax(prediction)
    @param candSize: number of destroy node set candidates model search before actual LNS procedure
    @param device: device
    @return: average cost improvement for each NLNS & LNS (0 ~ 1)
    """
    random.seed(42)  # evaluating on fixed maps

    " Load model "
    model = DestroyEdgewise(device=device)
    model.load_state_dict(torch.load(dir))
    model.eval()

    nlnsPerformance = 0
    lnsPerformance = 0

    mapIndex = list(range(100))
    random.shuffle(mapIndex)
    for mapVersion in mapIndex[:evalNum]:
        " EECBS solver directory setup "
        solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/eecbs')
        save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/{}/'.
                                format(datetime.now().strftime("%m%d_%H%M%S")))
        n_dir = [solver_dir, save_dir, 'nlns', mapVersion]

        try:
            if not os.path.exists(n_dir[1]):
                os.makedirs(n_dir[1])
        except OSError:
            print("Error: Cannot create the directory.")

        " Load initial solution "
        if dataDist == 'out':
            with open('evalData/evalData_{}.pkl'.format(mapVersion), 'rb') as f:
                info, graph, lns = pickle.load(f)
        elif dataDist == 'in':
            with open('data/dataDestroy_{}.pkl'.format(mapVersion), 'rb') as f:
                graph, destroy = pickle.load(f)
        else:
            raise NotImplementedError('test data distribution -> in or out')

        " Adapt model into LNS procedure (Actual Evaluation) "
        task_idx, assign = copy.deepcopy(info['lns'])
        pre_cost = info['init_cost']
        results = [pre_cost]

        for itr in range(100):
            temp_assign = copy.deepcopy(assign)
            temp_graph = copy.deepcopy(graph)

            num_tasks = len([i for i in temp_graph.nodes() if temp_graph.ndata['type'][i] == 2])

            destroyCand = [c for c in combinations(range(num_tasks), 3)]
            candDestroy = random.sample(destroyCand, candSize)
            removal_idx = model.act(temp_graph, candDestroy, evalMode, device)
            removal_idx = list(removal_idx)

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
                dir=n_dir
            )

            if cost == 'error':
                pass

            else:
                if cost < pre_cost:
                    pre_cost = cost
                    assign = temp_assign
                    results.append(pre_cost)
                    coordination = [[a] for a in info['agents'].tolist()]
                    for i, coords in enumerate(assign.values()):
                        temp_schedule = [list(c.values())[0][0] for c in coords]
                        coordination[i].extend(temp_schedule)
                    task_idx = assignment_to_id(len(info['agents']), assign)
                    next_nx_graph = convert_to_nx(task_idx, coordination, info['grid'].shape[0])
                    next_graph = dgl.from_networkx(
                        next_nx_graph,
                        node_attrs=['coord', 'type', 'idx', 'graph_id'],
                        edge_attrs=['dist', 'connected']
                    ).to(device)
                    next_graph.edata['dist'] = next_graph.edata['dist'].to(torch.float32)
                    graph = next_graph

                elif cost >= pre_cost:
                    results.append(pre_cost)

        nlnsPerformance += (results[0] - results[-1]) / results[0]
        lnsPerformance += lns

        try:
            if os.path.exists(n_dir[1]):
                shutil.rmtree(n_dir[1])
        except OSError:
            print("Error: Cannot remove the directory.")
    return nlnsPerformance / evalNum, lnsPerformance / evalNum


def multiEval(evalMode='greedy', candSize=10, device='cuda:3', returnDict=dict, evalID=0, threshold=10):
    """
        @param evalMode: greedy -> argmax model prediction, sample -> sample from softmax(prediction)
        @param candSize: number of destroy node set candidates model search before actual LNS procedure
        @param device: device
        @param returnDict: dictionary for the multiprocessing output
        @param evalID: map index
        @param threshold: time budget of algorithm
        @return: average cost improvement for each NLNS & LNS (0 ~ 1)
        """
    random.seed(42)

    " Load model "
    model = DestroyEdgewise(device=device)
    model.load_state_dict(torch.load('models/0510_192046/destroyEdgewise_topK_100.pt'))
    model.eval()

    " EECBS solver directory setup "
    solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/eecbs')
    save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/{}/'.format(evalID))
    n_dir = [solver_dir, save_dir, 'nlns', evalID]

    try:
        if not os.path.exists(n_dir[1]):
            os.makedirs(n_dir[1])
    except OSError:
        print("Error: Cannot create the directory.")

    " Load initial solution "
    with open('evalData/550/evalData_{}.pkl'.format(evalID), 'rb') as f:
        info, graph, _ = pickle.load(f)
        graph = graph.to(device)

    " Adapt model into LNS procedure (Actual Evaluation) "
    task_idx, assign = copy.deepcopy(info['lns'])
    pre_cost = info['init_cost']
    results = [pre_cost]

    algo_start_time = time.time()

    for itr in range(100_000_000):
        temp_assign = copy.deepcopy(assign)
        temp_graph = copy.deepcopy(graph)

        num_tasks = len([i for i in temp_graph.nodes() if temp_graph.ndata['type'][i] == 2])

        destroyCand = [c for c in combinations(range(num_tasks), 3)]
        candDestroy = random.sample(destroyCand, candSize)
        removal_idx = model.act(temp_graph, candDestroy, evalMode, device)
        removal_idx = list(removal_idx)

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
            dir=n_dir
        )

        if cost == 'error':
            pass

        else:
            algo_current_time = time.time()
            if algo_current_time - algo_start_time <= threshold:
                if cost < pre_cost:
                    pre_cost = cost
                    assign = temp_assign
                    results.append(pre_cost)
                    coordination = [[a] for a in info['agents'].tolist()]
                    for i, coords in enumerate(assign.values()):
                        temp_schedule = [list(c.values())[0][0] for c in coords]
                        coordination[i].extend(temp_schedule)
                    task_idx = assignment_to_id(len(info['agents']), assign)
                    next_nx_graph = convert_to_nx(task_idx, coordination, info['grid'].shape[0])
                    next_graph = dgl.from_networkx(
                        next_nx_graph,
                        node_attrs=['coord', 'type', 'idx', 'graph_id'],
                        edge_attrs=['dist', 'connected']
                    ).to(device)
                    next_graph.edata['dist'] = next_graph.edata['dist'].to(torch.float32)
                    graph = next_graph

                elif cost >= pre_cost:
                    results.append(pre_cost)

            else:
                break

    nlns = (results[0] - results[-1]) / results[0] * 100

    try:
        if os.path.exists(n_dir[1]):
            shutil.rmtree(n_dir[1])
    except OSError:
        print("Error: Cannot remove the directory.")

    # returnDict[nlns] = nlns
    return nlns


def heuristicEval(evalID=0, threshold=10):
    random.seed(42)

    " EECBS solver directory setup "
    solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/eecbs')
    save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'EECBS/{}/'.format(evalID))
    h_dir = [solver_dir, save_dir, 'lns', evalID]
    try:
        if not os.path.exists(h_dir[1]):
            os.makedirs(h_dir[1])
    except OSError:
        print("Error: Cannot create the directory.")

    " Load initial solution "
    with open('evalData/550/evalData_{}.pkl'.format(evalID), 'rb') as f:
        info, _, _ = pickle.load(f)

    " Adapt model into LNS procedure (Actual Evaluation) "
    task_idx, assign = info['lns']
    pre_cost = info['init_cost']
    results = [pre_cost]
    time_log = None

    algo_start_time = time.time()

    for itr in range(100_000_000):
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
            dir=h_dir
        )

        if cost == 'error':
            pass
        else:
            algo_current_time = time.time()
            if algo_current_time - algo_start_time <= threshold:
                if cost < pre_cost:
                    pre_cost = cost
                    assign = temp_assign
                    results.append(pre_cost)
                elif cost >= pre_cost:
                    results.append(pre_cost)
            else:
                break

    try:
        if os.path.exists(h_dir[1]):
            shutil.rmtree(h_dir[1])
    except OSError:
        print("Error: Cannot remove the directory.")

    return (results[0] - results[-1]) / results[0] * 100


if __name__ == '__main__':
    train_ = False

    if train_:
        train(epochs=100, dataSize=10000, batchNum=100, method='topK', device='cuda:3', wandb=True, midEval=False)

    else:
        import numpy as np
        import time

        random.seed(42)
        mapIndex = list(range(100))
        random.shuffle(mapIndex)
        mapIndex = mapIndex[:100]

        # lnsList = []
        # for mapID in mapIndex:
        #     with open('evalData/550/evalData_{}.pkl'.format(mapID), 'rb') as f:
        #         _, _, lnsResult = pickle.load(f)
        #         lnsList.append(lnsResult)

        for threshold in [5, 10, 30, 60, 300, 600]:
            budgetTestList = []
            for mapID in tqdm(mapIndex):
                nlns = multiEval(evalMode='greedy',
                                 candSize=10,
                                 device='cuda:1',
                                 returnDict=dict,
                                 evalID=mapID,
                                 threshold=threshold)
                budgetTestList.append(nlns)
            print('{} || {:.4f} +- {:.4f}'.format(threshold, np.mean(budgetTestList), np.std(budgetTestList)))

        for threshold in [5, 10, 30, 60, 300, 600]:
            _budgetTestList = []
            for mapID in tqdm(mapIndex):
                nlns = heuristicEval(evalID=mapID,
                                     threshold=threshold)
                _budgetTestList.append(nlns)
            print('{} || {:.4f} +- {:.4f}'.format(threshold, np.mean(_budgetTestList), np.std(_budgetTestList)))

        # totalList = []
        # for evalItr in range(1):
        #     manager = multiprocessing.Manager()
        #     output = manager.dict()
        #     process = []
        #
        #     for mapID in mapIndex[evalItr * 1: evalItr * 1 + 1]:
        #         p = Process(target=multiEval, args=(*['greedy', 10, 'cuda:1', output] + [mapID],))
        #         process.append(p)
        #         p.start()
        #
        #     for proc in process:
        #         proc.join()
        #     nlnsList = list(output.values())
        #     totalList.extend(nlnsList)
        #
        # print('NLNS: {:.4f}'.format(np.mean(totalList)))
