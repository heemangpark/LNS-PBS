import itertools
import os
import pickle
from pathlib import Path

import dgl
import numpy as np

curr_dir = os.path.realpath(__file__)
data_dir = os.path.join(Path(curr_dir).parent.parent, 'data/')


def load(type=None, exp_n=None):
    """
    output format: assignment with index
    """
    if exp_n is None:
        exp_n = []
    output = []
    for e in exp_n:
        if type == 'p':
            with open(data_dir + 'prev_{}.pkl'.format(e), 'rb') as f:
                data = pickle.load(f)
            output.append(data)
        elif type == 'f':
            with open(data_dir + 'final_{}.pkl'.format(e), 'rb') as f:
                data = pickle.load(f)
            output.append(data)

    return output


def softmax(x):
    y = np.exp(x - np.max(x))
    return y / y.sum(axis=0)


def diff(data_p, data_f):
    """
    return index of tasks
    """
    # output = []
    s_gaps = []
    for p, f in zip(data_p, data_f):
        max_len_p, max_len_a = max([len(i) for i in list(p.values())]), max([len(i) for i in list(f.values())])
        mat_p = np.zeros((len(list(p.keys())), max_len_p), dtype=object)
        mat_f = np.zeros((len(list(f.keys())), max_len_a), dtype=object)

        factor = 1
        for row in range(mat_p.shape[0]):
            for col in range(mat_p.shape[1]):
                if col == len(p[row]):
                    break
                mat_p[row][col] = (p[row][col], np.array([row * factor, col]))

        keys = [data_p[0] for data_p in mat_p[mat_p != 0].reshape(-1)]
        values = [data_p[1] for data_p in mat_p[mat_p != 0].reshape(-1)]
        dict_p = dict(zip(keys, values))
        keys_p = list(dict_p.keys())

        for row in range(mat_f.shape[0]):
            for col in range(mat_f.shape[1]):
                if col == len(f[row]):
                    break
                else:
                    mat_f[row][col] = (f[row][col], np.array([row * factor, col]))

        keys = [data_f[0] for data_f in mat_f[mat_f != 0].reshape(-1)]
        values = [data_f[1] for data_f in mat_f[mat_f != 0].reshape(-1)]
        dict_f = dict(zip(keys, values))

        task_idx = list(range(sum([len(v) for v in p.values()])))
        # gaps = sorted(dict(zip(task_idx, [sum(abs(dict_p[t_id] - dict_f[t_id])) for t_id in task_idx])).items(),
        #               key=lambda x: x[1], reverse=True)
        # output.append([gap[0] for gap in gaps[:3]])

        s_gap = softmax([sum(abs(dict_p[t_id] - dict_f[t_id])) for t_id in keys_p])
        s_gaps.append(s_gap)

    return np.array(s_gaps)


def load_score(datas):
    task_idx_list = []
    init_graph_list = []
    memory_list = []

    for data in datas:
        with open(data, 'rb') as f:
            d = pickle.load(f)
        assign, decrement, graph, removal = d[0], d[1], d[2][0], d[3]

        task_idx = []
        task_idx.extend(assign[0].values())
        t_id = list(itertools.chain(*task_idx))
        memory = dict(zip(t_id, [0 for _ in range(len(t_id))]))

        for actions, score in zip(removal[int(len(removal) * .2):], decrement[int(len(decrement) * .2):]):
            for a in actions:
                memory[a] += score

        task_idx_list.append(t_id)

        init_graph_list.append(dgl.from_networkx(graph,
                                                 node_attrs=['coord', 'type'],
                                                 edge_attrs=['a_dist', 'dist', 'obs_proxy']).to('cuda:1'))

        memory_list.append(softmax(list(memory.values())))

    return task_idx_list, init_graph_list, memory_list
