from copy import deepcopy

import networkx as nx
import numpy as np


def gen_graph(size=32, obs=20, rand_coord=False):
    instance = np.zeros((size, size))
    obstacle = np.random.random((size, size)) <= obs / 100
    instance[obstacle] = 1
    g = tool(instance, rand_coord=rand_coord)
    components = [c for c in nx.connected_components(g)]

    while len(components) != 1:
        instance = np.zeros((size, size))
        obstacle = np.random.random((size, size)) <= obs / 100
        instance[obstacle] = 1
        g = tool(instance, rand_coord=rand_coord)
        components = [c for c in nx.connected_components(g)]

    return instance, g


def tool(instance, rand_coord=False):
    instance = deepcopy(instance)
    m, n = instance.shape[0], instance.shape[1]
    g = nx.grid_2d_graph(m, n)

    if rand_coord:
        rand_interval_x = np.random.random(m - 1) + .5
        rand_x = np.array([rand_interval_x[:i].sum() for i in range(m)])
        rand_x /= rand_x.sum()

        rand_interval_y = np.random.random(n - 1) + .5
        rand_y = np.array([rand_interval_y[:i].sum() for i in range(n)])
        rand_y /= rand_y.sum()

    else:
        rand_x = np.array([i for i in range(m)])
        rand_y = np.array([j for j in range(n)])

    xs = np.array(list(rand_x) * n)
    ys = rand_y[::-1].repeat(m)
    coords = np.stack([xs, ys], -1)

    for id, n_id in enumerate(g.nodes()):
        g.nodes[n_id]['loc'] = coords[id].tolist()
        # g.nodes[n_id]['type'] = int(instance.reshape(-1)[id])

    for r, c in zip(instance.nonzero()[0], instance.nonzero()[1]):
        g.remove_node((r, c))

    for id, e_id in enumerate(g.edges()):
        man = np.abs(g.nodes[e_id[0]]['loc'][0] - g.nodes[e_id[1]]['loc'][0]) + \
              np.abs(g.nodes[e_id[0]]['loc'][1] - g.nodes[e_id[1]]['loc'][1])
        g.edges[e_id]['dist'] = man

    return g
