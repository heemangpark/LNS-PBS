from copy import deepcopy

import dgl
import networkx as nx
import numpy as np
import torch


def grid_to_dgl(world, rand_coord=True, four_dir=True):
    world = deepcopy(world)
    m, n = world.shape
    g = dgl.graph(list())
    g.add_nodes(m * n)

    if rand_coord:
        # node position
        rand_interval_x = torch.rand(m - 1) + .5
        rand_x = [rand_interval_x[:i].sum() for i in range(m)]
        rand_x = torch.Tensor(rand_x)
        rand_x /= rand_x.sum()

        rand_interval_y = torch.rand(n - 1) + .5
        rand_y = [rand_interval_y[:i].sum() for i in range(n)]
        rand_y = torch.Tensor(rand_y)
        rand_y /= rand_y.sum()
    else:
        rand_x = torch.Tensor([0 + i / (n - 1) for i in range(m)])
        rand_y = torch.Tensor([0 + i / (m - 1) for i in range(n)])

    xs = rand_x.repeat(n, 1).reshape(-1)
    ys = rand_y.flip(-1).repeat(m, 1).T.reshape(-1)

    g.ndata['loc'] = torch.stack([xs, ys], -1)
    g.ndata['type'] = torch.Tensor(world.reshape(-1, 1))

    # add edge
    matrix = np.arange(m * n).reshape(m, -1)
    v_from = matrix[:-1].reshape(-1)
    v_to = matrix[1:].reshape(-1)

    g.add_edges(v_from, v_to)
    g.add_edges(v_to, v_from)

    h_from = matrix[:, :-1].reshape(-1)
    h_to = matrix[:, 1:].reshape(-1)

    g.add_edges(h_from, h_to)
    g.add_edges(h_to, h_from)

    if not four_dir:
        dig_from = matrix[:-1, :-1].reshape(-1)
        dig_to = matrix[1:, 1:].reshape(-1)
        g.add_edges(dig_from, dig_to)
        g.add_edges(dig_to, dig_from)

        ddig_from = matrix[1:, :-1].reshape(-1)
        ddig_to = matrix[:-1, 1:].reshape(-1)
        g.add_edges(ddig_from, ddig_to)
        g.add_edges(ddig_to, ddig_from)

    # compute ef
    g.apply_edges(lambda edges: {'dist': ((edges.src['loc'] - edges.dst['loc']) ** 2).sum(-1).reshape(-1, 1) ** .5})

    # remove obstacle
    obs_idx = world.reshape(-1).nonzero()[0]
    g.remove_nodes(obs_idx)

    return g


def validGraph(size=32, obs=20, rand_coord=False):
    instance = np.zeros((size, size))
    obstacle = np.random.random((size, size)) <= obs / 100
    instance[obstacle] = 1
    g = createGraph(instance, rand_coord=rand_coord)
    components = [c for c in nx.connected_components(g)]

    while len(components) != 1:
        instance = np.zeros((size, size))
        obstacle = np.random.random((size, size)) <= obs / 100
        instance[obstacle] = 1
        g = createGraph(instance, rand_coord=rand_coord)
        components = [c for c in nx.connected_components(g)]

    return instance, g


def createGraph(instance, rand_coord=False):
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


def convert_to_nx(assign_idx, coord_schedule, size):
    coords = [item for sublist in coord_schedule for item in sublist]
    norm_coords = [[c[0] / size, c[1] / size] for c in coords]
    sch_nx = nx.complete_graph(len(norm_coords))
    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, norm_coords)), 'coord')

    AG_type, TASK_type = 1, 2
    types = []
    for c in coord_schedule:
        types.extend([AG_type] + [TASK_type for _ in range(len(c) - 1)])
    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, types)), 'type')

    graph_assign_id = []
    for idx in assign_idx:
        graph_assign_id.extend([-1] + idx)
    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, graph_assign_id)), 'idx')

    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, range(sch_nx.number_of_nodes()))), 'graph_id')

    # a_dist = [int(graph_astar(graph, coords[i], coords[j])[1]) for i, j in sch_nx.edges]
    norm_dist = [np.abs(coords[i][0] - coords[j][0]) + np.abs(coords[i][1] - coords[j][1]) / size
                 for i, j in sch_nx.edges]
    # obs = [i - j for i, j in zip(a_dist, dist)]

    # nx.set_edge_attributes(sch_nx, dict(zip(sch_nx.edges, a_dist)), 'a_dist')
    nx.set_edge_attributes(sch_nx, dict(zip(sch_nx.edges, norm_dist)), 'dist')
    # nx.set_edge_attributes(sch_nx, dict(zip(sch_nx.edges, obs)), 'obs_proxy')
    nx.set_edge_attributes(sch_nx, dict(zip(sch_nx.edges, [0] * sch_nx.number_of_edges())), 'connected')

    start_node_idx = 0
    for schedule in coord_schedule:
        n_schedule = len(schedule)
        node_indices = range(start_node_idx, start_node_idx + n_schedule)
        for i, j in zip(node_indices[:-1], node_indices[1:]):
            sch_nx.edges[i, j]['connected'] = 1

        start_node_idx += n_schedule

    return sch_nx.to_directed()
