import dgl
import networkx as nx
import torch

AG_type = 1
TASK_type = 2


def process_graph(nx_g):
    dgl_g = dgl.from_networkx(nx_g, node_attrs=['loc', 'type'])
    # Manhatten dist; useless as all the edge dists are the same
    # dgl_g.apply_edges(lambda edges: {'dist_m': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})

    return dgl_g


def convert_dgl(nx_g, agent_pos, tasks, agent_traj, task_finished=[]):
    di_nx_g = nx.DiGraph(nx_g)  # default networkx graph is undirected
    # set default edge attribute
    nx.set_edge_attributes(di_nx_g, 0, 'traj')
    nx.set_node_attributes(di_nx_g, 0, 'type')

    # set visited traj edge attribute
    for t in agent_traj:
        for _f, _t in zip(t[:-1], t[1:]):
            # TODO: 'hold' action
            if _f == _t: continue
            di_nx_g.edges[tuple(_f), tuple(_t)]['traj'] = 1

    for p in agent_pos:
        di_nx_g.nodes[tuple(p)]['type'] = AG_type

    for finished, t in zip(task_finished, tasks):
        for _t in t:
            if not finished:
                di_nx_g.nodes[tuple(_t)]['type'] = TASK_type

    di_dgl_g = dgl.from_networkx(di_nx_g, node_attrs=['loc', 'type'], edge_attrs=['traj'])
    node_idx_dict = dict()
    coord_x = []
    coord_y = []
    for i, node in enumerate(di_nx_g.nodes()):
        node_idx_dict[tuple(node)] = i
        coord_x.append(node[0])
        coord_y.append(node[1])

    ag_node_indices = []
    for a in agent_pos:
        ag_node_indices.append(node_idx_dict[tuple(a)])

    task_node_indices = []
    assert len(tasks[0]) == 1, "task size > 1 not supported yet"
    for task in tasks:
        task_node_indices.append(node_idx_dict[tuple(task[0])])

    di_dgl_g.ndata['x'] = torch.tensor(coord_x)
    di_dgl_g.ndata['y'] = torch.tensor(coord_y)

    return di_dgl_g, ag_node_indices, task_node_indices


if __name__ == '__main__':
    from utils.generate_scenarios import load_scenarios

    M, N = 10, 10
    scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(M, N))
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]

    di_dgl_g = convert_dgl(graph, agent_pos, total_tasks, [])
