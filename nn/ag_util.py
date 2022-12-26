import dgl
import networkx as nx

AG_type = 1
TASK_type = 2


def process_graph(nx_g):
    dgl_g = dgl.from_networkx(nx_g, node_attrs=['loc', 'type'])
    # Manhatten dist; useless as all the edge dists are the same
    # dgl_g.apply_edges(lambda edges: {'dist_m': (abs(edges.src['loc'] - edges.dst['loc'])).sum(-1)})

    return dgl_g


def embed_traj(nx_g, agent_pos, tasks, agent_traj):
    di_nx_g = nx.DiGraph(nx_g)
    # set default edge attribute
    nx.set_edge_attributes(di_nx_g, 0, 'traj')
    nx.set_node_attributes(di_nx_g, 0, 'type')

    # set visited traj edge attribute
    for t in agent_traj:
        for _f, _t in zip(t[:-1], t[1:]):
            di_nx_g.edges[tuple(_f), tuple(_t)]['traj'] = 1

    for p in agent_pos:
        di_nx_g.nodes[tuple(p)]['type'] = AG_type

    for t in tasks:
        for _t in t:
            di_nx_g.nodes[tuple(_t)]['type'] = TASK_type

    di_dgl_g = dgl.from_networkx(di_nx_g, node_attrs=['loc', 'type'], edge_attrs=['traj'])
    return di_dgl_g
