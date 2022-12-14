import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

curr_path = os.path.realpath(__file__)
fig_dir = os.path.join(Path(curr_path).parent.parent, 'fig')


def vis_graph(graph):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    nx.draw(graph, pos=pos, with_labels=False, node_size=50)

    try:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    except OSError:
        print("Error: Cannot create the directory.")
    plt.savefig(fig_dir + '/graph.png')
    plt.clf()


def vis_init_assign(graph, agents, tasks):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    nx.draw(graph, pos=pos, nodelist=[tuple(a) for a in agents], node_color='r', node_size=100)
    for j in range(len(tasks)):
        nx.draw(graph, pos=pos, nodelist=[tuple(t) for t in tasks[j]], node_color='b', node_size=100)

    try:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    except OSError:
        print("Error: Cannot create the directory.")
    plt.savefig(fig_dir + '/init_assign.png')
    plt.clf()


def vis_assign(graph, agents, tasks, itr):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    for a in tasks:
        nodelist = [tuple(agents[a])]
        nx.draw(graph, pos=pos, nodelist=nodelist, node_size=100)
        # nx.draw(graph, pos=pos, nodelist=nodelist, node_size=100,
        #         node_color=['red', 'orange', 'green', 'blue', 'purple'][a])
        for b in range(len(tasks[a])):
            for c in tasks[a][b].values():
                nodelist += [tuple(d) for d in c]
        nx.draw(graph, pos=pos, nodelist=nodelist, node_size=100, node_shape='X')
        # nx.draw(graph, pos=pos, nodelist=nodelist, node_size=100, node_shape='X',
        #         node_color=['red', 'orange', 'green', 'blue', 'purple'][a])

    try:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    except OSError:
        print("Error: Cannot create the directory.")
    plt.savefig(fig_dir + '/assign_{}.png'.format(itr))
    plt.clf()
