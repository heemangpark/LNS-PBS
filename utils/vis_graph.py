import os

import matplotlib.pyplot as plt
import networkx as nx


def vis_graph(graph):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    nx.draw(graph, pos=pos, with_labels=False, node_size=50)
    plt.savefig('graph.png')
    plt.clf()


def vis_init(graph, agents, tasks):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    nx.draw(graph, pos=pos, nodelist=[tuple(a) for a in agents], node_color='r', node_size=100)
    for j in range(len(tasks)):
        nx.draw(graph, pos=pos, nodelist=[tuple(t) for t in tasks[j]], node_color='b', node_size=100)

    dir = './fig/'
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Cannot create the directory.")
    plt.savefig(dir + 'init.png')
    plt.clf()


def vis_assign(graph, agents, tasks, itr):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    for a in tasks:
        nodelist = [tuple(agents[a])]
        nx.draw(graph, pos=pos, nodelist=nodelist, node_size=100,
                node_color=['red', 'orange', 'green', 'blue', 'purple'][a])
        for b in range(len(tasks[a])):
            for c in tasks[a][b].values():
                nodelist += [tuple(d) for d in c]
        nx.draw(graph, pos=pos, nodelist=nodelist, node_size=100, node_shape='X',
                node_color=['red', 'orange', 'green', 'blue', 'purple'][a])

    dir = './fig/'
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Cannot create the directory.")
    plt.savefig(dir + 'assign_{}.png'.format(itr))
    plt.clf()
