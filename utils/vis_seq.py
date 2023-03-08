from matplotlib import pyplot as plt


def vis_ta(graph, agents, tasks, itr):
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    task_nodes = list()
    colors = [plt.cm.get_cmap('rainbow')(i / len(agents)) for i in range(len(agents))]
