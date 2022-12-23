from utils.astar import graph_astar


def cost(agent_pos, solution, graph):
    agent_cost_list = list()
    for i in solution:
        path = list()
        agent_cost = 0 if len(solution[i]) == 0 else graph_astar(graph, agent_pos[i],
                                                                 list(solution[i][0].values())[0][0])[1]
        for a in solution[i]:
            for b in a.values():
                path += b
        for s, g in zip(path[:-1], path[1:]):
            agent_cost += graph_astar(graph, s, g)[1]
        agent_cost_list.append(agent_cost)

    return sum(agent_cost_list), max(agent_cost_list)
