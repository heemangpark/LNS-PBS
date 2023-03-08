import copy
import os
import pickle
import shutil
import subprocess
from itertools import combinations, permutations
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from LNS.shaw import removal
from utils.astar import graph_astar
from utils.generate_scenarios import load_scenarios


def intra_costs(tau_1, tau_2):
    mat_1 = np.zeros((len(tau_1), len(tau_1)))
    mat_2 = np.zeros((len(tau_2), len(tau_2)))
    for i in range(len(tau_1)):
        for j in range(i + 1, len(tau_1)):
            mat_1[i][j] = int(graph_astar(info['graph'], tau_1[i], tau_1[j])[1])
    for i in range(1, len(tau_1)):
        for j in range(i):
            mat_1[i][j] = mat_1[j][i]
    for i in range(len(tau_2)):
        for j in range(i + 1, len(tau_2)):
            mat_2[i][j] = int(graph_astar(info['graph'], tau_2[i], tau_2[j])[1])
    for i in range(1, len(tau_2)):
        for j in range(i):
            mat_2[i][j] = mat_2[j][i]
    # mat_1[0][:], mat_1[:, 0] = np.ones(len(tau_1)) * 99999, 0
    # mat_2[0][:], mat_2[:, 0] = np.ones(len(tau_2)) * 99999, 0

    # mat_1[0][:], mat_1[1:, 0] = np.zeros(len(tau_1)), 99999
    # mat_2[0][:], mat_2[1:, 0] = np.zeros(len(tau_2)), 99999

    mat_1[1:, 0], mat_2[1:, 0] = 0, 0

    return mat_1.astype(int), mat_2.astype(int)


def create_data_model(mat):
    return {'distance_matrix': mat, 'num_vehicles': 1, 'depot': 0}


def output_solution(data, manager, routing, solution):
    # print(f'Objective: {solution.ObjectiveValue()}')
    output = [[], []]
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            output[0].append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        # output[0].append(manager.IndexToNode(index)), output[1].append(route_distance)
        # output[1].append(route_distance)
    return output[0]


def or_tools(mat):
    data = create_data_model(mat)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        return output_solution(data, manager, routing, solution)


def seq_solver(instance, agents, tasks, solver_params):
    s_agents = copy.deepcopy(agents)
    todo = copy.deepcopy(tasks)
    seq_paths = [[list(agents[a])] for a in range(len(agents))]
    total_cost, itr = 0, 0

    while sum([len(t) for t in todo]) != 0:
        itr += 1
        s_tasks = list()
        for a, t in zip(s_agents, todo):
            if len(t) == 0:
                s_tasks.append([list(a)])
            else:
                s_tasks.append([t[0]])
        save_map(instance, exp_name)
        save_scenario(s_agents, s_tasks, exp_name, instance.shape[0], instance.shape[1])

        c = [solver_dir,
             "-m",
             save_dir + exp_name + '.map',
             "-a",
             save_dir + exp_name + '.scen',
             "-o",
             save_dir + exp_name + ".csv",
             "--outputPaths",
             save_dir + exp_name + "_paths_{}.txt".format(itr),
             "-k", "{}".format(len(s_agents)),
             "-t", "{}".format(solver_params[0]),
             "--suboptimality={}".format(solver_params[1])]
        process_out = subprocess.run(c, capture_output=True)
        text_byte = process_out.stdout.decode('utf-8')
        if text_byte[37:44] != 'Succeed':
            return 'error', 'error'

        traj = read_trajectory(save_dir + exp_name + "_paths_{}.txt".format(itr))
        len_traj = [len(t) - 1 for t in traj]
        d_len_traj = [l for l in len_traj if l not in {0}]
        next_t = np.min(d_len_traj)

        fin_id = list()
        for e, t in enumerate(traj):
            if len(t) == 1:
                fin_id.append(False)
            else:
                fin_id.append(t[next_t] == s_tasks[e][0])
        fin_ag = np.array(range(len(s_agents)))[fin_id]

        for a_id in range(len(s_agents)):
            if a_id in fin_ag:
                if len(todo[a_id]) == 0:
                    pass
                else:
                    ag_to = todo[a_id].pop(0)
                    s_agents[a_id] = ag_to
            else:
                if len_traj[a_id] == 0:
                    pass
                else:
                    s_agents[a_id] = traj[a_id][next_t]

            seq_paths[a_id] += traj[a_id][1:next_t + 1]

        total_cost += next_t * len(d_len_traj)

    return total_cost, seq_paths


def save_map(grid, filename):
    f = open(save_dir + '{}.map'.format(filename), 'w')
    f.write('type four-directional\n')
    f.write('height {}\n'.format(grid.shape[0]))
    f.write('width {}\n'.format(grid.shape[1]))
    f.write('map\n')

    # creating map from grid
    map_dict = {0: '.', 1: '@'}
    for r in range(grid.shape[0]):
        line = grid[r]
        l = []
        for g in line:
            l.append(map_dict[g])
        f.write(''.join(l) + '\n')

    f.close()


def save_scenario(agent_pos, total_tasks, scenario_name, row, column):
    f = open(save_dir + '{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    for a, t in zip(agent_pos, total_tasks):
        task = t[0]
        dist = abs(np.array(a) - np.array(t)).sum()  # Manhattan dist
        line = '1 \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}'.format('{}.map'.format(scenario_name), row, column, a[1],
                                                                  a[0], task[1], task[0], dist)
        f.write(line + "\n")
    f.close()


def read_trajectory(path_file_dir):
    f = open(path_file_dir, 'r')
    lines = f.readlines()
    agent_traj = []

    for i, string in enumerate(lines):
        curr_agent_traj = []
        split_string = string.split('->')
        for itr, s in enumerate(split_string):
            if itr == len(split_string) - 1:
                continue
            if itr == 0:
                tup = s.split(' ')[-1]
            else:
                tup = s

            ag_loc = [int(i) for i in tup[1:-1].split(',')]
            curr_agent_traj.append(ag_loc)
        agent_traj.append(curr_agent_traj)

    f.close()

    return agent_traj


def to_solver(task_in_seq, assignment):
    s_in_tasks = [[] for _ in range(len(assignment))]
    for a, t in assignment.items():
        if len(t) == 0:
            pass
        else:
            __t = list()
            for _t in t:
                __t += task_in_seq[list(_t.keys())[0]]
            s_in_tasks[a] = __t
    return s_in_tasks


def relatedness(routes, explore):
    div = dict()
    R = 10000
    for i, j in combinations(range(len(routes)), 2):  # route must include agent & task sequence's total coords
        dist = 0
        for a in routes[i]:
            for b in routes[j]:
                dist += (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]))
        div[(i, j)] = dist / (len(routes[i]) * len(routes[j]))
    div = sorted(div.items(), key=lambda x: x[1])
    rd = np.random.choice([r for r in range(1, len(div))][:R])

    return div[0][0] if not explore else div[rd][0]


def swap(grid, routes, tau_1, tau_2, itr, process=None):
    pos = dict()
    for i in range(len(grid)):
        pos[list(grid.nodes)[i]] = grid.nodes[list(grid.nodes)[i]]['loc']

    t_1 = [(tuple(i), tuple(j)) for i, j in zip(routes[tau_1][:-1], routes[tau_1][1:])]
    t_2 = [(tuple(i), tuple(j)) for i, j in zip(routes[tau_2][:-1], routes[tau_2][1:])]

    nx.draw_networkx_nodes(grid, pos=pos, nodelist=[t_1[0][0]], node_shape='o', node_color='r', node_size=50)
    nx.draw_networkx_edges(grid, pos=pos, edgelist=t_1, edge_color='r')
    nx.draw_networkx_nodes(grid, pos=pos, nodelist=[t_1[-1][-1]], node_shape='X', node_color='r', node_size=50)

    nx.draw_networkx_nodes(grid, pos=pos, nodelist=[t_2[0][0]], node_shape='o', node_color='b', node_size=50)
    nx.draw_networkx_edges(grid, pos=pos, edgelist=t_2, edge_color='b')
    nx.draw_networkx_nodes(grid, pos=pos, nodelist=[t_2[-1][-1]], node_shape='X', node_color='b', node_size=50)

    if process == 'before':
        plt.savefig('{}_before_swap.png'.format(itr))
        plt.clf()
    else:
        plt.savefig('{}_after_swap.png'.format(itr))
        plt.clf()


def CE(info):
    a = info['ce_assign']
    init_cost = info['init_cost']
    before_cost = init_cost
    explore = False
    results = []
    check_swap = []

    for itr in range(1, 101):

        # SelectTour
        swap_a = copy.deepcopy(a)
        for i in range(len(swap_a)):
            swap_a[i].insert(0, list(info['agents'][i]))
        swap_1, swap_2 = relatedness(swap_a, explore)
        check_swap.append([swap_1, swap_2])

        # substrings
        substring_length = 1
        sub_1 = [f for f in permutations(range(len(a[swap_1])), 2)]
        sub_2 = [s for s in permutations(range(len(a[swap_2])), 2)]
        trunc_1 = [cand for cand in sub_1 if np.abs(cand[0] - cand[1]) <= substring_length]
        trunc_2 = [cand for cand in sub_2 if np.abs(cand[0] - cand[1]) <= substring_length]
        sub_1, sub_2 = trunc_1, trunc_2

        # cross exchange
        output = [[], []]
        for s1 in sub_1:
            _s1 = np.arange(s1[0], s1[1] + 1) if s1[0] < s1[1] else np.arange(s1[0], s1[1] - 1, -1)
            for s2 in sub_2:
                _s2 = np.arange(s2[0], s2[1] + 1) if s2[0] < s2[1] else np.arange(s2[0], s2[1] - 1, -1)

                if max(_s1) >= len(a[swap_1]) or max(_s2) >= len(a[swap_2]):
                    pass
                else:
                    f_h, f_b, f_t = np.array(a[swap_1])[:min(_s1)], np.array(a[swap_1])[_s1], np.array(a[swap_1])[max(_s1) + 1:]
                    s_h, s_b, s_t = np.array(a[swap_2])[:min(_s2)], np.array(a[swap_2])[_s2], np.array(a[swap_2])[max(_s2) + 1:]

                    test_a = copy.deepcopy(a)
                    test_a[swap_1] = np.concatenate([f_h, s_b, f_t]).tolist()
                    test_a[swap_2] = np.concatenate([s_h, f_b, s_t]).tolist()

                    # intra operation
                    mat = intra_costs([info['agents'][swap_1].tolist()] + test_a[swap_1],
                                      [info['agents'][swap_2].tolist()] + test_a[swap_2])
                    sol_1, sol_2 = or_tools(mat[0]), or_tools(mat[1])

                    test_a[swap_1] = np.array(test_a[swap_1])[(np.array(sol_1[1:]) - 1)].tolist()
                    test_a[swap_2] = np.array(test_a[swap_2])[(np.array(sol_2[1:]) - 1)].tolist()

                    cost, s_routes = seq_solver(info['grid'], info['agents'], test_a, [1, 1.2])

                    if cost == 'error':
                        pass
                    else:
                        output[0].append(cost), output[1].append(test_a)

        # improved
        if before_cost > min(output[0]):
            before_cost = min(output[0])
            a = output[1][np.argmin(output[0])]
            explore = False

        # local optima
        elif (len(results) >= 3) and all(np.array(results[len(results) - 3:]) == results[-1]):
            explore = True

            # 'perturbation - 1a'
            # np.random.shuffle(a[swap_1]), np.random.shuffle(a[swap_2])
            #
            # 'perturbation - 1b'
            # a[swap_1][-1], a[swap_1][-2] = a[swap_1][-2], a[swap_1][-1]
            # a[swap_2][-1], a[swap_2][-2] = a[swap_2][-2], a[swap_2][-1]
            #
            # 'perturbation - 2a'
            # others = list(set(range(len(info['agents']))) - {swap_1, swap_2})
            # o = np.random.choice(others)
            # np.random.shuffle(a[o])
            #
            # 'perturbation - 2a'
            # others = list(set(range(len(info['agents']))) - {swap_1, swap_2})
            # o = np.random.choice(others)

            # o_1, o_2 = np.random.choice(others, 2, replace=False)
            # a[swap_1][-1], a[o_1][-1] = a[o_1][-1], a[swap_1][-1]
            # a[swap_2][-1], a[o_2][-1] = a[o_2][-1], a[swap_2][-1]
            # o = np.random.choice(others)
            # a[o][-1], a[o][-2] = a[o][-2], a[o][-1]
            # o_idx = np.random.choice(range(len(a[o])), 2, replace=False)
            # a[o][o_idx[0]], a[o][o_idx[1]] = a[o][o_idx[1]], a[o][o_idx[0]]

        results.append(before_cost)
        print('\n', results)
    return results, check_swap


def LNS(info):
    task_idx, assign = info['lns_assign'][0], info['lns_assign'][1]
    init_cost = info['init_cost']
    results = []

    for itr in range(100):

        removal_idx = removal(task_idx, info['tasks'], info['graph'], N=2)
        for i, t in enumerate(assign.values()):
            for r in removal_idx:
                if {r: info['tasks'][r]} in t:
                    assign[i].remove({r: info['tasks'][r]})

        while len(removal_idx) != 0:
            f = f_ijk(assign, info['agents'], removal_idx, info['tasks'], info['graph'])
            regret = get_regret(f)
            regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
            re_ins = list(regret.keys())[0]
            re_a, re_j = regret[re_ins][1], regret[re_ins][2]
            removal_idx.remove(re_ins)
            to_insert = {re_ins: info['tasks'][re_ins]}
            assign[re_a].insert(re_j, to_insert)

        cost, _ = seq_solver(info['grid'], info['agents'], to_solver(info['tasks'], assign), [1, 1.2])
        if cost == 'error':
            pass
        else:
            results.append(cost)

    return results


if __name__ == '__main__':
    np.random.seed(42)
    mode = 'exp'
    # mode = 'plot'

    if mode == 'exp':
        exp_num = 1
        curr_dir = os.path.realpath(__file__)
        solver_dir = os.path.join(Path(curr_dir).parent, 'EECBS/eecbs')
        save_dir = os.path.join(Path(curr_dir).parent, 'EECBS/exp_{}/'.format(exp_num))
        exp_name = 'eval_select'

        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print("Error: Cannot create the directory.")

        scenario = load_scenarios('202020_5_25/scenario_{}.pkl'.format(exp_num))
        info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
        assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
        info['ce_assign'] = to_solver(info['tasks'], assign)
        info['lns_assign'] = (assign_id, assign)
        info['init_cost'], info['init_routes'] = seq_solver(info['grid'], info['agents'], info['ce_assign'], [1, 1.2])

        ce_results, ce_id = CE(info)
        lns_results = LNS(info)

        with open('exp_opt_{}.pkl'.format(exp_num), 'wb') as f:
            pickle.dump([ce_results, lns_results], f)

        try:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        except OSError:
            print("Error: Cannot remove the directory.")

    elif mode == 'plot':
        plot_num = 1
        with open('exp_opt_{}.pkl'.format(plot_num), 'rb') as f:
            data = pickle.load(f)
        plt.plot(np.arange(100), data[0], label='cross exchange')
        plt.plot(np.arange(100), data[1], label='lns')
        plt.xlabel('iteration'), plt.ylabel('route length')
        plt.legend(loc='upper right')
        plt.savefig('202020_5_15_{}.png'.format(plot_num))

    else:
        raise NotImplementedError
