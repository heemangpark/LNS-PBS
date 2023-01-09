import os
import subprocess
from copy import deepcopy

import torch

from utils.vis_graph import vis_ta

import numpy as np

from nn.ag_util import convert_dgl
from nn.agent import Agent
from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory

VISUALIZE = False
solver_path = "EECBS/"
M, N = 10, 10
if not os.path.exists('scenarios/323220_1_{}_{}/'.format(M, N)):
    save_scenarios(size=32, M=M, N=N)

scenario_name = 'test1'
# scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(M, N))
# grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
# save_map(grid, scenario_name)
# vis_dist(graph, agent_pos, total_tasks)


agent = Agent(batch_size=3)

for e in range(10000):
    save_scenarios(size=32, M=M, N=N)
    scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(M, N))
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
    save_map(grid, scenario_name)

    itr = 0
    episode_timestep = 0

    agent_traj = []
    # `task_finished` defined for each episode
    task_finished_bef = np.array([False for _ in range(N)])
    di_dgl_g, bipartite_g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks, agent_traj,
                                                                            task_finished_bef)
    joint_action = []

    while True:
        """ 1.super-agent coordinates agent&task pairs """
        # `task_selected` initialized as the `task_finished` to jointly select task at each event
        task_selected = deepcopy(task_finished_bef)
        curr_tasks_solver = []
        agent_pos_solver = []
        selected_ag_idx, joint_action = agent(di_dgl_g, bipartite_g, task_finished_bef, ag_node_indices,
                                              task_node_indices)

        # convert action to solver format
        for ag_idx, action in zip(selected_ag_idx, joint_action):
            task_node_idx = task_node_indices[action]
            # convert action to solver format (task pos)
            _x = di_dgl_g.nodes[task_node_idx].data['x'].item()
            _y = di_dgl_g.nodes[task_node_idx].data['y'].item()

            agent_pos_solver.append(agent_pos[ag_idx])
            curr_tasks_solver.append([[_x, _y]])

        # non-selected agents
        for ag_idx in set(range(M)) - set(selected_ag_idx):
            agent_pos_solver.append(agent_pos[ag_idx])
            curr_tasks_solver.append([agent_pos[ag_idx]])

        # visualize
        if VISUALIZE:
            vis_ta(graph, agent_pos_solver, curr_tasks_solver, str(itr) + "_assigned")

        """ 2.pass created agent-task pairs to low level solver """
        # convert action to the solver input formation
        save_scenario(agent_pos_solver, curr_tasks_solver, scenario_name, grid.shape[0], grid.shape[1])

        # Run solver
        c = [solver_path + "eecbs",
             "-m",
             solver_path + scenario_name + '.map',
             "-a",
             solver_path + scenario_name + '.scen',
             "-o",
             solver_path + scenario_name + ".csv",

             "--outputPaths",
             solver_path + scenario_name + "_paths.txt",
             "-k", "{}".format(len(agent_pos_solver)), "-t", "1", "--suboptimality=1.1"]

        # process_out.stdout format
        # runtime, num_restarts, num_expanded, num_generated, solution_cost, min_sum_of_costs, avg_path_length
        process_out = subprocess.run(c, capture_output=True)
        text_byte = process_out.stdout.decode('utf-8')
        sum_costs = int(text_byte.split('Succeed,')[-1].split(',')[-3])

        # Read solver output
        agent_traj = read_trajectory(solver_path + scenario_name + "_paths.txt")
        agent_traj = agent_traj[:len(selected_ag_idx)]

        # Mark finished agent, finished task
        next_t = np.min([len(t) for t in agent_traj])

        finished_ag = np.array([len(t) for t in agent_traj]) == next_t  # as more than one agent may finish at a time

        finished_task_idx = np.array(joint_action)[finished_ag]
        task_finished_aft = deepcopy(task_finished_bef)
        task_finished_aft[finished_task_idx] = True
        episode_timestep += next_t

        # overwrite output
        agent_pos_new = deepcopy(agent_pos)
        for i, ag in enumerate(selected_ag_idx):
            agent_pos_new[ag] = agent_traj[i][next_t - 1]

        agent_pos = agent_pos_new

        # Replay memory 에 transition 저장. Agent position 을 graph 의 node 형태로
        # NOTE: solver out cost == 아래의 cost - M
        # costs = [len(t) for t in agent_traj]
        # print("itr:{}, cum_cost:{}, curr_complete_time:{}".format(itr, episode_timestep,
        #                                                           episode_timestep - next_t + sum_costs))

        # agent_traj = []
        terminated = all(task_finished_aft)
        agent.push(di_dgl_g, bipartite_g, ag_node_indices, task_node_indices, selected_ag_idx, joint_action,
                   deepcopy(task_finished_bef), next_t, terminated)

        if VISUALIZE:
            vis_ta(graph, agent_pos, curr_tasks_solver, str(itr) + "_finished")

        if terminated:
            fit_res = agent.fit()
            torch.save(agent.state_dict(), 'saved.th')
            print('E:{}, loss:{:.5f}, return:{}'.format(e, fit_res['loss'], episode_timestep))
            break

        task_finished_bef = task_finished_aft
        di_dgl_g, bipartite_g, ag_node_indices, _ = convert_dgl(graph, agent_pos, total_tasks, agent_traj,
                                                                task_finished_bef)

        # visualize
        itr += 1
