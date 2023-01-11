import subprocess
from collections import deque
from copy import deepcopy
from datetime import datetime
import dgl

import numpy as np
import torch

from nn.ag_util import convert_dgl
from nn.agent import Agent
from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory
from utils.vis_graph import vis_ta

exp_name = datetime.now().strftime("%Y%m%d_%H%M")
scenario_name = exp_name

VISUALIZE = False
solver_path = "EECBS/"
M, N = 10, 20

agent = Agent()
avg_return = deque(maxlen=50)
n_batch = 7

for e in range(10000):
    save_scenarios(size=32, M=M, N=N, itr=n_batch)

    g_batch = []
    bipartite_g_batch = []
    ag_node_indices_batch = []
    task_node_indices_batch = []

    task_finished_bef = np.full((n_batch, N), False)

    for i in range(n_batch):
        scenario = load_scenarios('323220_1_{}_{}/scenario_{}.pkl'.format(M, N, i + 1))
        grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
        save_map(grid, scenario_name + "_" + str(i))

        di_dgl_g, bipartite_g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks, [],
                                                                                task_finished_bef[i])
        g_batch.append(di_dgl_g)
        bipartite_g_batch.append(bipartite_g)
        ag_node_indices_batch.append(ag_node_indices)
        task_node_indices_batch.append(task_node_indices)

    g_size = [g.number_of_nodes() for g in g_batch]
    g_size = np.array(g_size)
    cumsum_g_size = np.cumsum(g_size) - g_size[0]

    g_batch = dgl.batch(g_batch)
    bipartite_g_batch = dgl.batch(bipartite_g_batch)
    ag_node_indices_batch = np.stack(ag_node_indices_batch)  # shape = (batch, M)
    task_node_indices_batch = np.stack(task_node_indices_batch)  # shape = (batch, N)

    # convert index into batched graph index
    ag_node_indices_batch = ag_node_indices_batch + cumsum_g_size.reshape(-1, 1)
    task_node_indices_batch = task_node_indices_batch + cumsum_g_size.reshape(-1, 1)

    itr = 0
    episode_timestep = 0

    agent_traj = []
    # `task_finished` defined for each episode
    joint_action = []

    while True:
        """ 1.super-agent coordinates agent & task pairs """
        # `task_selected` initialized as the `task_finished` to jointly select task at each event
        task_selected = deepcopy(task_finished_bef)
        selected_ag_idx, joint_action = agent(g_batch, bipartite_g_batch, task_finished_bef, ag_node_indices_batch,
                                              task_node_indices_batch)

        agent_pos_solver_batch = [[] * n_batch]
        curr_tasks_solver_batch = [[] * n_batch]
        # convert action to solver format

        for b in range(n_batch):
            for _ in range(N):
                # DEBUG ongoing
                pass

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
        try:
            sum_costs = int(text_byte.split('Succeed,')[-1].split(',')[-3])
        except:
            agent.replay_memory.memory = []
            break

        # Read solver output
        agent_traj = read_trajectory(solver_path + scenario_name + "_paths.txt")
        agent_traj = agent_traj[:len(selected_ag_idx)]

        # TODO makespan -> sum of cost
        # TODO batch training loop

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
            vis_ta(graph, agent_pos[selected_ag_idx + list(set(range(M)) - set(selected_ag_idx))], curr_tasks_solver,
                   str(itr) + "_finished")

        if terminated:
            avg_return.append(episode_timestep)
            torch.save(agent.state_dict(), 'saved/{}.th'.format(exp_name))
            fit_res = agent.fit(baseline=sum(avg_return) / len(avg_return))
            print('E:{}, loss:{:.5f}, return:{}'.format(e, fit_res['loss'], episode_timestep))
            break

        task_finished_bef = task_finished_aft
        di_dgl_g, bipartite_g, ag_node_indices, _ = convert_dgl(graph, agent_pos, total_tasks, [],
                                                                task_finished_bef)
        itr += 1
