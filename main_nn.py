import os
import subprocess
import random
import torch
import wandb
import numpy as np

from collections import deque
from copy import deepcopy
from datetime import datetime
from nn.ag_util import convert_dgl
from nn.agent import Agent
from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory
from utils.vis_graph import vis_ta

exp_name = datetime.now().strftime("%Y%m%d_%H%M")
scenario_name = exp_name

VISUALIZE = True
solver_path = "EECBS/"
M, N = 10, 20
T_threshold = 10  # N step fwd
if not os.path.exists('scenarios/323220_1_{}_{}/'.format(M, N)):
    save_scenarios(size=32, M=M, N=N)

agent = Agent(batch_size=3)
avg_return = deque(maxlen=50)

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
    g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks, task_finished_bef)
    joint_action_prev = np.array([0] * M)
    ag_order = np.arange(M).reshape(1, -1)
    continuing_ag = np.array([False for _ in range(M)])

    while True:
        print(itr)
        """ 1.super-agent coordinates agent&task pairs """
        # `task_selected` initialized as the `task_finished` to jointly select task at each event
        task_selected = deepcopy(task_finished_bef)
        curr_tasks = [[] for _ in range(M)]  # in order of init agent idx
        joint_action = agent(g, ag_order, continuing_ag, joint_action_prev)
        # ordered_joint_action = np.empty(shape=(M,), dtype=int)
        ordered_joint_action = [0] * M

        # convert action to solver format
        # TODO: batch

        for ag_idx, action in zip(ag_order[0], joint_action):
            if action < N:
                task_node_idx = action + M
                task_loc = g.nodes[action + M].data['original_loc'].squeeze().tolist()
            else:
                task_loc = agent_pos[ag_idx].tolist()

            curr_tasks[ag_idx] = [task_loc]
            ordered_joint_action[ag_idx] = action

        # visualize
        if VISUALIZE:
            vis_ta(graph, agent_pos, curr_tasks, str(itr) + "_assigned", total_tasks=total_tasks,
                   task_finished=task_finished_bef)

        """ 2.pass created agent-task pairs to low level solver """
        # convert action to the solver input formation
        save_scenario(agent_pos, curr_tasks, scenario_name, grid.shape[0], grid.shape[1])

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
             "-k", str(M), "-t", "1", "--suboptimality=1.1"]

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
        T = np.array([len(t) for t in agent_traj])

        # TODO makespan -> sum of cost
        # TODO batch training loop

        # Mark finished agent, finished task
        next_t = T[T > 1].min()

        finished_ag = T == next_t  # as more than one agent may finish at a time
        finished_task_idx = np.array(ordered_joint_action)[finished_ag]
        task_finished_aft = deepcopy(task_finished_bef)
        task_finished_aft[finished_task_idx] = True
        episode_timestep += next_t

        # overwrite output
        agent_pos_new = deepcopy(agent_pos)
        for ag_idx in ag_order[0]:
            if T[ag_idx] > 1:
                agent_pos_new[ag_idx] = agent_traj[ag_idx][next_t - 1]

        agent_pos = agent_pos_new
        # Replay memory 에 transition 저장. Agent position 을 graph 의 node 형태로
        terminated = all(task_finished_aft)

        # TODO: training detail
        agent.push(g, ag_node_indices, task_node_indices, ordered_joint_action, ag_order,
                   deepcopy(task_finished_bef), next_t, terminated)

        if VISUALIZE:
            vis_ta(graph, agent_pos, curr_tasks, str(itr) + "_finished", total_tasks=total_tasks,
                   task_finished=task_finished_aft)

        if terminated:
            avg_return.append(episode_timestep)
            torch.save(agent.state_dict(), 'saved/{}.th'.format(exp_name))
            fit_res = agent.fit(baseline=sum(avg_return) / len(avg_return))
            print('E:{}, loss:{:.5f}, return:{}'.format(e, fit_res['loss'], episode_timestep))
            wandb.log({'loss': fit_res['loss'], 'return': episode_timestep})
            break

        # agent with small T maintains previous action
        continuing_ag = (0 < T - next_t) * (T - next_t < T_threshold)
        continuing_ag_idx = continuing_ag.nonzero()[0].tolist()
        remaining_ag = list(set(range(M)) - set(continuing_ag_idx))
        random.shuffle(remaining_ag)

        ag_order = np.array([continuing_ag_idx + remaining_ag])
        joint_action_prev = np.array(ordered_joint_action, dtype=int)

        task_finished_bef = task_finished_aft
        g, ag_node_indices, _ = convert_dgl(graph, agent_pos, total_tasks, task_finished_bef)
        itr += 1
