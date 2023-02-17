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
from utils.generate_scenarios import load_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory, compute_astar
from utils.vis_graph import vis_ta

solver_path = "EECBS/"
dec_solver_path = "DecAstar/"


def run_episode(agent, M, N, exp_name, T_threshold, sample=True, scenario_dir=None, VISUALIZE=False, heuristic=False,
                n_sample=1):
    scenario = load_scenarios(scenario_dir)
    task_finished_bef = np.array([False for _ in range(N)])
    grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
    save_map(grid, exp_name)
    shortest_paths = compute_astar(agent_pos, total_tasks, exp_name, task_finished_bef)

    itr = 0
    episode_timestep = 0

    # `task_finished` defined for each episode
    g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks, task_finished_bef, shortest_paths)
    joint_action_prev = np.array([0] * M)
    ag_order = np.arange(M)
    continuing_ag = np.array([False for _ in range(M)])

    while True:
        """ 1.super-agent coordinates agent&task pairs """
        # `task_selected` initialized as the `task_finished` to jointly select task at each event
        best_T = None
        best_ordered_joint_action = None
        best_agent_traj = None
        best_curr_tasks = None
        best_joint_action = None
        max_T = 100

        for _ in range(n_sample):
            curr_tasks = [[] for _ in range(M)]  # in order of init agent idx
            if heuristic:
                joint_action = agent.forward_heuristic(g, ag_order, continuing_ag, joint_action_prev, sample=sample)
            else:
                joint_action = agent(g, ag_order, continuing_ag, joint_action_prev, sample=sample)
            ordered_joint_action = [0] * M

            # convert action to solver format
            # TODO: batch

            for ag_idx, action in zip(ag_order, joint_action):
                if action < N:
                    task_loc = g.nodes[action + M].data['original_loc'].squeeze().tolist()
                else:
                    task_loc = agent_pos[ag_idx].tolist()

                curr_tasks[ag_idx] = [task_loc]
                ordered_joint_action[ag_idx] = action

            """ 2.pass created agent-task pairs to low level solver """
            # convert action to the solver input formation
            save_scenario(agent_pos, curr_tasks, exp_name, grid.shape[0], grid.shape[1])

            # Run solver
            c = [solver_path + "eecbs",
                 "-m",
                 solver_path + exp_name + '.map',
                 "-a",
                 solver_path + exp_name + '.scen',
                 "-o",
                 solver_path + exp_name + ".csv",
                 "--outputPaths",
                 solver_path + exp_name + "_paths.txt",
                 "-k", str(M), "-t", "1", "--suboptimality=1.1"]

            # process_out.stdout format
            # runtime, num_restarts, num_expanded, num_generated, solution_cost, min_sum_of_costs, avg_path_length
            process_out = subprocess.run(c, capture_output=True)
            text_byte = process_out.stdout.decode('utf-8')
            try:
                sum_costs = int(text_byte.split('Succeed,')[-1].split(',')[-3])
            except:
                agent.replay_memory.memory = []
                return None, itr

            if itr > N:
                return None, itr

            # Read solver output
            agent_traj = read_trajectory(solver_path + exp_name + "_paths.txt")
            T = np.array([len(t) for t in agent_traj])

            if max_T > max(T):
                best_T = T
                best_curr_tasks = curr_tasks
                best_agent_traj = agent_traj
                best_ordered_joint_action = ordered_joint_action
                max_T = max(T)
                best_joint_action = joint_action

        # TODO makespan -> sum of cost
        # TODO batch training loop

        # Mark finished agent, finished task
        next_t = best_T[best_T > 1].min()

        finished_ag = (best_T == next_t) * (
                np.array(best_ordered_joint_action) < N)  # as more than one agent may finish at a time
        finished_task_idx = np.array(best_ordered_joint_action)[finished_ag]
        task_finished_aft = deepcopy(task_finished_bef)
        task_finished_aft[finished_task_idx] = True
        episode_timestep += next_t

        # overwrite output
        agent_pos_new = deepcopy(agent_pos)
        for ag_idx in ag_order:
            if best_T[ag_idx] > 1:
                agent_pos_new[ag_idx] = best_agent_traj[ag_idx][next_t - 1]

        # Replay memory 에 transition 저장. Agent position 을 graph 의 node 형태로
        terminated = all(task_finished_aft)

        # TODO: training detail
        if sample:
            agent.push(g, best_ordered_joint_action, ag_order, deepcopy(task_finished_bef), next_t, terminated)

        if VISUALIZE:
            # visualize
            vis_ta(graph, agent_pos, best_curr_tasks, str(itr) + "_assigned", total_tasks=total_tasks,
                   task_finished=task_finished_bef)
            vis_ta(graph, agent_pos_new, best_curr_tasks, str(itr) + "_finished", total_tasks=total_tasks,
                   task_finished=task_finished_aft)

        if terminated:
            return episode_timestep, itr

        # agent with small T maintains previous action
        continuing_ag = (0 < best_T - next_t) * (best_T - next_t < T_threshold)
        continuing_ag_idx = continuing_ag.nonzero()[0].tolist()
        remaining_ag = list(set(range(M)) - set(continuing_ag_idx))

        # option 1. randomly select remaining ag
        # random.shuffle(remaining_ag)

        # option 2. sort remaining ag by remaining task dist
        dists = g.edata['dist'].reshape(-1, M).T
        finished = task_finished_aft.nonzero()[0]
        reserved = np.array(best_joint_action)[continuing_ag_idx]
        dists[:, finished] = 9
        dists[:, reserved] = 9

        # option 2-1 sort by mean
        remaining_ag_dist = dists[remaining_ag].mean(-1)
        # option 2-2 sort by min
        # remaining_ag_dist = dists[remaining_ag, :-1].min(-1).values
        remaining_order = remaining_ag_dist.sort().indices
        remaining_ag_idx = np.array(remaining_ag)[remaining_order].tolist()
        if type(remaining_ag_idx) == int:
            remaining_ag_idx = [remaining_ag_idx]
        # ========================
        ag_order = np.array(continuing_ag_idx + remaining_ag_idx)
        assert len(set(ag_order)) == M
        joint_action_prev = np.array(best_ordered_joint_action, dtype=int)

        agent_pos = agent_pos_new
        task_finished_bef = task_finished_aft
        shortest_paths = compute_astar(agent_pos, total_tasks, exp_name, task_finished_bef)
        g, ag_node_indices, _ = convert_dgl(graph, agent_pos, total_tasks, task_finished_bef, shortest_paths)
        itr += 1


if __name__ == '__main__':
    from tqdm import tqdm

    epoch = 1000
    sample_per_epoch = 50

    M, N = 10, 20
    T_threshold = 10  # N step fwd
    agent = Agent()
    # agent.load_state_dict(torch.load('saved/20230201_1611.th'))
    n_eval = 20
    best_perf = 1000000

    exp_name = datetime.now().strftime("%Y%m%d_%H%M")
    wandb.init(project='etri-mapf', entity='curie_ahn', name=exp_name + "_20")

    for e in range(epoch):
        epoch_perf = []
        epoch_loss = []
        epoch_itr = []
        eval_performance = []
        eval_itr = []

        # train
        for sample_idx in tqdm(range(sample_per_epoch)):
            scenario_dir = '323220_1_{}_{}/scenario_{}.pkl'.format(M, N, sample_idx + 1)
            episode_timestep, itr = run_episode(agent, M, N, exp_name, T_threshold, sample=True,
                                                scenario_dir=scenario_dir)
            if episode_timestep is not None:
                fit_res = agent.fit()
                epoch_perf.append(episode_timestep)
                epoch_itr.append(itr)
                epoch_loss = fit_res['loss']

        #
        for i in tqdm(range(n_eval)):
            scenario_dir = '323220_1_{}_{}_eval/scenario_{}.pkl'.format(M, N, i + 1)
            episode_timestep, itr = run_episode(agent, M, N, exp_name, T_threshold, sample=False,
                                                scenario_dir=scenario_dir)
            if episode_timestep is not None:
                eval_performance.append(episode_timestep)
                eval_itr.append(itr)

        wandb.log({'epoch_loss_mean': np.mean(epoch_loss),
                   'epoch_cost_mean': np.mean(epoch_perf),
                   'e': e,
                   'eval_cost_mean': np.mean(eval_performance),
                   'epoch_itr_mean': np.mean(epoch_itr),
                   'eval_itr_mean': np.mean(eval_itr),
                   'n_sample': len(epoch_itr)})

        if np.mean(eval_performance) < best_perf:
            torch.save(agent.state_dict(), 'saved/{}.th'.format(exp_name))
            best_perf = np.mean(eval_performance)
