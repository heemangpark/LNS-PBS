import os
import subprocess
import numpy as np

from nn.agent import Agent
from nn.ag_util import convert_dgl
from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory
from utils.vis_graph import vis_dist, vis_ta
from copy import deepcopy

solver_path = "EECBS/"
M, N = 5, 5
# if not os.path.exists('scenarios/323220_1_{}_{}/'.format(M, N)):
# save_scenarios(size=32, M=M, N=N)

scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(M, N))
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
# vis_dist(graph, agent_pos, total_tasks)

scenario_name = 'test1'
save_map(grid, scenario_name)

ag = Agent()

"""
TODO total_tasks format
태스크 전체 (순서대로) 쭉 나열된 일종의 task 집합 -> 'agent to task' 할당 집합으로 바꿔야 함 (main.py에서 tasks에 해당)
"""
itr = 0

agent_traj = []
# `task_finished` defined for each episode
task_finished = [False for _ in range(N)]
di_dgl_g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks, agent_traj)
joint_action = []

while not all(task_finished):
    print("ITR:{}".format(itr))
    if itr != 100:
        # `task_selected` initialized as the `task_finished` to jointly select task at each event
        task_selected = deepcopy(task_finished)
        total_tasks_solver = []
        agent_pos_solver = []
        joint_action = []
        # TODO: agent sequence
        for i, ag_node_idx in enumerate(ag_node_indices):
            if not all(task_selected):
                # index of task. action \in [1, ..., N]
                action = ag(di_dgl_g, ag_node_idx, task_node_indices, task_selected)
                task_node_idx = task_node_indices[action]

                # convert action to executable format (task pos)
                _x = di_dgl_g.nodes[task_node_idx].data['x'].item()
                _y = di_dgl_g.nodes[task_node_idx].data['y'].item()
                total_tasks_solver.append([[_x, _y]])
                agent_pos_solver.append(agent_pos[i])
                task_selected[action] = True
            else:
                action = None
                # maintain current position
                # total_tasks_bef.append([agent_pos[i]])
                # total_tasks_bef.append([agent_pos[i]])

            joint_action.append(action)
    else:
        raise NotImplementedError
        total_tasks_bef = []

    # visualize
    vis_ta(graph, agent_pos_solver, total_tasks_solver, str(itr)+"_assigned")

    # convert action to the solver input formation
    save_scenario(agent_pos_solver, total_tasks_solver, scenario_name, grid.shape[0], grid.shape[1])

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
         "-k", "{}".format(len(agent_pos_solver)), "-t", "60", "--suboptimality=1.1"]
    subprocess.run(c)

    # Read solver output
    agent_traj = read_trajectory(solver_path + scenario_name + "_paths.txt")

    # Mark finished agent, finished task
    finished_ag_idx = np.argmin([len(t) for t in agent_traj])
    next_t = np.min([len(t) for t in agent_traj])
    finished_task_idx = joint_action[finished_ag_idx]
    task_finished[finished_task_idx] = True

    # overwrite output
    agent_pos = [traj[next_t - 1] for traj in agent_traj]

    # Replay memory 에 transition 저장. Agent position 을 graph 의 node 형태로
    # TODO
    agent_traj = []

    di_dgl_g, ag_node_indices, _ = convert_dgl(graph, agent_pos, total_tasks, agent_traj)

    costs = [len(t) for t in agent_traj]
    print("itr:{}, cost:{}".format(itr, sum(costs)))
    # visualize
    vis_ta(graph, agent_pos, total_tasks_solver, str(itr)+"_finished")
    itr += 1
