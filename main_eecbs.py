import os
import subprocess

from nn.agent import Agent
from nn.ag_util import convert_dgl
from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory
from utils.vis_graph import vis_dist

solver_path = "EECBS/"
M, N = 10, 10
# if not os.path.exists('scenarios/323220_1_{}_{}/'.format(M, N)):
# save_scenarios(size=32, M=M, N=N)

scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(M, N))
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
vis_dist(graph, agent_pos, total_tasks)

scenario_name = 'test1'
save_map(grid, scenario_name)
total_tasks_bef = total_tasks

ag = Agent()

"""
TODO total_tasks format
태스크 전체 (순서대로) 쭉 나열된 일종의 task 집합 -> 'agent to task' 할당 집합으로 바꿔야 함 (main.py에서 tasks에 해당)
"""
itr = 0

di_dgl_g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks_bef, [])
selected_task = [False for _ in range(N)]
while True:
    if itr == 0:
        total_tasks_bef = []
        for ag_idx in ag_node_indices:
            action = ag(di_dgl_g, ag_idx, task_node_indices, selected_task)
            task_node_idx = task_node_indices[action]

            # convert action to executable format (task pos)
            _x = di_dgl_g.nodes[task_node_idx].data['x'].item()
            _y = di_dgl_g.nodes[task_node_idx].data['y'].item()
            total_tasks_bef.append([_x, _y])
            selected_task[action] = True
    else:
        raise NotImplementedError
        total_tasks_bef = []

    save_scenario(agent_pos, total_tasks_bef, scenario_name, grid.shape[0], grid.shape[1])
    # c = [solver_path + "eecbs",
    #      "-m",
    #      solver_path + scenario_name + '.map',
    #      "-a",
    #      solver_path + scenario_name + '.scen',
    #      "-o",
    #      solver_path + scenario_name + ".csv",
    #      "--outputPaths",
    #      solver_path + scenario_name + "_paths.txt",
    #      "-k", "{}".format(M), "-t", "60", "--suboptimality=1.2"]
    # subprocess.run(c)
    # agent_traj = read_trajectory(solver_path + scenario_name + "_paths.txt")

    # Agent position 을 index 로 저장
    agent_traj = []

    di_dgl_g, ag_node_indices, task_node_indices = convert_dgl(graph, agent_pos, total_tasks_bef, agent_traj)
    total_tasks_after = ag(di_dgl_g, ag_node_indices[0], task_node_indices)
    total_tasks_bef = total_tasks_after

    costs = [len(t) for t in agent_traj]
    print("itr:{}, cost:{}".format(itr, sum(costs)))
    itr += 1
