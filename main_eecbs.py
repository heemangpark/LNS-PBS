import subprocess
import os

from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario, read_trajectory
from utils.vis_graph import vis_dist
from nn.agent import Agent

solver_path = "EECBS/"
M, N = 10, 10
if not os.path.exists('scenarios/323220_1_{}_{}/'.format(N, M)):
    save_scenarios(size=32, M=M, N=N)

scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(N, M))
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
vis_dist(graph, agent_pos, total_tasks)

scenario_name = 'test1'
save_map(grid, scenario_name)
total_tasks_bef = total_tasks

ag = Agent()

for i in range(10):
    save_scenario(agent_pos, total_tasks_bef, scenario_name, grid.shape[0], grid.shape[1])
    c = [solver_path + "eecbs",
         "-m",
         solver_path + scenario_name + '.map',
         "-a",
         solver_path + scenario_name + '.scen',
         "-o",
         solver_path + scenario_name + ".csv",
         "--outputPaths",
         solver_path + scenario_name + "_paths.txt",
         "-k", "{}".format(M), "-t", "60", "--suboptimality=1.2"]
    subprocess.run(c)
    agent_traj = read_trajectory(solver_path + scenario_name + "_paths.txt")
    costs = [len(t) for t in agent_traj]

    total_tasks_after = ag(graph, agent_pos, total_tasks_bef)
    total_tasks_bef = total_tasks_after

    print("cost:{}".format(sum(costs)))
