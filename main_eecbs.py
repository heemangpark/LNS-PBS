import subprocess
import os

from utils.generate_scenarios import load_scenarios, save_scenarios
from utils.solver_util import save_map, save_scenario
from utils.vis_graph import vis_dist

M, N = 50, 50
if not os.path.exists('scenarios/323220_1_{}_{}/'.format(N, M)):
    save_scenarios(size=32, M=M, N=N)

scenario = load_scenarios('323220_1_{}_{}/scenario_1.pkl'.format(N, M))
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
vis_dist(graph, agent_pos, total_tasks)

scenario_name = 'test1'
save_map(grid, scenario_name)
save_scenario(agent_pos, total_tasks, scenario_name, grid.shape[0], grid.shape[1])

solver_path = "EECBS/"
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

f = open(solver_path + scenario_name + "_paths.txt", 'r')
