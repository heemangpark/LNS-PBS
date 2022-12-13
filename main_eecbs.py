import os
import subprocess
import sys
from utils.solver_util import save_map, save_scenario
from utils.generate_scenarios import load_scenarios, save_scenarios

save_scenarios(M=10, N=10)
scenario = load_scenarios('./scenarios/323220_1_10_10/scenario_2.pkl')
# scenario = load_scenarios('./instance_scenarios/16_16_0.2/scenario_1.pkl')
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]

directory = 'data/'
scenario_name = 'mapfile'

save_map(grid, directory + scenario_name)
save_scenario(agent_pos, total_tasks, directory+scenario_name, grid.shape[0], grid.shape[1])

sys.path.append(os.path.abspath('EECBS'))

solver_path = "EECBS/"
c = [solver_path + "eecbs",
     "-m",
     # solver_path + "random-32-32-20.map",
     directory + scenario_name + '.map',
     "-a",
     # solver_path + "random-32-32-20-random-1.scen",
     directory + scenario_name + '.scen',
     "-o",
     solver_path + "test.csv",
     "--outputPaths=paths.txt",
     "-k", "50", "-t", "60", "--suboptimality=1.2"]

subprocess.run(c)

# def invoke_EECBS(cmd):
#     subprocess.Popen(cmd)
#
#
# if __name__ == '__main__':
#     cmd = "./eecbs -m random-32-32-20.map -a random-32-32-20-random-1.scen -o test.csv --outputPaths=paths.txt -k 50 -t 60 --suboptimality=1.2 "
#     print('A')
#     invoke_EECBS(cmd)
