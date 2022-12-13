import os
import subprocess
import sys

sys.path.append(os.path.abspath('EECBS'))

solver_path = "EECBS/"
c = [solver_path + "eecbs",
     "-m", solver_path + "random-32-32-20.map",
     "-a", solver_path + "random-32-32-20-random-1.scen",
     "-o", solver_path + "test.csv",
     "--outputPaths=paths.txt",
     "-k", "50", "-t", "60", "--suboptimality=1.2"]

subprocess.run(c)
