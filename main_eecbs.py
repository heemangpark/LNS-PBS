import os, sys
import subprocess

sys.path.append(os.path.abspath('EECBS'))

solver_path = "EECBS/"
c = [solver_path + "eecbs",
     "-m",
     solver_path + "random-32-32-20.map",
     "-a",
     solver_path + "random-32-32-20-random-1.scen",
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
