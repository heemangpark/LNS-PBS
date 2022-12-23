import pickle

with open('data/1010grid_3', 'rb') as f:
    grid = pickle.load(f)

with open('data/1010graph_1', 'rb') as f:
    graph = pickle.load(f)

with open('data/1010AP_1', 'rb') as f:
    agent = pickle.load(f)

with open('data/1010TA_1', 'rb') as f:
    task = pickle.load(f)

print('d')
