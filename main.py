import os
import pickle
import random
import sys
from pathlib import Path

import torch
import wandb as w
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from nn.agent import NeuroRepair

wandb = False
if wandb:
    w.init()

data_save_dir = os.path.join(Path(os.path.realpath(__file__)).parent, 'dataset/')
file_list = list(os.walk(data_save_dir))[0][-1]
file_tags = sorted([int(file.split('dataset')[1].split('.')[0]) for file in file_list])
sorted_file_list = ['dataset_{}.pkl'.format(f_id) for f_id in file_tags]

model = NeuroRepair()

for epoch in trange(100):
    random.shuffle(sorted_file_list)
    epoch_loss = 0

    for file in sorted_file_list:
        with open(data_save_dir + file, 'rb') as f:
            assign, decrement, graph, removal = pickle.load(f)

        loss = model.train(assign, decrement, graph, removal)
        epoch_loss += loss

        if wandb:
            w.log({'epoch': epoch,
                   'file_idx': int(file.split('_')[-1].split('.')[0]),
                   'loss': loss})

    if wandb:
        w.log({'epoch_loss': epoch_loss / len(sorted_file_list)})

    torch.save(model.state_dict(), 'NLNS_{}.pt'.format(epoch + 1))
