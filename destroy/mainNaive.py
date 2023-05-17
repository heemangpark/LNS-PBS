import os
import pickle
import random
import sys

import dgl
import numpy as np
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from nn.destroyNaive import DestroyNaive

if __name__ == '__main__':
    train = True
    if train:
        random.seed(42)
        model = DestroyNaive()

        wandb = False
        if wandb:
            import wandb

            wandb.init()

        epochs = 100
        data_size = 10000
        batch_size = 100
        batch_num = data_size // batch_size
        batch_config = [data_size, batch_size, batch_num]
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        for e in range(epochs):
            shuffle_data = random.sample(range(data_size), data_size)

            batch_loss = 0
            for b_num in range(batch_num):
                train_idx = shuffle_data[b_num * batch_size:(b_num + 1) * batch_size]
                batch_graph, batch_destroy = [], []

                for t_id in train_idx:
                    with open('data/dataDestroy_{}.pkl'.format(t_id), 'rb') as f:
                        graph, destroy = pickle.load(f)

                    batch_graph.append(graph)
                    batch_destroy.append(destroy)

                graphs = dgl.batch(batch_graph).to(device)
                destroys = batch_destroy

                loss = model.learn(graphs, destroys, batch_graph, batch_config)
                batch_loss += loss

            if e % 10 == 0:
                torch.save(model.state_dict(), 'models/destroyNaiveL1_{}.pt'.format(e))

            if wandb:
                wandb.log({'epoch_loss': batch_loss / batch_num})
            else:
                print('Epoch {} || Loss: {}'.format(e, batch_loss / batch_num))

    else:
        random.seed(42)
        model = DestroyNaive()
        model.load_state_dict(torch.load('destroyNaive_0.pt'))
        model.eval()
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        for idx in trange(100):
            with open('data/evalDestroy_{}.pkl'.format(idx), 'rb') as f:
                graph, destroy = pickle.load(f)

            pred = model.act(graph, destroy).reshape(10, 10)
            cost = np.array([v for v in destroy.values()]).reshape(10, 10) / 64

            print(pred, cost)
