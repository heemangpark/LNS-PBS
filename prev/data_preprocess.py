import os
from pathlib import Path


def preprocess():
    data_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'data/')
    file_list = [d[2] for d in os.walk(data_dir)][0]
    final_list = [file.split('final_')[1].split('.pkl')[0] for file in file_list if file.startswith('final')]

    for file in file_list:
        if file.startswith('prev_'):
            if file.split('prev_')[1].split('.pkl')[0] not in final_list:
                os.remove(data_dir + file)
        elif file.startswith('sch_space_'):
            if file.split('sch_space_')[1].split('.pkl')[0] not in final_list:
                os.remove(data_dir + file)

    idx = list(map(int, final_list))
    idx.sort()
    for id, sorted_id in zip(idx, range(len(idx))):
        os.rename(data_dir + 'prev_{}.pkl'.format(id), data_dir + 'prev_{}.pkl'.format(sorted_id))
        os.rename(data_dir + 'final_{}.pkl'.format(id), data_dir + 'final_{}.pkl'.format(sorted_id))
        os.rename(data_dir + 'sch_space_{}.pkl'.format(id), data_dir + 'sch_space_{}.pkl'.format(sorted_id))
