import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from modules_sl import *


if __name__ == '__main__':
    # set random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    parser.add_argument('--jobid', type = str, default = 'em', help = 'job id')
    parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results_sl'), help = 'path to store results')

    # environment parameters
    parser.add_argument('--num_items', type = int, default = 3, help = 'number of items')
    parser.add_argument('--num_targets', type = float, default = 6, help = 'number of targets')
    parser.add_argument('--t_delay', type = float, default = 1, help = 'delay time')

    args = parser.parse_args()

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')

    # initialize model
    net = torch.load(os.path.join(exp_path, f'net.pth'))

    # initialize dataset
    dataset = MemoryDataset(
        size = 0,
        num_items = args.num_items,
        num_targets = args.num_targets,
        t_delay = args.t_delay,
    )

    # initialize recording
    data = {
        'items': [],
        'hidden_seqs': [],
        'key_seqs': [],
        'value_seqs': [],
        'query_seqs': [],
    }

    # simulate
    num_trials = 10000
    net.eval()
    with torch.no_grad():
        for _ in range(num_trials):
            inputs, targets = dataset.generate_data()
            inputs = inputs.unsqueeze(0) # add batch dimension (1, seq_len, input_size)

            outputs, hiddens = net(inputs) # (batch_size, seq_len, output_size/hidden_size)

            keys = torch.cat([net.fc_key(hiddens[:, i, :]) for i in range(hiddens.shape[1])]) # (seq_len, key_size)
            queries = torch.cat([net.fc_query(hiddens[:, i, :]) for i in range(hiddens.shape[1])]) # (seq_len, query_size)

            # record data
            data['items'].append(dataset.items.numpy()) # (num_items,)
            data['hidden_seqs'].append(hiddens.squeeze(0).numpy()) # (seq_len, hidden_size)
            data['key_seqs'].append(keys.numpy()) # (seq_len, key_size)
            data['query_seqs'].append(queries.numpy()) # (seq_len, key_size)
    
    with open(os.path.join(exp_path, f'data_simulation.p'), 'wb') as f:
        pickle.dump(data, f)
