import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from modules import *


if __name__ == '__main__':
    # set random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    parser.add_argument('--jobid', type = str, default = '0', help = 'job id')
    parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results'), help = 'path to store results')

    # nework parameters
    parser.add_argument('--hidden_size', type = int, default = 32, help = 'hidden size')

    # environment parameters
    parser.add_argument('--num_items', type = int, default = 3, help = 'number of items')
    parser.add_argument('--num_targets', type = float, default = 6, help = 'number of targets')
    parser.add_argument('--t_delay', type = float, default = 1, help = 'delay time')

    # training parameters
    parser.add_argument('--dataset_size', type = int, default = 50000, help = 'dataset size')
    parser.add_argument('--num_epochs', type = int, default = 60, help = 'number of epochs')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch_size')

    args = parser.parse_args()

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # initialize model
    net = WorkingMemoryNet(
        input_size = 2,
        output_size = args.num_targets,
        hidden_size = args.hidden_size,
    )

    # initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = args.lr)

    # training loop
    for epoch in range(args.num_epochs):
        # initialize dataset
        dataloader = DataLoader(
            dataset = MemoryDataset(
                size = args.dataset_size,
                num_items = args.num_items,
                num_targets = args.num_targets,
                t_delay = args.t_delay,
            ),
            batch_size = args.batch_size,
            shuffle = True,
        )

        # training
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = criterion(
                outputs[:, -args.num_items:, :].reshape(-1, args.num_targets),
                targets[:, -args.num_items:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss.item():.4f}')
    
    # save net
    torch.save(net, os.path.join(exp_path, f'net.pth'))