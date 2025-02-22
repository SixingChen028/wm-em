import os
import argparse
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from modules_rl import *


if __name__ == '__main__':

    # set random seed
    seed = 15
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    parser.add_argument('--jobid', type = str, default = 'em', help = 'job id')
    parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results_rl'), help = 'path to store results')

    # nework parameters
    parser.add_argument('--hidden_size', type = int, default = 32, help = 'hidden size')

    # environment parameters
    parser.add_argument('--num_items', type = int, default = 3, help = 'number of items')
    parser.add_argument('--num_targets', type = float, default = 6, help = 'number of targets')
    parser.add_argument('--t_delay', type = float, default = 1, help = 'delay time')


    # training parameters
    parser.add_argument('--num_episodes', type = int, default = 200000, help = 'training episodes')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'batch_size')
    parser.add_argument('--gamma', type = float, default = 0.9, help = 'temporal discount')
    parser.add_argument('--lamda', type = float, default = 1.0, help = 'generalized advantage estimation coefficient')
    parser.add_argument('--beta_v', type = float, default = 0.1, help = 'value loss coefficient')
    parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
    parser.add_argument('--max_grad_norm', type = float, default = 1.0, help = 'gradient clipping')

    args = parser.parse_args()

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')

    # load net
    net = torch.load(os.path.join(exp_path, f'net.pth'))

    # set environment
    env = IndexSerialRecallEnv(
        num_items = args.num_items,
        num_targets = args.num_targets,
        t_delay = args.t_delay,
        # seed = seeds[i],
    )

    # simulate
    num_trial = 10000
    simulator = Simulator(net = net, env = env)
    simulator.simulate(
        num_trial = num_trial,
        # include_hidden = True,
        # include_logits = True,
        # include_policy = True,
    )
    simulator.save_data(os.path.join(exp_path, f'data_simulation.p'))