import os
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from modules_rl import *


# note: vectorizing wrapper only works under this protection
if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    parser.add_argument('--jobid', type = str, default = 'em', help = 'job id')
    parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results_rl'), help = 'path to store results')

    # nework parameters
    parser.add_argument('--hidden_size', type = int, default = 32, help = 'hidden size')
    parser.add_argument('--key_size', type = int, default = 64, help = 'key size')

    # environment parameters
    parser.add_argument('--num_items', type = int, default = 3, help = 'number of items')
    parser.add_argument('--num_targets', type = float, default = 6, help = 'number of targets')
    parser.add_argument('--t_delay', type = float, default = 1, help = 'delay time')

    # training parameters
    parser.add_argument('--num_episodes', type = int, default = 80000, help = 'training episodes')
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
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # set environment
    seeds = [random.randint(0, 1000) for _ in range(args.batch_size)]
    env = gym.vector.AsyncVectorEnv([
        lambda: IndexSerialRecallEnv(
            num_items = args.num_items,
            num_targets = args.num_targets,
            t_delay = args.t_delay,
            # seed = seeds[i],
        )
        for i in range(args.batch_size)
    ])

    # set net
    net = KeyValueMemoryActorCriticPolicy(
        input_size = env.single_observation_space.shape[0],
        key_size = args.key_size,
        memory_size = args.num_targets,
        hidden_size = args.hidden_size,
    )

    # set model
    model = BatchMaskA2C(
        net = net,
        env = env,
        lr = args.lr,
        batch_size = args.batch_size,
        gamma = args.gamma,
        lamda = args.lamda,
        beta_v = args.beta_v,
        beta_e = args.beta_e,
        max_grad_norm = args.max_grad_norm,
    )

    # train network
    data = model.learn(
        num_episodes = args.num_episodes,
        print_frequency = 10
    )

    # save net and data
    model.save_net(os.path.join(exp_path, f'net.pth'))
    # model.save_data(os.path.join(exp_path, f'data_training.p'))

    # visualization
    plt.figure()
    plt.plot(np.array(data['episode_reward']).reshape(100, -1).mean(axis = 1))
    plt.show()