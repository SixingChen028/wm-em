import numpy as np
import random
import pickle
import torch


class Simulator:
    """
    A simulator.
    """

    def __init__(self, net, env):
        """
        Construct an simulator.
        """
        self.net = net
        self.env = env

        # reset simulator
        self.reset()
    

    def reset(self):
        """
        Reset simulator.
        """
        # reset environment
        self.env.reset()

        # reset data
        self.data = {
            'items': [],
            'action_seqs': [],
        }
    

    def simulate(
            self,
            num_trial,
            greedy = False,
            include_hidden = True,
            include_logits = False,
            include_policy = False,
        ):
        """
        Simulate data
        """

        # reset simulator
        self.reset()

        # add additional keys
        if include_hidden:
            self.data['hidden_seqs'] = []
        if include_logits:
            self.data['logits_seqs'] = []
        if include_policy:
            self.data['policy_seqs'] = []
        
        # get net type
        net_type = type(self.net).__name__

        # iterate through trials
        for _ in range(num_trial):

            # initialize trial recordings
            action_seq_ep = []
            if include_hidden:
                hidden_seq_ep = []
            if include_logits:
                logits_seq_ep = []
            if include_policy:
                policy_seq_ep = []

            # initialize a trial
            done = False
            states = None

            # reset environment
            obs, info = self.env.reset()
            obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
            action_mask = torch.tensor(info['mask']) # (action_dim,)

            with torch.no_grad():
                # iterate through a trial
                while not done:

                    # step the net
                    action, policy, log_prob, entropy, value, states = self.net(
                        obs, states, action_mask
                    )
                    if greedy:
                        action = torch.argmax(policy)

                    # step the env
                    obs, reward, done, truncated, info = self.env.step(action.item())
                    obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
                    action_mask = torch.tensor(info['mask']) # (action_dim,)

                    # record results for the timestep
                    action_seq_ep.append(int(action))
                    if include_hidden:
                        hidden_seq_ep.append(self.process_hidden(states, net_type))
                    if include_logits:
                        logits_seq_ep.append(self.net.policy_net.logits.squeeze().tolist())
                    if include_policy:
                        policy_seq_ep.append(policy.squeeze().tolist())

                # record results for the trial
                self.data['items'].append(self.env.items)
                self.data['action_seqs'].append(action_seq_ep)
                if include_hidden:
                    self.data['hidden_seqs'].append(hidden_seq_ep)
                if include_logits:
                    self.data['logits_seqs'].append(logits_seq_ep)
                if include_policy:
                    self.data['policy_seqs'].append(policy_seq_ep)
    

    def process_hidden(self, states, net_type):
        """
        Get hidden state.
        """
        hidden_processed = states.squeeze().tolist() # (num_hidden,)
        
        return hidden_processed


    def pull(self, index, *keys):
        """
        Pull data according to keys.
        """
        return [self.data[key][index] for key in keys]
    

    def save_data(self, path):
        """
        Save data.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)


    def load_data(self, path):
        """
        Load data.
        """
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

