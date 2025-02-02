import numpy as np
import random
import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete


class ImmediateSerialRecallEnv(gym.Env):
    """
    A immediate serial recall environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_items = 3,
            num_targets = 6,
            t_delay = 1,
            seed = None,
        ):
        """
        Construct an environment.
        """

        self.num_items = num_items # number of items
        self.num_targets = num_targets # number of targets
        self.t_delay = t_delay # delay time

        # set random seed
        self.set_random_seed(seed)

        # initialize coordinates
        angles = np.linspace(0, 2 * np.pi, self.num_targets, endpoint = False) # radius = 1
        self.coordinates = np.column_stack((np.cos(angles), np.sin(angles)))

        # initialize action space
        self.action_space = Discrete(self.num_targets)

        # initialize observation space
        observation_shape = (
            2 + # coordinate (2,)
            1, # (stage)
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape,)


    def reset(self):
        """
        Reset the environment.
        """

        # reset timer
        self.t_elapsed = 0
        self.stage = 0

        # select items (shuffled by default) (without replacement)
        self.items = np.random.choice(self.num_targets, self.num_items, replace = False)

        # get observation
        item = self.items[self.t_elapsed]
        coordinate = self.coordinates[item]
        
        obs = np.hstack([
            coordinate,
            self.stage,
        ])
    
        # get info
        info = {
            'items': self.items,
            't_elapsed': self.t_elapsed,
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }

        return obs, info


    def step(self, action):
        """
        Step the environment.
        """

        done = False
        reward = 0.0 # initialize reward
        item = None

        # memory stage
        if self.stage == 0:
            item = self.items[self.t_elapsed]
            coordinate = self.coordinates[item]
        
        # delay stage
        elif self.stage == 1:
            coordinate = np.array([0., 0.])
        
        # decision stage
        elif self.stage == 2:
            coordinate = np.array([0., 0.])
            if action == self.items[self.t_elapsed - self.num_items - self.t_delay]:
                reward += 1

        # update timer
        self.t_elapsed += 1

        # update stage
        if self.t_elapsed == self.num_items:
            self.stage = 1

        elif self.t_elapsed == self.num_items + self.t_delay:
            self.stage = 2

        # done
        if self.t_elapsed == self.num_items + self.t_delay + self.num_items:
            done = True

        # wrap observation
        obs = np.hstack([
            coordinate,
            self.stage,
        ])

        # get info
        info = {
            'items': self.items,
            't_elapsed': self.t_elapsed,
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    

    def get_action_mask(self):
        """
        Get action mask.

        Note:
            no batching is considered here. batching is implemented by vectorzation wrapper.
            if no batch training is used, add the batch dimension and transfer the mask to torch.tensor in trainer.
            if batch training is used, concatenate batches and transfer the mask to torch.tensor in trainer.
        """

        mask = np.ones((self.action_space.n,), dtype = bool)
        
        return mask


    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot



class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        # initialize previous variables
        self.init_prev_variables()

        # define new observation space
        new_observation_shape = (
            self.env.observation_space.shape[0] + # obs
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        # concatenate previous variables into observation
        obs_wrapped = self.wrap_obs(obs)

        # update previous variables
        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        # initialize previous physical action and reward
        self.init_prev_variables()

        # concatenate previous physical action and reward into observation
        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs, # current obs
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward,
        ])
        return obs_wrapped
    



if __name__ == '__main__':
    # test single environment

    import warnings
    warnings.filterwarnings('ignore')
    
    env = ImmediateSerialRecallEnv()
    env = MetaLearningWrapper(env)

    for i in range(50):

        obs, info = env.reset()
        done = False

        print('items:', env.env.items)
        print('initial obs:', obs.shape)
        
        while not done:
            # sample action
            action = env.action_space.sample()

            # step env
            obs, reward, done, truncated, info = env.step(action)

            print(
                'action:', action, '|',
                'reward:', np.round(reward, 3), '|',
                'next obs:', obs.shape, '|',
                'stage:', env.env.stage, '|',
                'elapsed time:', env.env.t_elapsed, '|',
                'done:', done, '|',
            )
        print()