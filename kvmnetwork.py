import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    """
    A torch Categorical class with action masking.
    """

    def __init__(self, logits, mask):
        self.mask = mask

        # set mask value to minimum possible value
        self.mask_value = torch.tensor(
            torch.finfo(logits.dtype).min, dtype = logits.dtype
        )

        # replace logits of invalid actions with the minimum value
        logits = torch.where(self.mask, logits, self.mask_value)

        super(CategoricalMasked, self).__init__(logits = logits)


    def entropy(self):
        p_log_p = self.logits * self.probs

        # compute entropy with possible actions only (not really necessary)
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype = p_log_p.dtype, device = p_log_p.device),
        )

        return -torch.sum(p_log_p, axis = 1)
    

class FlattenExtractor(nn.Module):
    """
    A flatten feature extractor.
    """
    def forward(self, x):
        # keep the first dimension while flatten other dimensions
        return x.view(x.size(0), -1)


class ValueNet(nn.Module):
    """
    Value baseline network.
    """
    
    def __init__(self, input_size):
        super(ValueNet, self).__init__()
        self.fc_value = nn.Linear(input_size, 1)
    
    def forward(self, x):
        value = self.fc_value(x) # (batch_size, 1)

        return value


class ActionNet(nn.Module):
    """
    Action network.
    """

    def __init__(self, input_size, output_size):
        super(ActionNet, self).__init__()
        self.fc_action = nn.Linear(input_size, output_size)
    
    def forward(self, x, mask = None):
        self.logits = self.fc_action(x) # record logits for later analyses

        # no action masking
        if mask is None:
            dist = Categorical(logits = self.logits)
        
        # with action masking
        elif mask is not None:
            dist = CategoricalMasked(logits = self.logits, mask = mask)
        
        policy = dist.probs # (batch_size, output_dim)
        action = dist.sample() # (batch_size,)
        log_prob = dist.log_prob(action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)
        
        return action, policy, log_prob, entropy


class KeyValueMemoryActorCriticPolicy(nn.Module):
    """
    A key-value network with shared actor and critic.
    """

    def __init__(
            self,
            feature_size,
            action_size,
            key_size,
            hidden_size = 128,
        ):
        super(KeyValueMemoryActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_size = feature_size
        self.action_size = action_size
        self.key_size = key_size
        self.hidden_size = hidden_size

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.gru = nn.GRUCell(feature_size, hidden_size)

        # policy and value net
        self.policy_net = ActionNet(hidden_size, action_size)
        self.value_net = ValueNet(hidden_size)

        # keys and values
        self.


    def forward(self, obs, states = None, mask = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # initialize hidden states
        if states is None:
            states = torch.zeros(features.size(0), self.gru.hidden_size, device = obs.device)
        
        # iterate one step
        hidden = self.gru(features, states)

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden, mask)

        # compute value
        value = self.value_net(hidden)

        return action, policy, log_prob, entropy, value, hidden



if __name__ == '__main__':
    # testing

    feature_size = 60
    action_size = 3
    batch_size = 16


    # net = LSTMRecurrentActorCriticPolicy(
    #     feature_size = feature_size,
    #     action_size = action_size,
    # )

    # net = SharedLSTMRecurrentActorCriticPolicy(
    #     feature_size = feature_size,
    #     action_size = action_size,
    # )

    # net = GRURecurrentActorCriticPolicy(
    #     feature_size = feature_size,
    #     action_size = action_size,
    # )

    net = SharedGRURecurrentActorCriticPolicy(
        feature_size = feature_size,
        action_size = action_size,
    )

    # generate random test input
    test_input = torch.randn((batch_size, feature_size))
    test_mask = torch.randint(0, 2, size = (batch_size, action_size), dtype = torch.bool)

    # forward pass through the network
    action, policy, log_prob, entropy, value, states_lstm = net(test_input, mask = test_mask)

    print('action:', action)
    print('policy:', policy)
    print('log prob:', log_prob)
    print('entropy:', entropy)
    print('value:', value)
    print('lstm states:', states_lstm)
