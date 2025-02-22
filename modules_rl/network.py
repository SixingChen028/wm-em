import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ValueNet(nn.Module):
    """
    Value baseline network.
    """
    
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.fc_value = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        value = self.fc_value(x) # (batch_size, 1)

        return value


class ActionNet(nn.Module):
    """
    Action network.
    """

    def __init__(self, input_dim, output_dim):
        super(ActionNet, self).__init__()
        self.fc_action = nn.Linear(input_dim, output_dim)
    
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


class WorkingMemoryActorCriticPolicy(nn.Module):
    """
    GRU recurrent actor-critic policy with shared actor and critic.
    """

    def __init__(
            self,
            input_size,
            action_size,
            hidden_size = 128,
        ):
        super(WorkingMemoryActorCriticPolicy, self).__init__()

        # network parameters
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # recurrent neural network
        self.gru = nn.GRUCell(input_size, hidden_size)

        # policy and value net
        self.policy_net = ActionNet(hidden_size, action_size)
        self.value_net = ValueNet(hidden_size)


    def forward(self, input, state = None, mask = None):
        """
        Forward the net.
        """

        # initialize hidden states
        if state is None:
            state = torch.zeros(input.size(0), self.gru.hidden_size, device = input.device)
        
        # iterate one step
        hidden = self.gru(input, state)

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden, mask)

        # compute value
        value = self.value_net(hidden)

        return action, policy, log_prob, entropy, value, hidden



class KeyValueMemoryActorCriticPolicy(nn.Module):
    """
    GRU key-value actor-critic policy with shared actor and critic.
    """

    def __init__(
            self,
            input_size,
            key_size = 64,
            memory_size = 3,
            hidden_size = 128,
        ):
        super(KeyValueMemoryActorCriticPolicy, self).__init__()

        # network parameters
        self.input_size = input_size
        self.key_size = key_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        
        # recurrent neural network
        self.gru = nn.GRUCell(input_size, hidden_size)

        # fc layers
        self.fc_key = nn.Linear(hidden_size, key_size)
        self.fc_query = nn.Linear(hidden_size, key_size)

        # value net
        self.value_net = ValueNet(hidden_size)


    def forward(self, input, state = None, mask = None):
        """
        Forward the net.
        """

        # initialize hidden states
        if state is None:
            # get sizes
            batch_size, _ = input.size()

            # initialize state
            state = torch.zeros(batch_size, self.hidden_size)

            # initialize keys
            self.keys = torch.zeros(batch_size, self.memory_size, self.key_size) # (batch_size, memory_size, key_size)

            # reset counter
            self.counter = 0
        
        # iterate one step
        hidden = self.gru(input, state)

        # compute action
        action, policy, log_prob, entropy = self.read_memory(hidden)

        # store new memory
        self.write_memory(hidden)

        # compute value
        value = self.value_net(hidden)

        return action, policy, log_prob, entropy, value, hidden
    

    def write_memory(self, hidden):
        """
        Add memory.
        """

        # only add if there's space
        # only add keys
        if self.counter < self.memory_size:
            # get batch size
            batch_size = hidden.shape[0]

            key = self.fc_key(hidden) # (batch_size, key_size)

            # generate one-hot index for writing memory
            one_hot_index = F.one_hot(torch.tensor(self.counter), num_classes = self.memory_size).float()  # (memory_size,)
            one_hot_index = one_hot_index.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (batch_size, memory_size, 1)

            # store memory
            # (batch_size, memory_size, key_size/output_size)
            self.keys = self.keys + one_hot_index * key.unsqueeze(1) # (batch_size, memory_size, key_size)

            # increment counter
            self.counter += 1
        
    
    def read_memory(self, hidden):
        """
        Retrieve memory.
        """

        # compute query
        query = self.fc_query(hidden) # (batch_size, key_size)
        
        # compute similarity
        # (batch_size, 1, key_size) * (batch_size, key_size, memory_size) = (batch_size, 1, memory_size)
        scores = torch.bmm(query.unsqueeze(1), self.keys.transpose(1, 2))

        # apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim = -1).squeeze(1) # (batch_size, memory_size)
        attention_weights = attention_weights.squeeze(1) # (batch_size, memory_size)
        
        # retrieve memory by sampling
        dist = Categorical(probs = attention_weights)
        policy = dist.probs # (batch_size, memory_size)
        action = dist.sample() # (batch_size,)
        log_prob = dist.log_prob(action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)
        
        return action, policy, log_prob, entropy



if __name__ == '__main__':
    # testing

    input_size = 60
    action_size = 3
    batch_size = 16

    # net = WorkingMemoryActorCriticPolicy(
    #     input_size = input_size,
    #     action_size = action_size,
    # )

    net = KeyValueMemoryActorCriticPolicy(
        input_size = input_size,
    )

    # generate random test input
    test_input = torch.randn((batch_size, input_size))
    test_mask = torch.randint(0, 2, size = (batch_size, action_size), dtype = torch.bool)

    # forward pass through the network
    action, policy, log_prob, entropy, value, state = net(test_input, mask = test_mask)

    print('action:', action)
    print('policy:', policy)
    print('log prob:', log_prob)
    print('entropy:', entropy)
    print('value:', value)
    print('hidden state:', state)




    