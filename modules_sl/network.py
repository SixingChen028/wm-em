import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkingMemoryNet(nn.Module):
    """
    GRU working memory network.
    """

    def __init__(
            self,
            input_size,
            output_size,
            hidden_size = 32,
        ):
        super(WorkingMemoryNet, self).__init__()

        # network parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # recurrent neural network
        self.gru = nn.GRUCell(input_size, hidden_size)

        # output head
        self.fc = nn.Linear(hidden_size, output_size)
    

    def forward(self, inputs):
        """
        Forward the net.

        Args:
            inputs: a torch.tensor with shape (batch_size, seq_len, input_size).

        Returns:
            outputs: a torch.tensor with shape (batch_size, seq_len, output_size)
            hiddens: a torch.tensor with shape (batch_size, seq_len, hidden_size)
        """

        # get sizes
        batch_size, seq_len, _ = inputs.size()

        # initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size)
        
        # initialize recordings
        outputs = []
        hiddens = []

        # loop through an episode
        for t in range(seq_len):
            # update hidden state
            hidden = self.gru(inputs[:, t, :], hidden)

            # record output and hidden
            outputs.append(self.fc(hidden))
            hiddens.append(hidden.detach())
        
        # stack outputs
        outputs = torch.stack(outputs, dim = 1) # (batch_size, seq_len, output_size)
        hiddens = torch.stack(hiddens, dim = 1) # (batch_size, seq_len, hidden_size)

        return outputs, hiddens



class KeyValueMemoryNet(nn.Module):
    """
    A key-value memory network.
    """

    def __init__(
            self,
            input_size,
            output_size,
            key_size = 64,
            memory_size = 3,
            hidden_size = 32,
        ):
        super(KeyValueMemoryNet, self).__init__()

        # network parameters
        self.input_size = input_size
        self.output_size = output_size
        self.key_size = key_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size

        # recurrent neural network
        self.gru = nn.GRUCell(input_size, hidden_size)

        # fc layers
        self.fc_key = nn.Linear(hidden_size, key_size)
        self.fc_query = nn.Linear(hidden_size, key_size)
        self.fc_value = nn.Linear(hidden_size, output_size)


    def forward(self, inputs):
        """
        Forward the net.

        Args:
            inputs: a torch.tensor with shape (batch_size, seq_len, input_size).

        Returns:
            outputs: a torch.tensor with shape (batch_size, seq_len, output_size)
            hiddens: a torch.tensor with shape (batch_size, seq_len, hidden_size)
        """

        # get sizes
        batch_size, seq_len, _ = inputs.size()

        # initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size)

        # initialize keys and values
        self.keys = torch.zeros(batch_size, self.memory_size, self.key_size) # (batch_size, memory_size, key_size)
        self.values = torch.zeros(batch_size, self.memory_size, self.output_size) # (batch_size, memory_size, output_size)

        # initialize counter
        self.counter = 0

        # initialize recordings
        outputs = []
        hiddens = []

        # loop through an episode
        for t in range(seq_len):
            # update hidden state
            hidden = self.gru(inputs[:, t, :], hidden)
            
            # retrieve memory based on current hidden state
            retrieved_memory = self.read_memory(hidden) # (batch_size, output_size)

            # store new memory
            self.write_memory(hidden)

            # record output and hidden
            outputs.append(retrieved_memory)
            hiddens.append(hidden.detach())

        # stack outputs
        outputs = torch.stack(outputs, dim = 1)  # (batch_size, seq_len, output_size)
        hiddens = torch.stack(hiddens, dim = 1)  # (batch_size, seq_len, hidden_size)

        return outputs, hiddens


    def write_memory(self, hidden):
        """
        Add memory.

        Args:
            hidden: a torch.tensor with shape (batch_size, hidden_size).
        """

        #only add if there's space
        if self.counter < self.memory_size:
            # get batch size
            batch_size = hidden.shape[0]

            key = self.fc_key(hidden) # (batch_size, key_size)
            value = self.fc_value(hidden) # (batch_size, output_size)

            # generate one-hot index for writing memory
            one_hot_index = F.one_hot(torch.tensor(self.counter), num_classes = self.memory_size).float()  # (memory_size,)
            one_hot_index = one_hot_index.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (batch_size, memory_size, 1)

            # store memory
            # (batch_size, memory_size, key_size/output_size)
            self.keys = self.keys + one_hot_index * key.unsqueeze(1) # (batch_size, memory_size, key_size)
            self.values = self.values + one_hot_index * value.unsqueeze(1) # (batch_size, memory_size, key_size)

            # increment counter
            self.counter += 1


    def read_memory(self, hidden):
        """
        Retrieve memory.

        Args:
            hidden: a torch.tensor with shape (batch_size, hidden_size).
        
        Returns:
            retrieved_memory: a torch.tensor of shape (batch_size, output_size).
        """

        # compute query
        query = self.fc_query(hidden) # (batch_size, key_size)

        # no memory stored yet
        if self.counter == 0:
            return torch.zeros(query.shape[0], self.output_size)
        
        # compute similarity
        # (batch_size, 1, key_size) * (batch_size, key_size, counter) = (batch_size, 1, counter)
        scores = torch.bmm(query.unsqueeze(1), self.keys[:, :self.counter, :].transpose(1, 2))

        # apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim = -1)  # (batch_size, 1, counter)
        
        # retrieve memory
        # (batch_size, 1, counter) * (batch_size, counter, output_size) = (batch_size, 1, output_size)
        retrieved_memory = torch.bmm(attention_weights, self.values[:, :self.counter, :])
        retrieved_memory = retrieved_memory.squeeze(1) # (batch_size, output_size)

        return retrieved_memory



if __name__ == '__main__':
    # testing

    input_size = 10
    output_size = 3
    seq_len = 6
    batch_size = 16

    # net = WorkingMemoryNet(
    #     input_size = input_size,
    #     output_size = output_size,
    # )

    net = KeyValueMemoryNet(
        input_size = input_size,
        output_size = output_size,
    )

    # generate random test input
    test_inputs = torch.randn((batch_size, seq_len, input_size))

    # forward pass through the network
    test_outputs, test_hiddens = net(test_inputs)

    print('outputs:', test_outputs.shape)
    print('hiddens:', test_hiddens.shape)
