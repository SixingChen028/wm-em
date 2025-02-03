import torch
import torch.nn as nn


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
            hiddens: a list where each element with shape (batch_size, hidden_size)
        """

        # get sizes
        batch_size, seq_len, _ = inputs.size()

        # initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size)
        
        # loop through an episode
        outputs = []
        hiddens = []
        for t in range(seq_len):
            hidden = self.gru(inputs[:, t, :], hidden)
            outputs.append(self.fc(hidden))
            hiddens.append(hidden.detach())
        
        # stack outputs
        outputs = torch.stack(outputs, dim = 1) # (batch_size, seq_len, output_size)
        hiddens = torch.stack(hiddens, dim = 1) # (batch_size, seq_len, hidden_size)

        return outputs, hiddens




if __name__ == '__main__':
    # testing

    input_size = 10
    output_size = 3
    seq_len = 6
    batch_size = 16

    net = WorkingMemoryNet(
        input_size = input_size,
        output_size = output_size,
    )

    # generate random test input
    test_inputs = torch.randn((batch_size, seq_len, input_size))

    # forward pass through the network
    test_outputs, test_hiddens = net(test_inputs)

    print('outputs:', test_outputs.shape)
    print('hiddens:', test_hiddens.shape)
