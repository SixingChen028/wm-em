import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MemoryDataset(Dataset):
    """
    A dataset for the sequential memory task
    """

    def __init__(
            self,
            size,
            num_items = 3,
            num_targets = 6,
            t_delay = 1,
        ):
        """
        Initialize the dataset.
        """

        # initialize parameters
        self.size = size # dataset size
        self.num_items = num_items # number of items
        self.num_targets = num_targets # number of targets
        self.t_delay = t_delay # delay time

        # compute sequence length
        self.seq_len = 2 * self.num_items + 1

        # initialize coordinates
        angles = torch.linspace(0, 2 * torch.pi * (self.num_targets - 1) / self.num_targets, self.num_targets) # radius = 1
        self.coordinates = torch.column_stack((torch.cos(angles), torch.sin(angles)))

        # generate data
        self.data = [self.generate_data() for _ in range(size)]
    

    def generate_data(self):
        """
        Generate sample.
        """

        # initialize input and target
        inputa = torch.zeros(self.seq_len, 2)
        targets = torch.zeros(self.seq_len, dtype = torch.long)
        
        # select items
        self.items = torch.randperm(self.num_targets)[:self.num_items] 

        # incode input sequence (first num_items steps)
        for t in range(self.num_items):
            item = self.items[t]
            inputa[t, :] = self.coordinates[item]
        
        # target sequence (last num_items steps)
        targets[-self.num_items:] = self.items
        
        return inputa, targets
    

    def __len__(self):
        return self.size
    

    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == '__main__':
    # testing

    dataset = MemoryDataset(1000)
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

    for inputs, targets in dataloader:
        print(inputs.shape, targets.shape)