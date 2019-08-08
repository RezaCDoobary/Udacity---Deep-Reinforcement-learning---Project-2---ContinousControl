import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable


class batch_generator:
    """Batching for stochastic gradient ascent/descent"""
    def __init__(self, batch_size = None, index_data =[ None]):
        """Initialize parameters and build model.

        Params
        ======
            batch_size (int): The desired size of each batch.
            index_data (array): The array to be batch sampled from.
        """
        self.batch_size = batch_size
        self.index_data = index_data
        self.data_size = len(index_data)
        if batch_size == 0:
            self.stop = True
        else:
            self.stop = False
        self.generator = None
        self.next_is_stop = False

    def get_next_is_stop(self):
        """Return true if the next batch is the last one"""
        return self.next_is_stop
        
    def shuffle(self):
        """Shuffles the indices before batch sampling"""
        np.random.shuffle(self.index_data)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_index_data(self, index_data):
        self.index_data = index_data

    def createGenerator(self):
        """Create a generator object that will repeatedly generate the next sample"""

        index_list = range(0, self.data_size, self.batch_size)
        self.shuffle()
        for i in range(0,len(index_list)):
            if index_list[i] == index_list[-1]:
                self.stop = True
                yield self.index_data[index_list[i]:]
                
            else:
                if index_list[i] == index_list[-2]:
                    self.next_is_stop = True
                yield self.index_data[index_list[i]:index_list[i+1]]
                
    def end(self):
        """Returns true is current sampling is the last one"""

        return self.stop
    
    def get_iter(self):
        """Sets up the generator for batch sampling, note that each call with reshuffle the indices"""

        self.generator = self.createGenerator()

    def sample_idxs(self):
        """Sample the indices"""

        idxs = next(self.generator)
        batch_indices = torch.Tensor(idxs).long()
        return batch_indices
    
    def sample(self, *arrays):
        """Will sample a list of arrays subject the generator

        Params
        ======
            *arrays (variable list of arrays): A variable number of lists to be sampled from.
        """
        idxs = next(self.generator)
        batch_indices = torch.Tensor(idxs).long()
        return [a[batch_indices] for a in arrays]