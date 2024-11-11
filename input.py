import torch
import torch.nn as nn
from torch.nn import functional as F

import pickle
import math
import random
import numpy as np

class Input_Data:
    # pm25_data is a list of pm2.5 values
    # station_data is a list of NAPS_IDs, storing the id for each pm2.5 value
    # station_ids is a dict converting NAPS_IDs to 0 - len(stations) for use in embedding
    def __init__(self, data, station_ids):
        self.data = data
        self.station_ids = station_ids

    def save_data(self, filepath):
        f = open(filepath, "wb")
        pickle.dump(self, f)
        f.close()

    # used like a constructor when using the data
    @classmethod
    def load_data(self, filepath):
        f = open(filepath, 'rb')
        self = pickle.load(f)
        f.close()

        return self

    def init_split(self):
        self.data[0] = np.where(self.data[0] < 0, 0, self.data[0])

        n = int(0.75*len(self.data[0])) # first 90% will be train, rest val
        t = int(0.9*len(self.data[0]))

        train_array = np.array([row[:n] for row in self.data])

        self.train = torch.tensor(train_array)
        self.val = torch.tensor(np.array([row[n:t] for row in self.data]))
        self.test = torch.tensor(np.array([row[t:] for row in self.data]))

        self.pm25_mean = np.mean(train_array)
        self.pm25_std = np.std(np.array([row[:n] for row in self.data]))

    # data loading
    def get_batch(self, split, batch_size, block_size, device):
        # generate a small batch of data of inputs x and targets y
        data = self.train_pm25 if split == 'train' else self.val_pm25
        station_data = self.train_stn if split == 'train' else self.val_stn
        
        if split == 'test':
            data = self.test_pm25
            station_data = self.test_stn

        ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        naps_stn = torch.stack([station_data[i:i+block_size] for i in ix])

        x, y, naps_stn = x.to(device), y.to(device), naps_stn.to(device)
        return x, y, naps_stn
    
    def get_batch2(self, split, data, block_size, batch_size, device):
        # generate a small batch of data of inputs x and targets y
        pm25_data = data.train[0] if split == 'train' else data.val[0]
        station_data = data.train[2] if split == 'train' else data.val[2]
        
        if split == 'test':
            pm25_data = data.test[0]
            station_data = data.test[2]

        block_num = math.floor(len(pm25_data)/(batch_size * block_size))
        index = random.randint(0, block_num - 1)

        start = index * batch_size * block_size
        end = (index + 1) * batch_size * block_size

        ix = torch.arange(start, end, block_size)

        context = torch.stack([pm25_data[i:i+block_size] for i in ix])
        target = torch.stack([pm25_data[i+1:i+block_size+1] for i in ix])
        stns = torch.stack([station_data[i:i+block_size] for i in ix])
        time = torch.stack([data.train[1][i:i+block_size] for i in ix])

        batch = torch.stack((context, stns, time, target))

        return batch.to(device)
    
    def get_block(self, split, data, block_size, device):
                # generate a small batch of data of inputs x and targets y
        pm25_data = data.train[0] if split == 'train' else data.val[0]
        station_data = data.train[2] if split == 'train' else data.val[2]
        
        if split == 'test':
            pm25_data = data.test[0]
            station_data = data.test[2]

        block_num = math.floor(len(pm25_data)/block_size)

        index = random.randint(0, block_num - 1)
        start = index * block_size

        half = math.floor(block_size / 2)

        context = torch.stack((pm25_data[start:start+half],))
        stns = torch.stack((station_data[start:start+half],))
        time = torch.stack((data.train[1][start:start+half],))
        target = torch.stack((pm25_data[start+1:start+block_size+1],))

        batch = torch.stack((context, stns, time))

        return batch.to(device), target.to(device)