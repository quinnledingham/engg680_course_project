import torch
import torch.nn as nn
from torch.nn import functional as F

import pickle
import math
import random
import numpy as np
import pandas as pd
from naps import Naps

class Input_Data:
    # pm25_data is a list of pm2.5 values
    # station_data is a list of NAPS_IDs, storing the id for each pm2.5 value
    # station_ids is a dict converting NAPS_IDs to 0 - len(stations) for use in embedding
    #def __init__(self, data, station_ids):
        #self.data = data
        #self.station_ids = station_ids

    def save_data(self, filepath):
        f = open(filepath, "wb")
        pickle.dump(self, f)
        f.close()

    # used like a constructor when using the data
    def load_data(self, filepath):
        #f = open(filepath, 'rb')
        #self = pickle.load(f)
        #f.close()
        self.df = pd.read_csv('./data_cache/station_features.csv')

    def init_split(self):
        self.df = self.df.drop(self.df.columns[0], axis=1)
        self.num_of_stations = len(self.df.columns)
        self.data = torch.tensor(self.df.to_numpy(), dtype=torch.long)
        #self.data[0] = np.where(self.data[0] < 0, 0, self.data[0])

        n = int(0.75*len(self.data)) # first 90% will be train, rest val
        t = int(0.9*len(self.data))

        self.train = self.data[:n, :]
        self.val = self.data[n:t, :]
        self.test = self.data[t:, :]

        train_array = np.array([row[:n] for row in self.data])

        self.pm25_mean = np.mean(train_array)
        self.pm25_std = np.std(train_array)
        self.pm25_min = np.min(train_array)
        self.pm25_max = np.max(train_array)

        '''
        naps = Naps()
        df = naps.data(year=year)
        station_ids = {}
        # Create a lookup table for station ids
        for row_id in df['NAPS ID//Identifiant SNPA'].unique():
            station_ids[row_id] = num
            num += 1

        filtered_stations_df = naps.stations_df[naps.stations_df['NAPS_ID'].isin(station_ids.keys())]
        # Cache closest stations for each station ID
        closest_stations_cache = {
            row['NAPS_ID']: naps.find_5_closest(filtered_stations_df, naps.station_coords(row))
            for _, row in filtered_stations_df.iterrows()
        }
        '''
    
    def get_batch(self, split, batch_size, block_size, device):
        if split == 'train':
            data = self.train
        elif split == 'val':
            data = self.val
        elif split == 'test':
            data = self.test

        ix = torch.randint(len(data) - block_size, (batch_size,))

        data = torch.unsqueeze(data, 2)
        x = torch.stack([data[i:i+block_size] for i in ix],)
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        return (x.to(device), y.to(device), ix)

    def get_block(self, split, batch_size, block_size, device):
        if split == 'train':
            data = self.train
        elif split == 'val':
            data = self.val
        elif split == 'test':
            data = self.test

        ix = torch.randint(len(data) - block_size, (1,))
        ix[0] = 0

        data = torch.unsqueeze(data, 2)
        split = int(block_size-20)
        x = torch.stack([data[i:i+split] for i in ix],)
        y = torch.stack([data[i+split:i+block_size] for i in ix])

        print(x)
        return (x.to(device), y.to(device))
    