import torch
import torch.nn as nn
from torch.nn import functional as F

import pickle
import math
import random
import numpy as np
import pandas as pd
from naps import Naps

stations_in_batch = 1
num_of_features = 2

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
        self.num_of_stations = int(len(self.df.columns)/num_of_features)
        self.data = torch.tensor(self.df.to_numpy(), dtype=torch.long)


        T, SF = self.data.shape
        self.data = self.data.reshape(T, num_of_features, self.num_of_stations)
        self.data = self.data.permute(0, 2, 1)

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

    def select(self, x, stn_indices):
        B, T, S, F = x.shape # SF is stations * features
        
        hours = int(T / stations_in_batch)

        all_indices = torch.arange(S, device=x.device)        
        mask = torch.ones(S, dtype=torch.bool, device=x.device)
        mask[stn_indices] = False  # Mark the `stn_indices` as False

        not_stn_indices = all_indices[mask]

        selected = x[:, :, stn_indices, :]
        
        # Separate the data not at `stn_indices`
        not_selected = x[:, :, not_stn_indices, :]

        # Reshape if necessary
        selected = selected.view(B, T, len(stn_indices) * F)
        not_selected = not_selected.view(B, T, len(not_stn_indices) * F)

        #out = x[:, :hours, stn_indices, :]
        #out = out.view(B, T, stations_in_batch * F)

        return selected, not_selected

    def get_batch(self, split, batch_size, block_size, device):
        if split == 'train':
            data = self.train
        elif split == 'val':
            data = self.val
        elif split == 'test':
            data = self.test

        ix = torch.randint(len(data) - block_size, (batch_size,))

        #data = torch.unsqueeze(data, 2)
        x = torch.stack([data[i:i+block_size] for i in ix],)
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        start_ix = torch.randint(self.num_of_stations - (stations_in_batch-1), (1,))  # Ensure we have room for 10 stations
        stn_indices = torch.arange(start_ix.item(), start_ix.item() + stations_in_batch)

        x, other = self.select(x, stn_indices)
        y, _ = self.select(y, stn_indices)

        return (x.to(device), y.to(device), ix, other.to(device))

    def get_block(self, split, hours, block_size, device):
        if split == 'train':
            data = self.train
        elif split == 'val':
            data = self.val
        elif split == 'test':
            data = self.test

        ix = torch.randint(len(data) - block_size, (1,))

        #data = torch.unsqueeze(data, 2)
        split = int(block_size-hours)
        x = torch.stack([data[i:i+split] for i in ix],)
        y = torch.stack([data[i+split:i+block_size] for i in ix])

        start_ix = torch.randint(self.num_of_stations - (stations_in_batch-1), (1,))  # Ensure we have room for 10 stations
        stn_indices = torch.arange(start_ix.item(), start_ix.item() + stations_in_batch)

        x, other = self.select(x, stn_indices)
        y, _ = self.select(y, stn_indices)

        return (x.to(device), y.to(device), ix, other.to(device))
    