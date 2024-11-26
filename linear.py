import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np
import time
import math
from sklearn.metrics import r2_score

from input import Input_Data, stations_in_batch, num_of_features
from gpt import estimate_loss, target_size, sinusoidal_position_embedding

start_time = time.time()

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 36 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 0.01
eval_interval = 500
eval_iters = 200
dropout = 0.2
n_embd = 384

class LinearModel(nn.Module):
    def __init__(self, data):
        super().__init__()

        self.pm25_min = data.pm25_min
        self.pm25_max = data.pm25_max
        self.num_of_stations = data.num_of_stations\

        self.scaler = StandardScaler()
        self.scaler.fit(data.train[:, :, 0])

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.pm25_projection = nn.Linear(self.num_of_stations, n_embd)
        self.station_embedding = nn.Embedding(self.num_of_stations, n_embd)
        self.fire_embedding = nn.Embedding(10, n_embd)

        self.linear = nn.Linear(block_size, block_size) 
        #self.relu = nn.ReLU()  
        #self.output = nn.Linear(64, 1) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def normalize(self, x):
        return (x - self.pm25_min) / (self.pm25_max - self.pm25_min)

    def denormalize(self, x):
        return x * (self.pm25_max - self.pm25_min) + self.pm25_min

    def forward(self, idx):
        idx, targets, ix, stn_ix = idx

        B, T, S, _ = idx.shape

        pm25, fire = torch.split(idx, 1, dim=-1)

        pm25 = pm25[:, :, stn_ix, :]
        pm25 = pm25.view(B, T)
        logits = self.linear(self.normalize(pm25))
        #x = self.relu(x)
        #logits = self.output(x)

        loss = None
        mse_loss = None
        r2 = None
        if targets is not None:
            targets, _ = torch.split(targets, 1, dim=-1)
            logits = logits.view(B*T)
            targets = targets.view(B*T)
            norm_targets = self.normalize(targets.float())
            mse_loss = self.mse_loss(logits, norm_targets)
            loss = self.l1_loss(logits, norm_targets)
            r2 = r2_score(torch.flatten(norm_targets).tolist(), torch.flatten(logits).tolist())

        return logits, mse_loss, r2
    
    def generate(self, idx, max_new_tokens):
        idx, targets, ix, stn_ix = idx
        B, T, S, _ = idx.shape
        predictions = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Predict the next value
                next_pred, loss, r2 = self((idx, None, ix, stn_ix)) # next_pred = (B, T)
                
                next_pred_station = next_pred[:, -1:]  # Take the last predicted timestep
                next_pred_station = self.denormalize(next_pred_station)

                # Append the denormalized predicted value
                predictions.append(next_pred_station)

                # Expand `next_pred_station` to match `idx` shape for concatenation
                next_pred_expanded = next_pred_station.repeat(1, S, 2)
                next_pred_expanded = next_pred_expanded.view(B, 1, S, 2)

                # Update te input tensor by appending the new prediction and removing the oldest timestep
                idx = torch.cat([idx[:, 1:, :, :], next_pred_expanded], dim=1)

        predictions = torch.cat(predictions, dim=1)  # Combine along the sequence dimension (time steps)

        return predictions

def main_linear():
    data = Input_Data()
    data.load_data('./data_cache/station_features_v2.csv')
    data.init_split()

    torch.manual_seed(1337)

    model = LinearModel(data)
    model.apply(model._init_weights)
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    losses_total = []

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data)
            losses_total.append(losses)
            print(f"step {iter}: train loss {losses['train'][0]:.8f}, val loss {losses['val'][0]:.8f}, time {time.time() - start_time}")
            #scheduler.step(losses['val'])

        # sample a batch of data
        batch = data.get_batch('train', batch_size, block_size, target_size, device)

        # evaluate the loss
        logits, loss, r2 = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, losses_total

if __name__ == '__main__':
    main_linear()