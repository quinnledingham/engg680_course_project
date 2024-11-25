import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np
import time
import math

from input import Input_Data, stations_in_batch, num_of_features

start_time = time.time()

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 0.01
eval_interval = 500
eval_iters = 200
dropout = 0.2

class LinearModel(nn.Module):
    def __init__(self, data):
        super().__init__()

        self.pm25_min = data.pm25_min
        self.pm25_max = data.pm25_max

        self.scaler = StandardScaler()
        self.scaler.fit(data.train[:, :, 0])

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.hidden = nn.Linear(1, 64) 
        self.relu = nn.ReLU()  
        self.output = nn.Linear(64, 1) 

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
        idx, targets, ix, stn_ix, other = idx

        pm25, fire = torch.split(idx, 1, dim=-1)
        
        B, T, F = pm25.shape # SF is stations * features
        norm = self.normalize(pm25.float())

        x = self.hidden(norm)
        x = self.relu(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            targets, _ = torch.split(targets, 1, dim=-1)
            logits = logits.view(B*T)
            targets = targets.view(B*T)
            mse_loss = self.mse_loss(logits, self.normalize(targets.float()))
            loss = self.l1_loss(logits, self.normalize(targets.float()))

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        generated_values = torch.tensor([0] * max_new_tokens)

        x, targets, ix, stn_ix, other = idx

        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # get the predictions
            logits, loss = self((x, None, ix, stn_ix, other))
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            values = self.denormalize(logits)
            generated_values[i] = values.round().int().flatten()

            next_input = values.unsqueeze(2).repeat(1, 1, 2)
            
            #next_input[:, :, 0] = targets[:, i, 0]
            next_input[:, :, 1] = targets[:, i, 1]

            x = torch.cat((x, next_input.round().int()), dim=1)

        return generated_values

@torch.no_grad()
def estimate_loss(model, data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = data.get_batch(split, batch_size, block_size, device)
            logits, loss = model(batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
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
            print(f"step {iter}: train loss {losses['train']:.8f}, val loss {losses['val']:.8f}, time {time.time() - start_time}")
            #scheduler.step(losses['val'])

        # sample a batch of data
        batch = data.get_batch('train', batch_size, block_size, device)

        # evaluate the loss
        logits, loss = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    #torch.save(model.state_dict(), "./data_cache/five_years_linear.model")
    return model, losses_total

if __name__ == '__main__':
    main()