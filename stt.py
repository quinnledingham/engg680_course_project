# spatial temporal transformer

#%%
# Help from https://github.com/karpathy/ng-video-lecture
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#import intel_extension_for_pytorch as ipex

import pickle
import numpy as np
import time
import math
from sklearn.metrics import r2_score

from input import Input_Data, stations_in_batch, num_of_features

start_time = time.time()

#%%
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 36 # what is the maximum context length for predictions?
target_size = 12
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate =1e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1
# ------------

pos_range = torch.arange(0, 24, device=device).repeat(int(block_size / stations_in_batch))
day_range = torch.arange(0, 365, device=device).repeat_interleave(24)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape

        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # self attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

def sinusoidal_position_embedding(seq_len, dim):
    # Initialize the position and dimension terms
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * 
                        -(math.log(10000.0) / dim))  # (dim // 2)

    # Compute sinusoidal embeddings
    sinusoidal_embedding = torch.zeros(seq_len, dim, device=device)
    sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)  # Even indices: sine
    sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cosine

    return sinusoidal_embedding

class PM25TransformerModel(nn.Module):

    def __init__(self, data):
        super().__init__()
        self.pm25_mean = data.pm25_mean
        self.pm25_std = data.pm25_std
        self.pm25_min = data.pm25_min
        self.pm25_max = data.pm25_max
        self.num_of_stations = data.num_of_stations

        self.pm25_projection = nn.Linear(self.num_of_stations, n_embd)

        self.station_embedding = nn.Embedding(self.num_of_stations, n_embd)
        self.fire_embedding = nn.Embedding(10, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        self.lm_head = nn.Sequential(
            nn.Linear(n_embd, 1),
            nn.Sigmoid()
        )

        self.mse_loss = nn.MSELoss()

    def forward(self, idx):
        idx, targets, ix, stn_ix = idx

        B, T, S, _ = idx.shape

        pm25, fire = torch.split(idx, 1, dim=-1)

        pm25 = pm25.view(B, T, S)
        pm25_emb = self.pm25_projection(self.normalize(pm25.float())) # (B, T, n_embd)

        fire = fire[:, :, stn_ix, :]
        fire = fire.view(B, T)
        fire_emb = self.fire_embedding(fire)

        spatial_emb = self.station_embedding(stn_ix)  # (B, S, n_embd)
        spatial_emb = spatial_emb.expand(B, T, n_embd)
        
        pos_emb = sinusoidal_position_embedding(T, n_embd)

        x = pm25_emb + fire_emb + spatial_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,1)

        mse_loss = None
        r2 = None
        if targets is not None:
            targets, _ = torch.split(targets, 1, dim=-1)
            norm_targets = self.normalize(targets.float())
            mse_loss = self.mse_loss(logits, norm_targets)
            r2 = r2_score(targets.detach().cpu().numpy().flatten(), self.denormalize(logits).detach().cpu().numpy().flatten())

        return self.denormalize(logits), mse_loss, r2

    def generate(self, idx, max_new_tokens):
        x, targets, ix, stn_ix = idx
        B, T, S, _ = x.shape
        predictions = []

        # idx is (B, T) array of indices in the current context
        with torch.no_grad():
            for i in range(max_new_tokens):
                # get the predictions
                logits, loss, r2 = self((x, None, ix, stn_ix))
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                next_pred_station = logits

                predictions.append(next_pred_station)

                next_pred_expanded = next_pred_station.repeat(1, S, 2)
                next_pred_expanded = next_pred_expanded.view(B, 1, S, 2)

                next_pred_expanded[:, :, :, 1] = targets[:, i, 1].unsqueeze(1).unsqueeze(2).repeat(1, 1, 210)

                idx = torch.cat([x[:, 1:, :, :], next_pred_expanded], dim=1)

        predictions = torch.cat(predictions, dim=1)  # Combine along the sequence dimension (time steps)

        return predictions

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # He initialization
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def normalize(self, x):
        return (x - self.pm25_min) / (self.pm25_max - self.pm25_min)

    def denormalize(self, x):
        return x * (self.pm25_max - self.pm25_min) + self.pm25_min

@torch.no_grad()
def estimate_loss(model, data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        r2_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = data.get_batch(split, batch_size, block_size, target_size, device)
            logits, loss, r2 = model(batch)
            losses[k] = loss.item()
            r2_losses[k] = r2
        out[split] = [losses.mean(), r2_losses.mean()]
    model.train()
    return out

def main():
    data = Input_Data()
    data.load_data('./data_cache/station_features_v2.csv')
    data.init_split()

    torch.manual_seed(1337)

    model = PM25TransformerModel(data)
    model.apply(model._init_weights)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    losses_total = []

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data)
            losses_total.append(losses)
            print(f"step {iter}: train loss {losses['train'][0]:.8f}, val loss {losses['val'][0]:.8f}, time {time.time() - start_time}")

        # sample a batch of data
        batch = data.get_batch('train', batch_size, block_size, target_size, device)

        # evaluate the loss
        logits, loss, r2 = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        #scheduler.step(loss)

    return model, losses_total

if __name__ == '__main__':
    main()