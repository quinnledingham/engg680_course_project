#%%
# Help from https://github.com/karpathy/ng-video-lecture
import torch
import torch.nn as nn
from torch.nn import functional as F

import pickle
import numpy as np
import time

from input import Input_Data

start_time = time.time()

#%%
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 120 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'xpu' if torch.xpu.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 2000
# ------------

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
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
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
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPSSpatialEmbedding(nn.Module):
    def __init__(self, lat_bins=1800, lon_bins=3600):
        super(GPSSpatialEmbedding, self).__init__()
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        # Define embeddings for latitude and longitude bins
        self.lat_embedding = nn.Embedding(lat_bins, n_embd)
        self.lon_embedding = nn.Embedding(lon_bins, n_embd)

    def discretize_coordinates(self, latitude, longitude):
        # Convert latitude from [-90, 90] to indices [0, lat_bins - 1]
        lat_idx = ((latitude + 90) * (self.lat_bins / 180)).long().clamp(0, self.lat_bins - 1)
        
        # Convert longitude from [-180, 180] to indices [0, lon_bins - 1]
        lon_idx = ((longitude + 180) * (self.lon_bins / 360)).long().clamp(0, self.lon_bins - 1)
        
        return lat_idx, lon_idx

    def forward(self, lat, lon):
        lat_idx, lon_idx = self.discretize_coordinates(lat, lon)

        # Get embeddings for latitude and longitude
        lat_embed = self.lat_embedding(lat_idx)
        lon_embed = self.lon_embedding(lon_idx)
        
        # Combine embeddings (e.g., by summing or concatenating)
        spatial_embedding = lat_embed + lon_embed
        
        return spatial_embedding

class GPTLanguageModel(nn.Module):

    def __init__(self, num_of_stations, pm25_mean, pm25_std):
        super().__init__()
        self.pm25_mean = pm25_mean
        self.pm25_std = pm25_std

        # each token directly reads off the logits for the next token from a lookup table
        #self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pm25_projection = nn.Linear(1, n_embd)
        self.position_embedding_table = nn.Embedding(24, n_embd)
        self.naps_station_embedding_table = nn.Embedding(num_of_stations, n_embd)
        #self.spatial_embedding_table = GPSSpatialEmbedding()

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        #self.lm_head = nn.Linear(n_embd, vocab_size)
        self.lm_head = nn.Linear(n_embd, 1)

        self.criterion = nn.MSELoss()

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        D, B, T = idx.shape # D is datatype

        # idx and targets are both (B,T) tensor of integers
        normalized_pm25 = (idx[0].float() - self.pm25_mean) / self.pm25_std
        pm25_emb = self.pm25_projection(normalized_pm25.unsqueeze(-1))
        #pm25_emb = self.pm25_projection(idx[0].float().unsqueeze(-1))
        stn_emb = self.naps_station_embedding_table(idx[1])
        pos_emb = self.position_embedding_table(idx[2]) # (T,C)
        
        x = pm25_emb + pos_emb # (B,T,C)
        if idx[1] is not None:
            x += stn_emb

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if idx[3] is None:
            loss = None
        else:
            #B, T, C = logits.shape
            normalized_targets = (idx[3].float() - self.pm25_mean) / self.pm25_std
            #loss = F.mse_loss(logits.squeeze(-1), targets)
            loss = self.criterion(logits.squeeze(-1), normalized_targets)

        return logits, loss

    def generate(self, block, max_new_tokens):
        generated_values = None
        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # get the predictions
            logits, loss = self(block)
            # focus only on the last time step
            logits = logits[:, -1] # becomes (B, C)

            denormalized_value = (logits * self.pm25_std) + self.pm25_mean
            if generated_values is None:
                generated_values = denormalized_value.round().int()
            else:
                generated_values = torch.cat((generated_values, denormalized_value.round().int()), dim=1)

            # block.size() = 3, 1, 60
            # input: data type(pm25, station id, time), batch, context length
            station = torch.tensor([[block[1][0][i]]], dtype=torch.long, device=device)
            time = torch.tensor([[12 + (i / 5)]], dtype=torch.long, device=device)

            new_block0 = torch.cat((block[0], denormalized_value.round().int()), dim=1)
            new_block1 = torch.cat((block[1], station), dim=1) # (B, T+1)
            new_block2 = torch.cat((block[2], time), dim=1) # (B, T+1)

            block = torch.stack((new_block0, new_block1, new_block2))

        return generated_values

@torch.no_grad()
def estimate_loss(model, data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = data.get_batch2(split, data, block_size, batch_size, device)
            logits, loss = model(batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    data = Input_Data.load_data("./data_cache/new.data")
    data.init_split()

    torch.manual_seed(1337)

    model = GPTLanguageModel(len(data.station_ids), data.pm25_mean, data.pm25_std)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {time.time() - start_time}")

        # sample a batch of data
        batch = data.get_batch2('train', data, block_size, batch_size, device)

        # evaluate the loss
        logits, loss = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(m.state_dict(), "./data_cache/year2_gpt.model")

if __name__ == '__main__':
    main()

#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))