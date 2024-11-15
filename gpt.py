#%%
# Help from https://github.com/karpathy/ng-video-lecture
import torch
import torch.nn as nn
from torch.nn import functional as F
import intel_extension_for_pytorch as ipex

import pickle
import numpy as np
import time

from input import Input_Data

start_time = time.time()

#%%
# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'xpu' if torch.xpu.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 8
n_layer = 6
dropout = 0.2
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

    def forward(self, x, x_kv=None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape

        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)
        if x_kv is not None:
            k = self.query(x_kv)
            v = self.query(x_kv)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if x_kv is None: # masked attention
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

    def forward(self, x, x_kv=None):
        out = torch.cat([h(x, x_kv) for h in self.heads], dim=-1)
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
        #self.ca = MultiHeadAttention(n_head, head_size) # cross attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)


    def forward(self, inp):
        x, encoder_output = inp
        x = x + self.sa(self.ln1(x))
        #x = x + self.ca(self.ln2(x), encoder_output)
        x = x + self.ffwd(self.ln3(x))
        return (x, encoder_output)

class EncoderBlock(nn.Module):
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

stations_in_batch = 1

class PM25TransformerModel(nn.Module):

    def __init__(self, data):
        super().__init__()
        self.pm25_mean = data.pm25_mean
        self.pm25_std = data.pm25_std
        self.pm25_min = data.pm25_min
        self.pm25_max = data.pm25_max
        self.num_of_stations = data.num_of_stations

        # each token directly reads off the logits for the next token from a lookup table
        self.pm25_projection = nn.Linear(1, n_embd)
        self.hour_embedding_table = nn.Embedding(24, n_embd)
        self.day_embedding_table = nn.Embedding(365, n_embd)
        self.station_embedding = nn.Embedding(self.num_of_stations, n_embd)

        #self.encoder_blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, 1)

        self.criterion = nn.MSELoss()

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def select(self, x, stn_indices):
        B, T, S, F = x.shape # SF is stations * features

        hours = int(T / stations_in_batch)

        x = x.view(B, T, S * F)
        x = x[:, :hours, stn_indices]
        x = x.view(B, T)
        #x = self.normalize(x.float().unsqueeze(2))
        x = x.float().unsqueeze(2)

        return x

    def forward(self, idx):
        idx, targets, ix = idx

        hours = int(idx.shape[1] / stations_in_batch)

        start_ix = torch.randint(self.num_of_stations - (stations_in_batch-1), (1,))  # Ensure we have room for 10 stations
        stn_indices = torch.arange(start_ix.item(), start_ix.item() + stations_in_batch, device=device)

        if targets is not None:
            idx = self.select(idx, stn_indices)
            targets = self.select(targets, stn_indices)

        station_range = stn_indices.repeat(int(block_size / stations_in_batch))

        start_hours = torch.remainder(ix, 24)
        start_days = torch.div(ix, 24, rounding_mode="floor")
        
        pos_range = torch.arange(0, 24, device=device).repeat(int(block_size / stations_in_batch))
        day_range = torch.arange(0, 365, device=device).repeat_interleave(24)

        hour_ranges = torch.stack([pos_range[h:idx.shape[1]+h] for h in start_hours])
        day_ranges = torch.stack([day_range[d:idx.shape[1]+d] for d in start_days])

        pm25_emb = self.pm25_projection(idx) # (B, T, n_embd)
        hour_emb = self.hour_embedding_table(hour_ranges)
        day_emb = self.day_embedding_table(day_ranges)
        stn_emb = self.station_embedding(station_range[0:idx.shape[1]])

        x = pm25_emb + hour_emb + day_emb # (B,T,C)

        #encoder_output = self.encoder_blocks(x)
        encoder_output = None
        x, _ = self.blocks((x, encoder_output)) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = self.criterion(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens):
        generated_values = []

        #x = self.normalize(x)

        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # get the predictions
            logits, loss = self((x, None))
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            #values = self.denormalize(logits)
            values = logits
            generated_values.append(values.round().int().flatten().tolist())

            logits = logits.unsqueeze(2)
            x = torch.cat((x, logits), dim=1)

        return generated_values

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
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
        for k in range(eval_iters):
            batch = data.get_batch(split, batch_size, block_size, device)
            logits, loss = model(batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    data = Input_Data()
    data.load_data("./data_cache/new.data")
    data.init_split()

    torch.manual_seed(1337)

    model = PM25TransformerModel(data)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data)
            print(f"step {iter}: train loss {losses['train']:.8f}, val loss {losses['val']:.8f}, time {time.time() - start_time}")

        # sample a batch of data
        batch = data.get_batch('train', batch_size, block_size, device)

        # evaluate the loss
        logits, loss = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(m.state_dict(), "./data_cache/all_stns_v2.model")

if __name__ == '__main__':
    main()

#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
# %%
