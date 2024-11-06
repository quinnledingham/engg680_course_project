#%%
# Help from https://github.com/karpathy/ng-video-lecture
import torch
import torch.nn as nn
from torch.nn import functional as F

import pickle
from naps import Naps


#%%
'''
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
'''

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
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

f = open("./data_cache/test.data", 'rb')
input = pickle.load(f)

naps_stations = input.station_ids

# Train and test splits
data = torch.tensor(input.pm25_data, dtype=torch.long)
station_data = torch.tensor([naps_stations[stn] for stn in input.station_data], dtype=torch.long)

n = int(0.75*len(data)) # first 90% will be train, rest val
t = int(0.9*len(data))

def split_data(data, n, t):
    train= data[:n]
    val = data[n:t]
    test = data[t:]
    return train, val, test

train_data, val_data, test_data = split_data(data, n, t)
stn_train, stn_val, stn_test = split_data(station_data, n, t)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    station_data = stn_train if split == 'train' else stn_val

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    naps_stn = torch.stack([station_data[i:i+block_size] for i in ix])
    #lat = torch.stack([gps_data[i:i+block_size, 0] for i in ix])
    #lon = torch.stack([gps_data[i:i+block_size, 1] for i in ix])

    x, y, naps_stn = x.to(device), y.to(device), naps_stn.to(device)
    return x, y, naps_stn

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, naps_stn = get_batch(split)
            logits, loss = model(X, Y, naps_stn)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.naps_station_embedding_table = nn.Embedding(len(naps_stations), n_embd)
        #self.spatial_embedding_table = GPSSpatialEmbedding()

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        #self.lm_head = nn.Linear(n_embd, vocab_size)
        self.lm_head = nn.Linear(n_embd, 1)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, naps_stn=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        stn_emb = self.naps_station_embedding_table(naps_stn)
        
        #if lat is not None:
        #    spatial_emb = self.spatial_embedding_table(lat, lon)
        #    x = tok_emb + pos_emb + stn_emb
        #else:
        x = tok_emb + pos_emb + stn_emb # (B,T,C)

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            #B, T, C = logits.shape
            #logits = logits.view(B*T, C)
            #targets = targets.view(B*T)
            #loss = F.cross_entropy(logits, targets)
            loss = F.mse_loss(logits.squeeze(-1), targets.float())

        return logits, loss

    def generate(self, idx, naps_stn, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond, None, naps_stn)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

torch.manual_seed(1337)

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb, stn = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb, stn)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m.state_dict(), "./data_cache/gpt.model")

#if __name__ == '__main__':
#   main()

#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))