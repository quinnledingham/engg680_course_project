#%%
# Help from https://github.com/karpathy/ng-video-lecture
import torch
import torch.nn as nn
from torch.nn import functional as F

import pickle
from naps import Naps

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 24 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'xpu' if torch.xpu.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

f = open("./data_cache/test.data", 'rb')
input = pickle.load(f)

vocab_size = 2000

# Train and test splits
data = torch.tensor(input.pm25_data, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "./data_cache/bigram.model")


#%%
torch.no_grad()
model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load("./data_cache/bigram.model", weights_only=True))
m = model.to(device)
m.eval()

# generate from the model
context = torch.ones((1, 1), dtype=torch.long, device=device)
print(m.generate(context, max_new_tokens=24)[0].tolist())

#%%
context = torch.tensor([[15, 10, 4, 1, 8, 11, 9, 9, 7, 6, 6, 5, 5, 5, 1000, 5, 6, 7, 10, 8, 7, 9, 9, 8]], dtype=torch.long, device=device)
print(m.generate(context, max_new_tokens=24)[0].tolist())

# %%