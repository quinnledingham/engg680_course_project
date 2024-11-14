#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from importlib import reload 

from gpt import PM25TransformerModel, batch_size, block_size
from input import Input_Data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input = Input_Data.load_data("./data_cache/new.data")
input.init_split()

torch.no_grad()
#model = nn.Transformer(nhead=8, d_model=256, dropout=0.2, num_decoder_layers=6)
model = PM25TransformerModel(input)
model.load_state_dict(torch.load("./data_cache/all_stns_v2.model", weights_only=True))
m = model.to(device)
m.eval()

#%%
block = input.get_block('train', batch_size, block_size, device)
hours = 20
gen = m.generate(block, max_new_tokens=hours)

#gen = model.normalize(gen)
#target = model.normalize(block[1])

print(block[0])
print(block[1])

targets = model.select(block[1])
#targets = model.denormalize(targets)

for i in range(hours):
    print(f'{gen[i][0]},', end="")
print(' ')
print(targets)
print(len(gen)) 
print(len(targets))

err = 0
for i in range(hours):
    #print(targets[i] - gen[i][0])
    err += (targets[0][i][0] - gen[i][0])**2
    #print(f'{target[i]} = {gen[i]}')
print(f"err {err/hours}")

print("End")

#%%
out = {}

losses = torch.zeros(200)
for k in range(200):
    batch = input.get_batch('test', batch_size, block_size, device)
    logits, loss = model(batch)
    losses[k] = loss.item()
print(losses)
out['test'] = losses.mean()
model.train()
print(out)


#%%
model.eval()
batch = input.get_batch3('test', batch_size, block_size, device)
logits, loss = model(batch)
print(loss)
# %%
