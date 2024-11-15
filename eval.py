#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from importlib import reload 

from gpt import PM25TransformerModel, batch_size, block_size, stations_in_batch
from input import Input_Data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = Input_Data()
data.load_data("./data_cache/new.data")
data.init_split()

torch.no_grad()
#model = nn.Transformer(nhead=8, d_model=256, dropout=0.2, num_decoder_layers=6)
model = PM25TransformerModel(data)
model.load_state_dict(torch.load("./data_cache/all_stns_v2.model", weights_only=True))
m = model.to(device)
m.eval()

#%%
block = data.get_block('train', batch_size, block_size, device)

x, targets = block
hours = int(x.shape[1] / stations_in_batch)

start_ix = torch.randint(model.num_of_stations - (stations_in_batch-1), (1,))  # Ensure we have room for 10 stations
stn_indices = torch.arange(start_ix.item(), start_ix.item() + stations_in_batch, device=device)

x = model.select(x, stn_indices)
targets = model.select(targets, stn_indices)

hours = 20
gen = m.generate(x, max_new_tokens=hours)

print(gen)
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
    batch = data.get_batch('test', batch_size, block_size, device)
    logits, loss = model(batch)
    losses[k] = loss.item()
print(losses)
out['test'] = losses.mean()
model.train()
print(out)


#%%
model.eval()
batch = data.get_batch3('test', batch_size, block_size, device)
logits, loss = model(batch)
print(loss)
# %%
