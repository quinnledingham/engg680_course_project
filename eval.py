#%%
import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt import PM25TransformerModel, batch_size, block_size, stations_in_batch
from input import Input_Data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = Input_Data()
data.load_data("./data_cache/new.data")
data.init_split()

torch.no_grad()
#model = nn.Transformer(nhead=8, d_model=256, dropout=0.2, num_decoder_layers=6)
model = PM25TransformerModel(data)
model.load_state_dict(torch.load("./data_cache/linear.model", weights_only=True))
m = model.to(device)
m.eval()

#%%
hours = 24
block = data.get_block('train', hours, block_size, device, torch.tensor([4693]), torch.tensor([201]))
#block = data.get_block('train', hours, block_size, device)

x, targets, ix, stn_ix, other = block

gen = m.generate(block, max_new_tokens=hours)
targets, _ = torch.split(targets, 1, dim=-1)
targets = targets.squeeze(-1).squeeze(-1)


print(x)
print(gen)
print(targets)
print(len(gen)) 
print(len(targets))

err = 0
for i in range(hours):
    #print(targets[i] - gen[i][0])
    err += (targets[0][i] - gen[i])**2
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
batch = data.get_batch('test', batch_size, block_size, device)
logits, loss = model(batch)
print(loss)


# %%
