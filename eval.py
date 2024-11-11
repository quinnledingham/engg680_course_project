#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from importlib import reload 

from gpt import GPTLanguageModel, batch_size, block_size
from input import Input_Data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input = Input_Data.load_data("./data_cache/new.data")
input.init_split()

torch.no_grad()
#model = nn.Transformer(nhead=8, d_model=256, dropout=0.2, num_decoder_layers=6)
model = GPTLanguageModel(len(input.station_ids), input.pm25_mean, input.pm25_std)
model.load_state_dict(torch.load("./data_cache/year2_gpt.model", weights_only=True))
m = model.to(device)
m.eval()

#%%
'''
out = {}
model.eval()
losses = torch.zeros(200)
for k in range(200):
    batch = input.get_batch2('test', input, block_size, batch_size, device)
    logits, loss = model(batch)
    losses[k] = loss.item()
out['test'] = losses.mean()
model.train()
print(out)
'''
#%%
block, target = input.get_block('train', input, block_size, device)
gen = m.generate(block, max_new_tokens=60)[0]
target = target[0, -60:]
for i in range(60):
    print(f'{target[i]} = {gen[i]}')

#%%
'''
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
beijing_pm2_5 = fetch_ucirepo(id=381) 
  
# data (as pandas dataframes) 
X = beijing_pm2_5.data.features 
y = beijing_pm2_5.data.targets 
  
# metadata 
print(beijing_pm2_5.metadata) 
  
# variable information 
print(beijing_pm2_5.variables) 
'''