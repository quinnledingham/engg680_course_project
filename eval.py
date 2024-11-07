#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from importlib import reload 
from gpt import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

f = open("./data_cache/test.data", 'rb')
input = pickle.load(f)
input.init_split()

torch.no_grad()
#model = nn.Transformer(nhead=8, d_model=256, dropout=0.2, num_decoder_layers=6)
model = GPTLanguageModel(len(input.station_ids), input.pm25_mean, input.pm25_std)
model.load_state_dict(torch.load("./data_cache/gpt_with_stations.model", weights_only=True))
m = model.to(device)
m.eval()
#%%
'''
context = torch.tensor([[15, 10, 4, 1, 8, 11, 9, 9, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 10, 8, 7, 9, 9, 8]], dtype=torch.long, device=device)
station = torch.tensor([1], dtype=torch.long, device=device)
station = station.repeat(1, len(context[0]))
print(context.size())
print(station.size())
print(torch.tensor([[station[0][0].tolist()]]))
print(m.generate(context, station, max_new_tokens=12)[0].tolist())


err = torch.zeros(1, 12).to(device)
test_loss = 0
iterations = 0
while iterations < 100:
    result = m.generate(X, stn, max_new_tokens=12).to(device)
    err = torch.add(err, torch.sub(y, result))
    #err += y - result[12:]

    iterations += 1

err /= iterations
print(err.tolist())
'''
#%%
out = {}
model.eval()
losses = torch.zeros(200)
for k in range(200):
    X, Y, naps_stn = input.get_batch('test')
    logits, loss = model(X, Y, naps_stn)
    losses[k] = loss.item()
out['test'] = losses.mean()
model.train()
print(out)
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