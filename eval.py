#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from importlib import reload 
from gpt import GPTLanguageModel

f = open("./data_cache/test.data", 'rb')
input = pickle.load(f)
input.init_split()

X, y, stn = input.get_batch('test')
print(X)
print(y)
print(stn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.no_grad()
#model = nn.Transformer(nhead=8, d_model=256, dropout=0.2, num_decoder_layers=6)
model = GPTLanguageModel(len(input.station_ids))
#model.load_state_dict(torch.load("./data_cache/gpt.model", weights_only=True))
m = model.to(device)
m.eval()
#%%
context = torch.tensor([[15, 10, 4, 1, 8, 11, 9, 9, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 10, 8, 7, 9, 9, 8]], dtype=torch.long, device=device)
print(m(context)[0].tolist())

#%%
# generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(m.generate(context, max_new_tokens=500)[0].tolist())

context = torch.tensor([[15, 10, 4, 1, 8, 11, 9, 9, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 10, 8, 7, 9, 9, 8]], dtype=torch.long, device=device)
print(m.generate(context, max_new_tokens=24)[0].tolist())

#%%
err = torch.zeros(12).to(device)
test_loss = 0
i = 0
iterations = 0
while iterations < 100:
    x = torch.tensor([test_data[i:i+12].tolist()], dtype=torch.long, device=device)
    lat = torch.tensor(gps_data[i, 0].tolist(), dtype=torch.long, device=device)
    lon = torch.tensor(gps_data[i, 1].tolist(), dtype=torch.long, device=device)
    target = (test_data[i+12:i+24]).to(device)
    result = m.generate(x, lat, lon, max_new_tokens=12)[0].to(device)
    #print(test_data[i:i+24])
    #print(x)
    #print(target)

    #test_loss += F.nll_loss(result[:12], target, size_average=False).item()

    err += target - result[12:]
    
    iterations += 1
    i += 24

err /= iterations
print(err)

# %%
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

#%%
print(y)

# %%
context = torch.tensor([[15, 10, 4, 1, 8, 11, 9, 9, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 10, 8, 7, 9, 9, 8]], dtype=torch.long, device=device)
station = torch.tensor([1], dtype=torch.long, device=device)
station = station.repeat(1, len(context[0]))
print(context.size())
print(station.size())
print(torch.tensor([[station[0][0].tolist()]]))
print(m.generate(context, station, max_new_tokens=12)[0].tolist())

# %%
