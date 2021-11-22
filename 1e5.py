import torch
import torch.nn as nn
import numpy as np 
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pdb
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
BATCH_SIZE = 2560 
EPOCHS = 100000

def obs_convert(edge_index, obs, device):
  data_list = []        
  f1_list = []
  f2_list = []        
  for i in range(len(obs)):    
    x = obs[i][:-2]
    x = torch.reshape(x, (118, 17)).to(torch.float32)    
    f1 = obs[i][-2]
    f2 = obs[i][-1] 
    data = Data(x=x, edge_index=edge_index)
    data_list.append(data)
    f1_list.append(f1)
    f2_list.append(f2)  
  train_loader = DataLoader(data_list, batch_size=len(data_list))
  for data in train_loader: 
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch = data.batch.to(device)
    f1_list = torch.tensor(f1_list, dtype=torch.long).to(device)
    f2_list = torch.tensor(f2_list, dtype=torch.long).to(device)
  return [x , edge_index, batch, f1_list, f2_list] 
  

class Model(nn.Module):
    def __init__(self, feature_num):
        super(Model, self).__init__()   
        self.conv1 = GATv2Conv(feature_num, 128)
        self.conv2 = GATv2Conv(128, 128)
        self.conv3 = GATv2Conv(128, 128)        
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 256)        
        self.l3 = nn.Linear(256, 1)        
        self.day_embedding = nn.Embedding(7, 64)
        self.hour_embedding = nn.Embedding(24, 64)

    def forward(self, obs):
        x = obs[0].to(torch.float32)
        edge_index = obs[1]
        batch = obs[2]        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  
        day = obs[-2]
        hour = obs[-1]
        day_emb = self.day_embedding(day)
        hour_emb = self.hour_embedding(hour)
        x = torch.cat((x, day_emb, hour_emb), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))        
        x = self.l3(x)        
        return x


def train(model, train_loader, optimizer):
    model.train()
    num_batch = 0   
    train_loss_sum = 0     
    for train_batch in train_loader:
      train_x = train_batch[:, :-1]      
      shape_tmp = len(train_batch[:, -1])      
      train_y = train_batch[:, -1].reshape((shape_tmp, 1)).to(torch.float32).to(device)
      train_x = obs_convert(edge_index, train_x, device)
      optimizer.zero_grad() 
      predict = model(train_x).to(torch.float32)
      train_loss = loss_func(predict, train_y)          
      train_loss_sum += train_loss.detach().item()        
      train_loss.backward()  
      optimizer.step()
      num_batch+=1
      if num_batch%10 == 0: 
          print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
              epoch, int(num_batch * BATCH_SIZE), train_size,
              train_loss.item()/BATCH_SIZE)) 
    train_loss_mean= train_loss_sum/num_batch/BATCH_SIZE
    writer_train.add_scalar('loss',train_loss_mean, epoch) 


def test(model,test_loader):
    model.eval()
    num_batch = 0 
    test_loss = 0 
    
    with torch.no_grad():
        for test_batch in test_loader:  
            test_x = test_batch[:, :-1] 
            shape_tmp = len(test_batch[:, -1])               
            test_y = test_batch[:, -1].reshape((shape_tmp, 1)).to(torch.float32).to(device)
            test_x = obs_convert(edge_index, test_x, device)
            predict = model(test_x).to(torch.float32)
            test_loss += loss_func(predict, test_y).item()  
            num_batch+=1
            if num_batch%10 == 0:
                print('Test [{}/{}]\tLoss: {:.6f}'.format(
                    int(num_batch * BATCH_SIZE), test_size,
                    test_loss/num_batch/BATCH_SIZE ))
        test_loss_mean = test_loss/num_batch/BATCH_SIZE         
        writer_test.add_scalar('loss',test_loss_mean , epoch)
        print('*************************')
        print('test_loss', test_loss_mean)                
        print('*************************')
        if min_loss>test_loss_mean:
          torch.save(model.state_dict(), best_model_dir)
        torch.save(model.state_dict(), last_model_dir)



if __name__ == '__main__':
    writer_train = SummaryWriter('cl/learning_rate_1e5_train')
    writer_test = SummaryWriter('cl/learning_rate_1e5_test')
    best_model_dir = 'model/1e5/best_model.pth'
    last_model_dir = 'model/1e5/last_model.pth'
    device = torch.device("cuda:1" if torch.cuda.
                            is_available() else "cpu")    
    data = np.load('new_data_100000.npy')
    np.random.shuffle(data)
    edge_index = np.load('edge_index.npy')    
    print('*****begin learning*****')
    min_loss = float('+inf')
    edge_index = torch.from_numpy(edge_index).to(torch.long)    
    train_size = int(len(data)*0.8)
    test_size = len(data) - train_size 
    train_dataset = data[:train_size]    
    test_dataset = data[train_size:]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = Model(17).to(device)    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    loss_func = nn.MSELoss()
    for epoch in range(EPOCHS):
      train(model, train_loader, optimizer) 
      test(model, test_loader)
    writer_train.close()
    writer_test.close()