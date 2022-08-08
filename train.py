import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import VGG3D, make_layers
np.random.seed(2022)


parser = argparse.ArgumentParser()
parser.add_argument('--rock_type', type=str, default='sandstone', help='rock type')
opt = parser.parse_args()


# hyper-parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_channels = 1
n_epochs = 50
lr = 1e-4
batch_size = 64


# dataloader
base_dir = '/scratch/users/PermNet'
data_dir = os.path.join(base_dir, opt.rock_type, 'train')
perm_path = f'/home/users/workbench/code/PermNet/data/poro_perm_{opt.rock_type}_train.npy'
perms = np.load(perm_path)[2]
IDs = np.load(perm_path)[0].astype(np.int32)

idx = np.random.permutation(len(IDs))
perms = perms[idx]
IDs = IDs[idx]

val_ratio = 0.1
val_num = int(len(idx) * val_ratio)
perms_train = perms[val_num:]
IDs_train = IDs[val_num:]
perms_val = perms[:val_num]
IDs_val = IDs[:val_num]

print(len(idx), val_num)

train_dataset = MyDataset(data_dir, perms_train, IDs_train)
valid_dataset = MyDataset(data_dir, perms_val, IDs_val)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=32)


# model
cfg = [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256]
model = VGG3D(make_layers(cfg, batch_norm=True)).to(device)
model_dir = os.path.join(base_dir, opt.rock_type, 'saved_models')
os.makedirs(model_dir, exist_ok=True)


# training
criterior = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

train_loss_history = np.zeros(n_epochs)
valid_loss_history = np.zeros(n_epochs)

for epoch in range(n_epochs):
    train_loss = 0
    for i, (img, true_perm) in enumerate(train_dataloader):
        img = img.to(device, dtype=torch.float)
        true_perm = true_perm.to(device, dtype=torch.float)
        optimizer.zero_grad()
        pred_perm = model(img)
        loss = criterior(pred_perm, true_perm)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('epoch: {}, batch: {} | training loss: {:.4f} '.format(epoch, i, loss))
        
    with torch.no_grad():
        valid_loss = 0
        for img, true_perm in valid_dataloader:
            img = img.to(device, dtype=torch.float)
            true_perm = true_perm.to(device, dtype=torch.float)
            pred_perm = model(img)
            loss = criterior(pred_perm, true_perm)
            valid_loss += loss.item()
    
    train_loss_history[epoch] = train_loss/len(train_dataloader)
    valid_loss_history[epoch] = valid_loss/len(valid_dataloader)
    print('epoch: {} | training loss: {:.4f} | validation loss: {:.4f}'.format(epoch, train_loss_history[epoch],
                                                                                valid_loss_history[epoch]))
    scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_%d.pth'%epoch))
    np.save(os.path.join(model_dir, 'train_loss_history.npy'), train_loss_history)
    np.save(os.path.join(model_dir, 'valid_loss_history.npy'), valid_loss_history)