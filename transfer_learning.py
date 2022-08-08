import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset
from model import VGG3D, make_layers
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


dx = 1.65
rock_type = 'sandstone'
sample_name = 'Bentheimer'
base_dir = '/scratch/users/PermNet'
data_dir = os.path.join(base_dir, 'test', sample_name, 'subvols')
perm_path = os.path.join(base_dir, 'test', sample_name, 'poro_perm.npy')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_channels = 1
n_epochs = 20
lr = 1e-4
batch_size = 32


perms = np.load(perm_path)[2]/(dx*dx)
IDs = np.load(perm_path)[0].astype(np.int32)
idx = np.random.permutation(len(IDs))
perms = perms[idx]
IDs = IDs[idx]

train_num = 700
perms_train = perms[:train_num]
IDs_train = IDs[:train_num]
np.save(os.path.join(base_dir, 'test', sample_name, 'IDs_train.npy'), IDs_train)
perms_val = perms[train_num::5]
IDs_val = IDs[train_num::5]
print(len(IDs_train))

train_dataset = MyDataset(data_dir, perms_train, IDs_train)
valid_dataset = MyDataset(data_dir, perms_val, IDs_val)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=3)


# model
base_dir = '/scratch/users/PermNet'
cfg = [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256]
model = VGG3D(make_layers(cfg, batch_norm=True)).to(device)

model_dir = os.path.join(base_dir, rock_type, 'saved_models')
model.load_state_dict(torch.load(os.path.join(model_dir, f'model_{rock_type}.pth')))

for param in model.features.parameters():
    param.requires_grad = False

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
        true_perm = true_perm.reshape(-1, 1).to(device, dtype=torch.float)
        optimizer.zero_grad()
        pred_perm = model(img)
        loss = criterior(pred_perm, true_perm)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    with torch.no_grad():
        valid_loss = 0
        for img, true_perm in valid_dataloader:
            img = img.to(device, dtype=torch.float)
            true_perm = true_perm.reshape(-1, 1).to(device, dtype=torch.float)
            pred_perm = model(img)
            loss = criterior(pred_perm, true_perm)
            valid_loss += loss.item()
    
    train_loss_history[epoch] = train_loss/len(train_dataloader)
    valid_loss_history[epoch] = valid_loss/len(valid_dataloader)
    print('epoch: {} | training loss: {:.4f} | validation loss: {:.4f}'.format(epoch, train_loss_history[epoch],
                                                                                valid_loss_history[epoch]))

perms_test = np.ones(7200)
IDs_test = np.arange(7200)

test_dataset = MyDataset(data_dir, perms_test, IDs_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
true_perms = []
pred_perms = []
with torch.no_grad():
    for img, true_perm in test_dataloader:
        img = img.to(device, dtype=torch.float)
        true_perm = true_perm.to(device, dtype=torch.float)
        pred_perm = model(img)
        true_perms.append(true_perm.cpu().detach().numpy())
        pred_perms.append(pred_perm.cpu().detach().numpy())

pred_perm_array = np.concatenate(pred_perms)[:, 0] + np.log(dx*dx)
true_perm_array = np.log(np.load(perm_path)[2] / 9.869233e-16)
IDs = np.load(perm_path)[0].astype(np.int32)
pred_perm_array = pred_perm_array[IDs]

correlation_matrix = np.corrcoef(true_perm_array.T, pred_perm_array.T)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print(r_squared)


fig = plt.figure(figsize=(8, 6))
plt.scatter(true_perm_array, pred_perm_array)
plt.plot(np.arange(-4, 16, 0.1), np.arange(-4, 16, 0.1), 'k--')
plt.xlim([-4, 16])
plt.ylim([-4, 16])
plt.xlabel('True log$K$ (mD)', fontsize=16)
plt.ylabel('Predicted log$K$ (mD)', fontsize=16)
plt.savefig(f'./figs/pred_lnk_{sample_name}_R2_{r_squared:.2f}.tiff', dpi=300)