import os
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, img_path, perms, list_IDs):
        'Initialization'
        self.img_path = img_path
        self.perms = perms
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img = np.fromfile(os.path.join(self.img_path, str(ID)+'.raw'), np.uint8).reshape(100, 100, 100)
        img = np.expand_dims(img, axis=0)
        perm = np.log(self.perms[index] / 9.869233e-16)
        perm = np.expand_dims(perm, axis=-1)

        return img, perm