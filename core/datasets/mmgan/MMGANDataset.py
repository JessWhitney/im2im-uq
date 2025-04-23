import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils
#TODO: Copy over the class and check if main is correct
class MMGANDataset(Dataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    def __init__(self, path, num_instances, normalize=None):
        print('loading dataset from ' + path + '...')
        x = None
        y = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:,:], self.y[idx,:,:,:]



if __name__ == "__main__":
  path = '/clusterfs/abc/angelopoulos/bsbcm/'
  dataset = MMGANDataset(path, num_instances='all', normalize='min-max')
  loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=8)

  for idx, sample in enumerate(loader):
      print(idx)

  pdb.set_trace()
  print("Hi!")
