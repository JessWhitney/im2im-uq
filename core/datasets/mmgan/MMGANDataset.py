import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils
import glob


class MMGANDataset(Dataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    #TODO: Don't make this hard coded
    def __init__(self, path, num_instances, normalize=None, gt_files=None, recon_files=None):
        print('loading dataset from ' + path + '...')
        if gt_files is not None and recon_files is not None:
            self.gt_files = gt_files
            self.recon_files = recon_files
        else:
            # GT and reconstructions
            self.gt_files = sorted(glob.glob(os.path.join(path, 'np_gt_[0-9]*.npy')))
            self.recon_files = sorted(glob.glob(os.path.join(path, 'np_avgs_[0-9]*.npy')))
            assert len(self.gt_files) == len(self.recon_files), "Mismatch between ground truth and recon files!"
        
        if num_instances != 'all':
            if num_instances <= len(self.gt_files):
                self.gt_files = self.gt_files[:num_instances]
                self.recon_files = self.recon_files[:num_instances]
            else:
                print(f'Dataset only has {len(self.gt_files)} instances, please try again.')
                exit(0)
        self.normalize = normalize
        self.params = None

        print(f'Loaded filepaths for {len(self.gt_files)} instances.')
    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        # ground truth
        x = np.load(self.gt_files[idx])
        recon = np.load(self.recon_files[idx])
        # Both are real images, need to add an extra channel
        if x.ndim==2:
            x = x[None, :, :]
            recon  = recon[None, :, :]
        x = torch.tensor(x, dtype=torch.float64)
        recon = torch.tensor(recon, dtype=torch.float64)

        if self.normalize:
            x, _ = utils.normalize(x, type=self.normalize, per_pixel=False, input_output='input')
            recon, _ = utils.normalize(recon, type=self.normalize, per_pixel=False, input_output='output')

        return x, recon


if __name__ == "__main__":
  path = "/share/gpu0/jjwhit/samples/real_output/"
  dataset = MMGANDataset(path, num_instances='all', normalize='min-max')
  loader = DataLoader(dataset, batch_size=9, shuffle=True, num_workers=4)

  for idx, sample in enumerate(loader):
      print(idx)

  pdb.set_trace()
  print("Hi!")
