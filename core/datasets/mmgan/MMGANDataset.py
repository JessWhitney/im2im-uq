import os,sys,inspect
base_dir = os.path.dirname(__file__)
rcgan_root = os.path.abspath(os.path.join(base_dir, '../../../../rcGAN'))
print("RcGAN path:", rcgan_root)
sys.path.insert(0, rcgan_root)
from data.lightning.MassMappingDataModule import MMDataTransform
rcps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(1, rcps_path) 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils
import glob


class MMGANDataset(Dataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    def __init__(self, path=None, num_instances=None, normalize=None, gt_files=None, samp_files=None, args=None):
        self.data_transform = MMDataTransform(args)
        print('loading dataset')
        if gt_files is not None and samp_files is not None:
            self.gt_files = gt_files
            self.samp_files = samp_files
        else:
            # GT and reconstructions
            self.gt_files = sorted(glob.glob(os.path.join(path, 'np_gt_[0-9]*.npy')))
            self.samp_files = sorted(glob.glob(os.path.join(path, 'np_samps_[0-9]*.npy')))
            assert len(self.gt_files) == len(self.samp_files), "Mismatch between ground truth and recon files!"
        
        if num_instances is not None and num_instances != 'all':
            if num_instances <= len(self.gt_files):
                self.gt_files = self.gt_files[:num_instances]
                self.samp_files = self.samp_files[:num_instances]
            else:
                print(f'Dataset only has {len(self.gt_files)} instances, please try again.')
                exit(0)
        self.normalize = normalize
        self.params = None

        print(f'Loaded filepaths for {len(self.gt_files)} instances.')
    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        # # ground truth and reconstruction SAMPLES
        # gt = np.load(self.gt_files[idx])
        # gt_torch = torch.from_numpy(gt)
        # # samp = np.load(self.samp_files[idx])
        # # recon = samp.mean(axis=1)
        # # recon_torch = torch.from_numpy(recon)

        # # model in is re + im of shear, re + im of KS
        # shear, normalized_gt,_,_ = self.data_transform(gt)
        
        # if self.normalize:
        #     shear, _ = utils.normalize(shear, type=self.normalize, per_pixel=False, input_output='input')
        #     normalized_gt, _ = utils.normalize(normalized_gt, type=self.normalize, per_pixel=False, input_output='output')
        # assert shear.shape == (4, self.data_transform.im_size, self.data_transform.im_size), \
        # f"Unexpected shape for shear: {shear.shape}"
        # return shear, normalized_gt
        # # return recon_torch, gt_torch
        sample_path = self.samp_files[idx]
        gt_path = self.gt_files[idx]

        samples = np.load(sample_path)  # shape [32, H, W]
        gt = np.load(gt_path)           # shape [H, W]

        samples = torch.tensor(samples).float()
        gt = torch.tensor(gt).float()

        return samples, gt


if __name__ == "__main__":
  path = "/share/gpu0/jjwhit/samples/real_output/"
  dataset = MMGANDataset(path, num_instances='all', normalize='standard')
  loader = DataLoader(dataset, batch_size=9, shuffle=True, num_workers=4)

  for idx, sample in enumerate(loader):
      print(idx)

  pdb.set_trace()
  print("Hi!")
