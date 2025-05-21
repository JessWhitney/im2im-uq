import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import psutil
import copy
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from core.calibration.bounds import HB_mu_plus
import pdb

def get_rcps_losses(model, dataset, rcps_loss_fn, lam, device):
  losses = []
  dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
  for batch in dataloader:
    sets = model.nested_sets_from_output(batch,lam) 
    losses = losses + [rcps_loss_fn(sets, labels),]
  return torch.cat(losses,dim=0)

def get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam, device):
  losses = []
  dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True) 
  model = model.to(device)
  for batch in dataloader:
    x, labels = batch
    sets = model.nested_sets_from_output(x.to(device),lam) 
    losses = losses + [rcps_loss_fn(sets, labels.to(device)).cpu(),]
  return torch.cat(losses,dim=0)

def get_rcps_metrics_from_outputs(model, out_dataset, rcps_loss_fn, device):
  losses = []
  sizes = []
  residuals = []
  spatial_miscoverages = []
  dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
  model = model.to(device)
  for batch in dataloader:
    x, labels = batch
    labels = labels.to(device)
    sets = model.nested_sets_from_output(x.to(device)) 
    losses = losses + [rcps_loss_fn(sets, labels),]
    sets_full = (sets[2]-sets[0]).flatten(start_dim=1).detach().cpu().numpy()
    size_random_idxs = np.random.choice(sets_full.shape[1],size=sets_full.shape[0])
    size_samples = sets_full[range(sets_full.shape[0]),size_random_idxs]
    residuals = residuals + [(labels - sets[1]).abs().flatten(start_dim=1)[range(sets_full.shape[0]),size_random_idxs]]
    spatial_miscoverages = spatial_miscoverages + [(labels > sets[2]).float() + (labels < sets[0]).float()]
    sizes = sizes + [torch.tensor(size_samples),]
  losses = torch.cat(losses,dim=0)
  sizes = torch.cat(sizes,dim=0)
  sizes = sizes + torch.rand(size=sizes.shape).to(sizes.device)*1e-6
  residuals = torch.cat(residuals,dim=0).detach().cpu().numpy() 
  spearman = spearmanr(residuals, sizes)[0]
  mse = (residuals*residuals).mean().item()
  spatial_miscoverage = torch.cat(spatial_miscoverages, dim=0).detach().cpu().numpy().mean(axis=0).mean(axis=0)
  size_bins = torch.tensor([0, torch.quantile(sizes, 0.25), torch.quantile(sizes, 0.5), torch.quantile(sizes, 0.75)])
  buckets = torch.bucketize(sizes, size_bins)-1
  stratified_risks = torch.tensor([losses[buckets == bucket].mean() for bucket in range(size_bins.shape[0])])
  print(f"Model output shape: {x.shape}, label shape: {labels.shape}, Sets shape: {sets[2].shape}, sizes: {sizes}, size_bins:{size_bins}, stratified_risks: {stratified_risks}, mse: {mse}")
  return losses, sizes, spearman, stratified_risks, mse, spatial_miscoverage

def evaluate_from_loss_table(loss_table,n,alpha,delta):
  with torch.no_grad():
    perm = torch.randperm(loss_table.shape[0])
    loss_table = loss_table[perm]
    calib_table, val_table = loss_table[:n], loss_table[n:]
    Rhats = calib_table.mean(dim=0)
    RhatPlus = torch.tensor([HB_mu_plus(Rhat, n, delta) for Rhat in Rhats])
    try:
        idx_lambda = (RhatPlus <= delta).nonzero()[0]
    except:
        print("No rejections made!")
        idx_lambda = 0
    return val_table[:,idx_lambda].mean()
  
def fraction_missed_loss(pset,label):
  misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[2].squeeze() < label.squeeze()).float()
  misses[misses > 1.0] = 1.0
  d = len(misses.shape)
  return misses.mean(dim=tuple(range(1,d)))

def get_rcps_loss_fn(config):
  string = config['rcps_loss']
  if string == 'fraction_missed':
    return fraction_missed_loss
  else:
    raise NotImplementedError

def calibrate_model(model, dataset, config):
  with torch.no_grad():
    print(f"Calibrating...")
    model.eval()
    alpha = config['alpha']
    delta = config['delta']
    device = config['device']
    print("Initialize lambdas")
    if config["uncertainty_type"] == "softmax":
      lambdas = torch.linspace(config['minimum_lambda_softmax'],config['maximum_lambda_softmax'],config['num_lambdas'])
    else:
      lambdas = torch.linspace(config['minimum_lambda'],config['maximum_lambda'],config['num_lambdas'])
    print("Initialize loss")
    rcps_loss_fn = get_rcps_loss_fn(config)
    print("Put model on device")
    model = model.to(device)
    print("Initialize labels")
    if config['dataset'] == 'temca':
      labels = torch.cat([x[1].unsqueeze(0).to('cpu') for x in iter(dataset)], dim=0)
      outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in iter(dataset)])
      print("Labels initialized.")
    else:
      # labels_shape = list(dataset[0][0].unsqueeze(0).shape)
      # labels_shape[0] = len(dataset)
      # labels = torch.zeros(tuple(labels_shape), device='cpu', dtype=torch.float64)
      # # outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
      # outputs_shape = list(dataset[0][1].unsqueeze(0).shape)
      # outputs_shape[0] = len(dataset)
      # outputs = torch.zeros(tuple(outputs_shape),device='cpu', dtype=torch.float64)
      n_samples = len(dataset)
      sample_label = dataset[0][0] # [300, 300]
      sample_output = dataset[0][1] # [32, 300, 300]
      labels = torch.zeros((n_samples, *sample_label.shape), dtype=torch.float64)
      outputs = torch.zeros((n_samples, *sample_output.shape), dtype=torch.float64)
      print("shape of outputs", outputs.shape, "and of labels", labels.shape, flush=True)
      print("Collecting dataset")
      tempDL = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], pin_memory=True)
      counter = 0
      for batch in tqdm(tempDL):
        batch_size = batch[0].shape[0]
        # outputs[counter:counter+batch[0].shape[0],:,:,:] = model(batch[0].to(device)).cpu()
        outputs[counter:counter+batch_size, :, :, :] = batch[1]
        labels[counter:counter+batch_size] = batch[0]
        counter += batch_size

    print("Output dataset", flush=True) # this prints
    out_dataset = TensorDataset(outputs,labels.cpu())
    dlambda = lambdas[1]-lambdas[0]
    model.set_lhat(lambdas[-1]+dlambda-1e-9)
    print("Computing losses", flush=True)
    calib_loss_table = torch.zeros((outputs.shape[0],lambdas.shape[0]))
    for lam in reversed(lambdas):
      # losses here is the failure rate
      losses = get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam-dlambda, device)
      calib_loss_table[:,np.where(lambdas==lam)[0]] = losses[:,None]
      # Aim is to match alpha
      Rhat = losses.mean()
      RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], delta)
      print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}  ",end='', flush=True)
      if Rhat >= alpha or RhatPlus > alpha:
        model.set_lhat(lam)
        print("")
        print(f"Model's lhat set to {model.lhat}", flush=True)
        break
    return model, calib_loss_table
