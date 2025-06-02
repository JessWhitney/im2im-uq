import os, sys, inspect
# to import model function from parent directory
base_dir = os.path.dirname(__file__)
rcgan_root = os.path.abspath(os.path.join(base_dir, '../../../../rcGAN'))
sys.path.insert(0, rcgan_root)
from models.lightning.mmGAN import mmGAN
from utils.parse_args import create_arg_parser
rcps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(1, rcps_path) 
import wandb
import random
import copy
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import warnings
import yaml
import glob
# import types

from core.scripts.eval import get_images, eval_net, get_loss_table, eval_set_metrics
from core.models.add_uncertainty import add_uncertainty_GAN
from core.calibration.calibrate_model import calibrate_model
from core.utils import fix_randomness
from core.datasets.mmgan import MMGANDataset

def nested_set_from_output_fn(model, output, lam=None):
    if lam == None:
        if model.lhat == None:
            raise Exception(
                "You have to specify lambda unless your model is already calibrated."
            )
        lam = model.lhat
    # GAN output is samples: (batch_size, samples, im_size, im_size)

    # TODO: Don't hardcode this - checking if I need to renormalise
    kappa_mean = 0.00015744
    kappa_std = 0.02968585
    output = output * kappa_std + kappa_mean

    prediction = output.mean(axis=1)
    std = output.std(dim=1)
    upper_edge = lam * std + prediction
    lower_edge = -lam * std + prediction

    return lower_edge, prediction, upper_edge

class TempArgs:
    def __init__(self, im_size):
        self.im_size = im_size
        # Fix the randomness
        fix_randomness()
        warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # importing base model args
    print("Entered main method.")
    wandb.init()
    print("wandb init.")
    config = wandb.config
    im_size = config.im_size
    temp_args = TempArgs(im_size)
    # Check if results exist already
    output_dir = wandb.config["output_dir"]
    results_fname = (
        output_dir
        + f"results_"
        + wandb.config["dataset"]
        + "_"
        + wandb.config["uncertainty_type"].replace(".", "_")
        + ".pkl"
    )
    if os.path.exists(results_fname):
        print(f"Results already precomputed and stored in {results_fname}!")
        os._exit(os.EX_OK)
    else:
        print("Computing the results from scratch!")
    # Otherwise compute results

    curr_method = wandb.config["uncertainty_type"]
    curr_dataset = wandb.config["dataset"]
    wandb.run.name = f"{curr_method}, {curr_dataset}"
    wandb.run.save()
    params = {key: wandb.config[key] for key in wandb.config.keys()}
    batch_size = wandb.config["batch_size"]
    params["batch_size"] = batch_size
    print("wandb save run.")

    # DATASET LOADING
    # Make sure each point is a tuple (input, target).
    # Input = Data not GT! So should be re + im shear, re + im KS recon.
    random.seed(42)
    data_path = "/share/gpu0/jjwhit/samples/real_output/"
    gt_files = sorted(glob.glob(os.path.join(data_path, 'np_gt_[0-9]*.npy')))
    samp_files = sorted(glob.glob(os.path.join(data_path, 'np_samps_[0-9]*.npy')))
    # avg_files = sorted(glob.glob(os.path.join(data_path, 'np_avgs_[0-9]*.npy')))
    assert len(gt_files) == len(samp_files)

    # pair samples with gt, and shuffle
    paired = list(zip(gt_files, samp_files))
    random.shuffle(paired)
    gt_files, samp_files = zip(*paired)

    #split into calibration, validation, and testing
    n = len(gt_files)
    n_cal = int(0.4*n)
    n_val = int(0.05*n)
    n_end = int(0.05*n)

    calib_dataset = MMGANDataset(gt_files=gt_files[:n_cal], samp_files=samp_files[:n_cal], normalize='standard', args=temp_args)
    val_dataset = MMGANDataset(gt_files=gt_files[n_cal:n_cal+n_val], samp_files=samp_files[n_cal:n_cal+n_val], normalize='standard', args=temp_args)
    test_dataset = MMGANDataset(gt_files=gt_files[n_cal+n_val:n_cal+n_val+n_end], samp_files=samp_files[n_cal+n_val:n_cal+n_val+n_end], normalize='standard', args=temp_args)

    # batch if needed

    # MODEL LOADING
    # TODO: Note, mb this shouldn't be hard coded either?
    trunk = mmGAN.load_from_checkpoint(checkpoint_path="/share/gpu0/jjwhit/mass_map/mm_models/mmgan_training_real_output/checkpoint-epoch=93.ckpt")
    trunk.cuda() 
    trunk.eval()

    model = add_uncertainty_GAN(trunk, nested_set_from_output_fn)
    model.eval()
    with torch.no_grad():
        print("Calibrate the model.")
        model, calib_loss_table = calibrate_model(model, calib_dataset, params)
        print(f"Model calibrated! lambda hat = {model.lhat}")
        # Save the loss tables
        if output_dir != None:
            try:
                os.makedirs(output_dir, exist_ok=True)
                print("Created output directory")
            except OSError:
                pass
        torch.save(
            calib_loss_table, 
            output_dir
            + f"/loss_table_"
            + wandb.config["dataset"]
            + "_"
            + wandb.config["uncertainty_type"].replace(".", "_")
            + ".pth",
        )
        print("Loss table saved!")
        # Get the prediction sets and properly organize them
        (
            examples_input,
            examples_lower_edge,
            examples_prediction,
            examples_upper_edge,
            examples_ground_truth,
            examples_ll,
            examples_ul,
            raw_images_dict,
        ) = get_images(
            model,
            val_dataset,
            wandb.config["device"],
            list(range(wandb.config["num_validation_images"])),
            params,
        )
        # Log everything
        wandb.log(
            {"examples_input": examples_input}
        )
        wandb.log(
            {"Lower edge": examples_lower_edge}
        )
        wandb.log(
            {"Predictions": examples_prediction}
        )
        wandb.log(
            {"Upper edge": examples_upper_edge}
        )
        wandb.log(
            {"Ground truth": examples_ground_truth}
        )
        wandb.log({ "Lower length": examples_ll})
        wandb.log({"Upper length": examples_ul})
        # Get the risk and other metrics
        print("GET THE METRICS INCLUDING SPATIAL MISCOVERAGE")
        risk, sizes, spearman, stratified_risk, mse, spatial_miscoverage = (
            eval_set_metrics(model, val_dataset, params)
        )
        print("DONE")

        data = [[label, val] for (label, val) in zip(["Easy","Easy-medium", "Medium-Hard", "Hard"], stratified_risk.numpy())]
        table = wandb.Table(data=data, columns = ["Difficulty", "Empirical Risk"])
        wandb.log({"Size-Stratified Risk Barplot" : wandb.plot.bar(table, "Difficulty","Empirical Risk", title="Size-Stratified Risk") })

        print(
            f"Risk: {risk}  |  Mean size: {sizes.mean()}  |  Spearman: {spearman}  |  Size-stratified risk: {stratified_risk} | MSE: {mse} | Spatial miscoverage: (mu, sigma, min, max) = ({spatial_miscoverage.mean()}, {spatial_miscoverage.std()}, {spatial_miscoverage.min()}, {spatial_miscoverage.max()})"
        )
        wandb.log(
            {
                "risk": risk,
                "mean_size": sizes.mean(),
                "Spearman": spearman,
                "Size-Stratified Risk": stratified_risk,
                "mse": mse,
                "spatial_miscoverage": spatial_miscoverage,
            }
        )

        # Save outputs for later plotting
        print("Saving outputs for plotting")
        if output_dir != None:
            try:
                os.makedirs(output_dir, exist_ok=True)
                print("Created output directory")
            except OSError:
                pass
            results = {
                "risk": risk,
                "sizes": sizes,
                "spearman": spearman,
                "size-stratified risk": stratified_risk,
                "mse": mse,
                "spatial_miscoverage": spatial_miscoverage,
            }
            results.update(raw_images_dict)
            with open(results_fname, "wb") as handle:
                pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

            print(f"Results saved to file {results_fname}!")

        print(f"Done with {str(params)}")
