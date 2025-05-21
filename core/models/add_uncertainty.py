import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], "../../"))
import torch
import torch.nn as nn
from core.utils import standard_to_minmax
import json


class ModelWithUncertainty(nn.Module):
    def __init__(self, baseModel, in_nested_sets_from_output_fn):
        """ Given a model, and a function which returns lower/upper bounds of predictions,
        this returns a new model which incorporates both
        """
        super(ModelWithUncertainty, self).__init__()
        self.baseModel = baseModel
        self.register_buffer("lhat", None)
        self.in_nested_sets_from_output_fn = in_nested_sets_from_output_fn

    def forward(self, x):
        return self.baseModel(x)

    # Always outputs [0,1] valued nested sets
    def nested_sets_from_output(self, output, lam=None):
        lower_edge, prediction, upper_edge = self.in_nested_sets_from_output_fn(
            self, output, lam
        )
        upper_edge = torch.maximum(
            upper_edge, prediction + 1e-6
        )  # set a lower bound on the size.
        lower_edge = torch.minimum(lower_edge, prediction - 1e-6)

        return lower_edge, prediction, upper_edge

    # def nested_sets(self, x, lam=None):
    #     if lam == None:
    #         if self.lhat == None:
    #             raise Exception(
    #                 "You have to specify lambda unless your model is already calibrated."
    #             )
    #         lam = self.lhat
    #     output = self(*x)
    #     return self.nested_sets_from_output(output, lam=lam)

def nested_sets(self, output_or_input, lam=None, is_output=True):
    if lam is None:
        if self.lhat is None:
            raise Exception(
                "You have to specify lambda unless your model is already calibrated."
            )
        lam = self.lhat

    if is_output:
        output = output_or_input  # it's already posterior samples
    else:
        output = self(output_or_input)  # need to run the model

    return self.nested_sets_from_output(output, lam=lam)

    def set_lhat(self, lhat):
        self.lhat = lhat


def add_uncertainty_GAN(model, nested_sets_from_output_fn):
    return ModelWithUncertainty(model, nested_sets_from_output_fn)
