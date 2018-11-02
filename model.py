from statistics import mean
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch

def calc_centroid(matrix):
    """
    Calculate centroid of numpy matrix

    Returns: 1D torch tensor.
    """

    return torch.mean(torch.from_numpy(matrix), dim=0)
