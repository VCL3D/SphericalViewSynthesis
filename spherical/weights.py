import torch

from .grid import *

def phi_confidence(sgrid): # fading towards horizontal singularities
    return torch.abs(torch.sin(phi(sgrid)))

def theta_confidence(sgrid): # fading towards vertical singularities
    return torch.abs(torch.cos(theta(sgrid)))

def spherical_confidence(sgrid, zero_low=0.0, one_high=1.0):
    weights = phi_confidence(sgrid) * theta_confidence(sgrid)
    weights[weights < zero_low] = 0.0
    weights[weights > one_high] = 1.0
    return weights