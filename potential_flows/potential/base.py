"""Basic definitions for the potential module."""

import numpy as np
import torch
from torch import nn
from typing import Iterable, Union
import sys
sys.path.append("../")
from potential_flows import transforms

class ConjugateNotAvailable(Exception):
    """Exception to be thrown when the conjugate of potential function is not available."""
    pass

class Potential(nn.Module):
    """Base class for input convex potentials."""

    def __init__(self,
                transform: Union[transforms.Transform, None]):

        super(Potential, self).__init__()

        transforms_list = [transforms.IdentityTransform()]
        self.non_identity_transform = False
        if transform:
            transforms_list.append(transform)
            self.non_identity_transform = True
        self.Tmap = transforms.CompositeTransform(transforms_list)


    def integral(self, inputs, context=None):
        raise NotImplementedError()

    def conjugate(self, input, context=None):
        raise ConjugateNotAvailable()

    def gradient(self, input, context=None):
        raise NotImplementedError()

    def gradient_inv(self, input, context=None):
        raise NotImplementedError()
