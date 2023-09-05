"""Implementations of linear transforms."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, init
from typing import Callable
from .utils import is_positive_int, is_bool, random_orthogonal
from potential_flows import transforms


class LinearCache(object):
    """Helper class to store the cache of a linear transform.

    The cache consists of: the weight matrix, its inverse and its log absolute determinant.
    """

    def __init__(self):
        self.weight = None
        self.inverse = None

    def invalidate(self):
        self.weight = None
        self.inverse = None


class Linear(transforms.Transform):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        if not is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        # Caching flag and values.
        self.using_cache = using_cache
        self.cache = LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            return outputs
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None:
            self.cache.weight = self.weight()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            return outputs
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()

    def train(self, mode=True):
        if mode:
            # If training again, invalidate cache.
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        if not is_bool(mode):
            raise TypeError('Mode must be boolean.')
        self.using_cache = mode

    def weight(self):
        return self.weight()

    def weight_inverse(self):
        return self.weight_inverse()

    def forward_no_cache(self, inputs):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def jacobian(self, inputs, context=None):
        n = inputs.shape[0]
        return torch.stack([self.weight()]*n, dim=0)


class NaiveLinear(Linear):
    """A general linear transform that uses an unconstrained weight matrix.

    This transform explicitly computes the log absolute determinant in the forward direction
    and uses a linear solver in the inverse direction.

    Both forward and inverse directions have a cost of O(D^3), where D is the dimension
    of the input.
    """

    def __init__(self, features, orthogonal_initialization=True, using_cache=False):
        """Constructor.

        Args:
            features: int, number of input features.
            orthogonal_initialization: bool, if True initialize weights to be a random
                orthogonal matrix.

        Raises:
            TypeError: if `features` is not a positive integer.
        """
        super().__init__(features, using_cache)

        if orthogonal_initialization:
            self._weight = nn.Parameter(random_orthogonal(features))
        else:
            self._weight = nn.Parameter(torch.empty(features, features))
            stdv = 1.0 / np.sqrt(features)
            init.uniform_(self._weight, -stdv, stdv)

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = F.linear(inputs, self._weight, self.bias)
        return outputs

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^3 + D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = inputs - self.bias
        outputs, lu = torch.gesv(outputs.t(), self._weight)  # Linear-system solver.
        outputs = outputs.t()
        return outputs

    def weight(self):
        """Cost:
            weight = O(1)
        """
        return self._weight

    def weight_inverse(self):
        """
        Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        return torch.inverse(self._weight)

    def jacobian(self, inputs, context=None):
        n = inputs.shape[0]
        return torch.stack([self.weight()]*n, dim=0)

class PositiveLinear(Linear):
    """A general linear transform that uses an all positive elements weight matrix.

    This transform explicitly computes the log absolute determinant in the forward direction
    and uses a linear solver in the inverse direction.

    The forward directions have a cost of O(D^2), where D is the dimension
    of the input.

    The reverse directions have a cost of O(D^3), where D is the dimension
    of the input.
    """

    def __init__(self, 
                features: int, 
                non_neg_transform: Callable = nn.functional.softplus, 
                orthogonal_initialization: bool = True, 
                using_cache: bool = False):
        
        """Constructor.

        Args:
            features: int, number of input features.
            orthogonal_initialization: bool, if True initialize weights to be a random
                orthogonal matrix.

        Raises:
            TypeError: if `features` is not a positive integer.
        """
        super().__init__(features, using_cache)

        if orthogonal_initialization:
            self._weight = nn.Parameter(random_orthogonal(features))
        else:
            self._weight = nn.Parameter(torch.empty(features, features))
            stdv = 1.0 / np.sqrt(features)
            init.uniform_(self._weight, -stdv, stdv)
        
        self.non_neg_transform = non_neg_transform

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = F.linear(inputs, self.non_neg_transform(self._weight), self.bias)
        return outputs

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^3 + D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = inputs - self.bias
        outputs, lu = torch.gesv(outputs.t(), self.non_neg_transform(self._weight))  # Linear-system solver.
        outputs = outputs.t()
        return outputs

    def weight(self):
        """Cost:
            weight = O(1)
        """
        return self.non_neg_transform(self._weight)

    def weight_inverse(self):
        """
        Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        return torch.inverse(self.non_neg_transform(self._weight))

    def jacobian(self, inputs, context=None):
        n = inputs.shape[0]
        return torch.stack([self.weight()]*n, dim=0)
