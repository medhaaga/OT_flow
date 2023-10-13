"""Basic definitions for the transforms module."""

import numpy as np
import torch
from torch import nn
from potential_flows import transforms

class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""
    pass

class JacobianNotAvailable(Exception):
    """Exception to be thrown when a transform does not have a jacobian."""
    pass


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass


class Transform(nn.Module):
    """Base class for all transform objects."""

    def __init__(self):
        super(Transform, self).__init__()

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()

    def jacobian(self, inputs, context=None):
        raise JacobianNotAvailable


class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        outputs = inputs

        for func in funcs:
            outputs = func(outputs, context)
        return outputs

    @staticmethod
    def jac_cascade(inputs, funcs, context):
        dim = np.prod(inputs.shape[1:])
        n = inputs.shape[0]
        jac = torch.stack([torch.eye(dim)]*n, dim=0)
        outputs = inputs

        for func in funcs:
            func_jac = func.jacobian(outputs, context)
            jac = torch.einsum("bij,bjk->bik", func_jac, jac)
            outputs = func(outputs, context)
        return jac

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)
    
    def jacobian(self, inputs, context=None):
        funcs = self._transforms
        return self.jac_cascade(inputs, funcs, context)


class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform):
        """Constructor.

        Args:
            transform: An object of type `Transform`.
        """
        super().__init__()
        self._transform = transform

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context)

class IdentityTransform(Transform):
    """Transform that leaves input unchanged."""
    def __init__(self):
        super().__init__()

    def forward(self, inputs, context=None):
        return inputs

    def inverse(self, inputs, context=None):
        return inputs

    def jacobian(self, inputs, context=None):
        dim = np.prod(inputs.shape[1:])
        n = inputs.shape[0]
        return torch.stack([torch.eye(dim)]*n, dim=0)
