"""Implementations of linear transforms."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from typing import Callable, Iterable, Union
from potential_flows import potential
from potential_flows import transforms
from transforms.rational_quadratic import normalize_spline_parameters, gather_inputs, RQ_bin_integral


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3 

class ICRQ(potential.Potential):
    def __init__(self, 
                 transforms_list: Union[transforms.Transform, None],
                 data_shape: tuple = (2,),
                 num_bins: int = 10,
                 normalization: Callable = F.softplus,
                 tail_bound: float = 1.,
                 min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative: float = DEFAULT_MIN_DERIVATIVE) -> None:
        super().__init__(transforms_list)
        self.num_bins = num_bins
        self.widths = nn.Parameter(torch.randn(*data_shape, self.num_bins))
        self.heights = nn.Parameter(torch.randn(*data_shape, self.num_bins))
        self.derivatives = nn.Parameter(torch.randn(*data_shape, self.num_bins+1))
        self.normalization = normalization
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.data_shape = data_shape

    def integral(self, inputs, context=None):
        assert inputs.device == self.device()
        inputs = self.Tmap(inputs)
        inside_interval_mask = (inputs > -self.tail_bound) & (inputs < self.tail_bound)
        outputs = torch.zeros_like(inputs)
        inputs_cpy = torch.clone(inputs)
        inputs_cpy[~inside_interval_mask] = 0
        self.bin_checks()

        lesser_than_B_mask = inputs <= -self.tail_bound
        greater_than_B_mask = inputs >= self.tail_bound
        inputs_cpy[lesser_than_B_mask], inputs_cpy[greater_than_B_mask] = -self.tail_bound+1e-5, self.tail_bound - 1e-5
        widths, cumwidths, heights, cumheights, derivatives, deltas = self.normalized_all_parameters()
        bin_idx = transforms.searchsorted(cumwidths, inputs_cpy)[..., None]
        # print(f'Bin index: {bin_idx}, shape: {bin_idx.shape}')
        if ((bin_idx < 0) | (bin_idx > self.num_bins-1)).any():
            print(bin_idx)
            raise transforms.InputOutsideDomain()
        input_cumwidths, input_bin_widths, input_cumheights, input_bin_heights, input_deltas, input_derivatives, input_derivatives_plus_one = gather_inputs(bin_idx, widths, cumwidths, heights, cumheights, derivatives, deltas)
    

        bin_integrals = RQ_bin_integral(cumwidths[...,:-1], cumwidths[...,1:], cumheights[...,:-1], widths, heights, 
                                        derivatives[...,:-1], derivatives[...,1:])

        cum_bin_integrals = torch.cumsum(bin_integrals, dim=-1) # shape: 1 * d * K
        cum_bin_integrals = F.pad(cum_bin_integrals[None,...], pad=(1,0), mode='constant', value=0.0) # shape: 1 * d * K+1
        outputs = torch.cat([cum_bin_integrals.clone()]*bin_idx.shape[0], dim=0).gather(-1, bin_idx)[..., 0] 
        outputs += RQ_bin_integral(input_cumwidths, inputs, input_cumheights, input_bin_widths, input_bin_heights, 
                                    input_derivatives, input_derivatives_plus_one)

        outputs[lesser_than_B_mask] = (inputs[lesser_than_B_mask].pow(2) - 4*(self.tail_bound**2))/2
        outputs[inside_interval_mask] -= (3*(self.tail_bound**2))/2
        outputs[greater_than_B_mask] = (torch.cat([cum_bin_integrals] * bin_idx.shape[0], dim=0)[..., -1])[greater_than_B_mask]
        outputs[greater_than_B_mask] += (inputs[greater_than_B_mask].pow(2) - 4*(self.tail_bound**2))/2


        return torch.sum(outputs.view(inputs.shape[0],-1), dim=-1)

    def conjugate(self, inputs_to, context=None):
        assert inputs_to.device == self.device()
        if not self.non_identity_transform:
            n = inputs_to.shape[0]
            reverse_inputs = self.inverse(inputs_to)
            if torch.isnan(reverse_inputs).any():
                raise ValueError
            integral_conjugates = torch.sum(
                (reverse_inputs*inputs_to).view(n,-1), dim=-1) - self.integral(reverse_inputs)
            return integral_conjugates
        else:
            raise potential.base.ConjugateNotAvailable()

    def gradient(self, inputs_from, context=None):
        if not self.non_identity_transform:
            return self.spline_forward(inputs_from)
        else:
            inputs = self.Tmap(inputs_from) # T(x)
            # return torch.einsum("bi,bij->bj", self.spline_forward(inputs), self.Tmap.jacobian(inputs_from))
            return self.spline_forward(inputs)

    def gradient_inv(self, inputs_to):
        if not self.non_identity_transform:
            return self.spline_reverse(inputs_to)
        else:
            raise NotImplementedError

    def spline_forward(self, inputs, context=None):
        assert inputs.device == self.device()
        inside_interval_mask = (inputs > -self.tail_bound) & (inputs < self.tail_bound)
        outputs = torch.zeros_like(inputs)
        inputs_cpy = torch.clone(inputs)
        outputs[~inside_interval_mask] = inputs[~inside_interval_mask]
        inputs_cpy[~inside_interval_mask] = 0
        self.bin_checks()

        widths, cumwidths, heights, cumheights, derivatives, deltas = self.normalized_all_parameters()
        bin_idx = transforms.searchsorted(cumwidths, inputs_cpy)[..., None]
        if ((bin_idx < 0) | (bin_idx > self.num_bins-1)).any():
            raise transforms.InputOutsideDomain()

        input_cumwidths, input_bin_widths, input_cumheights, input_bin_heights, input_deltas, input_derivatives, input_derivatives_plus_one = gather_inputs(bin_idx, widths, cumwidths, heights, cumheights, derivatives, deltas)
        theta = (inputs_cpy - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_bin_heights * (input_deltas * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_deltas + ((input_derivatives + input_derivatives_plus_one - 2 * input_deltas) * theta_one_minus_theta)
        outputs[inside_interval_mask] = (input_cumheights + numerator / denominator)[inside_interval_mask]

        return outputs
    ## This function calculates the integral of one bin using Monte Carlo approximation

    def spline_reverse(self, inputs, context=None):
        assert inputs.device == self.device()
        inside_interval_mask = (inputs > -self.tail_bound) & (inputs < self.tail_bound)
        outputs = torch.zeros_like(inputs)
        inputs_cpy = torch.clone(inputs)
        outputs[~inside_interval_mask] = inputs[~inside_interval_mask]
        inputs_cpy[~inside_interval_mask] = 0
        self.bin_checks()

        widths, cumwidths, heights, cumheights, derivatives, deltas = self.normalized_all_parameters()
        bin_idx = transforms.searchsorted(cumheights, inputs_cpy)[..., None]
        if ((bin_idx < 0) | (bin_idx > self.num_bins-1)).any():
            raise transforms.InputOutsideDomain()
        input_cumwidths, input_bin_widths, input_cumheights, input_bin_heights, input_deltas, input_derivatives, input_derivatives_plus_one = gather_inputs(bin_idx, widths, cumwidths, heights, cumheights, derivatives, deltas)
    

        a = (((inputs_cpy - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_deltas)
                                                + input_bin_heights * (input_deltas - input_derivatives)))
        b = (input_bin_heights * input_derivatives - (inputs_cpy - input_cumheights) * (input_derivatives
                                                                                    + input_derivatives_plus_one - 2 * input_deltas))
        c = - input_deltas * (inputs_cpy - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs[inside_interval_mask] = (root * input_bin_widths + input_cumwidths)[inside_interval_mask]
        return outputs

    def plot_flow(self, n_points=100, log_dir=None, save_fig=False):
 
        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1)
        inputs = inputs.to(self.device())
        d = np.prod(self.data_shape)
        forward_vals = self.gradient(inputs).to(self.device())
        plt.figure(figsize=(5, 3))
        for i in range(d):
            plt.plot(inputs.detach().cpu().numpy()[:, i], forward_vals.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
        plt.title(r'$\nabla f$')
        plt.legend()
        plt.show()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                plt.savefig(os.path.join(log_dir, 'spline.png'))
        
        plt.close()
    
    def plot_Tmap(self, n_points=100, log_dir=None, save_fig=False):
        
        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1).to(self.device())
        d = np.prod(self.data_shape)
        forward_vals = self.Tmap(inputs)
        plt.figure(figsize=(5, 3))
        for i in range(d):
            plt.plot(inputs.detach().cpu().numpy()[:, i], forward_vals.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
        plt.title(r'$ T$')
        plt.legend()
        plt.show()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                plt.savefig(os.path.join(log_dir, 'spline.png'))
        
        plt.close()

        

    def bin_checks(self):
        if self.min_bin_width * self.num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if self.min_bin_height * self.num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

    def normalized_all_parameters(self):
    
        widths, cumwidths = normalize_spline_parameters(self.widths, tail_bound=self.tail_bound, normalization=self.normalization, min_param=self.min_bin_width)
        heights, cumheights = normalize_spline_parameters(self.heights, tail_bound=self.tail_bound, normalization=self.normalization, min_param=self.min_bin_height)
        derivatives = self.min_derivative + F.softplus(self.derivatives)
        derivatives[...,0] = 1
        derivatives[...,-1] = 1
        deltas = heights/widths
        return widths, cumwidths, heights, cumheights, derivatives, deltas

    def x_knots(self):
        return normalize_spline_parameters(self.widths, tail_bound=self.tail_bound, normalization=self.normalization, min_param=self.min_bin_width)
    
    def y_knots(self):
        return normalize_spline_parameters(self.heights, tail_bound=self.tail_bound, normalization=self.normalization, min_param=self.min_bin_height)
    
    def der_knots(self):
        return normalize_spline_parameters(self.derivatives, tail_bound=self.tail_bound, normalization=self.normalization, min_param=self.min_derivative)
   
    def device(self):
        return self.widths.device