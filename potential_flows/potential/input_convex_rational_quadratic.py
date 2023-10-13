
"""
Class of rational quadratic splines based potential flow.

The potential function is parameterized as composition of 
input-convex component-wise RQ splines S: R^D --> R and 
function T: R^D --> R^D

The class implements RQ spline forward, reverse, integral,
and conjugate of integral.

All of the above methods are O(D) where D is the problem dim.

"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from typing import Callable, Iterable, Union
from potential_flows import potential
from potential_flows import transforms
from potential_flows.transforms import RQ_bin_integral, normalize_spline_parameters, gather_inputs



DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3 

class ICRQ(potential.Potential):
    def __init__(self, 
                 transforms_list: Union[transforms.Transform, None] = None,
                 data_shape: tuple = (2,),
                 num_bins: int = 10,
                 normalization: Callable = F.softplus,
                 tail_bound: float = 1.,
                 min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative: float = DEFAULT_MIN_DERIVATIVE) -> None:

        """Constructor.

        Args:
            transforms_list: Union[transforms.Transform, None], composition of transforms
            that are applied on input before feeding into the input convex RQ spline.
            data shape: needs to be a tuple of data-shape. In case of tabular data, it is (D,)
            num_bins: number of bins in all D RQ splines.
            normalization: method fornormalizing the bin widths and heights.
            tail_bound: end points for all D splines. Should be chosen according to data range


        Raises:
            TypeError: if `data_shape` is not a tuple with positive product
        """
        super().__init__(transforms_list)
        if not isinstance(data_shape, tuple):
            raise TypeError
        if np.prod(data_shape) <=0:
            raise TypeError

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

    def forward(self, inputs, context=None):
        """
        Calculates S(T(X)) 
        """
        assert inputs.device == self.device()
        inputs = self.Tmap(inputs)
        outputs = self.spline_integral(inputs)
        return torch.sum(outputs.view(inputs.shape[0],-1), dim=-1)

    def conjugate(self, inputs_to, context=None):
        """"
        Calculates S*(Y) when T is identity function.
        Note implemented otherwise.

        """
        assert inputs_to.device == self.device()
        if not self.non_identity_transform:
            n = inputs_to.shape[0]
            reverse_inputs = self.gradient_inv(inputs_to)
            if torch.isnan(reverse_inputs).any():
                raise ValueError
            integral_conjugates = torch.sum(
                (reverse_inputs*inputs_to).view(n,-1), dim=-1) - self.forward(reverse_inputs)
            return integral_conjugates
        else:
            raise potential.base.ConjugateNotAvailable()

    def gradient(self, inputs_from, context=None):
        """
        Calculates gradient of S(T(X)) with respect to X.

        If T is identity, then gradient of S(X) is the 
        spline forward.

        If T is not identity, then the gradient is calculated 
        using vector-Jacobian product.

        """
        assert inputs_from.device == self.device()
        if not self.non_identity_transform:
            return self.spline_forward(inputs_from)
        else:
            inputs_from.requires_grad = True
            T_x = self.Tmap(inputs_from)
            v = self.spline_forward(T_x)
            T_x.backward(gradient=v, retain_graph=True)
            return inputs_from.grad
            

    def gradient_inv(self, inputs_to):
        """
        Calculates (\nabla S)^{-1}(X) when T is an identity functsion.

        Gives NotImplementedError otherwise.
        """
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

    def spline_integral(self, inputs, context=None):
        assert inputs.device == self.device()
        
        ## mask for inputs within tail bounds
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
        if ((bin_idx < 0) | (bin_idx > self.num_bins-1)).any():
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


        return outputs.view(inputs.shape[0],-1)


    def plot_flow(self, n_points=100, log_dir=None, show_figure=True, save_fig=False):
 
        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1)
        inputs = inputs.to(self.device())
        inputs.requires_grad = True
        forward_vals = self.gradient(inputs)
        d = np.prod(self.data_shape)
        plt.figure(figsize=(5, 3))
        for i in range(d):
            plt.plot(inputs.detach().cpu().numpy()[:, i], forward_vals.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
        plt.title(r'$\nabla f$')
        plt.legend()
        if show_figure:
            plt.show()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                plt.savefig(os.path.join(log_dir, 'spline.png'))
        plt.close()

    def plot_potential(self, n_points=100, log_dir=None, save_fig=False):

        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1)
        inputs = inputs.to(self.device())
        forward_vals = self.forward(inputs)
        d = np.prod(self.data_shape)
        plt.figure(figsize=(5, 3))
        for i in range(d):
            plt.plot(inputs.detach().cpu().numpy()[:, i], forward_vals.detach().cpu().numpy(),  label='Dim-{}'.format(i+1), linewidth=2)
        plt.title(r'$f$')
        plt.legend()
        plt.show()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                plt.savefig(os.path.join(log_dir, 'potential.png'))
        
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

    def plot_grad_potential(self, n_points=100, log_dir=None, save_fig=False):
        
        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1).to(self.device())
        d = np.prod(self.data_shape)
        forward_vals = self.spline_forward(inputs)
        plt.figure(figsize=(5, 3))
        for i in range(d):
            plt.plot(inputs.detach().cpu().numpy()[:, i], forward_vals.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
        plt.title(r'$ \nabla \phi$')
        plt.legend()
        plt.show()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                plt.savefig(os.path.join(log_dir, 'spline.png'))
        
        plt.close()

    def grad_phi_T(self, n_points=100, log_dir=None, save_fig=False):

        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1).to(self.device())

        inputs = self.Tmap(inputs)
        d = np.prod(self.data_shape)
        forward_vals = self.spline_forward(inputs)
        plt.figure(figsize=(5, 3))
        for i in range(d):
            plt.plot(inputs.detach().cpu().numpy()[:, i], forward_vals.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
        plt.title(r'$ \nabla \phi(T(x))$')
        plt.legend()
        plt.show()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                plt.savefig(os.path.join(log_dir, 'spline.png'))
        
        plt.close()

    def plot_spline_integral(self, n_points=100, log_dir=None, save_fig=False):

        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1).to(self.device())
        T_x = self.Tmap(inputs)
        spline_integrals = self.spline_integral(inputs)
        integral_T_x = self.spline_integral(T_x)
        forward_vals = self.forward(inputs)
        d = np.prod(self.data_shape)
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(d):
            axs[0].plot(inputs.detach().cpu().numpy()[:, i], T_x.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
            axs[0].set_title(r"$T(x)$")
            axs[1].plot(inputs.detach().cpu().numpy()[:, i], spline_integrals.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
            axs[1].set_title(r"$\phi_i(x)$")
            axs[2].plot(inputs.detach().cpu().numpy()[:, i], integral_T_x.detach().cpu().numpy()[:, i], label='Dim-{}'.format(i+1), linewidth=2)
            axs[2].set_title(r"$\phi_i(T(x))$")
            axs[3].plot(inputs.detach().cpu().numpy()[:, i], forward_vals.detach().cpu().numpy(), label='Dim-{}'.format(i+1), linewidth=2)
            axs[3].set_title(r"$\phi(T(x))$")
            plt.legend()
            
        
        plt.show()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                plt.savefig(os.path.join(log_dir, 'spline_integral.png'))
        
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