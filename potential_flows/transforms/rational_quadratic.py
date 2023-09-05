
import os
import sys
# sys.path.append('/Users/vroulet/Git/medha/NSF-KR')
sys.path.append('/home/medhaaga/NSF-KR')
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import statsmodels.api as smi
from potential_flows import transforms
from typing import Callable
import json
import socket
import time

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3 


class RQspline(transforms.Transform):
    def __init__(self, data_shape: tuple = (2,),
                 num_bins: int = 10,
                 normalization: Callable = F.softplus,
                 tail_bound: float = 1.,
                 min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative: float = DEFAULT_MIN_DERIVATIVE) -> None:
        super().__init__()

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


    def inverse(self, inputs, context=None):
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
                                                + input_bin_heights * (input_deltas - input_derivatives)))          # n * d
        b = (input_bin_heights * input_derivatives - (inputs_cpy - input_cumheights) * (input_derivatives
                                                                                    + input_derivatives_plus_one - 2 * input_deltas))
        c = - input_deltas * (inputs_cpy - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs[inside_interval_mask] = (root * input_bin_widths + input_cumwidths)[inside_interval_mask]
        return outputs

    def jacobian(self, inputs, context=None):
        n, d = inputs.shape[0], np.prod(inputs.shape[1:])
        inside_interval_mask = (inputs > -self.tail_bound) & (inputs < self.tail_bound)
        inputs_cpy = torch.clone(inputs)
        outputs = torch.zeros_like(inputs)
        outputs[~inside_interval_mask] = 1 
        inputs_cpy[~inside_interval_mask] = 0
        self.bin_checks()

        widths, cumwidths, heights, cumheights, derivatives, deltas = self.normalized_all_parameters()
        bin_idx = transforms.searchsorted(cumwidths, inputs_cpy)[..., None]
        if ((bin_idx < 0) | (bin_idx > self.num_bins-1)).any():
            raise transforms.InputOutsideDomain()
        input_cumwidths, input_bin_widths, input_cumheights, input_bin_heights, input_deltas, input_derivatives, input_derivatives_plus_one = gather_inputs(bin_idx, widths, cumwidths, heights, cumheights, derivatives, deltas)
        theta = (inputs_cpy - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_deltas.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2*input_deltas*theta_one_minus_theta + input_derivatives*(1-theta).pow(2))
        denominator = (input_deltas + (input_derivatives + input_derivatives_plus_one - 2*input_deltas)*theta_one_minus_theta).pow(2)
        outputs[inside_interval_mask] = (numerator/denominator)[inside_interval_mask]
        
        jac = torch.stack([torch.diag(outputs[i]) for i in range(n)], dim=0)
        return jac

    def integral(self, inputs, context=None):
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


        return torch.sum(outputs.view(inputs.shape[0],-1), dim=-1)

    def integral_conjugate(self, inputs_to, context=None):
        n = inputs_to.shape[0]
        reverse_inputs = self.inverse(inputs_to)
        if torch.isnan(reverse_inputs).any():
            raise ValueError
        integral_conjugates = torch.sum(
            (reverse_inputs*inputs_to).view(n,-1), dim=-1) - self.integral(reverse_inputs)
        return integral_conjugates


    def plot_flow(self, n_points=100, log_dir=None, save_fig=False):
 
        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1)
        forward_vals = self.forward(inputs)
        reverse_vals = self.inverse(inputs)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for i,c in zip(range(2), ['deepskyblue', 'navy']):
            axs[0].plot(inputs.detach().numpy()[:, i], forward_vals.detach().numpy()[:, i], label='Dim-{}'.format(i+1), color=c, linewidth=2-i)
        axs[0].set_title(r'$\nabla f$')
        axs[0].legend()
        for i,c in zip(range(2), ['deepskyblue', 'navy']):
            axs[1].plot(inputs.detach().numpy()[:, i], reverse_vals.detach().numpy()[:, i], label='Dim-{}'.format(i+1), color=c, linewidth=2-i)
        axs[1].set_title(r'$(\nabla f)^{-1}$')
        axs[1].legend()
        if save_fig:
            if log_dir is None:
                raise ValueError('No log directory provided for saving figure.')
            else:
                fig.savefig(os.path.join(log_dir, 'spline.png'))
        plt.show()
        plt.close(fig)

    def plot_phi_and_phi_conjugate(self, log_dir, n_points=100, save_fig=True):
        inputs = torch.cat([torch.linspace(-2*self.tail_bound, 2*self.tail_bound,
                           n_points).unsqueeze(-1)]*np.prod(self.data_shape), dim=1)
        phi = self.integral(inputs)
        phi_conjugate = self.integral_conjugate(inputs)

        fig, axs = plt.subplots(1, 2, figsize=(6,3))

        for i in range(2):
            axs[0].plot(inputs.detach().numpy()[:,i], phi.detach().numpy(), label='Dimension-{}'.format(i+1))
        axs[0].set_title(r'$\phi$')
        for i in range(2):
            axs[1].plot(inputs.detach().numpy()[:,i], phi_conjugate.detach().numpy(), label='Dimension-{}'.format(i+1))
        axs[1].set_title(r'$\phi^*$')

        plt.legend()
        if save_fig:
            plt.savefig(os.path.join(log_dir, 'potentials.png'))
        plt.show()
        plt.close(fig)
  
    def generate_data(self, log_dir, data, n_gen_samp=1000, save_fig=True):
        with torch.no_grad():
            noise = torch.randn(n_gen_samp, *self.data_shape)
            samples = self.inverse(noise)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            ax1.scatter(data[:, 0], data[:, 1], s=2, color='powderblue')
            ax2.scatter(noise[:, 0], noise[:, 1], s=2, color='powderblue')
            ax3.scatter(samples[:, 0], samples[:, 1], s=2, color='powderblue')
            ax1.set_title('Original Samples')
            ax2.set_title('Noise')
            ax3.set_title('Generated Samples')
            if save_fig:
                plt.savefig(os.path.join(log_dir, 'samples.png'))
            plt.show()
            plt.close(fig)

    def plot_normalize_data(self, log_dir, data, save_fig=True):
        with torch.no_grad():
            noise = self.forward(data)

            noise = noise.detach().numpy()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ax1.scatter(data[:, 0], data[:, 1], s=2, color='powderblue')
            ax2.scatter(noise[:, 0], noise[:, 1], s=2, color='powderblue')
            ax1.set_title('Original Samples')
            ax2.set_title('Normalized Samples')
            if save_fig:
                plt.savefig(os.path.join(log_dir, 'normalized.png'))
            plt.show()
            plt.close(fig)

            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            smi.qqplot(noise[:, 0], line='45', ax=axs[0])
            smi.qqplot(noise[:, 1], line='45', ax=axs[1])
            if save_fig:
                plt.savefig(os.path.join(log_dir, 'qq_plots.png'))
            plt.show()
            plt.close(fig)


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
    



def normalize_spline_parameters(unnormalized_params, tail_bound=1, normalization=F.softplus, min_param=DEFAULT_MIN_BIN_WIDTH):
    
    num_bins = unnormalized_params.shape[-1]
    params = normalization(unnormalized_params)/torch.sum(normalization(unnormalized_params), dim=-1).unsqueeze(-1)
    params = min_param + (1-min_param*num_bins)*params
    cum_params = torch.cumsum(params, dim=-1)
    cum_params = F.pad(cum_params, pad=(1, 0), value=0.0)
    cum_params = 2*tail_bound*cum_params - tail_bound
    params = cum_params[..., 1:] - cum_params[..., :-1]
    return params, cum_params
 

 
def gather_inputs(bin_idx, widths, cumwidths, heights, cumheights, derivatives, deltas):
    input_cumwidths = torch.cat([cumwidths[None,...]]*bin_idx.shape[0], dim=0).gather(-1, bin_idx)[..., 0] # shape: n * d 
    input_bin_widths = torch.cat([widths[None,...]]*bin_idx.shape[0], dim=0).gather(-1, bin_idx)[..., 0] # shape: n * d 
    input_cumheights = torch.cat([cumheights[None,...]]*bin_idx.shape[0], dim=0).gather(-1, bin_idx)[..., 0] # shape: n * d 
    input_bin_heights = torch.cat([heights[None,...]]*bin_idx.shape[0], dim=0).gather(-1, bin_idx)[..., 0]# shape: n * d 
    input_derivatives = torch.cat([derivatives[None,...]]*bin_idx.shape[0], dim=0).gather(-1, bin_idx)[..., 0] # shape: n * d 
    input_derivatives_plus_one = torch.cat([derivatives[None,...]]*bin_idx.shape[0], dim=0)[..., 1:].gather(-1, bin_idx)[..., 0]# shape: n * d 
    input_deltas = torch.cat([deltas[None,...]]*bin_idx.shape[0], dim=0).gather(-1, bin_idx)[..., 0]
    return input_cumwidths, input_bin_widths, input_cumheights, input_bin_heights, input_deltas, input_derivatives, input_derivatives_plus_one


## This function calculates the integral of one bin using Monte Carlo approximation

def RQ_bin_integral(x_k, x, y_k, w_k, h_k, der_k, der_k_plus_one, N=1000, eps=1e-5):
    device = x_k.device
    s_k = h_k/w_k
    tau_k = der_k + der_k_plus_one - 2 * s_k # shape: 1 * d
    x_minus_x_k = x - x_k
    bin_integral = y_k*x_minus_x_k
    line = torch.linspace(0,1-eps,N).to(device)
    mc = torch.cat([line]*np.prod(x_k.shape)).reshape(*x_k.shape, N)*(x_minus_x_k/w_k).unsqueeze(-1).to(device)
    mc_num = (s_k - der_k).unsqueeze(-1)*mc.pow(2) + der_k.unsqueeze(-1)*mc
    mc_denom = -tau_k.unsqueeze(-1)*mc.pow(2) + tau_k.unsqueeze(-1)*mc + s_k.unsqueeze(-1)
    bin_integral += h_k*x_minus_x_k*torch.sum(mc_num/mc_denom, dim=-1)/N
    return bin_integral
