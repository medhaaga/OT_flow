import numpy as np
import torch
import os
import sys
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--source_dist', type=str, default='gaussian',
                        choices=['banana', 'sine', 'crescent', 'two_spirals', 'checker', 'our_checker', 'eight_gmm', 'gaussian', 'four_gaussians', 'nine_gaussians', 'custom'],
                        help='Name of source dataset to use.')     
    parser.add_argument('--target_dist', type=str, default='gaussian',
                        help='target distribution name')
    parser.add_argument('--data_shape', type=tuple, default=(2,))           
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='number of samples.')
    parser.add_argument('--test_num_samples', type=int, default=500,
                        help='number of test samples')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Size of batch used for training.')
    parser.add_argument('--regularization', type=float, default=1e-5,
                        help='Size of batch used for training.')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer.')
    parser.add_argument('--max_inner_iter', type=int, default=5,
                        help='Inner iterations for minimization.')
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Number of pochs.')
    parser.add_argument('--anneal_learning_rate', type=str, default='none',
                        choices=['cosine', 'exponential', 'none'],
                        help='Whether to anneal the learning rate.')
    parser.add_argument('--grad_norm_clip_value', type=float, default=5.,
                        help='Value by which to clip norm of gradients.')
    parser.add_argument('--preload', type=int, default=0,
                        help='Do you want to preload a model or train from scratch?')
    parser.add_argument('--tolerance', type=float, default=0.0005,
                        help='tolerance for optimization algo')

    # flow details
    parser.add_argument('--num_bins', type=int, default=4,
                        help='Number of bins to use for piecewise transforms.')
    parser.add_argument('--tail_factor', type=float, default=1.5,
                        help='Factor times the max input value is tail bound.')
    parser.add_argument('--normalization', type=str, default='softplus',
                        help='method for normalizing the bin widths and heights.')


    # logging and checkpoints
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Interval in steps at which to report training stats.')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Print logs in shell?')
    parser.add_argument('--save_figure_interval', type=int, default=100)


    # reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for PyTorch and NumPy.')
    
    # optimization routine
    parser.add_argument('--method', type=str, default='dual',
                        choices = ['dual', 'minimax'])

    # minimax optimization specs
    parser.add_argument('--model_variant', type=str, default='alternate',
                        choices=['alternate', 'step_through_max'])
    
    # figure specifications
    parser.add_argument('--show_the_plot', type=bool, default=False)

    # device 
    parser.add_argument('--device_name', type=str, default='cpu')
    return parser

def parse_arguments():
    parser = get_parser()
    args = parser.parse_args()
    return args

def parse_notebook_arguments():
    parser = get_parser()
    args = parser.parse_args([])
    return args

# setting seed for reproducibility
def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

def get_exp_dir():
    root = os.getcwd()
    return os.path.join(root, 'experiments')