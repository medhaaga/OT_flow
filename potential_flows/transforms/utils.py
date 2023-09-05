import torch
import os
import socket
import time


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass

def searchsorted(bin_locations, inputs, eps=1e-6):
    #bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations,dim=-1) -1

def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if num_batch_dims<0:
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time

def on_cluster():
    hostname = socket.gethostname()
    return True if hostname == 'zh-gpu2' else False

def get_log_root(transport, dataset_name):
    project_dir = os.path.dirname(os.path.dirname(__file__))
    exp_folder = os.path.join(project_dir, 'experiments', transport, dataset_name)
    os.makedirs(exp_folder, exist_ok=True)
    
    return exp_folder

def is_positive_int(x):
    return isinstance(x, int) and x > 0

def is_bool(x):
    return isinstance(x, bool)


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q
