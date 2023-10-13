import numpy as np
import os
import sys
# sys.path.append('/home/medhaaga/nsf')
import torch
from torch.utils.data import Dataset
from potential_flows import potential


class BananaDataset(Dataset):
    def __init__(self, n):
        self.num_samples = n
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_samples

    def create_data(self):
        x1 = torch.randn(self.num_samples, 1)
        x2 = (x1**2)/2 + np.sqrt(0.5)*torch.randn(self.num_samples, 1)
        x = torch.cat([x1, x2], dim=1)
        return (x - torch.mean(x, dim=0))/torch.std(x, dim=0)

class FourGaussians(Dataset):
    def __init__(self, n):
        self.num_samples = n
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_samples

    def create_data(self):
        x1 = torch.cat([torch.randn(int(self.num_samples/2), 1)-3, torch.randn(int(self.num_samples/2), 1)+3])
        x2 = torch.cat([torch.randn(int(self.num_samples/2), 1)-3, torch.randn(int(self.num_samples/2), 1)+3])
        x = torch.cat([x1[torch.randperm(x1.size()[0])], x2[torch.randperm(x2.size()[0])]], dim=1)
        return (x - torch.mean(x, dim=0))/torch.std(x, dim=0)

class NineGaussians(Dataset):
    def __init__(self, n):
        self.num_samples = n
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_samples

    def create_data(self):
        x1 = torch.cat([torch.randn(int(self.num_samples/3), 1)-6, torch.randn(int(self.num_samples/3), 1), torch.randn(self.num_samples - 2*int(self.num_samples/3), 1)+6])
        x2 = torch.cat([torch.randn(int(self.num_samples/3), 1)-6, torch.randn(int(self.num_samples/3), 1), torch.randn(self.num_samples - 2*int(self.num_samples/3), 1)+6])
        x = torch.cat([x1[torch.randperm(x1.size()[0])], x2[torch.randperm(x2.size()[0])]], dim=1)
        return (x - torch.mean(x, dim=0))/torch.std(x, dim=0)


class SineWaveDataset(Dataset):
    def __init__(self, n):
        self.num_points = n
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sin(5 * x1)
        x2_var = torch.exp(-2 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        return torch.stack((x1, x2)).t()

class CrescentCubedDataset(Dataset):
    def __init__(self, n):
        self.num_points = n
        if self.num_points % 4 != 0:
            raise ValueError('Number of data points must be a multiple of four')
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    @staticmethod
    def create_circle(num_per_circle, std=0.1):
        u = torch.rand(num_per_circle)
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        return data

    def create_data(self):
        num_per_circle = self.num_points // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        return torch.cat(
            [self.create_circle(num_per_circle) - torch.Tensor(center)
             for center in centers]
        )

class TwoSpiralsDataset(Dataset):
    def __init__(self, n):
        self.num_points = n
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def create_data(self):
        n = torch.sqrt(torch.rand(self.num_points // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(self.num_points // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(self.num_points // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
        return x / 3 + torch.randn_like(x) * 0.1


class Eight_GMM(Dataset):
    """samples from four 2D gaussians."""

    def __init__(self,
                 n: int=2000,
                 scale: float=10.,
                 eps_noise: float=.5):
        self.num_points = n
        scale = scale
        self.eps_noise = eps_noise

        centers = [
			(1,0),
			(-1,0),
			(0,1),
			(0,-1),
			(1./np.sqrt(2), 1./np.sqrt(2)),
			(1./np.sqrt(2), -1./np.sqrt(2)),
			(-1./np.sqrt(2), 1./np.sqrt(2)),
			(-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        
        self.centers = [(scale*x,scale*y) for x,y in centers]
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def create_data(self):
        batch = []
        for i in range(self.num_points):
            
            point = np.random.randn(2) * self.eps_noise
            center = self.centers[i % 8]
            point[0] += center[0]
            point[1] += center[1]
            batch.append(point)
        batch = np.array(batch, dtype='float32')
        batch = torch.from_numpy(batch)
        return batch

class Our_Checkerboard(Dataset):
    """samples from one of two sets 2D squares (depending on alternate=T/F)."""

    def __init__(self,
                 n: int=256,
                 scale: float=5.0,
                 eps_noise: float=0.5,
                 five_squares: bool=False):
        self.num_points = n
        self.scale = scale
        self.eps_noise = eps_noise
        self.five_squares = five_squares
        if self.five_squares:
            centers = [ (0., 0.), (1., 1.), (-1., 1.),\
                    (-1., -1.), (1., -1.)
                ]
        else:
            centers = [ (1., 0.), (0., 1.), (-1., 0.), (0., -1.)
                ]

        self.centers = [(scale * x, scale * y) for x, y in centers]
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def create_data(self):
        batch = []
        for i in range(self.num_points):
            point = 2 * (np.random.rand(2)-0.5) * self.eps_noise
            num = 5 if self.five_squares else 4
            center = self.centers[i % num]
            point[0] += center[0]
            point[1] += center[1]
            batch.append(point)
        batch = np.array(batch, dtype='float32')
        batch = torch.from_numpy(batch)
        return batch

class Checkerboard(Dataset):
    """samples from one of two sets 2D squares (depending on alternate=T/F)."""

    def __init__(self,
                 n: int=256,
                 scale: float=1.5,
                 center_coor_min: float=-0.25,
                 center_coor_max: float=+0.25,
                 eps_noise: float=0.01,
                 alternate: bool=False,
                 simple: bool=False):

        self.num_points = n
        self.scale = scale
        self.eps_noise = eps_noise
        self.alternate = alternate
        self.simple = simple
        diag_len = np.sqrt(center_coor_min**2 + center_coor_max**2)
        center_coor_mid = (center_coor_max + center_coor_min)/2
        if self.simple:
            centers = [(center_coor_mid / diag_len, center_coor_mid / diag_len)]
        else:
            if self.alternate:
                centers = [
                    (center_coor_mid / diag_len, center_coor_max / diag_len),
                    (center_coor_mid / diag_len, center_coor_min / diag_len),
                    (center_coor_max / diag_len, center_coor_mid / diag_len),
                    (center_coor_min / diag_len, center_coor_mid / diag_len),
                    ]
            else:
                centers = [
                    (center_coor_max / diag_len, center_coor_max / diag_len),
                    (center_coor_min / diag_len, center_coor_min / diag_len),
                    (center_coor_max / diag_len, center_coor_min / diag_len),
                    (center_coor_min / diag_len, center_coor_max / diag_len),
                    (center_coor_mid / diag_len, center_coor_mid / diag_len)
                ]
        self.centers = [(scale * x, scale * y) for x, y in centers]
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points
        
    def create_data(self):
        batch = []
        for i in range(self.num_points):
            point = (np.random.rand(2)-0.5) * self.eps_noise
            num = 4 if self.alternate else 5
            center = self.centers[i % num]
            point[0] += center[0]
            point[1] += center[1]
            batch.append(point)
        batch = np.array(batch, dtype='float32')
        batch = torch.from_numpy(batch)
        return batch


class TwoGaussians(Dataset):
    def __init__(self, n, d):
        self.num_points = n
        self.shape = d
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def create_data(self):
        x1 = torch.randn(int(self.num_points/2), self.shape) + 3*torch.ones(1, self.shape)
        x2 = torch.randn(int(self.num_points/2), self.shape) - 3*torch.ones(1, self.shape)
        x = torch.cat([x1, x2], dim=0)
        x = x[torch.randperm(x.size()[0])]
        return (x - torch.mean(x, dim=0))/torch.std(x, dim=0)

class Gaussian(Dataset):
    def __init__(self, n, d):
        self.num_points = n
        self.shape = d
        self.data = self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def create_data(self):
        x = torch.randn(self.num_points, *self.shape)
        return x

class CustomDataset(Dataset):
    def __init__(self, data):
        self.num_points = data.shape[0]
        self.shape = data.shape[1:]
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

def create_custom_dataset(n):
    d = 2
    num_bins, tail_bound = 5, 2
    idx = 2*np.arange(num_bins//2) + 1
    potential_flow = potential.ICRQ(tail_bound=tail_bound, num_bins=num_bins, data_shape=(d,))
    widths, heights, derivatives = torch.ones(d, num_bins), torch.ones(d, num_bins), torch.ones(d, num_bins+1)
    widths[:,idx] = 10.
    heights[:,idx] = 2.
    idx = 2*np.arange((num_bins+1)//2) + 1
    derivatives[:,idx] = 0

    potential_flow.widths = torch.nn.Parameter(widths)
    potential_flow.heights = torch.nn.Parameter(heights)
    potential_flow.derivatives = torch.nn.Parameter(derivatives)
    from_data = potential_flow.gradient_inv(torch.randn(n,d)).detach()
    to_data = torch.randn(n,d)
    from_data = CustomDataset(from_data)
    to_data = CustomDataset(to_data)
    return from_data, to_data, potential_flow

def get_dataset(args, split="train"):

    planar_data_dict = {'banana': BananaDataset,
                 'sine': SineWaveDataset,
                 'crescent': CrescentCubedDataset,
                 'two_spirals': TwoSpiralsDataset,
                 'four_gaussians': FourGaussians,
                 'nine_gaussians': NineGaussians,
                 'eight_gmm': Eight_GMM
                 }
    twoD_data_dict = {'gaussian': Gaussian,
                    'two_gaussians': TwoGaussians,
                    'four_gaussians': FourGaussians,
                    'nine_gaussians': NineGaussians}

    if split == "train":
        n = args.num_samples
    else:
        n = args.test_num_samples

    ## source distribution 
    if args.source_dist in planar_data_dict:
        x = planar_data_dict[args.source_dist](n)
    elif args.source_dist in twoD_data_dict:
        x = twoD_data_dict[args.source_dist](n, args.data_shape)
    elif args.source_dist == 'custom':
        return create_custom_dataset(n)
    elif args.source_dist == 'checker':
        x = Checkerboard(n, eps_noise=0.5, alternate=True)
    elif args.source_dist == 'our_checker':
        x = Our_Checkerboard(n, eps_noise=0.5)
    else:
        raise NotImplementedError

    ## target distribution
    if args.target_dist in planar_data_dict:
        y = planar_data_dict[args.target_dist](n)
    elif args.target_dist in twoD_data_dict:
        y = twoD_data_dict[args.target_dist](n, args.data_shape)
    elif args.target_dist == 'checker':
        y = Checkerboard(n, eps_noise=0.5, alternate=False)
    elif args.target_dist == 'our_checker':
        y = Our_Checkerboard(n, eps_noise=0.5, five_squares=True)
    else:
        raise NotImplementedError
    return x,y
    
    