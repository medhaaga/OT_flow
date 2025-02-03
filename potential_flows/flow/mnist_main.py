import sys
sys.path.append("../../")

import torch
from torch.utils.data import DataLoader

from potential_flows import potential, data
from arguments import set_seed, parse_arguments
from potential_flows.flow.dual_OT_Trainers import DualOT_Trainer


def main():

    ## parse arguments
    args = parse_arguments()
    set_seed(args)

    ## get training and testing dataset
    true_potential = None
    args.source_dist = 'mnist'
    args.data_shape = (784,)

    ## loda dataset
    dataset_x = data.CustomMNISTDataset(root='../data/MNIST_data/', train=True)
    dataset_y = data.Gaussian(n=len(dataset_x), d=(784,))
    test_x = data.CustomMNISTDataset(root='./MNIST_data/', train=False)
    test_y = data.Gaussian(n=len(dataset_y), d=(784,))
    
    ## make data loaders for train dataset
    data_loader_X = DataLoader(dataset_x, batch_size=args.batch_size, shuffle=True)
    data_loader_Y = DataLoader(dataset_y, batch_size=args.batch_size, shuffle=True)    

    ## create the flow
    tail_bound = torch.max(torch.cat([torch.abs(dataset_x.data), torch.abs(dataset_y.data)]))
    potential_flow = potential.ICRQ(tail_bound=args.tail_factor*tail_bound, num_bins=args.num_bins, data_shape=args.data_shape)

    ## train
    OT_Trainer = DualOT_Trainer(potential_flow, args, dataset_x=data_loader_X, dataset_y=data_loader_Y, test_x=test_x, test_y=test_y, true_potential=true_potential)       
    OT_Trainer.learn()

if __name__=='__main__':
	main()     
