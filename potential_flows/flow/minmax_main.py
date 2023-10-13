import sys
sys.path.append("../../")

import torch
from torch.utils.data import DataLoader

from potential_flows import potential
from arguments import set_seed, parse_arguments
from potential_flows.flow.minmax_OT_Trainers import MinmaxOT_Trainer
from potential_flows.data.datasets import get_dataset


def main():

    ## parse arguments
    args = parse_arguments()
    set_seed(args)

    ## get training and testing dataset
    dataset_x, dataset_y = get_dataset(args, split="train")
    test_x, test_y = get_dataset(args, split="test")

    ## make data loaders for train dataset
    data_loader_X = DataLoader(dataset_x, batch_size=args.batch_size, shuffle=True)
    data_loader_Y = DataLoader(dataset_y, batch_size=args.batch_size, shuffle=True)    

    ## create the flow
    tail_bound = torch.max(torch.cat([torch.abs(dataset_x.data), torch.abs(dataset_y.data)]))
    potential_flow_x = potential.ICRQ(tail_bound=args.tail_factor*tail_bound, num_bins=args.num_bins, data_shape=args.data_shape)
    potential_flow_y = potential.ICRQ(tail_bound=args.tail_factor*tail_bound, num_bins=args.num_bins, data_shape=args.data_shape)

    ## train
    OT_Trainer = MinmaxOT_Trainer(potential_flow_x, potential_flow_y, args, dataset_x=data_loader_X, dataset_y=data_loader_Y, test_x=test_x, test_y=test_y)       
    OT_Trainer.learn()

if __name__=='__main__':
	main()     
