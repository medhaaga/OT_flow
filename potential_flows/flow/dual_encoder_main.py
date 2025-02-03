import sys
sys.path.append("../../")
import numpy as np
import torch
from torch.utils.data import DataLoader

from potential_flows import potential, data
from arguments import set_seed, parse_arguments
from potential_flows import flow, encoders


def main():

    ## parse arguments
    args = parse_arguments()
    set_seed(args)

    ## get training and testing dataset
    true_potential = None
    if args.source_dist == 'custom':
        dataset_x, dataset_y, true_potential = data.get_dataset(args, split="train")
        test_x, test_y, true_potential = data.get_dataset(args, split="test")
    else:
        dataset_x, dataset_y = data.get_dataset(args, split="train")
        test_x, test_y = data.get_dataset(args, split="test")

    ## make data loaders for train dataset
    data_loader_X = DataLoader(dataset_x, batch_size=args.batch_size, shuffle=True)
    data_loader_Y = DataLoader(dataset_y, batch_size=args.batch_size, shuffle=True)    

    # create encoders for source and target distribution

    encoder_x = encoders.Encoder(input_dim=np.prod(args.data_shape), hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    encoder_y = encoders.Encoder(input_dim=np.prod(args.data_shape), hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)

    # potential flow

    potential_flow = potential.ICRQ(tail_bound=1, num_bins=4, data_shape=(args.latent_dim,))

    # train

    Encoder_OT_Trainer = flow.DualEncoder_OT_Trainer(potential=potential_flow,
                                                encoder_x=encoder_x,
                                                encoder_y=encoder_y,
                                                args=args,
                                                dataset_x=data_loader_X,
                                                dataset_y=data_loader_Y,
                                                test_x=test_x,
                                                test_y=test_y,
                                                true_potential=true_potential)
    Encoder_OT_Trainer.learn()


if __name__=='__main__':
	main()     
