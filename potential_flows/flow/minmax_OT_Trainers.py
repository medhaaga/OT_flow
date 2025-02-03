import os
import json
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import time
from itertools import cycle
from potential_flows import potential
from typing import Union



class MinmaxOT_Trainer:

    def __init__(self, 
                 potential_x: potential.Potential,
                 potential_y: potential.Potential,
                 args: argparse.ArgumentParser,
                 dataset_x: DataLoader,
                 dataset_y: DataLoader,
                 test_x: Union[None, torch.Tensor, Dataset] = None,
                 test_y: Union[None, torch.Tensor, Dataset] = None,
                 true_potential: Union[None, potential.Potential] = None
                 ):
    

        ## potential flow

        self.potential_flow_x = potential_x
        self.potential_flow_y = potential_y
        self.args = args

        ## logging and saving

        if self.args.model_variant == 'alternate':
            self.exp_dir = f'experiments/{self.args.source_dist}_{self.args.target_dist}/minmax_alternate/dim_{np.prod(self.args.data_shape)}/num_bins_{self.args.num_bins}/lr_{self.args.learning_rate}'
        elif self.args.model_variant == 'step_through_max':
            self.exp_dir = f'experiments/{self.args.source_dist}_{self.args.target_dist}/minmax_step_through_max/dim_{np.prod(self.args.data_shape)}/num_bins_{self.args.num_bins}/lr_{self.args.learning_rate}'
        else:
            NotImplementedError

        os.makedirs(self.exp_dir, exist_ok=True)

        # remove tye log file if it exists already
    
        if os.path.exists(os.path.join(self.exp_dir, 'logger.log')):
            os.remove(os.path.join(self.exp_dir, 'logger.log'))
        
        if self.args.verbose:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')

        
        ## source dataset dataloaders

        self.dataloader_x = dataset_x
        self.dataloader_y = dataset_y

        ## creating test dataset tensors

        if (test_x is not None) and (test_y is not None):
            if isinstance(test_x, Dataset):
                self.test_x = test_x.data
                self.test_y = test_y.data
            else:
                self.test_x = test_x
                self.test_y = test_y
        else:
            logging.warning('Test dataset not provided. Part of training dataset will be used')
            iter_x, iter_y = iter(self.dataloader_x), iter(self.dataloader_y)
            test_x, test_y = [], []
            for i in range(int(self.args.test_num_samples/self.args.batch_size)):
                batch_x, batch_y = next(iter_x), next(iter_y)
                test_x.extend(batch_x)
                test_y.extend(batch_y)
            self.test_x = torch.stack(test_x)
            self.test_y = torch.stack(test_y)

        self.true_potential = true_potential

    def get_device(self):
        return 

    def from_loss(self, inputs_from, inputs_to):
        n = inputs_from.shape[0]
        from_forward = self.potential_flow_x.gradient(inputs_from)
        g_grad_fx = torch.mean(self.potential_flow_y(from_forward).view(n,-1))
        x_dot_grad_fx = torch.mean(inputs_from.view(n,-1) * from_forward.view(n,-1))

        if self.args.regularization:
            return g_grad_fx - x_dot_grad_fx + sum([self.args.regularization*torch.sum(param**2) for param in self.potential_flow_x.parameters()])
        return g_grad_fx - x_dot_grad_fx

    def to_loss(self, inputs_from, inputs_to):
        n = inputs_from.shape[0]
        from_forward = self.potential_flow_x.gradient(inputs_from)
        gy = torch.mean(self.potential_flow_y(inputs_to).view(n,-1))
        g_grad_fx = torch.mean(self.potential_flow_y(from_forward).view(n,-1))

        if self.args.regularization:
            return gy - g_grad_fx + sum([self.args.regularization*torch.sum(param**2) for param in self.potential_flow_y.parameters()])
        return gy - g_grad_fx
        
    def minimax_loss(self, inputs_from, inputs_to):

        n = inputs_from.shape[0]
        from_forward = self.potential_flow_x.gradient(inputs_from)
        value1 = self.potential_flow_y(inputs_to).view(n,-1)
        value2 = inputs_from.view(n,-1) * from_forward.view(n,-1)
        value3 = self.potential_flow_y(from_forward).view(n,-1)

<<<<<<< HEAD
    # inner minimization loop using Adam optimizer
    def approx_min(self, iter_from, iter_to, optimizer_from):

        for _ in range(self.args.max_inner_iter):
            optimizer_from.zero_grad()
            batch_from, batch_to = next(iter_from), next(iter_to)
            loss = self.minimax_loss(inputs_from=batch_from, inputs_to=batch_to)
=======
        if self.args.regularization:
            value4 = sum([self.args.regularization*torch.sum(param**2) for param in self.potential_flow_x.parameters()]) - sum([self.args.regularization*torch.sum(param**2) for param in self.potential_flow_y.parameters()])
            return -torch.mean(value1) - torch.mean(value2 - value3) + value4
        else:
            return -torch.mean(value1) - torch.mean(value2 - value3) 

    # inner minimization loop using Adam optimizer for minimizing over f
    def approx_min(self, iter_from, iter_to, optimizer_from):

        for _ in range(self.args.max_inner_iter):

            optimizer_from.zero_grad()
            batch_from, batch_to = next(iter_from), next(iter_to)
            loss = self.from_loss(inputs_from=batch_from, inputs_to=batch_to)
>>>>>>> 368995cc8faaf3b55343417ed0e1ccd6892e0730
            loss.backward(inputs = list(self.potential_flow_x.parameters()), create_graph=False)
            optimizer_from.step()

    # inner minimization loop using minibatch gradient descent
    def nested_approx_min(self, iter_from, iter_to):
        for _ in range(self.args.max_inner_iter):
            batch_from, batch_to = next(iter_from), next(iter_to)
            loss_from = self.minimax_loss(inputs_from=batch_from, inputs_to=batch_to)
            loss_from.backward(create_graph=True)
            for w in self.potential_flow_x.parameters():
                w = w - self.args.learning_rate * w.grad

        # Test grad dependency
        aux = torch.sum(list(self.potential_flow_x.parameters())[0])
        grad_to = torch.autograd.grad(aux, inputs=self.potential_flow_y.parameters())


    def learn(self):

        ## save_config
        filename = os.path.join(self.exp_dir, 'config.json')
        with open(filename, 'w') as file:
            json.dump(vars(self.args), file)

        ## create iterables for dataloaders
        iter_from, iter_to = cycle(iter(self.dataloader_x)), cycle(iter(self.dataloader_y))

        # create optimizer
        optimizer_from = optim.Adam(self.potential_flow_x.parameters(), lr=self.args.learning_rate)
        optimizer_to = optim.Adam(self.potential_flow_y.parameters(), lr=self.args.learning_rate)

        # create scheduler
        if self.args.anneal_learning_rate == 'cosine':
            scheduler_to = optim.lr_scheduler.CosineAnnealingLR(optimizer_to, self.args.num_epochs, 1e-4)
        elif self.args.anneal_learning_rate == 'exponential':
            scheduler_to = optim.lr_scheduler.ExponentialLR(optimizer_to, 0.9)
        else:
            scheduler_to = None

        ## training
        tbar = tqdm(range(self.args.num_steps))
        logs, test_loss_list = [], []
        best_val_loss = 1e5
        torch.autograd.set_detect_anomaly(True)
        start = time.time()
        for t in tbar:

            ## get source and target data batches
            batch_from, batch_to = next(iter_from), next(iter_to)

            ## calculate loss and gradient norm
            loss = self.minimax_loss(inputs_from=batch_from, inputs_to=batch_to)
            
            ## delete the parameter gradients
            # for (param_from, param_to) in zip(self.potential_flow_x.parameters(), self.potential_flow_y.parameters()):
            #     param_from.grad, param_to.grad = None, None
    
            ## take optimization and scheduler step
            if self.args.model_variant == "alternate":

                # update f
                self.approx_min(iter_from, iter_to, optimizer_from)

                # update g
                optimizer_to.zero_grad()
                loss_to = self.to_loss(inputs_from=batch_from, inputs_to=batch_to)
                loss_to.backward(inputs = list(self.potential_flow_y.parameters()))
                optimizer_to.step()

                if self.args.anneal_learning_rate != 'none':
                    scheduler_to.step()

            elif self.args.model_variant == "step_through_max":
                optimizer_to.zero_grad()
                self.nested_approx_min(iter_from, iter_to)
                loss_to = self.minimax_loss(inputs_from=batch_from, inputs_to=batch_to)
                loss_to.backward(inputs = list(self.potential_flow_y.parameters()))
                optimizer_to.step()
            
            ## saving figures
            if (t+1) % self.args.save_figure_interval == 0:
                self.save_figure()

            ## logging
            if (t+1) % self.args.log_interval == 0:

                # calculate test loss
                with torch.no_grad():
                    test_loss = self.minimax_loss(self.test_x, self.test_y)
                    test_loss_list.append(test_loss.item())

                # calculate mse if true potential is known
                if self.true_potential:
                    squared_diff = ((self.potential_flow_x.gradient(self.test_x) - self.true_potential.gradient(self.test_x)).reshape(self.args.test_num_samples,-1))**2
                    mse = torch.mean(torch.sum(squared_diff, dim=-1))
                    logs.append({'step': t+1, 'train_loss': loss.item(), 'test_loss': test_loss.item(), 'mse': mse.item()}) 
                else:
                    logs.append({'step': t+1, 'train_loss': loss.item(), 'test_loss': test_loss.item()}) 

                
                with open(self.exp_dir+"/training_metrics.json", "w") as json_file:
                    json.dump(logs, json_file)

                logging.info(logs[-1])

                # saving best model
                if test_loss < best_val_loss:
                    torch.save({'flow_from': self.potential_flow_x, 'flow_to': self.potential_flow_y}, self.exp_dir+'/best_flow.t')

                # stop iterations if the change in test loss is less than tolerance
                if len(test_loss_list) >=2:
                    self.plot_losses()
                    if (np.abs(test_loss_list[-1] - test_loss_list[-2]) < self.args.tolerance):
                        logging.info(f"Optimization stopped at {t+1} iterations.")
                        break
        


        stop = time.time()
        logging.info(f'Running {t+1} training steps takes {stop - start}s.')

    def save_figure(self):

        ## transport the test dataset

        with torch.no_grad():
            X_pred = self.potential_flow_y.gradient(self.test_y).detach().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        axs[0].scatter(self.test_x[:,0], self.test_x[:,1], color='darkgreen', alpha=0.5, s=10)
        axs[0].set_title(r'$X$')
        axs[1].scatter(self.test_y[:,0], self.test_y[:,1], color='goldenrod', alpha=0.5, s=10)
        axs[1].set_title(r'$Y$')
        axs[2].scatter(X_pred[:,0], X_pred[:,1], color='seagreen', alpha=0.5, s=10)
        axs[2].set_title(r'$(\nabla f)^{-1}(Y)$')

        if self.args.show_the_plot:
            plt.show()
        plt.close()

        fig.savefig(os.path.join(self.exp_dir, 'test_data.png'))

        self.potential_flow_x.plot_flow(n_points=1000, log_dir=self.exp_dir, show_figure=self.args.show_the_plot, save_fig=True)

    def plot_losses(self):

        # open the training metrics
        with open(os.path.join(self.exp_dir, 'training_metrics.json'), "r") as json_file:
            train_specs = json.load(json_file)
        train_specs = pd.DataFrame(train_specs)

        num_logs = train_specs.shape[0]

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(self.args.log_interval*(1+np.arange(num_logs)), train_specs['train_loss'])
        axs[0].set_xlabel('Training Steps')
        axs[0].set_ylabel('Train Loss')
        axs[1].plot(self.args.log_interval*(1+np.arange(num_logs)), train_specs['test_loss'])
        axs[1].set_xlabel('Training Steps')
        axs[1].set_ylabel('Test Loss')
        plt.tight_layout()
        if self.args.show_the_plot:
            plt.show()
        plt.savefig(os.path.join(self.exp_dir, 'losses.png'))
        plt.close()
        
        if self.true_potential:
            plt.figure(figsize=(4,3))
            plt.plot(self.args.log_interval*(1+np.arange(num_logs)), train_specs['mse'])
            plt.xlabel('Training Steps')
            plt.ylabel('MSE')
            if self.args.show_the_plot:
                plt.show()
            plt.savefig(os.path.join(self.exp_dir, 'mse.png'))
            plt.close()
        
        plt.close()




				


