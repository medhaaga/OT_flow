import os
import sys
import json
import argparse
import warnings
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
from potential_flows import transforms
from potential_flows import encoders
from typing import Union
from arguments import get_exp_dir



class DualOT_Trainer:

    """
    The Trainer for learning OT using dual method.
    potential: the potentail flow between source and target 
    
    args: tarining arguments
    dataset_x: DataLoader class for source distribution samples
    dataset_y: DataLoader class for target distribution samples
    test_x: Dataset class for source distribution test samples
    test_y: Dataset class for target distribution test samples
    true_potential: Potential class for the true Brenier potential
    """

    def __init__(self, 
                 potential: potential.Potential,
                 args: argparse.ArgumentParser,
                 dataset_x: DataLoader,
                 dataset_y: DataLoader,
                 test_x: Union[None, torch.Tensor, Dataset] = None,
                 test_y: Union[None, torch.Tensor, Dataset] = None,
                 true_potential: Union[None, potential.Potential] = None
                 ):
    

        ## potential flow
        self.potential_flow = potential
        self.args = args

        ## logging and saving
        root = get_exp_dir()
        self.exp_dir = os.path.join(root, f'{self.args.source_dist}_{self.args.target_dist}/dual/dim_{np.prod(self.args.data_shape)}/num_bins_{self.args.num_bins}/lr_{self.args.learning_rate}')
        os.makedirs(self.exp_dir, exist_ok=True)
        file_list = os.listdir(self.exp_dir)
        for file in file_list:
            os.remove(os.path.join(self.exp_dir, file))
        
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
        

    def objective(self, inputs_from, inputs_to):
        value1 = self.potential_flow(inputs_from)
        value2 = self.potential_flow.conjugate(inputs_to)
        value3 = sum([self.args.regularization*torch.sum(param**2) for param in self.potential_flow.parameters()])
        return torch.mean(value1) + torch.mean(value2) + value3


    def learn(self):

        ## save_config
        filename = os.path.join(self.exp_dir, 'config.json')
        with open(filename, 'w') as file:
            json.dump(vars(self.args), file)

        ## create iterables for dataloaders
        iter_from, iter_to = cycle(iter(self.dataloader_x)), cycle(iter(self.dataloader_y))

        ## create optimizers & scheduler
        optimizer = optim.Adam(self.potential_flow.parameters(), lr=self.args.learning_rate)
        if self.args.anneal_learning_rate == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.num_epochs, 1e-4)
        elif self.args.anneal_learning_rate == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        else:
            scheduler = None

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
            loss = self.objective(inputs_from=batch_from, inputs_to=batch_to)
            grads = torch.autograd.grad(loss, self.potential_flow.parameters(), retain_graph=True)
            grad_norm = torch.sqrt(sum([torch.sum(grad**2) for grad in grads]))

            ## take optimization and scheduler step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.args.anneal_learning_rate != 'none':
                scheduler.step(t)
            
            ## saving figures
            if (t+1) % self.args.save_figure_interval == 0:
                self.save_figure()

            ## logging
            if (t+1) % self.args.log_interval == 0:

                # calculate test loss
                test_loss = self.objective(self.test_x, self.test_y)
                test_loss_list.append(test_loss.item())

                # calculate mse if true potential is known
                if self.true_potential:
                    squared_diff = ((self.potential_flow.gradient(self.test_x) - self.true_potential.gradient(self.test_x)).reshape(self.args.test_num_samples,-1))**2
                    mse = torch.mean(torch.sum(squared_diff, dim=-1))
                    logs.append({'step': t+1, 'train_loss': loss.item(), 'gradient norm': grad_norm.item(), 'test_loss': test_loss.item(), 'mse': mse.item()}) 
                else:
                    logs.append({'step': t+1, 'train_loss': loss.item(), 'gradient norm': grad_norm.item(), 'test_loss': test_loss.item()}) 

                
                with open(self.exp_dir+"/training_metrics.json", "w") as json_file:
                    json.dump(logs, json_file)

                logging.info(logs[-1])

                # saving best model
                if test_loss < best_val_loss:
                    torch.save(self.potential_flow, self.exp_dir+'/best_flow.t')

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

        X_pred = self.potential_flow.gradient_inv(self.test_y).detach().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        axs[0].scatter(self.test_x[:,0], self.test_x[:,1], color='C1', alpha=0.5)
        axs[0].set_title(r'$X$')
        axs[1].scatter(self.test_y[:,0], self.test_y[:,1], color='C2', alpha=0.5)
        axs[1].set_title(r'$Y$')
        axs[2].scatter(X_pred[:,0], X_pred[:,1], color='C3', alpha=0.5)
        axs[2].set_title(r'$(\nabla f)^{-1}(Y)$')

        if self.args.show_the_plot:
            plt.show()
        plt.close()

        fig.savefig(os.path.join(self.exp_dir, 'test_data.png'))

        self.potential_flow.plot_flow(n_points=1000, log_dir=self.exp_dir, show_figure=self.args.show_the_plot, save_fig=True)

    def plot_losses(self):
        plot_losses_(self.exp_dir, self.args, self.true_potential)

class DualEncoderDecoder_OT_Trainer:

    """
    The base Trainer for learning OT using dual method between encoded 
    source and target distributions.

    potential: the potentail flow between source and target 
    transform_x: EncoderDecoder pair for the source distribution. Choices are Aurtoencoders or PCA
    transform_y: EncoderDecoder pair for the target distribution.
    args: tarining arguments
    dataset_x: DataLoader class for source distribution samples
    dataset_y: DataLoader class for target distribution samples
    test_x: Dataset class for source distribution test samples
    test_y: Dataset class for target distribution test samples
    true_potential: Potential class for the true Brenier potential
    """

    def __init__(self,
                 potential: potential.Potential,
                 transform_x: encoders.EncoderDecoder,
                 transform_y: encoders.EncoderDecoder,
                 args: argparse.ArgumentParser,
                 dataset_x: DataLoader,
                 dataset_y: DataLoader,
                 test_x: Union[None, torch.Tensor, Dataset] = None,
                 test_y: Union[None, torch.Tensor, Dataset] = None,
                 true_potential: Union[None, potential.Potential] = None):

        ## create PCA OT flow
        self.encoder_flow = encoders.EncoderDecoder_OT(transform_x=transform_x, transform_y=transform_y, potential=potential)

        self.args = args

        ## logging and saving
        self.exp_dir = None  
              
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
            for _ in range(int(self.args.test_num_samples/self.args.batch_size)):
                batch_x, batch_y = next(iter_x), next(iter_y)
                test_x.extend(batch_x)
                test_y.extend(batch_y)
            self.test_x = torch.stack(test_x)
            self.test_y = torch.stack(test_y)

        self.true_potential = true_potential

    def objective(self, x, y):

        ## encoded source and target data
        T_x = self.encoder_flow.encode_x(x)
        T_y = self.encoder_flow.encode_y(y)

        ## dual loss on encoded data
        value1 = self.encoder_flow.potential(T_x)
        value2 = self.encoder_flow.potential.conjugate(T_y)
        value3 = sum([self.args.regularization*torch.sum(param**2) for param in self.encoder_flow.potential.parameters()])
        dual_loss = torch.mean(value1) + torch.mean(value2) + value3

        return dual_loss 

    def learn(self):

        ## save_config
        assert self.exp_dir is not None, "Experiment directory not specified!"

        filename = os.path.join(self.exp_dir, 'config.json')
        with open(filename, 'w') as file:
            json.dump(vars(self.args), file)

        ## create iterables for dataloaders
        iter_from, iter_to = cycle(iter(self.dataloader_x)), cycle(iter(self.dataloader_y))

        ## create optimizers & scheduler
        optimizer = optim.Adam(self.encoder_flow.parameters(), lr=self.args.learning_rate)

        if self.args.anneal_learning_rate == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.num_epochs, 1e-4)
        elif self.args.anneal_learning_rate == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        else:
            scheduler = None

        ## training
        tbar = tqdm(range(self.args.num_steps))
        logs, test_loss_list = [], []
        best_val_loss = 1e5
        torch.autograd.set_detect_anomaly(True)
        start = time.time()
        for t in tbar:

            ## get source and target data batches
            batch_from, batch_to = next(iter_from), next(iter_to)

            ## calculate loss 
            loss = self.objective(x=batch_from, y=batch_to)
            grads = torch.autograd.grad(loss, self.encoder_flow.parameters(), retain_graph=True)
            grad_norm = torch.sqrt(sum([torch.sum(grad**2) for grad in grads]))
        

            ## delete gradients
            self.encoder_flow.zero_grad()
        
            ## take optimization and scheduler step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.args.anneal_learning_rate != 'none':
                scheduler.step(t)
            
            ## saving figures
            if (t+1) % self.args.save_figure_interval == 0:
                self.generate()
                self.plot_flow()

            ## logging
            if (t+1) % self.args.log_interval == 0:

                # calculate test loss
                with torch.no_grad():
                    test_loss = self.objective(self.test_x, self.test_y)
                    test_loss_list.append(test_loss.item())

                    logs.append({'step': t+1, 'train_loss': loss.item(), 'grad norm': grad_norm.item(),  'test_loss': test_loss.item()}) 

                
                with open(self.exp_dir+"/training_metrics.json", "w") as json_file:
                    json.dump(logs, json_file)

                logging.info(logs[-1])

                # saving best model
                if test_loss < best_val_loss:
                    torch.save(self.encoder_flow, self.exp_dir+'/best_flow.t')

                # stop iterations if the change in test loss is less than tolerance
                if len(test_loss_list) >=2:
                    self.plot_losses()
                    if (np.abs(test_loss_list[-1] - test_loss_list[-2]) < self.args.tolerance):
                        logging.info(f"Optimization stopped at {t+1} iterations.")
                        break
        
        stop = time.time()
        logging.info(f'Running {t+1} training steps takes {stop - start}s.')

    def plot_losses(self):
        plot_losses_(self.exp_dir, self.args, self.true_potential)

    def generate(self):

        ## transport the test dataset
        with torch.no_grad():
            x_pred = self.encoder_flow.decode_x(self.encoder_flow.potential.gradient_inv(self.encoder_flow.encode_y(self.test_y)))

        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        axs[0].scatter(self.test_x[:,0], self.test_x[:,1], color='C1', alpha=0.5)
        axs[0].set_title(r'$X$')
        axs[1].scatter(self.test_y[:,0], self.test_y[:,1], color='C2', alpha=0.5)
        axs[1].set_title(r'$Y$')
        axs[2].scatter(x_pred[:,0], x_pred[:,1], color='C3', alpha=0.5)
        axs[2].set_title(r'$(\nabla f)^{-1}(Y)$')

        if self.args.show_the_plot:
            plt.show()
        plt.close()

        fig.savefig(os.path.join(self.exp_dir, 'test_data.png'))

    def plot_flow(self, nums=[0,1]):

        x = torch.linspace(torch.min(self.test_x), torch.max(self.test_x), 1000)
        x = torch.cat([x.unsqueeze(-1)]*np.prod(self.args.data_shape), dim=1)
        with torch.no_grad():
            y = self.encoder_flow.potential.gradient(self.encoder_flow.encode_x(x))
            y = self.encoder_flow.decode_y(y)

        for i in nums:
            plt.plot(x[:,i], y[:,i], label='Component-{}'.format(i+1))
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, 'flow.png'))
        if self.args.show_the_plot:
            plt.show()
        plt.close()
        

class DualEncoder_OT_Trainer:

    """
    The base Trainer for learning OT using dual method between encoded 
    source and target distributions. Only encoder is used - no decoder - no reconstruction loss

    potential: the potentail flow between source and target 
    transform_x: Encoder for the source distribution. Choices are Aurtoencoders or PCA
    transform_y: Encoder for the target distribution.
    args: tarining arguments
    dataset_x: DataLoader class for source distribution samples
    dataset_y: DataLoader class for target distribution samples
    test_x: Dataset class for source distribution test samples
    test_y: Dataset class for target distribution test samples
    true_potential: Potential class for the true Brenier potential
    """

    def __init__(self, 
                 potential: potential.Potential,
                 encoder_x: encoders.Encoder,
                 encoder_y: encoders.Encoder,
                 args: argparse.ArgumentParser,
                 dataset_x: DataLoader,
                 dataset_y: DataLoader,
                 test_x: Union[None, torch.Tensor, Dataset] = None,
                 test_y: Union[None, torch.Tensor, Dataset] = None,
                 true_potential: Union[None, torch.Tensor, Dataset] = None
                 ):
    
        self.encoder_flow = encoders.Encoder_OT(encoder_x=encoder_x, encoder_y=encoder_y, potential=potential)

        self.args = args

        ## logging and saving
        self.exp_dir = f'experiments/{self.args.source_dist}_{self.args.target_dist}/dualEncoder/dim_{np.prod(self.args.data_shape)}/num_bins_{self.args.num_bins}/lr_{self.args.learning_rate}'
        os.makedirs(self.exp_dir, exist_ok=True)
        file_list = os.listdir(self.exp_dir)
        for file in file_list:
            os.remove(os.path.join(self.exp_dir, file))
        
        logging.basicConfig(level=logging.INFO, 
                                    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
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
            for _ in range(int(self.args.test_num_samples/self.args.batch_size)):
                batch_x, batch_y = next(iter_x), next(iter_y)
                test_x.extend(batch_x)
                test_y.extend(batch_y)
            self.test_x = torch.stack(test_x)
            self.test_y = torch.stack(test_y)

        self.true_potential = true_potential

        

    def objective(self, x, y):

        ## encoded source and target data
        T_x = self.encoder_flow.encode_x(x)
        T_y = self.encoder_flow.encode_y(y)

        ## dual loss on encoded data
        value1 = self.encoder_flow.potential(T_x)
        value2 = self.encoder_flow.potential.conjugate(T_y)
        value3 = sum([self.args.regularization*torch.sum(param**2) for param in self.encoder_flow.potential.parameters()])
        dual_loss = torch.mean(value1) + torch.mean(value2) + value3

        return dual_loss 


    def learn(self):

        ## save_config
        filename = os.path.join(self.exp_dir, 'config.json')
        with open(filename, 'w') as file:
            json.dump(vars(self.args), file)

        ## create iterables for dataloaders
        iter_from, iter_to = cycle(iter(self.dataloader_x)), cycle(iter(self.dataloader_y))

        ## create optimizers & scheduler
        optimizer = optim.Adam(self.encoder_flow.parameters(), lr=self.args.learning_rate)

        if self.args.anneal_learning_rate == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.num_epochs, 1e-4)
        elif self.args.anneal_learning_rate == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        else:
            scheduler = None

        ## training
        tbar = tqdm(range(self.args.num_steps))
        logs, test_loss_list = [], []
        best_val_loss = 1e5
        torch.autograd.set_detect_anomaly(True)
        start = time.time()
        for t in tbar:

            ## get source and target data batches
            batch_from, batch_to = next(iter_from), next(iter_to)

            ## calculate loss 
            loss = self.objective(x=batch_from, y=batch_to)
            grads = torch.autograd.grad(loss, self.encoder_flow.parameters(), retain_graph=True)
            grad_norm = torch.sqrt(sum([torch.sum(grad**2) for grad in grads]))
        

            ## delete gradients
            self.encoder_flow.zero_grad()
        
            ## take optimization and scheduler step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.args.anneal_learning_rate != 'none':
                scheduler.step(t)
            

            ## logging
            if (t+1) % self.args.log_interval == 0:

                # calculate test loss
                with torch.no_grad():
                    test_loss = self.objective(self.test_x, self.test_y)
                    test_loss_list.append(test_loss.item())

                    logs.append({'step': t+1, 'train_loss': loss.item(), 'grad norm': grad_norm.item(),  'test_loss': test_loss.item()}) 

                
                with open(self.exp_dir+"/training_metrics.json", "w") as json_file:
                    json.dump(logs, json_file)

                logging.info(logs[-1])

                # saving best model
                if test_loss < best_val_loss:
                    torch.save(self.encoder_flow, self.exp_dir+'/best_flow.t')

                # stop iterations if the change in test loss is less than tolerance
                if len(test_loss_list) >=2:
                    self.plot_losses()
                    if (np.abs(test_loss_list[-1] - test_loss_list[-2]) < self.args.tolerance):
                        logging.info(f"Optimization stopped at {t+1} iterations.")
                        break
        
        stop = time.time()
        logging.info(f'Running {t+1} training steps takes {stop - start}s.')

    def plot_losses(self):
        plot_losses_(self.exp_dir, self.args, self.true_potential)

    def learn_decoder(self, input_dim, hidden_dim=100, latent_dim=10, epochs=100, dir="source"):
        decoder = transforms.Decoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        opt = optim.Adam(decoder.parameters(), lr=self.args.learning_rate)
        if dir == "source":
            dataloader = self.dataloader_x
        else:
            dataloader = self.dataloader_y
       
        for epoch in tqdm(range(epochs)):
            for batch in dataloader:
                y = self.encoder_flow.encoder_x(batch)
                loss = ((batch - decoder(y))**2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, loss: {loss}')
        return decoder

    def generate(self, decoder_x):

        ## transport the test dataset
        with torch.no_grad():
            x_pred = decoder_x(self.encoder_flow.potential.gradient_inv(self.encoder_flow.encode_y(self.test_y)))

        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        axs[0].scatter(self.test_x[:,0], self.test_x[:,1], color='C1', alpha=0.5)
        axs[0].set_title(r'$X$')
        axs[1].scatter(self.test_y[:,0], self.test_y[:,1], color='C2', alpha=0.5)
        axs[1].set_title(r'$Y$')
        axs[2].scatter(x_pred[:,0], x_pred[:,1], color='C3', alpha=0.5)
        axs[2].set_title(r'$(\nabla f)^{-1}(Y)$')

        if self.args.show_the_plot:
            plt.show()
        plt.close()

        fig.savefig(os.path.join(self.exp_dir, 'test_data.png'))

    def plot_flow(self, decoder_y, nums=[0,1]):

        x = torch.linspace(torch.min(self.test_x), torch.max(self.test_x), 1000)
        x = torch.cat([x.unsqueeze(-1)]*np.prod(self.args.data_shape), dim=1)
        with torch.no_grad():
            y = self.encoder_flow.potential.gradient(self.encoder_flow.encode_x(x))
            y = decoder_y(y)

        for i in nums:
            plt.plot(x[:,i], y[:,i], label='Component-{}'.format(i+1))
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, 'flow.png'))
        if self.args.show_the_plot:
            plt.show()
        plt.close()


class DualPCA_OT_Trainer(DualEncoderDecoder_OT_Trainer):

    """
    DualEncoderDecoder_OT_Trainer subclass for PCA encoder-decoder
    """

    def __init__(self, 
                 potential: potential.Potential,
                 transform_x: encoders.PCA,
                 transform_y: encoders.PCA,
                 args: argparse.ArgumentParser,
                 dataset_x: DataLoader,
                 dataset_y: DataLoader,
                 test_x: Union[None, torch.Tensor, Dataset] = None,
                 test_y: Union[None, torch.Tensor, Dataset] = None,
                 true_potential: Union[None, torch.Tensor, Dataset] = None
                 ):

        assert hasattr(transform_x, "components_"), "PCA for x component must be fit before use."
        assert hasattr(transform_y, "components_"), "PCA for y component must be fit before use."

        super(DualPCA_OT_Trainer, self).__init__(potential=potential,
                                                transform_x=transform_x, 
                                                transform_y=transform_y, 
                                                args=args,
                                                dataset_x=dataset_x,
                                                dataset_y=dataset_y,
                                                test_x=test_x,
                                                test_y=test_y,
                                                true_potential=true_potential)

        
        ## logging and saving
        self.exp_dir = f'experiments/{self.args.source_dist}_{self.args.target_dist}/dualPCA/dim_{np.prod(self.args.data_shape)}/num_bins_{self.args.num_bins}/lr_{self.args.learning_rate}'
        os.makedirs(self.exp_dir, exist_ok=True)
        file_list = os.listdir(self.exp_dir)
        for file in file_list:
            os.remove(os.path.join(self.exp_dir, file))
        
        if self.args.verbose:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')
     

class DualAE_OT_Trainer(DualEncoderDecoder_OT_Trainer):

    """
    DualEncoderDecoder_OT_Trainer subclass for autoencoder 
    """

    def __init__(self, 
                 potential: potential.Potential,
                 transform_x: encoders.AE,
                 transform_y: encoders.AE,
                 args: argparse.ArgumentParser,
                 dataset_x: DataLoader,
                 dataset_y: DataLoader,
                 test_x: Union[None, torch.Tensor, Dataset] = None,
                 test_y: Union[None, torch.Tensor, Dataset] = None,
                 true_potential: Union[None, torch.Tensor, Dataset] = None
                 ):


        super(DualAE_OT_Trainer, self).__init__(potential=potential,
                                                transform_x=transform_x, 
                                                transform_y=transform_y, 
                                                args=args,
                                                dataset_x=dataset_x,
                                                dataset_y=dataset_y,
                                                test_x=test_x,
                                                test_y=test_y,
                                                true_potential=true_potential)

        
        ## logging and saving
        self.exp_dir = f'experiments/{self.args.source_dist}_{self.args.target_dist}/dualAE/dim_{np.prod(self.args.data_shape)}/num_bins_{self.args.num_bins}/lr_{self.args.learning_rate}'
        os.makedirs(self.exp_dir, exist_ok=True)
        file_list = os.listdir(self.exp_dir)
        for file in file_list:
            os.remove(os.path.join(self.exp_dir, file))
        
        if self.args.verbose:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')

    def objective(self, x, y):

        ## encoded source and target data
        T_x = self.encoder_flow.encode_x(x)
        T_y = self.encoder_flow.encode_y(y)

        ## approximate source and target data
        x_hat = self.encoder_flow.transform_x(x)
        y_hat = self.encoder_flow.transform_y(y)

        ## dual loss on encoded data
        value1 = self.encoder_flow.potential(T_x)
        value2 = self.encoder_flow.potential.conjugate(T_y)
        value3 = sum([self.args.regularization*torch.sum(param**2) for param in self.encoder_flow.potential.parameters()])
        dual_loss = torch.mean(value1) + torch.mean(value2) + value3

        ## AE loss
        reproduction_loss = (((x_hat - x)**2).sum())/x.shape[0] + (((y_hat - y)**2).sum())/y.shape[0]
        
        return dual_loss + reproduction_loss 
  
class DualVAE_OT_Trainer(DualEncoderDecoder_OT_Trainer):

    """
    DualEncoderDecoder_OT_Trainer subclass for VAE 
    """

    def __init__(self, 
                 potential: potential.Potential,
                 transform_x: encoders.VAE,
                 transform_y: encoders.VAE,
                 args: argparse.ArgumentParser,
                 dataset_x: DataLoader,
                 dataset_y: DataLoader,
                 test_x: Union[None, torch.Tensor, Dataset] = None,
                 test_y: Union[None, torch.Tensor, Dataset] = None,
                 true_potential: Union[None, torch.Tensor, Dataset] = None
                 ):


        super(DualVAE_OT_Trainer, self).__init__(potential=potential,
                                                transform_x=transform_x, 
                                                transform_y=transform_y, 
                                                args=args,
                                                dataset_x=dataset_x,
                                                dataset_y=dataset_y,
                                                test_x=test_x,
                                                test_y=test_y,
                                                true_potential=true_potential)

        
        ## logging and saving
        self.exp_dir = f'experiments/{self.args.source_dist}_{self.args.target_dist}/dualVAE/dim_{np.prod(self.args.data_shape)}/num_bins_{self.args.num_bins}/lr_{self.args.learning_rate}'
        os.makedirs(self.exp_dir, exist_ok=True)
        file_list = os.listdir(self.exp_dir)
        for file in file_list:
            os.remove(os.path.join(self.exp_dir, file))
        
        if self.args.verbose:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, 
                                handlers=[logging.FileHandler(os.path.join(self.exp_dir, 'logger.log'))],
                                format='%(asctime)s - %(levelname)s - %(message)s')

    def objective(self, x, y):

        ## encoded source and target data
        T_x = self.encoder_flow.encode_x(x)
        T_y = self.encoder_flow.encode_y(y)

        ## approximate source and target data
        x_hat, mean_x, log_var_x = self.encoder_flow.transform_x(x)
        y_hat, mean_y, log_var_y = self.encoder_flow.transform_y(y)

        ## dual loss on encoded data
        value1 = self.encoder_flow.potential(T_x)
        value2 = self.encoder_flow.potential.conjugate(T_y)
        value3 = sum([self.args.regularization*torch.sum(param**2) for param in self.encoder_flow.potential.parameters()])
        dual_loss = torch.mean(value1) + torch.mean(value2) + value3

        ## VAE loss
        reproduction_loss = ((x_hat - x)**2).sum() + ((y_hat - y)**2).sum()
        KLD = - 0.5 * torch.sum(1+ log_var_x - mean_x.pow(2) - log_var_x.exp()) - 0.5 * torch.sum(1+ log_var_y - mean_y.pow(2) - log_var_y.exp())
        return dual_loss + reproduction_loss + KLD
  


def plot_losses_(exp_dir, args, true_potential):

    # open the training metrics
    with open(os.path.join(exp_dir, 'training_metrics.json'), "r") as json_file:
        train_specs = json.load(json_file)
    train_specs = pd.DataFrame(train_specs)

    num_logs = train_specs.shape[0]

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot(args.log_interval*(1+np.arange(num_logs)), train_specs['train_loss'])
    axs[0].set_xlabel('Training Steps')
    axs[0].set_ylabel('Train Loss')
    axs[1].plot(args.log_interval*(1+np.arange(num_logs)), train_specs['test_loss'])
    axs[1].set_xlabel('Training Steps')
    axs[1].set_ylabel('Test Loss')
    plt.tight_layout()
    if args.show_the_plot:
        plt.show()
    plt.savefig(os.path.join(exp_dir, 'losses.png'))
    plt.close()
    
    if true_potential:
        plt.figure(figsize=(4,3))
        plt.plot(args.log_interval*(1+np.arange(num_logs)), train_specs['mse'])
        plt.xlabel('Training Steps')
        plt.ylabel('MSE')
        if args.show_the_plot:
            plt.show()
        plt.savefig(os.path.join(exp_dir, 'mse.png'))
        plt.close()
    
    plt.close()

