import torch
import numpy as np
import torch.nn as nn

## Encoder

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=100, latent_dim=10, device="cpu"):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
            )

    def forward(self, x):
        return self.encoder(x)


## Decoder

class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=100, latent_dim=10, device="cpu"):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            )

    def forward(self, x):
        return self.decoder(x)

## Autoencoder

class AE(nn.Module):

    def __init__(self, input_dim, hidden_dim=100, latent_dim=10, device="cpu"):
        super(AE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
            )
        

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
            )

        self.device = device
     
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


## Variational Autoencoder

class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=100, latent_dim=10, device="cpu"):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
            )

        self.device = device
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.decode(z)  
        return x_hat, mean, log_var
        
    def device(self):
        return self.device


