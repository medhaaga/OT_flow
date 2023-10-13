import torch
import numpy as np
import torch.nn as nn
from potential_flows import potential, encoders

## Encoder

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=100, latent_dim=10, device="cpu"):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

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

class AE(encoders.EncoderDecoder):

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

class VAE(encoders.EncoderDecoder):

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


## Encoder and Kantrovich Potential      

class Encoder_OT(nn.Module):

    """
    The class that combines the implementation of mapping source 
    and target distribution to the latent space using encoder 
    and using the potential flow to find the OT between the two
    latent spaces.

    encoder_x: Encoder for source distribution
    encoder_y: Encoder for target distribution
    potential_flow: between the source and target latent spaces
    """

    def __init__(self,
                encoder_x: Encoder,
                encoder_y: Encoder,
                potential: potential.Potential):
        super(Encoder_OT, self).__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.potential = potential

    def encode_x(self, x):
        return self.encoder_x(x)
    
    def encode_y(self, y):
        return self.encoder_y(y)

## Autoencoder and Kantrovich Potential      

class AE_OT(encoders.EncoderDecoder_OT):

    """
    The class that combines the implementation of mapping source 
    and target distribution to the latent space using autoencoder 
    and using the potential flow to find the OT between the two
    latent spaces.

    ae_x: Autoencoder for source distribution
    ae_y: Autoencoder for target distribution
    potential_flow: between the source and target latent spaces
    """
    def __init__(self,
                ae_x: AE,
                ae_y: AE,
                potential: potential.Potential):
        super(AE_OT, self).__init__(transform_x=ae_x, transform_y=ae_y, potential=potential)
    
        self.ae_x = ae_x
        self.ae_y = ae_y
        self.potential = potential


## Variational Autoencoder and Kantrovich Potential      

class VAE_OT(nn.Module):

    """
    The class that combines the implementation of mapping source 
    and target distribution to the latent space using VAE and
    using the potential flow to find the OT between the two
    latent spaces.

    vae_x: Autoencoder for source distribution
    vae_y: Autoencoder for target distribution
    potential_flow: between the source and target latent spaces
    """

    def __init__(self,
                vae_x: VAE,
                vae_y: VAE,
                potential: potential.Potential):
        super(VAE_OT, self).__init__(transform_x=vae_x, transform_y=vae_y, potential=potential)
    