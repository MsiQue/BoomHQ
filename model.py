"""
This module (model.py) defines helper classes, models, and utility functions
specifically for use by the main script 'HQ_single.py'.
"""

import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class Binary_Predictor(nn.Module):
    def __init__(self, hidden_dim, scalar_position_dim, scalar_magnitude_dim, output_dim):
        super(Binary_Predictor, self).__init__()
        self.scalar_position_f = nn.Linear(scalar_position_dim, hidden_dim)
        self.scalar_magnitude_f = nn.Linear(scalar_magnitude_dim, hidden_dim)
        self.pre_check_list_f = NN(3, hidden_dim, hidden_dim)
        self.index_info_f = NN(2, hidden_dim, hidden_dim)
        self.final = NN(hidden_dim * 4, hidden_dim, output_dim)

    def forward(self, query_scalar_position, query_scalar_magnitude, pre_check_list, index_info):
        f1 = self.pre_check_list_f(pre_check_list)
        f2 = self.index_info_f(index_info)
        f3 = self.scalar_position_f(query_scalar_position)
        f4 = self.scalar_magnitude_f(query_scalar_magnitude)
        f = torch.cat((f1, f2, f3, f4), dim=1)
        return self.final(f)

class Predictor(nn.Module):
    def __init__(self, VAE, hidden_dim, vae_output_dim, output_dim):
        super(Predictor, self).__init__()
        self.VAE = VAE
        self.pre_check_list_f = NN(3, hidden_dim, hidden_dim)
        self.index_info_f = NN(2, hidden_dim, hidden_dim)
        self.ef_search_list_f = NN(2, hidden_dim, hidden_dim)
        self.final = NN(hidden_dim * 3 + vae_output_dim, hidden_dim, output_dim)

    def forward(self, query_vector, query_scalar, pre_check_list, index_info, ef_search_list):
        x, recon_x, _, _ = self.VAE(query_vector, query_scalar)
        delta = x - recon_x
        f1 = self.pre_check_list_f(pre_check_list)
        f2 = self.index_info_f(index_info)
        f3 = self.ef_search_list_f(ef_search_list)
        f = torch.cat((f1, f2, f3, delta), dim=1)
        return self.final(f)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class vector_scalar(nn.Module):
    def __init__(self, vector_dim, scalar_dim, hidden_dim, latent_dim):
        super(vector_scalar, self).__init__()
        self.vector_nn = NN(vector_dim, hidden_dim, hidden_dim)
        self.scalar_nn = NN(scalar_dim, hidden_dim, hidden_dim)
        self.VAE = VAE(hidden_dim * 2, hidden_dim, latent_dim)

    def forward(self, vector, scalar):
        vector_emb = self.vector_nn(vector)
        scalar_emb = self.scalar_nn(scalar)
        emb = torch.cat([vector_emb, scalar_emb], dim=1)
        return (emb, *self.VAE(emb))