import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

class CVAEDataset(Dataset):
    def __init__(self,data):
        self.len = len(data)
        x_data = data.loc[:, data.columns != 'location'].values
        y_data = data['location'].values

        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(str(idx.device))
    onehot.scatter_(1, idx, 1)

    return onehot


class VAE(nn.Module):
    """
    Reference: https://github.com/timbmg/VAE-CVAE-MNIST
    """
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0, num_condition=3):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.num_condition = num_condition
        
        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels, num_condition)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels, num_condition)
    
    def encode(self, x, c=None):
        batch_size = x.size(0)
        means, log_var = self.encoder(x, c)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means
        
        return z
    
    def forward(self, x, c=None):
        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z
    
                
    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, num_condition):

        super().__init__()

        self.conditional = conditional
        self.num_condition = num_condition
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
#         print('layer_sizes :',layer_sizes)
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_condition)
            c = c.to(dtype=torch.float64).cuda()
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x.float())
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, num_condition):

        super().__init__()

        self.MLP = nn.Sequential()
        self.num_condition = num_condition
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(out_size))
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=self.num_condition)
            z = torch.cat((z, c), dim=-1)
        
        x = self.MLP(z)
        return x