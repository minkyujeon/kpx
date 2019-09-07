import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

from models import CVAEDataset, VAE

def main(args):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()
    new_data = pd.read_pickle("new_data_cvae.pkl")
    
    new_data = new_data.fillna(method='ffill')
    
    train, test = train_test_split(new_data, test_size=0.2)
    
    train_dataset = CVAEDataset(train)
    test_dataset = CVAEDataset(test)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)
    
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.mse_loss(
            recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=3 if args.conditional else 0,
        num_condition=args.num_condition).to(device)
    print('model:',vae)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        #tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)
            #print('z:',z.shape,z)
#             for i, yi in enumerate(y):
#                 id = len(tracker_epoch)
#                 tracker_epoch[id]['x'] = z[i, 0].item()
#                 tracker_epoch[id]['y'] = z[i, 1].item()
#                 tracker_epoch[id]['label'] = yi.item()
                
            loss = loss_fn(recon_x.float(), x.float(), mean.float(), log_var.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, 3).long().unsqueeze(1)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference(n=3)

    with torch.no_grad():
        z = []
        for iteration, (x,y) in enumerate(test_loader):

            test_x = x.to(device)
            test_y = y.to(device)
#             if args.conditional:
#                 test_recon_x, test_mean, test_log_var, test_z = vae(test_x,test_y)
#             else:
#                 test_recon_x, test_mean, test_log_var, test_z = vae(test_x)
            z += [vae.encode(test_x.to(device), test_y.to(device))]
                #test_y = test_y.detach().cpu().numpy()
                
        z = torch.cat(z, dim=0)
        #print('z:',z.shape,len(z))
        z = z.mean(dim=0).cpu().numpy()
        #print('z_after:',z.shape,len(z))
        z = z[:,:2]
        plt.scatter(x=test_z[:,0], y=test_z[:,1], c = test_y.cpu().numpy(), alpha=3)# , s='tab10')
        plt.colorbar()
        plt.savefig('./plot/latent_space'+'.png',format='png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[train.shape[1]-1, train.shape[1]-5, train.shape[1]//2])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[train.shape[1]//2,
                                                                    train.shape[1]-5,
                                                                    train.shape[1]-1])
    parser.add_argument("--latent_size", type=int, default=7)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--conditional", action='store_true', default=1)
    parser.add_argument("--num_condition", type=int, default=len(train['location'].unique()))

    args = parser.parse_args()

    main(args)