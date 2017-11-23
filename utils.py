# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:25:51 2017

@author: Yuxian Meng
"""

import argparse
import torch
from torchvision import datasets, transforms

#TODO: data augmentation
#def augmentation(x, max_shift=2):
#    _, _, height, width = x.size()
#
#    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
#    source_height_slice = slice(max(0, h_shift), h_shift + height)
#    source_width_slice = slice(max(0, w_shift), w_shift + width)
#    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
#    target_width_slice = slice(max(0, -w_shift), -w_shift + width)
#
#    shifted_image = torch.zeros(*x.size())
#    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, 
#                 target_height_slice, target_width_slice]
#    return shifted_image.float()

def get_dataloader(args):
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    return train_loader, test_loader


def get_args():
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_epochs', type=int, default=1)
    parser.add_argument('-lr', type=float, default=2e-2)
    parser.add_argument('-clip', type=float, default=5)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-disable_cuda', action='store_true',
                    help='Disable CUDA')
    parser.add_argument('-print_freq', type=int, default=10)
    parser.add_argument('-pretrained', type=str, default="")
    parser.add_argument('-gpu', type=int, default=0, help = "which gpu to use") 
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    return args


if __name__ == "__main__":
    args = get_args()
    loader,_ = get_dataloader(args)
    print(len(loader.dataset))
    for data in loader:
        x,y = data
        print(x[0,0,:,:])
        break
