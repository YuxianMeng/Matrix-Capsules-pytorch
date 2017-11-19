# -*- coding: utf-8 -*-

'''
The Capsules Network.

@author: Yuxian Meng
'''

#TODO:run on a dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#import time
#import numpy as np
#from six.moves import xrange

from Capsules import PrimaryCaps, ConvCaps
from utils import get_args, get_dataloader

class CapsNet(nn.Module):
    def __init__(self,A=32,B=32,C=32,D=32, E=10, r = 3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2)
        self.primary_caps = PrimaryCaps(A,B)
        self.convcaps1 = ConvCaps(B, C, kernel = 3, stride=2,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.convcaps2 = ConvCaps(C, D, kernel = 3, stride=1,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.classcaps = ConvCaps(D, E, kernel = 0, stride=1,iteration=r,
                                  coordinate_add=False, transform_share = True) #TODO:coordinate addition
        
        
    def forward(self,x,lambda_): #b,1,28,28
        x = F.relu(self.conv1(x)) #b,32,12,12
        x = self.primary_caps(x) #b,32*(4*4+1),12,12
        x = self.convcaps1(x,lambda_) #b,32*(4*4+1),5,5
        x = self.convcaps2(x,lambda_) #b,32*(4*4+1),3,3
        x = self.classcaps(x,lambda_).view(-1,10*16+10) #b,10*16+10
        return x
    
    def loss(self, x, target, m): #x:b,10 target:b
        b = x.size(0)
        a_t = torch.cat([x[i][target[i]] for i in range(b)]) #b
        a_t_stack = a_t.view(b,1).expand(b,10).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        mask = u.ge(0).float() #max(u,0) #b,10
        loss = ((mask*u)**2).sum()/b - m**2  #float
        return loss
    
    def loss2(self,x ,target):
        loss = F.cross_entropy(x,target)
        return loss
      

if __name__ == "__main__":
    args = get_args()
    train_loader, test_loader = get_dataloader(args)
    lambda_ = 1e-3 #TODO:increase with time
    m = 0.2  #TODO:increase with time
    A,B,C,D,E,r = 64,8,16,16,10,1 # a small CapsNet
    model = CapsNet(A,B,C,D,E,r)
    
    if args.use_cuda: 
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_mode = True
    for epoch in range(args.num_epochs):
        b = 0
        for data in train_loader:
            b += 1
            optimizer.zero_grad()
            imgs,labels = data #b,1,28,28; #b
            imgs,labels = Variable(imgs),Variable(labels)
            if args.use_cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            out = model(imgs,lambda_) #b,10,17
            out_poses, out_labels = out[:,:-10],out[:,-10:] #b,16*10; b,10
            loss = model.loss(out_labels, labels, m)
#            loss = model.loss2(out_labels, labels)
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            loss.backward()
            optimizer.step()
            print("loss:{}".format(loss.data[0]))
            if b % 10 == 0:
                print(out_labels.data[:1],labels.data[:1])
            
            
            

        
        
