# -*- coding: utf-8 -*-

'''
The Capsules layer.

@author: Yuxian Meng
'''
#TODO:run on a dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Capsules import PrimaryCaps, ConvCaps

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=5, stride=2)
        self.primary_caps = PrimaryCaps(B=32)
        self.convcaps1 = ConvCaps(B=32, C=32, kernel = 3, stride=2,iteration=3,
                                  coordinate_add=False, transform_share = False)
        self.convcaps2 = ConvCaps(B=32, C=32, kernel = 3, stride=1,iteration=3,
                                  coordinate_add=False, transform_share = False)
        self.classcaps = ConvCaps(B=32, C=10, kernel = 3, stride=1,iteration=3,
                                  coordinate_add=True, transform_share = True)
        
        
    def forward(self,x,lambda_): #b,1,28,28
        x = F.relu(self.conv1(x)) #b,32,12,12
        x = self.primary_caps(x) #b,32*(4*4+1),12,12
        x = self.convcaps1(x,lambda_) #b,32*(4*4+1),5,5
        x = self.convcaps2(x,lambda_) #b,32*(4*4+1),3,3
        x = self.classcaps(x,lambda_).squeeze().view(-1,10,17) #b,10*16
        return x

    
    

if __name__ == "__main__":
    lambda_ = Variable(torch.randn(1))
    x = Variable(torch.Tensor(10,1,28,28))
    model = CapsNet()
    out = model(x,lambda_) #10,10,17
    print(out.size())
        
        
