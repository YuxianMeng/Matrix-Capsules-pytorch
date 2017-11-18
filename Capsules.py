# -*- coding: utf-8 -*-

'''
The Capsules layer.

@author: Yuxian Meng
'''
#TODO:vectorize;use tensor.expand rather than stack the same tensors


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, pi
from torch.autograd import Variable


class PrimaryCaps(nn.Module):
    def __init__(self,B=32):
        super(PrimaryCaps, self).__init__()
        self.B = B
        self.capsules_pose = nn.ModuleList([nn.Conv2d(in_channels=32,out_channels=4*4,
                                                 kernel_size=1,stride=1) 
                                                 for i in range(self.B)])
        self.capsules_activation = nn.ModuleList([nn.Conv2d(in_channels=32,out_channels=1,
                                                 kernel_size=1,stride=1) for i 
                                                 in range(self.B)])

    def forward(self, x): #b,14,14,32
        poses = [self.capsules_pose[i](x) for i in range(self.B)]#(b,16,12,12) *32
        poses = torch.cat(poses, dim=1) #b,16*32,12,12
        activations = [self.capsules_activation[i](x) for i in range(self.B)] #(b,1,12,12)*32
        activations = torch.cat(activations, dim=1) #b,32,12,12
        output = torch.cat([poses, activations], dim=1)
        return output

class ConvCaps(nn.Module):
    def __init__(self, B=32, C=32, kernel = 3, stride=2,iteration=3,
                 coordinate_add=False, transform_share = False):
        super(ConvCaps, self).__init__()
        self.B =B
        self.C=C
        self.K=kernel
        self.stride = stride
        self.coordinate_add=coordinate_add
        self.transform_share = transform_share
        self.beta_v = nn.Parameter(torch.randn(1))
        self.beta_a = nn.Parameter(torch.randn(1)) 
        if not transform_share:
            self.W = nn.Parameter(torch.randn(kernel,kernel, 
                self.B, self.C, 4, 4)) #K*K*B*C*4*4
        else:
            self.W = nn.Parameter(torch.randn(self.B, self.C, 4, 4)) #B*C*4*4
        self.iteration=iteration


    def forward(self, x, lambda_,):
        b =  batch_size = x.size(0)
        use_cuda = next(self.parameters()).is_cuda
        if self.transform_share:
            W = torch.stack([self.W]*self.K**2,0).view(self.K,self.K,self.B,self.C,4,4)
        else:
            W = self.W
        pose = x[:,:-self.B,:,:] #b,16*32,12,12
        activation = x[:,-self.B:,:,:] #b,32,12,12                    
        width_in = x.size(2)  #12
        width_out = int((width_in-self.K)/self.stride+1) #5
        
        # Coordinate Addition
        if self.coordinate_add:
            add = Variable(torch.Tensor([[[[i/width_in,j/width_in] for i in range(width_in)] 
                                for j in range(width_in)] for w in range(b)])).permute(0,3,1,2) #b,2,12,12
            add = torch.cat([add,Variable(torch.zeros(b,16-2,width_in,width_in))],1) #b,16,12,12
            add = torch.stack([add]*32,dim=1)
            if use_cuda:
                add = add.cuda()
            pose = pose + add
            
        poses = [] #used to store every capsule i's poses in each capsule c's receptive field
        poses = [torch.cat([pose[:,:,self.stride*i:self.stride*i+self.K,
                       self.stride*j:self.stride*j+self.K].unsqueeze(-1)   #b,16*32,3,3,1 
                  for i in range(width_out)],dim = -1).unsqueeze(-1)  #b,16*32,3,3,5,1
                for j in range(width_out)]
        poses = torch.cat(poses, dim = -1) #b,16*B,3,3,5,5
        poses = poses.view(-1,4,4,self.B,self.K,self.K,width_out,width_out)#b,4,4,B,K,K,5,5
        poses = poses.permute(0,4,5,3,6,7,1,2).contiguous()#b,K,K,B,5,5,4,4
        W_hat = W[None,:,:,:,None,None,:,:,:]         #1,K,K,B,1,1,C,4,4
        poses_hat = poses.unsqueeze(6)                     #b,K,K,B,5,5,1,4,4
        
        votes = torch.matmul(W_hat, poses_hat) #b,K,K,B,5,5,C,4,4
        votes = votes.permute(0,3,1,2,4,5,6,7,8).contiguous()#b,B,K,K,5,5,C,4,4
#        print(votes)
        assert votes.size() == (b,self.B,self.K,self.K,width_out,width_out,self.C,4,4)
        #Start EM   
        #b,B,12,12,5,5,32
        R = Variable(torch.ones(b,self.B,width_in,width_in,width_out,width_out,self.C))/(width_out*width_out*self.C)
        if use_cuda:
            R = R.cuda()
        for iterate in range(self.iteration):
            mus = []
            sigmas = []
            activations = []            
            #M-step
            for i in range(width_out):
                for j in range(width_out):
                    for typ in range(self.C):
                        r = R[:,:,self.stride*i:self.stride*i+self.K,  #b,B,K,K  
                                self.stride*j:self.stride*j+self.K,i,j,typ]
                        a = activation[:,:,self.stride*i:self.stride*i+self.K,
                                self.stride*j:self.stride*j+self.K] #b,B,K,K
                        r_hat = r*a #b,B,K,K
#                        print(r[0,0,0,0])
                        sum_r_hat = torch.sum(r_hat.view(b,-1),1).view(b,1,1).expand(b,1,16) #b,1,16
#                        print(sum_r_hat)
                        r_hat_stack = torch.stack([r_hat]*16, dim=-1).view(b,-1,16) #b,B*K*K,16
                        V = votes[:,:,:,:,i,j,typ,:,:].contiguous().view(b,-1,16) #b,B*K*K,16
                        mu = torch.sum(r_hat_stack*V, 1, True)/sum_r_hat # b,1,16
                        mu_stack = torch.cat([mu]*self.B*self.K*self.K,dim=1) #b,B*K*K,16
#                        print(mu[0,0,:])
                        sigma = torch.sum(r_hat_stack*(V-mu_stack)**2,1,True)/sum_r_hat #b,1,16
#                        print(sigma[0,0,:])
#                        sigma = sigma.clamp(0.01) #prevent nan
#                        print(sigma)
#                        break
                        cost = (self.beta_v + torch.log(sigma)) * sum_r_hat #b,1,16
#                        print(cost)
                        
                        a_c = torch.sigmoid(lambda_*(self.beta_a-torch.sum(cost,2))) #b,1
#                        print(a_c)
                        mus.append(mu)
                        sigmas.append(sigma)
                        activations.append(a_c)
            mus = torch.cat(mus,1).view(-1,width_out,width_out,self.C,16) #b,5,5,C,16
#            print(mus[0,:,:,0,0])
            sigmas = torch.cat(sigmas,1).view(-1,width_out,width_out,self.C,16) #b,5,5,C,16
#            print(sigmas[0,0,0,:,:])
            activations = torch.cat(activations,1).view(-1,width_out,width_out,self.C) #b,5,5,C
            #E-step
            for i in range(width_in):
                #compute the x axis range of capsules c that i connect to.
                x_range = (max(floor((i-self.K)/self.stride)+1,0),min(i//self.stride+1,width_out))
                #without padding, some capsules i may not be convolutional layer catched, in mnist case, i or j == 11
                if x_range[0]>=x_range[1]: 
                    continue
                u = len(range(*x_range))
                for j in range(width_in):
                    y_range = (max(floor((j-self.K)/self.stride)+1,0),min(j//self.stride+1,width_out))
                    if y_range[0]>= y_range[1]:
                        continue
                    v = len(range(*y_range))
                    for typ in range(self.B):                 
                        mu = mus[:,x_range[0]:x_range[1],y_range[0]:y_range[1],:,:] #b,u,v,C,16
                        sigma = sigmas[:,x_range[0]:x_range[1],y_range[0]:y_range[1],:,:] #b,u,v,C,16
                        V = []; a = []  
                        for x in range(*x_range):
                            for y in range(*y_range):
                                #compute where is the V_ic
                                pos_x = self.stride*x - i
                                pos_y = self.stride*y - j
                                V.append(votes[:,typ,pos_x,pos_y,x,y,:,:,:]) #b,C,16
                                a.append(activations[:,x,y,:]) #b,C
                        V = torch.stack(V,dim=1).view(batch_size,u,-1,self.C,16) #b,u,v,C,16
                        a = torch.stack(a,dim=1).view(batch_size,u,-1,self.C)  #b,u,v,C
                        p = torch.exp(-(V-mu)**2)/torch.sqrt(2*pi*sigma) #b,u,v,C,16
                        p = p.prod(dim=4)#b,u,v,C
                        p_hat = a*p  #b,u,v,C
#                        print(p_hat[0,0,0,0])
#                        break
                        sum_p_hat = torch.sum(p_hat.view(b,-1),1).view(b,1,1,1).expand(b,u,v,self.C)
#                        print(sum_p_hat.size())
                        r = p_hat/sum_p_hat #b,u,v,C
#                        print(r)
                        
                        R[:,typ,i,j,x_range[0]:x_range[1],        #b,u,v,C
                          y_range[0]:y_range[1],:] = r
        
        mus = mus.permute(0,3,4,1,2).contiguous().view(b,self.C*16,width_out,-1)
        activations = activations.permute(0,3,1,2).contiguous().view(b,self.C*1,width_out,-1)
        output = torch.cat([mus,activations], 1) #b,C*17,5,5
        return output
                        

if __name__ == "__main__":
    lambda_ = Variable(torch.randn(1))
#    x = Variable(torch.Tensor(10,1,28,28))
#    c = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,stride=2)
#    y = c(x)#10,32,12,12
#    model = PrimaryCaps(B=32)
#    y =model(y) #10,32*17,12,12
#    convcaps1 = ConvCaps()
#    out = convcaps1(y,lambda_) #10,32*17,5,5
#    print(out[:,-2,:,:])
    
    x = F.relu(Variable(torch.randn(2,32*17,3,3)))
    model = ConvCaps(B=32, C=10, kernel = 3, stride=1,iteration=3,
                     coordinate_add=True, transform_share = True)
    y = model(x,lambda_).squeeze().view(2,10,17) #2,10,17
#    print(y[:,:,-1])
