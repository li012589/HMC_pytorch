import os
import sys
sys.path.append(os.getcwd())

from model.ring2d import Ring2d
from hmc import HMCSampler

from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptanceRate

import numpy as np
import torch

modelSize = 2
def prior(batchSize):
    return torch.randn(batchSize,modelSize)
model = Ring2d()
sampler = HMCSampler(model,prior,dynamicStepSize=True)
BatchSize = 100
Steps = 800
BurnIn = 300
bins = 2
print("start sampler on model: ",model.name)
print("Samples: ",BatchSize,". Steps: ",Steps)
print("BurnIn: ",BurnIn)
res = sampler.sample(Steps,BatchSize)
res=np.array(res)
#print(res)
z_o = res[BurnIn:,:]
z_ = np.reshape(z_o,[-1,modelSize])
z1_,z2_= z_[:,0],z_[:,1]
print("mean: ",np.mean(z1_))
print("std: ",np.std(z1_))
autoCorrelation,error =  autoCorrelationTimewithErr(z_o[:,:,0],bins)
acceptRate = acceptanceRate(z_o)
print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))
