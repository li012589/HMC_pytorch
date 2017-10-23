import os
import sys
sys.path.append(os.getcwd())

from model.phi4 import phi4
from hmc import HMCSampler

from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptanceRate

import numpy as np
import torch

dims = 3
l = 6
Kappa = [0.18]#[i/100 for i in range(15,23)]
lamb = 1.145
BatchSize = 100
Steps = 800
BurnIn = 300
bins = 2
modelSize = l**dims

res = []
errors = []

def prior(batchSize):
    return torch.randn(batchSize,modelSize)

for kappa in Kappa:
    model = phi4(modelSize,l,dims,kappa,lamb)
    sampler = HMCSampler(model,prior,dynamicStepSize=True)
    print("start sampler on model: ",model.name)
    print("Samples: ",BatchSize,". Steps: ",Steps)
    print("BurnIn: ",BurnIn)
    res = sampler.sample(Steps,BatchSize)
    res=np.array(res)
    #print(res)
    z_o = res[BurnIn:,:]
    m_abs = np.mean(z_o,2)
    m_abs = np.absolute(m_abs)

    m_abs_p = np.mean(m_abs)
    autoCorrelation,error =  autoCorrelationTimewithErr(m_abs,bins)
    acceptRate = acceptance_rate(z_o)
    print("kappa:",mod.kappa)
    print("lambda:",mod.lamb)
    res.append(m_abs_p)
    errors.append(error)
    print("measure: <|m|/V>",m_abs_p,"with error:",error)
    print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))


