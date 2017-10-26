import os
import sys
sys.path.append(os.getcwd())

from model.phi4 import phi4
from hmc import HMCSampler

from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptanceRate

import numpy as np
import torch

import argparse
parser = argparse.ArgumentParser(description='')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-show", action='store_true',  help="show figure right now")
group.add_argument("-outname", default="result.pdf",  help="output pdf file")

#explain these variables
dims = 3
l = 6
kappalist = np.arange(0.15,0.22,0.01).tolist()
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

for kappa in kappalist:
    model = phi4(modelSize,l,dims,kappa,lamb)
    sampler = HMCSampler(model,prior,dynamicStepSize=True)
    print("start sampler on model: ",model.name)
    print("Samples: ",BatchSize,". Steps: ",Steps)
    print("BurnIn: ",BurnIn)
    ret = sampler.sample(Steps,BatchSize)
    ret=np.array(ret)
    #print(res)
    z_o = ret[BurnIn:,:]
    m_abs = np.mean(z_o,2)
    m_abs = np.absolute(m_abs)

    m_abs_p = np.mean(m_abs)
    autoCorrelation,error =  autoCorrelationTimewithErr(m_abs,bins)
    acceptRate = acceptanceRate(z_o)
    print("kappa:",model.kappa)
    print("lambda:",model.lamb)
    res.append(m_abs_p)
    errors.append(error)
    print("measure: <|m|/V>",m_abs_p,"with error:",error)
    print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))

print("Results: ",res)
print("Errors: ",errors)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(kappalist,res,yerr=errors)
ax.set_title("\langle$|m|/V,\lambda = 1.145\rangle$")
ax.set_ylabel("$\langle|m|/V\rangle$")
ax.set_xlabel("$\kappa$")

if args.show:
    plt.show()
else:
    plt.savefig(args.outname, dpi=300, transparent=True)
