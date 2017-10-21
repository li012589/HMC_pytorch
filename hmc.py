import numpy as np
import torch
from torch.autograd import Variable
from utils.metropolis import metropolis
from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptanceRate

def hmcUpdate(z,v,model,stepSize,interSteps):
    force = model.backward(z)
    vp = v - 0.5*stepSize*force
    zp  = z + stepSize*vp
    for i in range(interSteps):
        force = model.backward(zp)
        vp -= stepSize*force
        zp += stepSize*vp
    force = model.backward(z)
    vp = v - 0.5*stepSize*force
    return zp,vp

def hamiltonian(energy,v):
    return energy+0.5*torch.sum(v**2,1)

class HMCSampler:
    def __init__(self,model,prior,stepSize=0.1,interSteps=10):
        pass
        self.model = model
        self.prior = prior
        self.stepSize = stepSize
        self.interSteps = interSteps

    def sample(self,steps,batchSize):
        z = self.prior(batchSize)
        zpack = [z.numpy()]
        for i in range(steps):
            v = torch.randn(z.size())
            zp,vp = hmcUpdate(z,v,self.model,self.stepSize,self.interSteps)
            accept = metropolis(hamiltonian(self.model(z),v),hamiltonian(self.model(zp),vp))
            accept = torch.stack([accept,accept],1)
            mask = 1-accept
            z = torch.from_numpy(z.numpy()*mask.numpy() +zp.numpy()*accept.numpy())
            zpack.append(z.numpy())
        return zpack

if __name__ == "__main__":
    from model.ring2d import Ring2d
    modelSize = 2
    def prior(batchSize):
        return torch.randn(batchSize,modelSize)
    model = Ring2d()
    sampler = HMCSampler(model,prior)
    BatchSize = 100
    Steps = 800
    BurnIn = 300
    bins = 2
    print("start sampler")
    print("")
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