import numpy as np
import torch
from torch.autograd import Variable
from utils.metropolis import metropolis
from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptanceRate


class HMCSampler:
    def __init__(self,model,prior,stepSize=0.1,interSteps=10,targetAcceptRate=0.65,stepSizeMin=0.001,stepSizeMax=1000,stepSizeChangeRatio=0.03,dynamicStepSize=False):
        pass
        self.model = model
        self.prior = prior
        self.stepSize = stepSize
        self.interSteps = interSteps
        self.dynamicStepSize = dynamicStepSize
        if self.dynamicStepSize:
            self.targetAcceptRate = targetAcceptRate
            self.stepSizeMax = stepSizeMax
            self.stepSizeMin = stepSizeMin
            self.stepSizeInc = stepSizeChangeRatio+1
            self.stepSizeDec = 1-stepSizeChangeRatio

    def updateStepSize(self,accept,stepSize):
        ratio = torch.sum(accept)/accept.shape[0]*accept[1]
        if ratio > self.targetAcceptRate:
            newStepSize = stepSize*self.stepSizeInc
        else:
            newStepSize = stepSize*self.stepSizeDec
        newStepSize = max(min(self.stepSizeMax,newStepSize),self.stepSizeMin)
        return newStepSize

    @staticmethod
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

    @staticmethod
    def hamiltonian(energy,v):
        return energy+0.5*torch.sum(v**2,1)

    def sample(self,steps,batchSize):
        z = self.prior(batchSize)
        zpack = [z.numpy()]
        for i in range(steps):
            v = torch.randn(z.size())
            zp,vp = self.hmcUpdate(z,v,self.model,self.stepSize,self.interSteps)
            accept = metropolis(self.hamiltonian(self.model(z),v),self.hamiltonian(self.model(zp),vp))
            if self.dynamicStepSize:
                self.stepSize = self.updateStepSize(accept,self.stepSize)
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