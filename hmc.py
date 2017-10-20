import numpy as np
import torch
from torch.autograd import Variable
from utils.metropolis import metropolis

def hmcUpdate(z,model,stepSize,interSteps):
    pass

class HMCSampler:
    def __init__(self,model,prior,stepSize,interSteps):
        pass
        self.model = model
        self.prior = prior
        self.stepSize = stepSize
        self.interSteps = interSteps

    def sample(self,steps,batchSize):
        z = self.prior(batchSize)
        zpack = [z]
        for i in range(steps):
            zp = hmcUpdate(z,self.model,self.stepSize,self.interSteps)
            accept = metropolis(self.model(z),self.model(zp))
            if accept:
                z = zp
            zpack.append(z)
        return zpack