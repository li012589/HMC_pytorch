import numpy as np
import torch
from torch.autograd import Variable
from model import energy
#import torch.nn as nn

class Ring2d(energy):
    def __init__(self,name="Ring2d"):
        super(Ring2d,self).__init__(2,name)
    def _forward(self,z):
        z1 = z[:,0]
        z2 = z[:,1]
        v = ((torch.sqrt(z1*z1+z2*z2)-2)/0.4)**2
        return v

if __name__ == "__main__":
    z = (torch.Tensor([[1,2],[3,4],[5,6]]))
    t = Ring2d()
    batchSize = z.size()[0]
    out = t(z)
    print(out)
    #t.zero_grad()
    grad = t.backward(z)
    print(grad)