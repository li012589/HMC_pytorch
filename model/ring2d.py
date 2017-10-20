import numpy as np
import torch
from torch.autograd import Variable
#import torch.nn as nn

class Ring2d():
    def __init__(self,name="Ring2d"):
        self.name = name
    def __call__(self,z):
        z = Variable(z,requires_grad=True)
        return self._forward(z).data
    def _forward(self,z):
        z1 = z[:,0]
        z2 = z[:,1]
        v = ((torch.sqrt(z1*z1+z2*z2)-2)/0.4)**2
        return v
    def backward(self,z):
        z = Variable(z,requires_grad=True)
        out = self._forward(z)
        batchSize = z.size()[0]
        out.backward(torch.ones(batchSize))
        return z.grad.data

if __name__ == "__main__":
    z = (torch.Tensor([[1,2],[3,4],[5,6]]))
    t = Ring2d()
    batchSize = z.size()[0]
    out = t(z)
    print(out)
    #t.zero_grad()
    grad = t.backward(z)
    print(grad)