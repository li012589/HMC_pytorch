import numpy as np
import torch
from torch.autograd import Variable

class energy():
    def __init__(self,size,name="model"):
        self.name = name
        self.size = size
    def __call__(self,z):
        z = Variable(z,requires_grad=True)
        return self._forward(z).data
    def _forward(self,z):
        raise NotImplementedError(str(type(self)))
    def backward(self,z):
        z = Variable(z,requires_grad=True)
        out = self._forward(z)
        batchSize = z.size()[0]
        out.backward(torch.ones(batchSize))
        return z.grad.data