import numpy as np
import torch
from torch.autograd import Variable

class Ring2d:
    def __init__(self,name="Ring2d"):
        self.name = name
    def __call__(self,z):
        z = Variable(z,requires_grad=True)
        z1 = z[:,0]
        z2 = z[:,1]
        v = ((torch.sqrt(z1*z1+z2*z2)-2)/0.4)**2
        return v

if __name__ == "__main__":
    z = torch.Tensor([[1,2],[3,4],[5,6]])
    t = Ring2d()
    res = t (z)
    print(res)