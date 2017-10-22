import numpy as np
import torch
from torch.autograd import Variable

class phi4():
    def __init__(self,n,l,d,kappa,lamb,name=None,):
        if name is None:
            self.name = "phi4_l"+str(l)+"_d"+str(d)+"_kappa"+str(kappa)+"_lamb"+str(lamb)
        else:
            self.name = name
        self.d = d
        self.n = n
        self.l = l
        self.kappa = kappa
        self.lamb = lamb
        self.hoppingTable = []
        for i in range(n):
            LK = n
            y = i
            self.hoppingTable.append([])
            for j in reversed(range(d)):
                LK = int(LK/l)
                xk = int(y/LK)
                y = y-xk*LK
                if xk < l-1:
                    self.hoppingTable[i].append(i + LK)
                else:
                    self.hoppingTable[i].append(i + LK*(1-l))
                if xk > 0:
                    self.hoppingTable[i].append(i - LK)
                else:
                    self.hoppingTable[i].append(i-LK*(1-l))
    def __call__(self,z):
        pass
    def _forward(self,z):
        pass
    def backward(self,z):
        pass

if __name__ == "__main__":
    pass