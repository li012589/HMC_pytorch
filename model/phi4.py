import numpy as np
import torch
from torch.autograd import Variable
from model import energy

class phi4(energy):
    def __init__(self,n,l,d,kappa,lamb,name=None,):
        if name is None:
            name = "phi4_l"+str(l)+"_d"+str(d)+"_kappa"+str(kappa)+"_lamb"+str(lamb)
        else:
            pass
        super(phi4,self).__init__(n,name)
        assert n == l**d
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
    def _forward(self,z):
        S = Variable(torch.zeros(z[:,0].shape))
        for i in range(self.n):
            tmp = Variable(torch.zeros(z[:,0].shape))
            for j in range(self.d):
                #print(z[:,self.hoppingTable[i][j*2]])
                tmp += z[:,self.hoppingTable[i][j*2]]
            #print("tmp: ",tmp)
            S += -2*self.kappa*tmp*z[:,i]
        S+=torch.sum(z**2,1)+self.lamb*torch.sum((z**2-1)**2,1)
        return S

if __name__ == "__main__":
    pass