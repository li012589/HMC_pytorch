# Hamiltonian Monte Carlo using Pytorch

A pytorch version of hamiltonian monte carlo

## Example

```python
from hmc import HMC
from phi4 import *
t = Phi4(2,4,2,1,1)
inital = torch.randn(10,2,4,4)
HMC(t.energy,inital,50,5,0.1)
```

