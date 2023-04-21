import numpy as np
from sampled_matricize import Sampled_Matricize
from matricize import matricize

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

X=np.random.rand(5,6,10,9,1,8)

for mode in range(X.ndim):
    S=np.random.choice(np.prod(X.shape)//X.shape[mode] ,10)
    # S=np.arange(np.prod(X.shape)//X.shape[mode])
    SX_n=Sampled_Matricize(S,X,mode)
    X_n=matricize(X,mode)
    if np.allclose(X_n[:,S],SX_n):
        print(bcolors.OKGREEN+"PASS for mode {}".format(mode))
    else:
        print(bcolors.FAIL+"FAIL for mode {}".format(mode))