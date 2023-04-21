import numpy as np
from sampled_khatri_rao_prod import SKR
from khatri_rao import khatri_rao

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

tensor =np.zeros((7,8,9,10))
# tensor=np.zeros((3,4,5))
R=2 # rank
#test for all modes
for mode in range(tensor.ndim):
    factors=[]
    for i in range(tensor.ndim):
        factors.append(np.random.rand(tensor.shape[i],R))
    #test for all entries of khatri rao
    # S=np.arange(np.prod(tensor.shape)//tensor.shape[mode])

    #test for 10 random entries of the khatri rao product
    S=np.random.choice(np.prod(tensor.shape)//tensor.shape[mode],10,replace=False)
    skr=SKR(S,factors,mode)

    #this worls for the reverse Khatri rao product of factor matrices as given in algorithm
    proper_kr=np.ones(shape=(1,R))
    for i in reversed(range(tensor.ndim)):
        if i == mode:
            continue
        proper_kr=khatri_rao(proper_kr,factors[i])
    if np.allclose(proper_kr[S,:],skr):
        print(bcolors.OKGREEN+"PASS for mode {}".format(mode))
    else:
        print(bcolors.FAIL+"FAIL for mode {}".format(mode))