import numpy as np
import index_manipulation as ind_man
import khatri_rao 

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

# def sampled_khatri_rao_product(*args, num_samples: int = 1):
#     """sampled_khatri_rao_product computes the outer product of a list of matrices and sums them.
#     Args:
#         *args (np.ndarray): List of matrices.
#         num_samples (int): Number of samples to take.
#     Returns:
#         np.ndarray: Khatri-Rao product of the matrices.
#     """
#     assert len(args) > 1, "At least two matrices are required."
#     assert all([A.ndim == 2 for A in args]), "All matrices must be 2D."
#     assert all([A.shape[1] == args[0].shape[1] for A in args]), "All matrices must have the same number of columns."

#     R = args[0].shape[1]
#     out = np.zeros(tuple([A.shape[0] for A in args]))
#     for i in range(R):
#         out += np.prod(np.ix_(*[A[:, i] for A in args]))

#     return out

# def sampled_khatri_rao(S, factors):
#     idxs = np.array(S, dtype=int)
#     n_factors = len(factors)
#     Z_S = np.ones((idxs.shape[0], factors[0].shape[1]))

#     for n, factor in enumerate(factors):
#         A_s = factor[idxs[:, n], :]
#         Z_S = np.multiply(Z_S, A_s)

#     return Z_S

# def sampled_khatri_rao2(factor_mat, n_samples, skip_mat_index, indices_list=None):
#     if skip_mat_index is not None:
#         factor_mat = np.delete(factor_mat, skip_mat_index, axis=0)
#     if indices_list is None:
#         indices_list = [np.random.choice(factor_mat.shape[0], n_samples) for _ in range(factor_mat.shape[0])]
    
#     rank = factor_mat.shape[1]

#     sampled_khatri_rao_mat = np.ones((n_samples, rank))
#     for index, matrix in zip(indices_list, factor_mat):
#         sampled_khatri_rao_mat = np.multiply(sampled_khatri_rao_mat, matrix[index, :])
    
#     return sampled_khatri_rao_mat


def SKR(S :np.ndarray,factors,mode: int):
    """Finds sampled Khatri rao for a given range of indices

    Args:
        S (np.ndarray): the row numbers on which sampled khatri rao is to be found on
        factors (_type_): the factor matrices used for khatri rao calculation in standard formula
        mode (int): the mode for which this calculation is being done

    Returns:
        _type_: len(S)xR matrix of all the given rows form the actual khatri rao which were selected
    """
    # column no should be same
    assert all([x.ndim == 2 for x in factors]), "All matrices must be 2D."
    assert all(factors[0].shape[1]==x.shape[1] for x in factors), "All matrices must have the same number of columns."
    R=factors[0].shape[1]
    N=len(factors)
    S_sz=len(S)
    tensor_shape=[factors[i].shape[0] for i in range(len(factors))]

    indices=np.zeros(shape=(S_sz,N),dtype=int)
    
    for i in range(S_sz):
        indices[i,:]= ind_man.convert_to_tensor_index(tensor_shape,mode,(0,S[i]))
    ret=np.ones(shape=(S_sz,R))
    for n in range(N):
        if n==mode :
            continue
        # for tmp in range(S_sz):
        #     ret[tmp,:]=np.multiply(ret[tmp,:],factors[n][indices[tmp,n],:])
        ret*=factors[n][indices[:,n],:]
    return ret

tensor =np.zeros((7,8,9,10))
# tensor=np.zeros((3,4,5))
R=2
for mode in range(tensor.ndim):
    factors=[]
    for i in range(tensor.ndim):
        factors.append(np.random.rand(tensor.shape[i],R))
    S=np.arange(np.prod(tensor.shape)//tensor.shape[mode])
    skr=SKR(S,factors,mode)

    proper_kr=np.ones(shape=(1,R))
    for i in reversed(range(tensor.ndim)):
        if i == mode:
            continue
        proper_kr=khatri_rao.khatri_rao(proper_kr,factors[i])
    if np.allclose(proper_kr,skr):
        print(bcolors.OKBLUE+"""I'm hooked on a feelin'
I'm high on believin'
That you're in love with me""")
        print(bcolors.OKGREEN+"\tPASS for mode {}\n".format(mode))
    else:
        print(bcolors.WARNING+"""Raahein aisi jinki
Manzil hi nahin
Dhoondho mujhe ab
Main rehta hoon wahin
Dil hai kahin aur
Dhadkan hai kahin
Saansein hai magar
Kyun zinda main nahin""")
        print(bcolors.FAIL+"\tFAIL for mode {}\n""".format(mode))
