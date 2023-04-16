import numpy as np

def sampled_khatri_rao_product(*args, num_samples: int = 1):
    """sampled_khatri_rao_product computes the outer product of a list of matrices and sums them.
    Args:
        *args (np.ndarray): List of matrices.
        num_samples (int): Number of samples to take.
    Returns:
        np.ndarray: Khatri-Rao product of the matrices.
    """
    assert len(args) > 1, "At least two matrices are required."
    assert all([A.ndim == 2 for A in args]), "All matrices must be 2D."
    assert all([A.shape[1] == args[0].shape[1] for A in args]), "All matrices must have the same number of columns."

    R = args[0].shape[1]
    out = np.zeros(tuple([A.shape[0] for A in args]))
    for i in range(R):
        out += np.prod(np.ix_(*[A[:, i] for A in args]))

    return out

def sampled_khatri_rao(S, factors):
    idxs = np.array(S, dtype=int)
    n_factors = len(factors)
    Z_S = np.ones((idxs.shape[0], factors[0].shape[1]))

    for n, factor in enumerate(factors):
        A_s = factor[idxs[:, n], :]
        Z_S = np.multiply(Z_S, A_s)

    return Z_S