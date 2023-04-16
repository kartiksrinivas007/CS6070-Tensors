"""
File for the Khatri-Rao product.

"""

import numpy as np

def khatri_rao(A: np.ndarray, B: np.ndarray):
    """khatriRao computes the Khatri-Rao product of two matrices A and B.

    Args:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.

    Returns:
        np.ndarray: Khatri-Rao product of A and B.

    """
    assert A.ndim == 2, "A must be a 2D matrix." 
    assert B.ndim == 2, "B must be a 2D matrix."

    # Check if the matrices have the same number of columns
    if A.shape[1] != B.shape[1]:
        raise ValueError("The number of columns of A and B must be the same.")

    # Compute the Khatri-Rao product
    prod = np.zeros((A.shape[0] * B.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        prod[:, i] = np.kron(A[:, i], B[:, i])

    return prod
