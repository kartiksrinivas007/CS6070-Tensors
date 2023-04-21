"""
Test file for khatri_rao.py
"""

import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from tensor.operation.khatri_rao import khatri_rao, _get_product_index, _get_factor_indices, sampled_khatri_rao

def test_khatri_rao():
    """Test for the khatri_rao function.
    """
    # Define the test matrices
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12]])

    # Compute the Khatri-Rao product    
    C = khatri_rao(A, B)
    # print(C)

    # Check if the result is correct
    assert np.allclose(C, np.array([[7, 16, 27], [10, 22, 36], [28, 40, 54], [40, 55, 72]]))

    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # Compute the Khatri-Rao product
    C = khatri_rao(A, B)
    # print(C)
    assert np.allclose(C, np.array([[7, 16, 27], [10, 22, 36], [13, 28, 45], [28, 40, 54], [40, 55, 72], [52, 70, 90]]))



def test_khatri_rao_error():
    """Test for the khatri_rao function.
    """
    # Define the test matrices
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # Compute the Khatri-Rao product
    with pytest.raises(ValueError):
        C = khatri_rao(A, B)


def test_get_product_index():
    """Test for the _get_product_index function.
    """

    I = np.array([1, 2, 3])
    Sz = np.array([4, 5, 6])

    index = _get_product_index(I, Sz)
    # index = 3 + 2 * 6 + 1 * 6 * 5 = 45

    assert index == 45


def test_get_factor_indices():
    """Test for the _get_factor_indices function.
    """

    index = 45
    Sz = np.array([4, 5, 6])

    I = _get_factor_indices(index, Sz)
    # I = [1, 2, 3]

    assert np.allclose(I, np.array([1, 2, 3]))


def test_sampled_khatri_rao():
    """Test for the sampled_khatri_rao function.
    """
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    C_true = khatri_rao(A, B)

    # select 3 sample indices from [0, 2*3]
    S = np.random.choice(2*3, 3, replace=False)
    C = sampled_khatri_rao(S,[A, B])

    assert np.allclose(C, C_true[S, :])







if __name__ == "__main__":

    test_khatri_rao()
    test_get_product_index()
    test_get_factor_indices()
    test_sampled_khatri_rao()
    print("ALL TESTS PASSED")