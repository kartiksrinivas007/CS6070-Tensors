"""
Test file for kruskal.py
"""

import numpy as np
# import pytest

from kruskal import kruskal 

def test_kruskal():
    X = np.array([[[1, -1], [0, 0]], [[0, 0], [1, 1]]])
    A = np.array([[1,0,-1], [0,1,1]])
    B = np.array([[1,0,1], [0,1,1]])
    C = np.array([[1,1,0], [-1,1,0]])

    Y = kruskal(A, B, C)
    assert np.allclose(Y, X)
    
if __name__ == "__main__":
    test_kruskal()

