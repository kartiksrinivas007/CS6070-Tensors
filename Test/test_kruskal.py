"""
Test file for kruskal.py
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from tensor.operation.kruskal import kruskal 

def test_kruskal():
    X = np.array([[[1, -1], [0, 0]], [[0, 0], [1, 1]]])
    A = np.array([[1,0,-1], [0,1,1]])
    B = np.array([[1,0,1], [0,1,1]])
    C = np.array([[1,1,0], [-1,1,0]])

    Y = kruskal(A, B, C)
    assert np.allclose(Y, X),"FAILED"
    print("PASS")

if __name__ == "__main__":
    test_kruskal()

