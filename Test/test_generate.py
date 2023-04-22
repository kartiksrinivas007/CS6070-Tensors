import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

from tensor.operation.generate import random_tensor

def test_random_tensor_shape():
    X = random_tensor(3, (10, 10, 8), 5, 0.1)
    assert X.shape == (10, 10, 8), "FAILED"
    print("PASS")

def test_random_tensor():
    X = random_tensor(5, (10, 10, 8, 9, 7), 5, 0.1)
    assert X.shape == (10, 10, 8, 9, 7), "FAILED"
    print("PASS")


if __name__ == "__main__":
    test_random_tensor()
    test_random_tensor_shape()
