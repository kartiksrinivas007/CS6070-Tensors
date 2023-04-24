import numpy as np
import matplotlib.pyplot as plt
from tensor.operation.kruskal import kruskal
from tensor.operation.khatri_rao import khatri_rao
from tensor.operation.matricize import matricize
from tensor.operation.stopping_criteria import StoppingCriteria

class CP_ALS:
    def __init__(self, tensor: np.ndarray, rank: int, max_iter=10000, eps=0.01, init_type: str = "random"):
        """
        PARAFAC decompositon, it initializes the maximum amount of iterations that 
        are done along with the tolerance value for the convergence
        it also takes the size of the core tensor `rank`
        """
        # Original Tensor
        self.tensor = tensor

        # Rank of decomposition being targeted
        self.rank = rank

        # Max iterations
        self.max_iter = max_iter

        # Stopping error value
        self.eps = eps

        # sets the type of initialisation
        self.init_type = init_type
        # Stores the errrors at each step
        self.errors = []
        # Factor matrixes
        self.factors = []

    def fit(self, check_convergence = True, stopping_criteria=None):
        self._init_factors()
        P = np.power(2, 14)
        for itr in range(self.max_iter):
            for mode in range(self.tensor.ndim):
                self._update_factors(mode)
            
            if check_convergence and stopping_criteria == "sampled":
                if StoppingCriteria(self.tensor, kruskal(*self.factors), P, threshold=self.eps):
                    break
            
            elif check_convergence and self._is_converged():
                break
        

    def _init_factors(self):
        if self.init_type == 'random':
            self.factors = [np.random.rand(self.tensor.shape[i], self.rank) for i in range(self.tensor.ndim)]
        elif self.init_type == 'hosvd':
            self.factors = []
            for i in range(self.tensor.ndim):
                M = np.linalg.svd(matricize(self.tensor, i))[0]
                if M.shape[1] < self.rank:
                    M_ = np.zeros((M.shape[0], self.rank - M.shape[1]))
                    M = np.concatenate((M, M_), axis=1)
                else:
                    M = M[:, :self.rank]
                print("M", M.shape)
                self.factors.append(M)
        else:
            raise Exception("Invalid initialisation method")


    def _update_factors(self, mode):
        khatriRaoProd = np.ones((1, self.rank))
        for j in range(self.tensor.ndim, 0, -1):
            if j != (mode + 1):
                khatriRaoProd = khatri_rao(khatriRaoProd, self.factors[j - 1])

        self.factors[mode] = matricize(self.tensor, mode) @ np.linalg.pinv(khatriRaoProd).T
        
    def _is_converged(self):
        return np.linalg.norm(self.tensor - kruskal(*self.factors)) < self.eps
    
    def plot_errors(self):
        plt.plot(self.errors)
        plt.legend(["Errors"])
        plt.xlabel("Iterations")
        plt.ylabel("Frobenius norm")
        return plt


    def decompose(X, rank, max_iter=10000, eps=0.01, init_type="random", check_convergence=True, stopping_criteria=None):
        """
        Decomposes the tensor X into a sum of rank rank tensors
        """
        cp = CP_ALS(X, rank, max_iter, eps, init_type)
        cp.fit(check_convergence, stopping_criteria=stopping_criteria)
        return cp.factors
        