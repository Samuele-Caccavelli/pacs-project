from dlroms import *
from dodlib import DOD
import torch

from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import random


def IdentityScaling(theta):
    """This function performs no transformation on the input. It is useful as a default scaling operator.
    
    Input:
            theta               (torch.Tensor)              Input data.
    
    Output:
            (torch.Tensor) Unchanged input.
    """
    return theta

def Frobenius_metric(V1, V2):
    """Compute the Frobenius distance between the projection matrices associated with two basis.

    The metric is defined as:
        d = || V₁ᵀV₁ - V₂ᵀV₂ ||_F
    where ||·||_F denotes the Frobenius norm.

    Input:
            V1              (torch.Tensor)              First basis matrix of size (n_basis, nA).
            V2              (torch.Tensor)              Second basis matrix.

    Output:
            (torch.Tensor) Scalar tensor representing the Frobenius distance.
    """

    projection1 = torch.matmul(torch.t(V1), V1)
    projection2 = torch.matmul(torch.t(V2), V2)

    return torch.norm(projection1-projection2, p='fro')

class LocalBasis():
    """LocalBasis class, used for computing the quantity K via different methods.
    #! is it really called K?
    
    Attributes:
            q               (int)                       Dimensions of the parameter space.
            pbad_index_list (list of int)               List of indexes corresponding to the bad parameters.
            #! maybe we can call it p_prime instead of pbad to be more coherent with the paper
            module          (function)                  A map from Θ to R^(n_basis, Na).
                                                        #! specify what these are?
                                                        It must have the signature:
                                                            module(theta: torch.Tensor) -> torch.Tensor
            scaling         (function)                  A map from [0,1]^q to Θ * Θ'.
                                                        It must have the signature:
                                                            scaling(theta: torch.Tensor) -> torch.Tensor
            metric          (function)                  Function between two basis spaces to compute a metric returned as a scalar tensor.
                                                        It must have the signature:
                                                            metric(V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor
    """

    def __init__(self, q, pbad_index_list, module, scaling = IdentityScaling, metric = Frobenius_metric):
        """Initialize an LocalBasis object.
        
        Attributes:
                q               (int)                       Dimensions of the parameter space.
                pbad_index_list (list of int)               List of indexes corresponding to the bad parameters.
                #! maybe we can call it p_prime instead of pbad to be more coherent with the paper
                module          (function)                  A map from Θ to R^(n_basis, Na).
                                                            #! specify what these are?
                                                            It must have the signature:
                                                                module(theta: torch.Tensor) -> torch.Tensor
                scaling         (function)                  A map from [0,1]^q to Θ * Θ'.
                                                            It must have the signature:
                                                                scaling(theta: torch.Tensor) -> torch.Tensor
                                                            Defaults to `IdentityScaling`
                metric          (function)                  Function between two basis spaces to compute a metric returned as a scalar tensor.
                                                            It must have the signature:
                                                                metric(V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor
                                                            Defaults to `Frobenius_metric`
        """
        
        self.q = q
        self.pbad_index_list = pbad_index_list
        self.module = module
        self.scaling = scaling
        self.metric = metric

        # We freeze the module, so that it is not anymore trainable (needed for module based on Neural Networks)
        # We then set its `trainable` attribute to True so that we will work in the ambient space
        self.module.freeze()
        self.module.trainable = True

    def __call__(self, theta):
        # Here we are considering a theta inside [0,1]^q
        ##! maybe we should put here a check on theta being inside the right interval
        ##! but this could slow down this function that is then called by all the others
        # So we first rescale it to be inside Θ * Θ', then we compute the basis
        params = self.scaling(theta)
        return self.module(params[self.pbad_index_list].unsqueeze(0)).squeeze()

    # Private method needed for K_h_j
    def __CheckH(self, theta_hat_j, h):
        # Checks if 0 <= theta_hat_j + h <= 1
        return (h < -theta_hat_j) and (h > 1-theta_hat_j)

    def K_h_j(self, j, theta_hat, h=1e-1):
        if(j<0 or j>theta_hat.size()[0]-1):
            raise RuntimeError("The given j is out of bound for the dimensions of theta")
        if(theta_hat[j] < 0 or theta_hat[j] > 1):
            raise RuntimeError("Theta hat should be normalized first")
        if(self.__CheckH(theta_hat[j], h)):
            raise RuntimeError("The given h for the computation of K_h_j make theta_hat go out of its space")

        theta_hat_var = theta_hat.clone()
        theta_hat_var[j] += h

        # We compute the two basis spaces
        V_theta_hat = self.__call__(theta_hat)
        V_theta_hat_var = self.__call__(theta_hat_var)

        distance = self.metric(V_theta_hat, V_theta_hat_var).item()

        return distance / abs(h)

    def K_sup_j(self, j, theta_hat):
        if(j<0 or j>theta_hat.size()[0]-1):
            raise RuntimeError("The given j is out of bound for the dimensions of theta")
        if(theta_hat[j] < 0 or theta_hat[j] > 1):
            raise RuntimeError("Theta hat should be normalized first")

        def objective(h):
            ##! if we use self.K_h_j here, we don't have to write twice the same things
            ##! but in this way we are doing the checks at the start of self.K_h_j each time the optimization algorithm
            ##! calls this function
            # Attention here: we need to return (-K_h_j)
            return -(self.K_h_j(j, theta_hat, h))

        # Define bounds directly
        lower_bound = -theta_hat[j].item()
        upper_bound = 1 - theta_hat[j].item()

        # Use minimize_scalar with bounded method
        result = minimize_scalar(objective, 
                                bounds=(lower_bound, upper_bound), 
                                method='bounded')

        # We are interested in the maximum, so we return (-minimum)
        return -result.fun

    # Private method
    # Monte Carlo estimate of K_sup_j
    def __K_sup_j_tot(self, j, S, verbose, theta_dataset):
        sum = 0

        if(theta_dataset is None):
            print(S, "random values of the parameters will be generated for the Monte Carlo estimate")
            theta_dataset = torch.rand(S, self.q)

        else:
            S = theta_dataset.size()[0]
            print(S, "non-random values of the parameters were given, those will be used for the Monte Carlo estimate")

        K_vector = []

        for iter in tqdm(range(S), disable=not verbose, desc="Monte Carlo Estimate progress"):
            theta_hat = theta_dataset[iter]
            value = self.K_sup_j(j, theta_hat)
            K_vector.append(value)

        return np.mean(K_vector), np.std(K_vector)

    # Private method
    # Monte Carlo estimate of K_h_j
    def __K_h_j_tot(self, j, h, S, verbose, theta_dataset):
        sum = 0

        if(theta_dataset is None):
            print(S, "random values of the parameters will be generated for the Monte Carlo estimate")
            theta_dataset = torch.rand(S, self.q)

        else:
            S = theta_dataset.size()[0]
            print(S, "non-random values of the parameters were given, those will be used for the Monte Carlo estimate")

        K_vector = []

        for iter in tqdm(range(S), disable=not verbose, desc="Monte Carlo Estimate progress"):
            theta_hat = theta_dataset[iter]
            
            # We transform the j component of theta_hat to ensure |h| < theta_hat[j] < 1-|h|, so that 
            # we won't have errors while computing K_h_j
            theta_hat[j] = theta_hat[j] * (1-2*abs(h)) + abs(h)

            # We compute K_h_J 50% of the times in direction h, 50% in direction -h
            if random.choice([True, False]):
                K_vector.append(self.K_h_j(j, theta_hat, h))
            else:
                K_vector.append(self.K_h_j(j, theta_hat, -h))

        return np.mean(K_vector), np.std(K_vector)

    # Public method
    # Wrapper for the two private methods
    def K_j_tot(self, j, h=None, S=1000, verbose = False, theta_dataset = None):
        if(h is None):
            return self.__K_sup_j_tot(j, S, verbose, theta_dataset)
        return self.__K_h_j_tot(j, h, S, verbose, theta_dataset)

    