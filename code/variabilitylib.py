import torch
from tqdm import tqdm
import numpy as np
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
            V2              (torch.Tensor)              Second basis matrix of size (n_basis, nA).

    Output:
            (torch.Tensor) Scalar tensor representing the Frobenius distance.
    """

    projection1 = torch.matmul(torch.t(V1), V1)
    projection2 = torch.matmul(torch.t(V2), V2)

    return torch.norm(projection1-projection2, p='fro')

class LocalBasis():
    """LocalBasis class, used for computing the adaptivity scores.
    
    Attributes:
            q                   (int)                       Dimensions of the parameter space.
            p_prime_index_list  (list of int)               List of indexes corresponding to the bad parameters.
            module              (function)                  A map from Θ to R^(n_basis, nA).
                                                            It must have the signature:
                                                                module(theta: torch.Tensor) -> torch.Tensor
            scaling             (function)                  A map from [0,1]^q to Θ * Θ'.
                                                            It must have the signature:
                                                                scaling(theta: torch.Tensor) -> torch.Tensor
            metric              (function)                  Function between two basis spaces to compute a metric returned as a scalar tensor.
                                                            It must have the signature:
                                                                metric(V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor
    """

    def __init__(self, q, p_prime_index_list, module, scaling = IdentityScaling, metric = Frobenius_metric):
        """Initialize an LocalBasis object.
        
        Attributes:
                q                   (int)                       Dimensions of the parameter space.
                p_prime_index_list  (list of int)               List of indexes corresponding to the bad parameters.
                module              (function)                  A map from Θ to R^(n_basis, nA).
                                                                It must have the signature:
                                                                    module(theta: torch.Tensor) -> torch.Tensor
                scaling             (function)                  A map from [0,1]^q to Θ * Θ'.
                                                                It must have the signature:
                                                                    scaling(theta: torch.Tensor) -> torch.Tensor
                                                                Defaults to `IdentityScaling`
                metric              (function)                  Function between two basis spaces to compute a metric returned as a scalar tensor.
                                                                It must have the signature:
                                                                    metric(V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor
                                                                Defaults to `Frobenius_metric`
        """
        
        self.q = q
        self.p_prime_index_list = p_prime_index_list
        self.module = module
        self.scaling = scaling
        self.metric = metric

        # We freeze the module, so that it is not anymore trainable (needed for module based on Neural Networks)
        # We then set its `trainable` attribute to True so that we will work in the ambient space
        self.module.freeze()
        self.module.trainable = True

    def __call__(self, theta):
        """Returns the basis related to a specific normalized theta.
        
        Input:
                theta               (torch.Tensor)              Parameter vector to consider when computing the space.

        Output:
                (torch.Tensor) Basis related to a specific theta. If `trainable` is set to True, the basis returned refers to the ambient space, otherwise it refers to the full space.
        """
        # Here we are considering a theta inside [0,1]^q
        # So we first rescale it to be inside Θ * Θ', then we compute the basis
        params = self.scaling(theta)
        return self.module(params[self.p_prime_index_list].unsqueeze(0)).squeeze()

    # K_h APPROACH

    # Private method needed for K_j_h
    def _CheckH(self, theta_j, h):
        # Checks if 0 <= theta_j + h <= 1
        return (h < -theta_j) and (h > 1-theta_j)

    def K_j_h(self, j, theta, h=1e-1):
        """Returns the score K_j^h defined as

            K(theta) = metric(V_theta, V_theta_prime) / |h|

        where V_theta and V_theta_prime are the adaptive basis corresponding to theta and theta + h*e_j.
        
        Input:
                j               (int)                       Direction to consider to compute the score.
                theta           (torch.Tensor)              Parameter vector to consider when computing the score.
                h               (float, optional)           Magnitude of the displacement.
                                                            Defaults to 1e-1.

        Output:
            (float) Value of the score.      
        """        
        if(j<0 or j>theta.size()[0]-1):
            raise RuntimeError("The given j is out of bound for the dimensions of theta")
        # if the direction is not one the module is adaptive against, K is 0
        if j not in self.p_prime_index_list:
            return 0.0
        if(theta[j] < 0 or theta[j] > 1):
            raise RuntimeError("Theta hat should be normalized first")
        if(self._CheckH(theta[j], h)):
            raise RuntimeError("The given h for the computation of K_j_h make theta go out of its space")

        theta_prime = theta.clone()
        theta_prime[j] += h

        # We compute the two basis spaces
        V_theta = self.__call__(theta)
        V_theta_prime = self.__call__(theta_prime)

        distance = self.metric(V_theta, V_theta_prime).item()

        return distance / abs(h)

    def K_j_sup(self, j, theta):
        """Returns the score K_j^sup defined as

            K(theta) = sup(metric(V_theta, V_theta_prime) / |h|)

        where V_theta and V_theta_prime are the adaptive basis corresponding to theta and theta + h*e_j.
        
        Input:
                j               (int)                       Direction to consider to compute the score.
                theta           (torch.Tensor)              Parameter vector to consider when computing the score.

        Output:
            (float) Value of the score.      
        """ 
        if(j<0 or j>theta.size()[0]-1):
            raise RuntimeError("The given j is out of bound for the dimensions of theta")
        # if the direction is not one the module is adaptive against, K is 0
        if j not in self.p_prime_index_list:
            return 0.0
        if(theta[j] < 0 or theta[j] > 1):
            raise RuntimeError("Theta hat should be normalized first")

        def objective(h):
            # Attention here: we need to return -K_j_h
            return -(self.K_j_h(j, theta, h))

        # Define bounds directly
        lower_bound = -theta[j].item()
        upper_bound = 1 - theta[j].item()

        # Use minimize_scalar with bounded method
        result = minimize_scalar(objective, 
                                bounds=(lower_bound, upper_bound), 
                                method='bounded')

        # We are interested in the maximum, so we return -minimum
        return -result.fun

    # Private method
    # Monte Carlo estimate of K_j_sup
    def _K_j_sup_tot(self, j, S, verbose, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        theta_dataset = torch.rand(S, self.q)           

        K_values = []

        for theta in tqdm(theta_dataset, disable=not verbose, desc="Monte Carlo Estimate progress"):
            value = self.K_j_sup(j, theta)
            K_values.append(value)

        return np.mean(K_values), np.std(K_values)

    # Private method
    # Monte Carlo estimate of K_j_h
    def _K_j_h_tot(self, j, h, S, verbose, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        theta_dataset = torch.rand(S, self.q)

        K_values = []

        for theta in tqdm(theta_dataset, disable=not verbose, desc="Monte Carlo Estimate progress"):
            # We transform the j component of theta to ensure |h| < theta[j] < 1-|h|, so that 
            # we won't have errors while computing K_j_h
            theta[j] = theta[j] * (1-2*abs(h)) + abs(h)

            # We compute K_j_h 50% of the times in direction h, 50% in direction -h
            if random.choice([True, False]):
                K_values.append(self.K_j_h(j, theta, h))
            else:
                K_values.append(self.K_j_h(j, theta, -h))

        return np.mean(K_values), np.std(K_values)

    # Public method
    # Wrapper for the two private methods
    def K_j_tot(self, j, h=None, S=1000, verbose = False, seed = None):
        """Returns the Monte Carlo estimate of score K_j^h or K_j^sup.
        
        Input:
                j               (int)                       Direction to consider to compute the score.
                h               (float, optional)           Magnitude of the displacement to compute K_j^h.
                                                            Defaults to None - in this case, the computed value will be K_j^sup.
                S               (int)                       Monte Carlo estimate sample size.
                                                            Defaults to 1e3.
                verbose         (bool)                      When True, displays a progress bar during the Monte Carlo estimate.
                                                            Defaults to False.
                seed            (int, optional)             Random seed for reproducibility.
                                                            Defaults to None.

        Output:
            (tuple of floats) Mean and standard deviation of the scores.      
        """ 
        if j not in self.p_prime_index_list:
            return 0.0, 0.0

        if(h is None):
            return self._K_j_sup_tot(j, S, verbose, seed)
        return self._K_j_h_tot(j, h, S, verbose, seed)

    # SENSITIVITY APPROACH

    # Private method
    # Compute the variance of a stack of basis
    def _compute_spaces_variance(self, Vs):
        l = len(Vs)

        dist = 0.0
        for i in range(l//2):
            dist += self.metric(Vs[i], Vs[-i-1])

        return (dist/2)/(l//2)

    def sensitivity(self, m=30, l=20, verbose=False, seed=None):
        """Returns the sensitivity score for all directions of the parameter space.
        
        Input:
                m               (int)                       Monte Carlo estimate sample size for the approximation of E[Var(V_theta|theta_j)].
                                                            Defaults to 30.
                l               (int)                       Monte Carlo estimate sample size for the approximation of Var(V_theta|theta_j=theta_j^(k)).
                                                            If a value less then 2 is given, an error is risen.
                                                            Defaults to 20.
                verbose         (bool)                      When True, displays a progress bar for the outer Monte Carlo estimate.
                                                            Defaults to False.
                seed            (int, optional)             Random seed for reproducibility.
                                                            Defaults to None.

        Output:
            (list of floats) Sensibility score for each direction.      
        """ 
        if l < 2:
            raise ValueError(f"Inner sample size 'l' must be at least 2 to compute variance.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        sensitivities = [] # one for each direction j
        tot_var = 0.0

        params_tot_var = self.scaling(torch.rand(m*l, self.q))[:, self.p_prime_index_list]
        tot_var = self._compute_spaces_variance(self.module(params_tot_var)).item()

        for j in range(self.q):
            # if the direction is not one the module is adaptive against, the sensitivity is 0
            if j not in self.p_prime_index_list:
                sensitivities.append(0.0)
                continue

            cond_var = 0.0
            variances = []
            for _ in tqdm(range(m), disable=not verbose, desc="Monte Carlo Estimate progress"):
                th = torch.rand(1)
                theta = torch.rand(l, self.q)
                theta[:, j] = th
                params = self.scaling(theta)[:, self.p_prime_index_list]
                Vs = self.module(params)
                variances.append(self._compute_spaces_variance(Vs))
            cond_var = np.mean(variances)

            # we use this max because if m and l are too small, the tot_var could end up being smaller then a cond_var
            # so the sensitivity could end up being negative
            sensitivities.append(max(0, 1 - cond_var/tot_var))

        return sensitivities