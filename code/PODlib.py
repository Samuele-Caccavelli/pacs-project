import torch
from scipy.linalg import svd
import numpy as np
from dlroms import euclidean
import matplotlib.pyplot as plt

def omega_weights(theta, theta_i, lambda_penalty = 1e-2, p_prime_index_list=None):
    """Compute an exponentially decaying weight based on the squared Euclidean distance between two parameter vectors.
    The weight follows a Gaussian-like decay:
        w = exp(-‖θ - θᵢ‖² / λ_penalty²)
    where the distance is computed only over the indices specified in `p_prime_index_list`.
    Closer vectors yield higher weights.
    
    Input:
            theta               (torch.Tensor)              First parameter vector.
            theta_i             (torch.Tensor)              Second parameter vector.
            lambda_penalty      (float, optional)           Scaling parameter controlling the rate of exponential decay.
                                                            Defaults to 1e-2.
            p_prime_index_list  (list of int, optional)     Indices specifying the components of the vectors to consider in the distance computation.
                                                            Defaults to None - in this case all components of the vectors are used.
    
    Output:
            (float) Exponentially decaying weight.
    """
    if(theta.size(0) != theta_i.size(0)):
        raise RuntimeError("The inputs given have an incompatible shape.")

    if(p_prime_index_list is None):
        p_prime_index_list = list(range(theta.size(0)))
    
    return np.exp((-euclidean(theta[p_prime_index_list]-theta_i[p_prime_index_list], squared=True)/lambda_penalty**2).item())

class weighted_POD:
    """Class implementing the weighted POD method.
    
    Attributes:
            A               (torch.Tensor)              Ambient space.
            U               (torch.Tensor)              Tensor containing the dataset.
            theta_full      (torch.Tensor)              Tensor containing the parameters related to the dataset of size (q, n_snapshot).
            n_basis         (int)                       Number of basis to be returned.
            omega_func      (function)                  Function to compute the weights.
                                                        It must have the signature:
                                                            omega_func(theta: torch.Tensor, theta_i: torch.Tensor) -> float
            trainable       (bool)                      Bool value describing if working in the ambient space or in the full space.
                                                        When True, the returned space is considered inside the ambient space.

    Private attributes:
            _s_values       (ndarray)                   The singular values, sorted in non-increasing order.
    """

    def __init__(self, A, U, theta_full, n_basis, omega_func):
        """Initialize a weighted_POD object.
        
        Input:
                A               (torch.Tensor)              Ambient space.
                U               (torch.Tensor)              Tensor containing the dataset.
                theta_full      (torch.Tensor)              Tensor containing the parameters related to the dataset of size (q, n_snapshot).
                n_basis         (int)                       Number of basis to be returned.
                omega_func      (function)                  Function to compute the weights.
                                                            It must have the signature:
                                                                omega_func(theta: torch.Tensor, theta_i: torch.Tensor) -> float
        """
        
        self.A = A
        self.U = U
        self.theta_full = theta_full
        self.n_basis = n_basis
        self.omega_func = omega_func
        self.trainable = True
        self._s_values = None

    def compute_space(self, theta):
        """Compute the basis related to a specific theta.
        
        Input:
                theta           (torch.Tensor)              Parameter vector to consider when computing the space.

        Output:
                (torch.Tensor) Basis related to a specific theta.
        """
        W = torch.empty(self.U.shape[0], self.U.shape[1])

        if(theta.ndim != 1):
            theta.squeeze(0)

        for iter in range(self.U.shape[1]):
            W[:,iter] = self.U[:,iter] * self.omega_func(theta, self.theta_full[:,iter])

        X,self._s_values,_ = svd(W, full_matrices=False)

        return torch.from_numpy(X[:,:self.n_basis])

    def __call__(self, theta):
        """Returns the basis related to a specific theta.
        If multiple theta stacked are given, the output will be the stack of the corresponding spaces.
        
        Input:
                theta           (torch.Tensor)              Parameter vector to consider when computing the space.

        Output:
                (torch.Tensor) Basis related to a specific theta. If `trainable` is set to True, the basis returned refers to the ambient space, otherwise it refers to the full space.
        """
        if(theta.ndim == 1):
            out = self.compute_space(theta)
            out = torch.t(out)
            return out if self.trainable else out.matmul(self.A)
        elif(theta.ndim == 2):
            return torch.stack([self.__call__(th) for th in theta], axis = 0)
        else:
            raise RuntimeError("The input given has the wrong shape.")

    def singular_values(self):
        """Returns the singular values related to the basis computation of a specific theta. It refers to the singular values computed in the last call of `compute_space`.

        Output:
                (ndarray or None) The singular values, sorted in non-increasing order. Returns None if `compute_space` has not been called yet.
        """
        return self._s_values

    def freeze(self):
        """Needed for symmetry with the DOD class. Set `trainable` to False.
        """
        self.trainable = False

    def unfreeze(self):
        """Needed for symmetry with the DOD class. Set `trainable` to True.
        """
        self.trainable = True

def n_choice_graphs(s_values, N_max=None, n_trajectories=None, which='all', figsize=(8,8)):
    """Plots graphs needed to choose the right number of basis.
        
    Input:
            s_values            (numpy.Tensor)              Matrix containing the singular values for different values of theta of size (n_snapshot, n2), where n2 = min(n_snapshots, nA).
            N_max               (int, optional)             Number of basis to plot in all the graphs.
                                                            Defaults to None - in this case, all basis are plotted.
            n_trajectories      (int, optional)             Number of trajectories to plot in the graph type 'trajectories'.
                                                            Defaults to None - in this case, all trajectories are plotted.
            which               (string, optional)          Which graphs to show. Choices are:
                                                            - 'delta'           : graph of the the mean and minimum relative difference of the singular values.
                                                            - 'range'           : graph of the range between the minimum and maximum singular value for each number of basis.
                                                            - 'trajectories'    : graph of the trajectories of the singular values for each theta.
                                                            - 'all'             : all of the above.
                                                            Defaults to 'all'.
            figsize             (tuple of floats, optional) Width and height of the plot in inches.
                                                            Defaults to (8,8).                                              
    """    
    if(N_max is None):
        N_max = s_values.shape[1]

    if(n_trajectories is None):
        n_trajectories = s_values.shape[0]

    x = np.arange(1, N_max+1)

    plots_to_show = []
    if which in ['all', 'delta']:
        plots_to_show.append('delta')
    if which in ['all', 'range']:
        plots_to_show.append('range')
    if which in ['all', 'trajectories']:
        plots_to_show.append('trajectories')

    n_subplots = len(plots_to_show)
    if n_subplots == 0:
        raise ValueError("Invalid 'which' argument, nothing to plot.")

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)

    if n_subplots == 1:
        axes = [axes]

    for ax, plot_name in zip(axes, plots_to_show):
        if plot_name == 'delta':
            relative_diff = (s_values[:, :-1] - s_values[:, 1:])/s_values[:, :-1]
            delta_n_mean = (relative_diff).mean(axis=0)
            delta_n_min = (relative_diff).min(axis=0)

            ax.loglog(x[:-1], delta_n_mean[:N_max-1], marker="o", color='orange', label='Mean')
            ax.loglog(x[:-1], delta_n_min[:N_max-1], marker="*", color='blue', label='Minimum')
            ax.legend()
            ax.set_title('δ(n)')
            ax.grid(True)
        
        elif plot_name == 'range':
            min_n = s_values.min(axis = 0)
            max_n = s_values.max(axis = 0)

            ax.fill_between(x, min_n[:N_max], max_n[:N_max], color='blue', alpha=0.2)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('Singular Values Range')
            ax.grid(True)
        
        elif plot_name == 'trajectories':
            ax.loglog(x, s_values.T[:N_max, :n_trajectories])
            ax.set_title('Trajectories')
            ax.grid(True)

    step = max(1, N_max // 8)

    ticks = np.arange(1, N_max + 1, step)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(ticks)
    axes[-1].tick_params(axis='x', rotation=45)
    axes[-1].set_xlabel('Number of basis')

    plt.tight_layout()

    return