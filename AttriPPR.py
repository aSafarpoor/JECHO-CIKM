""" Borroed some idea from:
Unsupervised Ranking using Graph Structures and Node Attributes
Chin-Chi Hsu, Yi-An Lai, Wen-Hao Chen, Ming-Han Feng, and Shou-De Lin
Web Search and Data Mining (WSDM), 2017"""
import numpy as np
from scipy.sparse import csr_matrix


def _standardize(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd

def compute_r(features): # The modified version of AttriRank
    if features is None:
        raise ValueError("features=None: cannot compute r from attributes")
    X = _standardize(features)                     # shape (N, K)
    N, K = X.shape
    gamma = 1.0 / K

    # w_i = exp(-gamma ||x_i||^2)
    norms2 = np.einsum("ij,ij->i", X, X)          # (N,)
    w = np.exp(-gamma * norms2)                    # (N,)

    # a = sum_j w_j
    a = w.sum()

    # b = 2*gamma * sum_j w_j x_j  -> shape (K,)
    b = 2.0 * gamma * np.einsum("i,ij->j", w, X)

    # C = 2*gamma^2 * sum_j w_j x_j x_j^T -> shape (K,K)
    C = 2.0 * (gamma**2) * np.einsum("i,ij,ik->jk", w, X, X)

    # hat{r}_i = w_i * (a + x_i^T b + x_i^T C x_i)
    xb = X @ b                                   # (N,)
    xCx = np.einsum("ij,jk,ik->i", X, C, X)      # (N,)
    r_hat = w * (a + xb + xCx)                   # (N,)

    # Normalize
    total = r_hat.sum()
    if total <= 0 or not np.isfinite(total):
        # fallback to uniform if numerical issues
        return np.full(N, 1.0 / N)
    return r_hat / total


def build_P_fixed(num_nodes, edges):
    edges = np.asarray(edges, dtype=int)
    if edges.size == 0:
        raise ValueError("Empty edge list")
    
    src = edges[:, 0]
    dst = edges[:, 1]
    
    # Calculate out-degrees efficiently
    out_degrees = np.bincount(src, minlength=num_nodes)
    dangling = np.where(out_degrees == 0)[0]
    
    # Create normalized weights directly (this is the key fix)
    weights = np.where(out_degrees[src] > 0, 
                      1.0 / out_degrees[src], 
                      0.0)
    
    # Build matrix with pre-normalized weights
    P = csr_matrix((weights, (dst, src)), shape=(num_nodes, num_nodes))
    
    return P, dangling

def seeds_to_v(num_nodes, seeds):
    """
    Make a probability vector v from seed(s).
    seeds can be an int or an iterable of ints.
    """
    v = np.zeros(num_nodes, dtype=float)
    if isinstance(seeds, (int, np.integer)):
        idx = [int(seeds)]
    else:
        idx = list(seeds)
    if len(idx) == 0:
        raise ValueError("Seed set is empty")
    for s in idx:
        if s < 0 or s >= num_nodes:
            raise ValueError(f"Seed index {s} out of range")
        v[s] = 1.0
    v /= v.sum()
    return v

class AttriPPR:
    def __init__(self,num_nodes,
                edges,
                features=None,
                a=0.15,
                b=0.5,
                tol=1e-10,
                max_iter=100000,
                init = "v"):

        self.num_nodes = num_nodes
        self.edges = edges
        self.features=features
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.P, self.dangling = build_P_fixed(self.num_nodes, self.edges)

        if features is None:
            self.r = np.full(self.num_nodes, 1.0 / self.num_nodes, dtype=float)
        else:
            self.r = compute_r(np.asarray(features, dtype=float))


    def compute_pi(
            self,
            seeds = -1,
            
    ):
        num_nodes = self.num_nodes
        features = self.features
        a = self.a
        b = self.b
        tol = self.tol
        max_iter = self.max_iter

        if seeds == -1:
            seeds = list(range(len(features)))

        """
        Compute stationary pi for:
            pi^{t+1} = a v + (1-a) [ b r + (1-b) P pi^{t} ]
        with dangling mass routed to v on the P branch.

        Args:
            num_nodes: number of nodes (0..num_nodes-1)
            edges: list/array of (src, dst) directed edges
            seeds: int or iterable of ints (teleport prior v is 1 on seeds, 0 otherwise, then normalized)
            features: None, (N,), or (N,K). If None, r defaults to uniform.
            a in (0,1), b in [0,1]
            tol: L2 convergence tolerance
            max_iter: iteration cap
            init: "uniform" or "v" initial vector

        Returns:
            pi: numpy array shape (N,)
        """
        if not (0 < a < 1):
            raise ValueError("a must be in (0,1)")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0,1]")

        
        # Build components
        P, dangling = self.P, self.dangling
        r = self.r
        v = seeds_to_v(num_nodes, seeds)

        # r from features; fall back to uniform if features is None
        

        # Initialization
        if self.init == "uniform":
            pi = np.full(num_nodes, 1.0 / num_nodes, dtype=float)
        elif self.init == "v":
            pi = seeds_to_v(num_nodes, seeds)
        else:
            raise ValueError(f"Unknown init: {self.init}")
        

        # Power iteration
        for _ in range(int(max_iter)):
            leak = pi[dangling].sum()               # sum of mass at dangling columns
            Ppi = P.dot(pi) + leak * v              # route dangling mass to v
            mix = b * r + (1.0 - b) * Ppi
            new_pi = a * v + (1.0 - a) * mix

            # Convergence check
            if np.linalg.norm(new_pi - pi) < tol:
                pi = new_pi
                break
            pi = new_pi

        # Normalize for safety
        s = pi.sum()
        if s <= 0 or not np.isfinite(s):
            # fallback to v if something went wrong numerically
            return v
        return pi / s
