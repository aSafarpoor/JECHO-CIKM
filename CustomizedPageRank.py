import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import diags as DIAGS

class OurPageRank:
    def __init__(
        self,
        adj_matrix,
        features=None,
        alpha=0.85,
        beta=0.5,
        use_features=False,
        personalized=False,
        seed_node=0,
    ):
        self.A = csr_matrix(adj_matrix).astype(float)
        self.features = features
        self.alpha = alpha
        self.beta = beta
        self.use_features = use_features
        self.personalized = personalized
        self.seed_node = seed_node
        self.n = self.A.shape[0]
        self.P = self._build_transition_matrix()
        self.r = self._build_restart_vector()

    def column_normalize_with_dangling(self, A):
        A = csr_matrix(A).astype(float)
        n = A.shape[0]
        col_sum = np.array(A.sum(axis=0)).flatten()
        dangling = col_sum == 0
        col_sum[dangling] = 1
        D_inv = DIAGS(1.0 / col_sum)
        P = A @ D_inv
        if np.any(dangling):
            P = P.toarray()
            P[:, dangling] = 1.0 / n
            P = csr_matrix(P)
        return P

    def _build_transition_matrix(self):
        P = self.column_normalize_with_dangling(self.A)
        if not self.use_features:
            return P

        Q = cosine_similarity(self.features)
        Q = np.maximum(Q, 0)
        col_sum = Q.sum(axis=0)
        col_sum[col_sum == 0] = 1
        Q = Q / col_sum[np.newaxis, :]
        P = P.toarray()
        P = (1 - self.beta) * Q + self.beta * P
        return csr_matrix(P)

    def _build_restart_vector(self):
        if not self.personalized:
            return np.ones(self.n) / self.n
        v = np.zeros(self.n)
        if isinstance(self.seed_node, (list, tuple, np.ndarray)):
            v[self.seed_node] = 1.0
            return v / v.sum()
        else:
            v[self.seed_node] = 1.0
            return v

    def compute_pagerank(self, max_iter=100, tol=1e-6):
        pi = np.ones(self.n) / self.n
        for _ in range(max_iter):
            pi_new = (1 - self.alpha) * self.r + self.alpha * self.P.dot(pi)
            if np.linalg.norm(pi_new - pi, 1) < tol:
                break
            pi = pi_new
        return pi