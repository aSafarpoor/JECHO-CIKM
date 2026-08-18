from typing import Dict, Optional, Tuple
import igraph as ig
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
class FJLargeScale:
    def __init__(
        self,
        G: ig.Graph,
        dtype=np.float32,
        chunk_size: int = 5_000_000,
        use_weights: bool = True,
        symmetrize_directed: bool = True,
    ):
       
        self.G = G
        self.n = int(G.vcount())
        self.dtype = dtype
        self.chunk_size = int(chunk_size)
        self.symmetrize = (G.is_directed()) and symmetrize_directed
        E = np.asarray(G.get_edgelist(), dtype=np.int64)
        if E.size == 0:
            raise ValueError("Graph has no edges.")
        self.u = E[:, 0].astype(np.int32, copy=False)
        self.v = E[:, 1].astype(np.int32, copy=False)
        del E  # free
        m = self.u.size
        if use_weights and "weight" in G.es.attributes():
            w = np.asarray(G.es["weight"], dtype=self.dtype)
            if w.size != m:
                raise ValueError("Edge weight array length mismatch.")
            self.w = w
        else:
            self.w = np.ones(m, dtype=self.dtype)
        n = self.n
        deg = np.bincount(self.u, weights=self.w.astype(np.float64), minlength=n)
        if self.symmetrize or not G.is_directed():
            deg += np.bincount(self.v, weights=self.w.astype(np.float64), minlength=n)
        else:
            deg += np.bincount(self.v, weights=self.w.astype(np.float64), minlength=n)
        self.deg = deg.astype(self.dtype, copy=False)
    def _apply_Ax(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute y = A x for the operator we chose (undirected or symmetrized).
        Uses chunked np.add.at to avoid building CSR.
        """
        n = self.n
        if out is None:
            y = np.zeros(n, dtype=self.dtype)
        else:
            y = out
            y.fill(0)
        u, v, w = self.u, self.v, self.w
        m = u.size
        cs = self.chunk_size
        # For undirected or symmetrized A: add in both directions.
        for start in range(0, m, cs):
            end = min(start + cs, m)
            uu = u[start:end]
            vv = v[start:end]
            ww = w[start:end]
            np.add.at(y, uu, ww * x[vv])
            np.add.at(y, vv, ww * x[uu])
        return y
    def _apply_Mx(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        y = (I + L) x = x + D x - A x
        """
        if out is None:
            y = x.astype(self.dtype, copy=True)
        else:
            y = out
            np.copyto(y, x, casting="no")
        # y <- x + D x
        y += self.deg * x
        # y <- y - A x
        Ax = self._apply_Ax(x)
        y -= Ax
        return y
    def solve(
        self,
        s: np.ndarray,
        tol: float = 1e-4,
        maxiter: int = 2000,
        x0: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        
        if s.shape[0] != self.n:
            raise ValueError("s has wrong length.")
        s = s.astype(self.dtype, copy=False)
   
        def matvec(x):
            x = np.asarray(x, dtype=self.dtype, order="C")
            return self._apply_Mx(x)
        Aop = LinearOperator(
            shape=(self.n, self.n),
            matvec=matvec,
            dtype=self.dtype,
        )
        inv_diag = 1.0 / (1.0 + self.deg)
        def prec_mv(x):
            return inv_diag * x
        Mop = LinearOperator(
            shape=(self.n, self.n),
            matvec=prec_mv,
            dtype=self.dtype,
        )
        residuals = []
        def cb(rk):
            if isinstance(rk, np.ndarray):
                res = float(np.linalg.norm(rk) / (np.linalg.norm(s) + 1e-30))
                residuals.append(res)
                if verbose and len(residuals) % 10 == 0:
                    print(f"[CG] iter={len(residuals):5d}  rel_resid={res:.3e}")
        if x0 is None:
            x0 = s 
#        x, info_code = cg(Aop, s, x0=x0, M=Mop, tol=tol, maxiter=maxiter, callback=cb)
        x, info_code = cg(
            Aop, s,
            x0=x0,
            M=Mop,
            maxiter=maxiter,
            callback=cb
        )

        final_resid = None
        if residuals:
            final_resid = residuals[-1]
        elif info_code == 0:
            r = s - Aop @ x
            final_resid = float(np.linalg.norm(r) / (np.linalg.norm(s) + 1e-30))
        if verbose:
            if info_code == 0:
                print(f"[CG] Converged. final_rel_resid={final_resid:.3e}")
            elif info_code > 0:
                print(f"[CG] Reached maxiter={maxiter}. final_rel_resid={final_resid:.3e}")
            else:
                print("[CG] Breakdown or numerical issue.")
        return x.astype(self.dtype, copy=False), {"niter": max(0, len(residuals)), "final_resid": final_resid, "info_code": info_code}
def opinions_dict_to_vector(n: int, opinions: Dict[int, float], dtype=np.float32) -> np.ndarray:
    s = np.zeros(n, dtype=dtype)
    for vid, val in opinions.items():
        if 0 <= vid < n:
            s[vid] = val
        else:
            raise ValueError(f"Vertex id {vid} out of range 0..{n-1}")
    return s
