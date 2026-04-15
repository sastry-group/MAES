import numpy as np
import scipy.sparse as sp
import cvxpy as cp

from typing import Optional
from env.grid import GridMC, EMPTY, OBSTACLE, ROI, START 



def lambda2(P):
    if sp.issparse(P):
        P_dense = P.toarray()
    else:
        P_dense = np.asarray(P)


    eigvals = np.linalg.eigvals(P_dense)
    eigvals = np.real(eigvals)

    eigvals_sorted = np.sort(eigvals)[::-1]

    return eigvals_sorted[1] 

def build_transition_matrix_MH(env, phi_vec: np.ndarray | None = None, 
                               eps: float = 1e-12, 
                               compute_lambda2:bool = True) -> np.ndarray:
        if env._edge_src is None: 
              env.build_map_vec()
        phi = np.maximum(phi_vec.astype(np.float64), 0.0)
        if not np.any(phi > 0): phi[:] = 1.0

        # unnormalized target weights over states (>=0). If all zeros, make uniform.
        # self._index_state_map maps index -> (r,c)
        src, dst, deg = env._edge_src, env._edge_dst, env._deg
        qij = 1.0 / deg[src]
        qji = 1.0 / deg[dst]
        ratio = ((phi[dst] + eps) * qji) / ((phi[src] + eps) * qij)
        a = np.minimum(1.0, ratio)
        off_vals = qij * a

        n = env.nstates # valid states do not include the obstacles
        P_off = sp.coo_matrix((off_vals, (src, dst)), shape=(n, n)).tocsr()
        row_sums = np.asarray(P_off.sum(axis=1)).ravel()
        diag = 1.0 - row_sums
        P = P_off.tolil(); 
        P.setdiag(diag)
        second_largest_eig = None
        if compute_lambda2:
            second_largest_eig = lambda2(P)
        # second_largest_eig = 0
        # print(f"Second largest eigenvalue of P: {second_largest_eig:.6f}")
        return P.tocsr().astype(np.float32), second_largest_eig
    


def outer_expr(a, b):
      a = cp.Expression.cast_to_const(a); b = cp.Expression.cast_to_const(b)
      a = cp.reshape(a, (a.size, 1)); b = cp.reshape(b, (1, b.size))
      return a @ b


def build_transition_matrix_REMC(
    env,
    phi_vec: np.ndarray | None = None,
    eps: float = 1e-5,
    solver: str = "SCS",
    verbose: bool = False,
    max_iters: int = 5_000,
) -> sp.csr_matrix:
    """
    Build a Rapidly Ergodic Markov Chain (REMC) transition matrix P for the grid env
    that targets the stationary distribution proportional to phi_vec (or env.true_phi),
    respecting the graph sparsity given by env._edge_src/_edge_dst.

    Returns: scipy.sparse.csr_matrix of shape (n, n), dtype float32
    """
    if env._edge_src is None or env._edge_dst is None or env._deg is None:
        if env._index_state_map is None:
            env._indexing()
        env._precompute_graph()

    n = env.nstates

    if phi_vec is None:
        phi_vec = env.build_map_vec(env.true_phi)
    phi = np.maximum(phi_vec.astype(np.float64), 0.0)
    if not np.any(phi > 0):
        phi[:] = 1.0

    # Smooth to avoid zeros, then normalize to a prob. vector w
    w = phi + eps
    w = w / w.sum()
    # q and diagonal scalings
    q = np.sqrt(w)
    Q = np.diag(q)
    Qinv = np.diag(1.0 / q)


    P = cp.Variable((n, n))
    ones = np.ones(n)
    constraints = [
        P >= 0,
        P @ ones == ones,     # each row sums to 1
        P.T @ w == w,         # stationarity under row convention
    ]

    # Sparsity: allow only existing directed edges and self-loops
    # Build a set of allowed (i,j): i <- j is allowed if edge j->i exists (column j -> row i)
    allowed = set(zip(env._edge_src.tolist(), env._edge_dst.tolist()))
    # Always allow self-loops
    for i in range(n):
        allowed.add((i, i))

    # Enforce zeros where not allowed
    # (CVXPY: add equality constraints P[i,j]==0 for disallowed entries)
    for i in range(n):
        for j in range(n):
            if (i, j) not in allowed:
                constraints.append(P[i, j] == 0)

    # M = cp.transpose(P)
    # tilde = Qinv @ M @ Q - 2.0 * outer_expr(q, q)
    # sym_tilde = 0.5 * (tilde + tilde.T)
    S = Q @ P @ Qinv - 2.0 * outer_expr(q, q)
    sym_S = 0.5 * (S + S.T)
    objective = cp.Minimize(cp.lambda_max(sym_S))

    prob = cp.Problem(objective, constraints)


    opts = {}
    if solver.upper() == "SCS":
        opts = dict(max_iters=max_iters, verbose=verbose)
        prob.solve(solver=cp.SCS, **opts)
    else:
        prob.solve(solver=getattr(cp, solver.upper()), verbose=verbose)

    if P.value is None:
        raise RuntimeError(f"REMC optimization failed (status={prob.status}).")

    P_dense = np.array(P.value, dtype=np.float64)

    P_dense[P_dense < 0] = 0.0
    for i in range(n):
        for j in range(n):
            if (i, j) not in allowed:
                P_dense[i, j] = 0.0

    row_sums = P_dense.sum(axis=1, keepdims=True)
    zero_rows = np.where(row_sums < eps)[0]
    if zero_rows.size > 0:
        for i in zero_rows:
            P_dense[i, :] = 0.0
            P_dense[i, i] = 1.0
        row_sums = P_dense.sum(axis=1, keepdims=True)
    P_dense /= row_sums

    second_largest_eig = lambda2(P_dense)
    print(f"Second largest eigenvalue of P: {second_largest_eig:.6f}")
    return sp.csr_matrix(P_dense.astype(np.float32)), second_largest_eig


# def FMMC(self, weight, transform=True):
#     w =weight/sum(weight)
#     n = self.num_regions
#     P= cp.Variable((n,n))


#     q=np.sqrt(w)
#     Q=np.diag(q)
#     constrains=[
#                 # P@w==w,
#                 P@np.diag(w) == np.diag(w)@P.T,
#                 cp.min(P)>=0,
#                 np.ones(n)@P==np.ones(n)]
    
#     for i in range(n):
#         for j in range(n):
#             if ([i,j] not in self.edges):
#                 constrains.append(P[j,i]==0)

#     if transform:
#         prob = cp.Problem(cp.Minimize(cp.norm(np.linalg.inv(Q)@P@(Q)-outer(q,q),2)),
#                             constrains)

#     else:
#             prob = cp.Problem(cp.Minimize(cp.norm(P-outer(w,np.ones(n)),2)),
#                             constrains)
    

#     prob.solve()
#     P=P.value

#     P[P<0]=0
#     for i in range(n):
#         for j in range(n):
#             if ([i,j] not in self.edges):
#                 P[j,i]=0
#     P=P/sum(P, 0)

#     return P