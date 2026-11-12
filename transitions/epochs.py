
import math
import numpy as np



def gamma_constant(P, pi):
    """Gamma = (1+max(0,lam2))/(1-max(0,lam2)); exact via symmetrized reversible P."""
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P, float)
    d = np.sqrt(pi)
    S = d[:, None] * P / d[None, :]
    lam = float(np.sort(np.linalg.eigvalsh((S + S.T) / 2))[-2])
    m = max(0.0, lam)
    return (1 + m) / (1 - m), lam


def epoch_cap(rho_bar, P, e, n_agents, alpha, delta2):
    """n_e = ceil(2 M_e/rho_min + Gamma*ell/(2 rho_min^2)), M_e = e^alpha,
    ell = log(R / (rho_min * delta_{2,e}^{1/M}))."""
    R, rmin = len(rho_bar), float(rho_bar.min()) # number of regins, smallest entry in the belief target
    G, lam = gamma_constant(P, rho_bar)  
    # gamma: (1 + max(0, lam2)) / (1 − max(0, lam2)), the mixing penalty in the Hoeffding bound (Lemma 46);
    #  large when lam is near 1
    Me = e ** alpha  # required visits to every region
    dm = (6 * delta2 / (math.pi ** 2 * e * e)) ** (1 / n_agents)
    ell = math.log(R / (rmin * dm)) #log(R / (rmin · dm)), the log term from the union bound over regions and the failure budget
    cap = math.ceil(2 * Me / rmin + G * ell / (2 * rmin ** 2)) # epoch length
    print(f"[epoch_cap] R:{R}, rmin:{rmin}, G:{G}, lam2:{lam}, Me:{Me}, dm{dm}, l_e:{ell}")
    return dict(cap=cap, Me=math.ceil(Me), rho_min=rmin, lam2=lam, Gamma=G, ell=ell)