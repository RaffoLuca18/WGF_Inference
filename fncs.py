####################################################################################################
####################################################################################################
#                                                                                                  #
# importing the libraries                                                                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



import numpy as np
import matplotlib.pyplot as plt



####################################################################################################
####################################################################################################
#                                                                                                  #
# score matching on gaussian models with KNOWN zero mean                                           #
#                                                                                                  #
# assumption: data are centered (mean zero known).                                                 #
# estimator: covariance = (X^T X) / n  (+ tiny ridge for stability).                               #
#                                                                                                  #
####################################################################################################
####################################################################################################



def score_matching_cov_zero_mean(X: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """
    inputs
    ------
    X     : (n,d) centered data (mean zero is assumed known)
    ridge : small diagonal shift to ensure SPD numerically

    output
    ------
    Sigma_hat : (d,d) covariance estimate
    """


    X = np.asarray(X, float)
    assert X.ndim == 2, "X must be (n,d)"
    n, d = X.shape

    # soft recentring if user forgot to center
    m = X.mean(axis=0)
    if np.linalg.norm(m) > 1e-8:
        X = X - m

    Sigma_hat = (X.T @ X) / max(n, 1)
    if ridge > 0:
        Sigma_hat = Sigma_hat + ridge * np.eye(d)

    return Sigma_hat



####################################################################################################
####################################################################################################
#                                                                                                  #
# kernels (used for the wasserstein-gradient MMD objectives)                                       #
#                                                                                                  #
####################################################################################################
####################################################################################################



def kernel_linear(X: np.ndarray, Y: np.ndarray):
    """k(x,y) = x^T y"""


    X = np.asarray(X, float); Y = np.asarray(Y, float)

    return X @ Y.T



####################################################################################################



def kernel_poly(X: np.ndarray, Y: np.ndarray, degree: int = 2, c: float = 1.0):
    """k(x,y) = (x^T y + c)^degree"""


    X = np.asarray(X, float); Y = np.asarray(Y, float)
    assert degree >= 1 and int(degree) == degree, "degree must be integer >= 1"

    return (X @ Y.T + float(c))**int(degree)



####################################################################################################



def kernel_rbf(X: np.ndarray, Y: np.ndarray, sigma: float):
    """k(x,y) = exp( -||x-y||^2 / (2 sigma^2) )"""


    X = np.asarray(X, float); Y = np.asarray(Y, float)
    assert sigma > 0.0, "sigma must be positive"
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)

    return np.exp(- D2 / (2.0 * sigma**2))



####################################################################################################
####################################################################################################
#                                                                                                  #
# utilities: sampling from zero-mean gaussian & SPD projection                                     #
#                                                                                                  #
####################################################################################################
####################################################################################################



def sample_zero_mean_gaussian(Sigma: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """
    draw n samples from N(0, Sigma)
    """


    Sigma = np.asarray(Sigma, float)
    d = Sigma.shape[0]
    rng = np.random.default_rng(seed)

    return rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)



####################################################################################################



def project_to_spd(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    symmetric projection + eigenvalue floor eps
    """


    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w_clipped = np.maximum(w, eps)

    return (V * w_clipped) @ V.T



####################################################################################################
####################################################################################################
#                                                                                                  #
# wasserstein-gradient MMD losses for ZERO-MEAN gaussian model                                     #
#                                                                                                  #
# we follow the formulas derived:                                                                  #
#   - linear kernel: L = 4 || E[Z] - E[Y] ||^2        -> with zero-mean model, becomes 4||E[Z]||^2 #
#   - polynomial:    L = 4 p^2 E_x || E_Z[ Z (x^T Z + c)^{p-1} ] - E_Y[ Y (x^T Y + c)^{p-1} ] ||^2 #
#   - rbf:           L = (4/sigma^4) E_x || E_Z[(Z-x)k(x,Z)] - E_Y[(Y-x)k(x,Y)] ||^2               #
# expectations over model (Y) are approximated by sampling Y ~ N(0, Sigma).                        #
#                                                                                                  #
####################################################################################################
####################################################################################################



def loss_wgrad_mmd_linear(X: np.ndarray,
                          Sigma: np.ndarray,
                          n_model: int = 2048,
                          seed: int | None = None) -> float:
    """
    with known zero-mean model, the linear-kernel objective reduces to 4 || E[X] ||^2.
    we keep a symmetric implementation that also samples from the model for completeness.
    """


    X = np.asarray(X, float)
    mX = X.mean(axis=0)
    # model mean is exactly zero, so no sampling needed

    return float(4.0 * np.dot(mX, mX))



####################################################################################################



def loss_wgrad_mmd_poly(X: np.ndarray,
                        Sigma: np.ndarray,
                        degree: int = 2,
                        c: float = 1.0,
                        n_model: int = 4096,
                        seed: int | None = None) -> float:
    """
    polynomial kernel loss via sampling from N(0, Sigma)
    """


    X = np.asarray(X, float); Sigma = np.asarray(Sigma, float)
    n, d = X.shape
    p = int(degree)
    assert p >= 1

    Y = sample_zero_mean_gaussian(Sigma, n_model, seed=seed)

    # precompute inner products
    XTZ = X @ X.T   # NOT needed; we need x^T z with z from X's distribution
    # we need E_Z[...] with Z ~ data, so we estimate using X itself (V-statistic)
    # A(x) = E_Z [ Z (x^T Z + c)^{p-1} ]  ≈ (1/n) sum_j X_j * (x^T X_j + c)^{p-1}
    # B(x) = E_Y [ Y (x^T Y + c)^{p-1} ]  ≈ (1/m) sum_m Y_m * (x^T Y_m + c)^{p-1}

    # A(x) for all x in X
    GxX = X @ X.T                      # (n,n)
    weights_A = (GxX + c)**(p-1)       # (n,n)
    A = (weights_A @ X) / n            # (n,d); since row i: sum_j weights_A[i,j] * X_j

    # B(x) for all x in X
    GxY = X @ Y.T                      # (n,m)
    weights_B = (GxY + c)**(p-1)       # (n,m)
    B = (weights_B @ Y) / n_model      # (n,d)

    diff = A - B                       # (n,d)
    val = 4.0 * (p**2) * np.mean(np.sum(diff*diff, axis=1))

    return float(val)



####################################################################################################



def loss_wgrad_mmd_rbf(X: np.ndarray,
                       Sigma: np.ndarray,
                       sigma: float = 1.0,
                       n_model: int = 4096,
                       seed: int | None = None) -> float:
    """
    rbf kernel loss via sampling from N(0, Sigma)
    L = (4/sigma^4) E_x || E_Z[(Z-x)k(x,Z)] - E_Y[(Y-x)k(x,Y)] ||^2
    """


    X = np.asarray(X, float); Sigma = np.asarray(Sigma, float)
    n, d = X.shape
    assert sigma > 0.0

    # empirical E_Z over data itself (V-statistic)
    K_xX = kernel_rbf(X, X, sigma)            # (n,n)
    A = ((X[np.newaxis, :, :] - X[:, np.newaxis, :]) * K_xX[:, :, None]).mean(axis=1)  # (n,d) mean over Z

    # model term: sample Y ~ N(0, Sigma)
    Y = sample_zero_mean_gaussian(Sigma, n_model, seed=seed)   # (m,d)
    K_xY = kernel_rbf(X, Y, sigma)            # (n,m)
    B = ((Y[np.newaxis, :, :] - X[:, np.newaxis, :]) * K_xY[:, :, None]).mean(axis=1)  # (n,d) mean over Y

    diff = A - B
    val = (4.0 / (sigma**4)) * np.mean(np.sum(diff*diff, axis=1))

    return float(val)



####################################################################################################
####################################################################################################
#                                                                                                  #
# simple covariance fitting by MMD (shrinkage along identity)                                      #
#                                                                                                  #
# model: Sigma(alpha) = (1-alpha) * S + alpha * tau * I,  alpha in [0,1], tau = trace(S)/d         #
# choose alpha that minimizes the chosen wasserstein-gradient MMD loss.                            #
#                                                                                                  #
####################################################################################################
####################################################################################################


def fit_cov_by_mmd_shrinkage(X: np.ndarray,
                             kernel: str = "rbf",
                             n_model: int = 4096,
                             seed: int = 0,
                             # rbf params
                             sigma_rbf: float = 1.0,
                             # poly params
                             degree: int = 2,
                             c: float = 1.0,
                             # grid
                             alphas: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """
    returns (Sigma_rec, alpha_best, best_value)
    """


    X = np.asarray(X, float)
    n, d = X.shape

    # base covariance from score matching (zero-mean)
    S = score_matching_cov_zero_mean(X, ridge=1e-12)
    tau = np.trace(S) / d

    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 31)

    best = np.inf
    best_alpha = 0.0
    best_Sigma = S.copy()

    for a in alphas:
        Sigma_a = (1.0 - a) * S + a * tau * np.eye(d)
        Sigma_a = project_to_spd(Sigma_a, eps=1e-10)

        if kernel == "linear":
            val = loss_wgrad_mmd_linear(X, Sigma_a, n_model=n_model, seed=seed)
        elif kernel == "poly":
            val = loss_wgrad_mmd_poly(X, Sigma_a, degree=degree, c=c,
                                      n_model=n_model, seed=seed)
        elif kernel == "rbf":
            val = loss_wgrad_mmd_rbf(X, Sigma_a, sigma=sigma_rbf,
                                     n_model=n_model, seed=seed)
        else:
            raise ValueError("kernel must be in {'linear','poly','rbf'}")

        if val < best:
            best = val
            best_alpha = float(a)
            best_Sigma = Sigma_a

    return best_Sigma, best_alpha, best



####################################################################################################
####################################################################################################
#                                                                                                  #
# heatmaps: true covariance vs reconstructed covariance                                            #
#                                                                                                  #
####################################################################################################
####################################################################################################



def plot_cov_heatmaps(Sigma_true: np.ndarray,
                      Sigma_rec:  np.ndarray,
                      vmax: float | None = None,
                      title_true: str = "true covariance",
                      title_rec:  str = "reconstructed covariance"):
    """
    display two separate heatmaps.
    """


    Sigma_true = np.asarray(Sigma_true, float)
    Sigma_rec  = np.asarray(Sigma_rec,  float)

    # automatic symmetric color scale
    if vmax is None:
        vmax = max(np.max(np.abs(Sigma_true)), np.max(np.abs(Sigma_rec)))

    # true covariance
    plt.figure(figsize=(4.6, 4.3))
    plt.imshow(Sigma_true, cmap="viridis", aspect="equal",
               vmin=-vmax, vmax=vmax, origin="upper")  # origin='upper' keeps diag top-left→bottom-right
    plt.colorbar()
    plt.title(title_true)
    plt.xlabel("index j")
    plt.ylabel("index i")
    plt.tight_layout()
    plt.show()

    # reconstructed covariance
    plt.figure(figsize=(4.6, 4.3))
    plt.imshow(Sigma_rec, cmap="viridis", aspect="equal",
               vmin=-vmax, vmax=vmax, origin="upper")
    plt.colorbar()
    plt.title(title_rec)
    plt.xlabel("index j")
    plt.ylabel("index i")
    plt.tight_layout()
    plt.show()
