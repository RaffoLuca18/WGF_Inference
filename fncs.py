####################################################################################################
####################################################################################################
#                                                                                                  #
# importing the libraries                                                                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, random
from jax.scipy.linalg import solve_triangular
import numpy as np
import matplotlib.pyplot as plt
import optax



####################################################################################################
####################################################################################################
#                                                                                                  #
# parameterization: precision Omega = L L^T  with L lower-triangular, diag(L)>0 via exp()          #
#                                                                                                  #
####################################################################################################
####################################################################################################



def theta_size(d: int) -> int:


    return d + d*(d-1)//2


####################################################################################################



def theta_to_L(theta: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    theta = [logdiag(L) (length d), strict-lower entries row-wise]
    """


    theta = jnp.asarray(theta)
    diag = jnp.exp(theta[:d])
    L = jnp.zeros((d, d))
    L = L.at[jnp.arange(d), jnp.arange(d)].set(diag)

    offs = theta[d:]
    k = 0
    for i in range(1, d):
        for j in range(i):
            L = L.at[i, j].set(offs[k])
            k += 1

    return L



####################################################################################################



def L_to_theta(L: jnp.ndarray) -> jnp.ndarray:


    d = L.shape[0]
    parts = [jnp.log(jnp.diag(L))]
    offs = []
    for i in range(1, d):
        for j in range(i):
            offs.append(L[i, j])
    parts.append(jnp.array(offs))

    return jnp.concatenate(parts, axis=0)


####################################################################################################



def omega_from_theta(theta: jnp.ndarray, d: int) -> jnp.ndarray:


    L = theta_to_L(theta, d)

    return L @ L.T                         # SPD precision



####################################################################################################



def sigma_from_theta(theta: jnp.ndarray, d: int) -> jnp.ndarray:


    # Sigma = Omega^{-1}; compute via solves rather than explicit inverse when sampling
    # (for plotting/metrics we may need the explicit matrix)
    # Here we form it explicitly using Cholesky-based inverse: inv(LL^T) = L^{-T} L^{-1}
    L = theta_to_L(theta, d)
    I = jnp.eye(d)
    Linv = jax.scipy.linalg.solve_triangular(L, I, lower=True)       # L * X = I
    Sigma = Linv.T @ Linv

    return Sigma



####################################################################################################
####################################################################################################
#                                                                                                  #
# score matching (gaussian, known zero mean)                                                       #
#                                                                                                  #
# assumption: data are centered; estimator = (X^T X)/n (tiny ridge optional).                      #
#                                                                                                  #
####################################################################################################
####################################################################################################



def score_matching_cov_zero_mean(
    X: jnp.ndarray,
    ridge: float = 1e-8,
    lr: float = 1e-1,
    steps: int = 2000,
    seed: int = 0,
) -> jnp.ndarray:
    """
    inputs
    ------
    X     : (n, d) jnp array, data (assumed/enforced mean zero)
    ridge : small diagonal shift / tikhonov
    lr    : optax learning rate
    steps : number of optimization steps
    seed  : rng seed (kept for api symmetry)

    output
    ------
    Sigma_hat : (d, d) covariance estimate via score matching
    """

    # center data
    X = X - jnp.mean(X, axis=0)
    n, d = X.shape

    # empirical second moment (covariance, non-bias corrected)
    S = (X.T @ X) / jnp.maximum(n, 1)
    S = S + ridge * jnp.eye(d, dtype=X.dtype)  # stabilize

    # parameter: theta encodes L (lower-triangular with exp diag); omega = L L^T
    theta0 = jnp.zeros(theta_size(d), dtype=X.dtype)

    # hyvärinen score objective for zero-mean gaussian:
    # J(theta) = 0.5 * tr(Omega^2 S) - tr(Omega) + (ridge/2) * ||Omega||_F^2
    def loss_fn(theta):
        Omega = omega_from_theta(theta, d)
        fit = 0.5 * jnp.trace(Omega @ Omega @ S) - jnp.trace(Omega)
        reg = 0.5 * ridge * jnp.sum(Omega * Omega)
        return fit + reg

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # optimizer
    opt = optax.adam(lr)
    opt_state = opt.init(theta0)

    @jax.jit
    def step(carry, _):
        theta, opt_state = carry
        loss, g = loss_and_grad(theta)
        updates, opt_state = opt.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return (theta, opt_state), loss

    # run optimization
    (theta_fin, _), _ = jax.lax.scan(step, (theta0, opt_state), None, length=steps)

    # map back to covariance
    Sigma_hat = sigma_from_theta(theta_fin, d)
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)  # symmetrize (numerical)
    return Sigma_hat




####################################################################################################
####################################################################################################
#                                                                                                  #
# sampling Y ~ N(0, Sigma_theta) with reparameterization (differentiable)                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



def sample_model_from_theta(key: jax.Array,
                            theta: jnp.ndarray,
                            d: int,
                            n_samples: int) -> jnp.ndarray:
    """
    eps ~ N(0,I); solve L^T u = eps  and then  L y = u  ->  y ~ N(0, (LL^T)^{-1})
    """


    L = theta_to_L(theta, d)                              # (d,d)
    eps = random.normal(key, shape=(n_samples, d))        # (m,d)
    # solve L^T U^T = eps^T  => U = solve(L^T, eps^T).T
    U = solve_triangular(L.T, eps.T, lower=False).T
    Y = solve_triangular(L, U.T, lower=True).T            # solve L Y^T = U^T

    return Y                                              # (m,d)



####################################################################################################
####################################################################################################
#                                                                                                  #
# kernels (jnp)                                                                                    #
#                                                                                                  #
####################################################################################################
####################################################################################################



def kernel_linear(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:


    return X @ Y.T



####################################################################################################



def kernel_poly(X: jnp.ndarray, Y: jnp.ndarray, degree: int = 2, c: float = 1.0) -> jnp.ndarray:


    return (X @ Y.T + float(c))**int(degree)



####################################################################################################



def kernel_rbf(X: jnp.ndarray, Y: jnp.ndarray, sigma: float) -> jnp.ndarray:


    X2 = jnp.sum(X*X, axis=1, keepdims=True)
    Y2 = jnp.sum(Y*Y, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)

    return jnp.exp(- D2 / (2.0 * (sigma**2)))



####################################################################################################
####################################################################################################
#                                                                                                  #
# wasserstein-gradient MMD losses (zero-mean gaussian model)                                       #
#                                                                                                  #
# linear:   L = 4 || E[X] - E[Y] ||^2; with known zero-mean model, this is ~4||mean(X)||^2.        #
# poly(p):  L = 4 p^2 E_x || E_Z[ Z (x^T Z + c)^{p-1} ] - E_Y[ Y (x^T Y + c)^{p-1} ] ||^2          #
# rbf:      L = (4/s^4) E_x || E_Z[(Z-x)k(x,Z)] - E_Y[(Y-x)k(x,Y)] ||^2                            #
#                                                                                                  #
####################################################################################################
####################################################################################################



def loss_wgrad_mmd_linear_theta(X: jnp.ndarray,
                                theta: jnp.ndarray,
                                d: int,
                                key: jax.Array | None = None,
                                n_model: int = 0) -> jnp.ndarray:
    

    mX = jnp.mean(X, axis=0)

    return 4.0 * (mX @ mX)   # does not depend on theta if mean=0



####################################################################################################



def loss_wgrad_mmd_poly_theta(X: jnp.ndarray,
                              theta: jnp.ndarray,
                              d: int,
                              key: jax.Array,
                              n_model: int = 4096,
                              degree: int = 2,
                              c: float = 1.0) -> jnp.ndarray:
    

    n = X.shape[0]
    p = int(degree)

    # A(x): E_Z over data (V-statistic)
    GxX = X @ X.T                                    # (n,n)
    weights_A = (GxX + c)**(p-1)                     # (n,n)
    A = (weights_A @ X) / n                          # (n,d)

    # B(x): E_Y over model samples Y
    key_y = random.split(key, 1)[0]
    Y = sample_model_from_theta(key_y, theta, d, n_model)     # (m,d)
    GxY = X @ Y.T                                    # (n,m)
    weights_B = (GxY + c)**(p-1)                     # (n,m)
    B = (weights_B @ Y) / n_model                    # (n,d)

    diff = A - B

    return 4.0 * (p**2) * jnp.mean(jnp.sum(diff*diff, axis=1))



####################################################################################################



def loss_wgrad_mmd_rbf_theta(X: jnp.ndarray,
                             theta: jnp.ndarray,
                             d: int,
                             key: jax.Array,
                             n_model: int = 4096,
                             sigma: float = 1.0) -> jnp.ndarray:
    

    n = X.shape[0]

    # A(x) = E_Z[(Z-x)k(x,Z)], Z=data
    K_xX = kernel_rbf(X, X, sigma)                               # (n,n)
    # broadcast: (n,1,d) - (n,n,d) using X[None,:,:]? we want (n,n,d): (Z - x)
    Z_minus_x = X[None, :, :] - X[:, None, :]                    # (n,n,d)
    A = jnp.mean(Z_minus_x * K_xX[:, :, None], axis=1)           # (n,d)

    # B(x) = E_Y[(Y-x)k(x,Y)], Y=model
    key_y = random.split(key, 1)[0]
    Y = sample_model_from_theta(key_y, theta, d, n_model)        # (m,d)
    K_xY = kernel_rbf(X, Y, sigma)                               # (n,m)
    Y_minus_x = Y[None, :, :] - X[:, None, :]                    # (n,m,d)
    B = jnp.mean(Y_minus_x * K_xY[:, :, None], axis=1)           # (n,d)

    diff = A - B

    return (4.0 / (sigma**4)) * jnp.mean(jnp.sum(diff*diff, axis=1))



####################################################################################################
####################################################################################################
#                                                                                                  #
# optimizer: simple Adam in JAX                                                                    #
#                                                                                                  #
####################################################################################################
####################################################################################################



def adam_update(grad_t, m, v, t, theta, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):


    m = b1 * m + (1 - b1) * grad_t
    v = b2 * v + (1 - b2) * (grad_t * grad_t)
    mhat = m / (1 - b1**t)
    vhat = v / (1 - b2**t)
    theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)

    return theta, m, v



####################################################################################################
####################################################################################################
#                                                                                                  #
# fitting wrapper: gradient descent on theta with JAX autodiff                                     #
#                                                                                                  #
####################################################################################################
####################################################################################################



def fit_cov_by_wgrad_mmd_jax(X_np: np.ndarray,
                             kernel: str = "rbf",
                             # kernel params
                             sigma_rbf: float = 1.0,
                             degree: int = 2, c: float = 1.0,
                             # sampling
                             n_model: int = 4096,
                             seed_samples: int = 7,
                             # optimizer
                             iters: int = 300,
                             lr: float = 5e-2,
                             b1: float = 0.9, b2: float = 0.999,
                             # init
                             init_from_sm: bool = True):
    """
    inputs
    ------
    X_np : numpy array (n,d), converted to jnp inside
    returns
    -------
    Sigma_hat (np), theta_star (np), hist (list of floats)
    """
    X = jnp.asarray(X_np, dtype=jnp.float32)
    n, d = X.shape

    # init theta
    if init_from_sm:
        S0 = score_matching_cov_zero_mean(X, ridge=1e-6)
        # precision init: inverse
        # (use cholesky inverse for stability)
        Ls = jnp.linalg.cholesky(S0 + 1e-6 * jnp.eye(d))
        # Omega0 = (Ls^{-T})(Ls^{-1})
        Linv = solve_triangular(Ls, jnp.eye(d), lower=True)
        Omega0 = Linv.T @ Linv
        L0 = jnp.linalg.cholesky(Omega0)
        theta0 = L_to_theta(L0)
    else:
        theta0 = L_to_theta(jnp.eye(d))

    # loss selector
    key0 = random.PRNGKey(seed_samples)

    if kernel == "rbf":
        def loss_fn(theta, key):
            return loss_wgrad_mmd_rbf_theta(X, theta, d, key, n_model=n_model, sigma=sigma_rbf)
    elif kernel == "poly":
        def loss_fn(theta, key):
            return loss_wgrad_mmd_poly_theta(X, theta, d, key, n_model=n_model, degree=degree, c=c)
    elif kernel == "linear":
        def loss_fn(theta, key):
            return loss_wgrad_mmd_linear_theta(X, theta, d, key=None, n_model=0)
    else:
        raise ValueError("kernel must be in {'rbf','poly','linear'}")

    val_and_grad = jax.jit(jax.value_and_grad(lambda th, ky: loss_fn(th, ky)))

    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    hist = []

    key = key0
    for t in range(1, iters+1):
        key, sub = random.split(key)
        val, g = val_and_grad(theta, sub)
        theta, m, v = adam_update(g, m, v, t, theta, lr=lr, b1=b1, b2=b2)
        hist.append(float(val))

    # build Sigma from theta
    Sigma_hat = np.array(sigma_from_theta(theta, d))

    return Sigma_hat, np.array(theta), hist



####################################################################################################
####################################################################################################
#                                                                                                  #
# plotting heatmaps                                                                                #
#                                                                                                  #
####################################################################################################
####################################################################################################



def plot_cov_heatmaps(Sigma_true: np.ndarray,
                      Sigma_rec:  np.ndarray,
                      vmax: float | None = None,
                      title_true: str = "true covariance",
                      title_rec:  str = "reconstructed covariance"):
    

    Sigma_true = np.asarray(Sigma_true)
    Sigma_rec  = np.asarray(Sigma_rec)
    if vmax is None:
        vmax = float(max(np.max(np.abs(Sigma_true)), np.max(np.abs(Sigma_rec))))

    plt.figure(figsize=(4.8, 4.4))
    plt.imshow(Sigma_true, aspect="equal", vmin=-vmax, vmax=vmax, origin="upper")
    plt.colorbar(); plt.title(title_true); plt.xlabel("j"); plt.ylabel("i")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(4.8, 4.4))
    plt.imshow(Sigma_rec, aspect="equal", vmin=-vmax, vmax=vmax, origin="upper")
    plt.colorbar(); plt.title(title_rec); plt.xlabel("j"); plt.ylabel("i")
    plt.tight_layout(); plt.show()
