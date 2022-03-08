from os import truncate

import basis
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import scipy.stats
from numpy.random import default_rng
from plotly.subplots import make_subplots

"""Create matrices with small singular values. Show what happens with noise,
worst case noise. Show improvement if you truncate the smallest q singular values.
Show improvement if you use Tikhonov regularization. Show error as function of
delta parameter. In practice, figure out a ratio between largest and smallest
singular value, or, an absolute threshold for the smallest singular value.
Because this will cause problems when reconstructing using pseudoinverse.
"""


def generate_matrix(sigmas, M, N):
    U = sp.stats.ortho_group.rvs(dim=M, random_state=basis.SEED)
    V = sp.stats.ortho_group.rvs(dim=N, random_state=basis.SEED)
    diag_mat = np.diag(sigmas)
    U = U[:, :N]
    return U @ diag_mat @ V.T, U, diag_mat, V


def add_noise(y, max_noise, noise=None):
    if noise is None:
        noise = np.random.random(y.shape[0])
    if noise.shape[0] != y.shape[0]:
        raise RuntimeError("Noise term dimension doesn't match y vector.")
    noise /= np.max(noise) / max_noise
    return y + noise, noise


def generate_xvec(N, lo, hi):
    rng = default_rng(basis.SEED)
    return rng.choice(hi - lo, N) + lo


def truncated_matrix(num_omit, U, S, V):
    M = U.shape[0]
    R = U.shape[1]
    N = V.shape[0]
    return U[:, : N - num_omit] @ S[:-num_omit, :-num_omit] @ V[:, : N - num_omit].T


def tikhonov_regularize(delta, A):
    r, c = A.shape
    return sp.linalg.inv(A.T @ A + np.diag([delta] * c)) @ A.T


if __name__ == "__main__":
    # Generate matrix with 1 bad singular value
    N = 8
    M = 13
    R = 8
    lo, hi = -5, 10
    min_sigma = 0.000005
    max_noise = 0.01
    x = generate_xvec(N, lo, hi)
    sigmas = np.sort(np.random.random((R,)))[::-1]
    sigmas[-1] = min_sigma
    print("Singular values: {}\n".format(sigmas))
    A, U, S, V = generate_matrix(sigmas, M, N)
    y = A @ x
    print("y: {}\n".format(y))
    y_noise, noise = add_noise(y, max_noise)
    print("Noise: {}\n".format(noise))
    A_pinv = sp.linalg.pinv(A)
    x_hat = A_pinv @ y
    x_hat_noise = A_pinv @ y_noise
    print("x_hat_noise: {}\n".format(x_hat_noise))
    print("x_hat: {}\n".format(x_hat))
    print("raw x: {}\n".format(x))

    # SVD Truncation
    A_trunc = truncated_matrix(1, U, S, V)
    print(A_trunc.shape)
    A_trunc_pinv = sp.linalg.pinv(A_trunc)
    x_hat_noise_trunc = A_trunc_pinv @ y_noise
    x_hat_trunc = A_trunc_pinv @ y
    print("x hat truncated with noise: {}\n".format(x_hat_noise_trunc))
    print("x hat truncated: {}\n".format(x_hat_trunc))

    # Tikhonov Regularization
    # If bad sigma is less than delta, than delta will dominate
    delta = 0.006
    A_tikh_inv = tikhonov_regularize(delta, A)
    x_hat_noise_tikh = A_tikh_inv @ y_noise
    x_hat_tikh = A_tikh_inv @ y
    print("x hat tikhonov with noise: {}\n".format(x_hat_noise_tikh))
    print("x hat tikhonov: {}\n".format(x_hat_tikh))

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(y=y, mode="lines", name="Clean y"), row=1, col=1)
    fig.add_trace(go.Scatter(y=y_noise, mode="lines", name="Noisy y"), row=1, col=1)
    fig.add_trace(go.Scatter(y=noise, mode="lines", name="Noise delta"), row=1, col=1)
    fig.add_trace(go.Scatter(y=x, mode="lines", name="Original x"), row=1, col=2)
    fig.add_trace(
        go.Scatter(y=x_hat, mode="lines", name="Reconstructed x from clean y"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(y=x_hat_noise, mode="lines", name="Reconstructed x from noisy y"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=x_hat_noise_trunc, mode="lines", name="Truncated A Reconstruction of x"
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=x_hat_noise_tikh,
            mode="lines",
            name="Tikhonov Regularized Reconstruction of x from noisy Y",
        ),
        row=1,
        col=2,
    )
    fig.show()
