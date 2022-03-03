import basis
import numpy as np
import plotly.graph_objects as go
import scipy as sp
from plotly.subplots import make_subplots
from scipy import signal

""" Demonstrates performing a deconvolution of a filter on an output signal to
recover the optimal input signal. In this problem, the A matrix will be a convolution
matrix (therefore toeplitz) incorporating both the coefficients of the filter H and
the basis vectors psi_n(t) evaluated at the filter sample times. Once again, verify
that complete signal recovery is possible only when:
1. input signal f(t) is an exact linear combination of psi_n(t) vectors.
2. There are more output observations y(t) (M) than there are basis vectors (N).
"""


def convolution_matrix(h, psi):
    """Creates an (M+N)xK matrix by creating a (M+N)xN toeplitz convolution matrix, and
    multiplying by NxK matrix psi, whose columns are orthonormal basis vectors of f(t).

    Args:
        h (np.ndarray): 1D array of length M. Represents the filter coefficients.
        psi (np.ndarray): NxK matrix psi. N = time axis, K = number of basis vectors.

    Returns:
        np.ndarray: (M+N)xK matrix which = h_toeplitz @ psi
    """
    r = np.concatenate((h, np.zeros((psi.shape[0] - 1,))))
    c = np.concatenate(([h[0]], np.zeros((psi.shape[0] - 1,))))
    H = sp.linalg.toeplitz(c, r)
    return H.T @ psi


if __name__ == "__main__":
    N = 21
    T = 4
    tstep = 0.1
    t = np.arange(0, 4, tstep)
    len_t = t.shape[0]
    psi = basis.fourier_series_basis(T, N, t=t)
    ft, xn = basis.generate_func(N, -2, 5, psi)
    h = np.asarray([0.1, 0.3, -0.1, 0.4, 0.2, 0.24, 0.1])
    yt = signal.convolve(h, ft[:, 0])
    H_psi = convolution_matrix(h, psi)
    xn_hat = sp.linalg.pinv(H_psi) @ yt
    ft_hat = basis.reconstruct(xn_hat, psi)

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(y=ft[:, 0], name="Original f(t)", mode="lines"), row=1, col=1
    )
    fig.add_trace(go.Scatter(y=yt, name="filtered y(t)", mode="lines"), row=1, col=1)
    fig.add_trace(
        go.Scatter(y=ft_hat, name="Deconvolved f(t)", mode="lines"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=xn, name="Original coeffs x[n]", mode="lines"), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=xn_hat, name="Deconvolved coeffs x[n]", mode="lines"), row=1, col=2
    )
    fig.show()
