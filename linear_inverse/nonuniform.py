import basis
import numpy as np
import plotly.graph_objects as go
from numpy.random import default_rng
from plotly.subplots import make_subplots

"""Demonstrates signal recovery with non-uniform samples of the signal.
Assumptions: f(t) lives in the subspace spanned by the psi_n(t) basis vectors.
In other words, f(t) is an exact linear combination of the psi_n(t) vectors.
We get perfect signal recovery when M (number of non-uniform samples) > N (
number of basis vectors.) Makes sense, because if we have fewer samples, the
linear inverse problem is under-determined. Check this by adjusting M to be
larger or smaller than N in the code below.
"""

if __name__ == "__main__":
    T = 2
    N = 101
    tstep = 0.01
    t = np.arange(0, T, tstep)
    len_t = t.shape[0]
    psi = basis.fourier_series_basis(T, N, tstep)
    ft, xn = basis.generate_func(N, -1.5, 1.5, psi)
    M = len_t // 10 * 7

    rng = default_rng(basis.SEED)
    nonuniform_idx = np.sort((rng.choice(len_t, M, replace=False) - 1))
    t_non = t[nonuniform_idx]
    ft_non = ft[nonuniform_idx, 0]
    psi_non = basis.fourier_series_basis(T, N, tstep, t=t_non)

    xn_non = basis.estimate_coeffs(ft_non, psi_non)
    ft_hat = basis.reconstruct(xn_non, psi)

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=t, y=ft[:, 0], name="Original f(t)", mode="lines"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ft_hat[:, 0],
            name="Reconstructed f(t) from non uniform samples",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_non, y=ft_non, name="Non uniform samples of f(t)", mode="markers"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=xn, name="Original x[n] coefficients", mode="lines"), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            y=xn_non[:, 0],
            name="Recoved x[n] coefficients from non uniform samples",
            mode="lines",
        ),
        row=1,
        col=2,
    )
    fig.show()
