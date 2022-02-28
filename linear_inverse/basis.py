import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal


def fourier_series_basis(T, N, t_step=0.05):
    """Generate fourier basis over interval T.
    Basis vectors are stored in columns.

    Args:
        T (float): Time interval of signal.
        N (int): Number of basis vectors. Must be odd.
        t_step (float, optional): T_step sample period for time series signal. Defaults to 0.05.

    Raises:
        RuntimeError: Error if T <= 0, non-positive.
        RuntimeError: Error if N is not odd.

    Returns:
        np.ndarray: T/t_stepxN matrix representing N basis vectors over time T.
    """
    if T <= 0:
        raise RuntimeError("T is non positive.")
    if N % 2 != 1:
        raise RuntimeError("N is not odd.")
    t = np.arange(0, T, t_step)
    psi = np.zeros((t.shape[0], N))
    for n in range(N):
        psi[:, n] = generate_psi_n(t, n + 1, N, T)
    return psi


def generate_psi_n(t, n, N, T):
    """Generate a single fourier series basis vector.

    Args:
        t (np.ndarray): 1D array representing time axis.
        n (int): n'th basis vector to generate.
        N (int): Total number of basis vectors N. Must be odd.
        T (int): Endpoint of the time interval of the signal.

    Raises:
        RuntimeError: N is not odd.
        RuntimeError: Selected n is out of bounds (non-positive or greater than N).

    Returns:
        _type_: _description_
    """
    if N % 2 != 1:
        raise RuntimeError("N is not odd.")
    if n > N or n <= 0:
        raise RuntimeError("Selected n is out of bounds.")
    psi_n = np.zeros((t.shape[0],))
    if n == 1:
        psi_n[:] = 1
    elif n <= (N + 1) // 2:
        psi_n[:] = np.cos(2 * np.pi * (n - 1) / T * t)
    elif n >= (N + 3) // 2:
        psi_n[:] = np.sin(2 * np.pi * (n - (N + 1) // 2) / T * t)
    return psi_n


def plot_basis(psi_mat, show):
    fig = go.Figure()
    for n in range(psi_mat.shape[1] // 2):
        fig.add_trace(go.Scatter(y=psi_mat[:, n + 1], mode="lines"))
    for n in range(psi_mat.shape[1] // 2):
        fig.add_trace(go.Scatter(y=psi_mat[:, n + (N + 1) // 2], mode="lines"))
    if show:
        fig.show()


def estimate_coeffs(f_t, psi_mat):
    if len(f_t.shape) == 1:
        f_t = f_t.reshape((f_t.shape[0], 1))
    return np.linalg.pinv(psi_mat) @ f_t


def reconstruct(coeffs, psi_mat):
    if len(coeffs.shape) == 1:
        coeffs = coeffs.reshape((coeffs.shape[0], 1))
    return psi_mat @ coeffs


# generate a single value in the basis matrix. Given t, and N
def generate_val(t, n, N, T):
    if n == 1:
        return 1
    elif n <= (N + 1) // 2:
        return np.cos(2 * np.pi * (n - 1) / T * t)
    elif n >= (N + 3) // 2:
        return np.sin(2 * np.pi * (n - (N + 1) // 2) / T * t)


if __name__ == "__main__":
    T = 2
    N = 101
    tstep = 0.01
    psi = fourier_series_basis(T, N, tstep)
    # plot_basis(psi, show=True)

    len_t = psi.shape[0]
    ft = np.zeros((len_t,))
    ft[0 : len_t // 4] = 1
    ft[len_t // 2 : len_t // 4 * 3] = 2
    xn = estimate_coeffs(ft, psi)
    ft_hat = reconstruct(xn, psi)
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(y=ft_hat[:, 0], name="f(t) proj onto subspace"), row=1, col=1
    )
    fig.add_trace(go.Scatter(y=ft, mode="lines+markers", name="f(t)"), row=1, col=1)
    fig.add_trace(
        go.Scatter(y=xn[:, 0], mode="lines+markers", name="basis coefficients"),
        row=1,
        col=2,
    )
    fig.show()
