import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

SEED = 42
np.random.seed(SEED)


def fourier_series_basis(T, N, t_step=0.05, t=None):
    """Generate Fourier basis matrix over interval T.
    Basis vectors are stored in columns.

    Args:
        T (float): Time interval of signal.
        N (int): Number of basis vectors. Must be odd.
        t_step (float, optional): T_step sample period for time series signal. Defaults to 0.05.
        t (np.ndarray, optional): 1D array representing time samples, of length M.
        Defaults to Uniformly spaced time samples.

    Raises:
        RuntimeError: Error if T <= 0, non-positive.
        RuntimeError: Error if N is not odd.

    Returns:
        np.ndarray: MxN matrix representing N basis vectors over time T.
    """
    if T <= 0:
        raise RuntimeError("T is non positive.")
    if N % 2 != 1:
        raise RuntimeError("N is not odd.")
    if t is None:
        t = np.arange(0, T, t_step)
    psi = np.zeros((t.shape[0], N))
    for n in range(N):
        psi[:, n] = generate_psi_n(t, n + 1, N, T)
    return psi


def generate_psi_n(t, n, N, T):
    """Generate a single Fourier series basis vector.

    Args:
        t (np.ndarray): 1D array representing time axis of length M.
        n (int): n'th basis vector to generate.
        N (int): Total number of basis vectors N. Must be odd.
        T (int): Endpoint of the time interval of the signal.

    Raises:
        RuntimeError: N is not odd.
        RuntimeError: Selected n is out of bounds (non-positive or greater than N).

    Returns:
        np.ndarray: 1D array representing a single basis vector, or length M.
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


def generate_val(t, n, N, T):
    """Generate a single value in the MxN Fourier basis matrix. Given t and N.

    Args:
        t (float): Single time sample to compute value.
        n (int): n'th basis vector.
        N (int): Total number of basis vectors N. Must be odd.
        T (int): Endpoint of the time interval of the signal.

    Returns:
        float: A single value to populate the PSI[t, n] element of psi matrix.
    """
    if n == 1:
        return 1
    elif n <= (N + 1) // 2:
        return np.cos(2 * np.pi * (n - 1) / T * t)
    elif n >= (N + 3) // 2:
        return np.sin(2 * np.pi * (n - (N + 1) // 2) / T * t)


def generate_func(N, lo, hi, psi):
    """Returns a function f(t) that is in the span of columns of psi.
    Generated using random basis coefficients and then reconstructed.

    Args:
        N (int): Total number of basis vectors N. Must be Odd.
        lo (int): Lower bound, above which random basis coefficients will be chosen.
        hi (int): Upper bound, below which random basis coefficients will be chosen.
        psi (np.ndarray): MxN psi matrix, whose columns are the basis vectors.

    Returns:
        np.ndarray: Generated function f(t), 1D sequence of length M.
        np.ndarray: Generated coefficients x[n], 1D sequence of length N.
    """
    xn = (hi - lo) * np.random.random_sample(N) + lo
    return psi @ xn.reshape(xn.shape[0], 1), xn


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
        f_t = f_t.reshape(f_t.shape[0], 1)
    return np.linalg.pinv(psi_mat) @ f_t


def reconstruct(coeffs, psi_mat):
    if len(coeffs.shape) == 1:
        coeffs = coeffs.reshape(coeffs.shape[0], 1)
    return psi_mat @ coeffs


if __name__ == "__main__":
    T = 2
    N = 101
    tstep = 0.01
    psi = fourier_series_basis(T, N, tstep)

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
