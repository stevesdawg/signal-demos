import basis
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import scipy.linalg
from plotly.subplots import make_subplots

""" Linear regression example:
Input data is x-coordinates, 1D. Outputs are y-coordinates, 1D.
A matrix is [1, X], we will learn the intercept, as well as the slope.
We are learning theta vector, a 2x1, mapping input to output. We will
estimate a new theta sample by sample. In this example, y0 will contain
3 samples (3 > # of degrees of freedom of our model, which is 2).

Signal basis example:
Given a time series signal Y, our A matrix will be [MxN]. M is continuously
tracking the number of samples Y. It will increment as new samples appear.
N is the number of basis vectors. Our unknown X are the coefficients that best
estimate the signal Y. X is Nx1. In this example, y0 will contain at least N samples
so we always have more samples than degrees of freedom of our model.
"""


def compute_rls(y_k, A_k, xhat_k1, P_k1):
    if not y_k.shape:
        y_k = np.asarray([y_k]).reshape((1, 1))
        A_k = A_k.reshape((1, A_k.shape[0]))
    elif len(y.shape) == 1:
        y_k = y_k.reshape((y_k.shape[0], 1))
    xhat_k1 = xhat_k1.reshape((xhat_k1.shape[0], 1))

    num_samples = A_k.shape[0]
    P_k = (
        P_k1
        - P_k1
        @ A_k.T
        @ sp.linalg.inv(np.eye(num_samples) + A_k @ P_k1 @ A_k.T)
        @ A_k
        @ P_k1
    )
    K_k = P_k @ A_k.T
    return xhat_k1 + K_k @ (y_k - A_k @ xhat_k1), P_k


if __name__ == "__main__":
    ### Linear regression example ###
    np.random.seed(basis.SEED)
    N = 200
    step = 3
    y = np.linspace(1, 6, N, endpoint=False)
    yrand = y + np.random.random(y.shape) - 0.5
    t1 = np.linspace(-2, 10, N, endpoint=False)
    Adata = np.ones((N, 2))
    Adata[:, 1] = t1[:]

    fig = make_subplots(rows=1, cols=2)
    y0 = yrand[:3]
    A0 = Adata[:3, :]
    P0 = sp.linalg.inv(A0.T @ A0)
    xhat_0 = P0 @ (A0.T @ y0.reshape((y0.shape[0], 1)))
    print(xhat_0)
    yhat_0 = Adata @ xhat_0
    fig.add_trace(
        go.Scatter(
            x=Adata[:, 1],
            y=yrand,
            mode="markers",
            name="Raw data",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=Adata[:, 1],
            y=yhat_0[:, 0],
            mode="lines",
            name="xhat 0 fit",
        ),
        row=1,
        col=1,
    )

    all_coeffs = np.zeros((N // step + 1, 2))
    all_coeffs[0, :] = xhat_0[:, 0]
    for i in range(3, N, step):
        xhat_k, P_k = compute_rls(
            yrand[i : i + step], Adata[i : i + step, :], xhat_0, P0
        )
        all_coeffs[i // 3, :] = xhat_k[:, 0]
        if i % 10 == 0:
            yhat_k = Adata @ xhat_k
            fig.add_trace(
                go.Scatter(
                    x=Adata[:, 1],
                    y=yhat_k[:, 0],
                    mode="lines",
                    name="xhat {} fit".format(i),
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Scatter(
            y=all_coeffs[:, 1],
            mode="lines+markers",
            name="Slope coefficient over time",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=all_coeffs[:, 0],
            mode="lines+markers",
            name="Intercept coefficient over time",
        ),
        row=1,
        col=2,
    )
    fig.show()
