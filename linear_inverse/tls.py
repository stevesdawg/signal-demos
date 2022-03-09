import basis
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import scipy.linalg
from plotly.subplots import make_subplots

"""Total Least Squares (TLS). Compare approximate solutions of LS vs TLS.
Standard least squares minimizes: choose x that minimizes ||y + dy|| = Ax.
TLS minimizes: choose x that minimizes ||y + dy|| = ||A + dA||x.
"""

np.random.seed(basis.SEED)

if __name__ == "__main__":
    ### Signal basis example ###
    pass

    ### Linear regression example ###
    y = np.arange(-1, 2, 0.05)
    yrand = y + np.random.random(y.shape)
    xdata = np.ones((y.shape[0], 4))
    t1 = np.arange(1, 5.2, 0.07)
    xdata[:, 1] = t1[:]
    xdata[:, 2] = np.sin(2 * np.pi * t1)
    xdata[:, 3] = np.cos(2 * np.pi * t1)
    xdata_rand = xdata + (np.random.random(xdata.shape) - 0.5) / 8
    xdata_rand[:, 0:2] = xdata[:, 0:2]
    xdata_pinv = sp.linalg.pinv(xdata_rand)
    theta_hat = xdata_pinv @ yrand
    y_hat = xdata_rand @ theta_hat
    y_hat_clean = xdata @ theta_hat

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter"}]]
    )
    fig.add_trace(
        go.Scatter3d(
            x=xdata_rand[:, 2],
            y=xdata_rand[:, 3],
            z=yrand,
            mode="markers",
            marker=dict(
                size=[2],
                sizemode="diameter",
            ),
            name="Sample data",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=xdata_rand[:, 2],
            y=xdata_rand[:, 3],
            z=y_hat,
            mode="lines",
            name="Standard Least Squares Best Fit",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t1,
            y=yrand,
            mode="markers",
            name="Sample data against linear parameter t",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t1,
            y=y_hat,
            mode="markers",
            name="Best fit against linear parameter t",
        ),
        row=1,
        col=2,
    )
    fig.show()
