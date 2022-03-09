import basis
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import scipy.linalg

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
    xdata = np.ones((y.shape[0], 3))
    xdata[:, 1] = np.arange(2, 5, 0.05)
    xdata[:, 2] = np.arange(-7, -1, 0.1)
    xdata_rand = xdata + np.random.random(xdata.shape) - 0.5
    xdata_rand[:, 0] = xdata[:, 0]
    xdata_rand.sort(axis=0)

    # Standard Least Squares
    xdata_pinv = sp.linalg.pinv(xdata_rand)
    theta_hat = xdata_pinv @ yrand
    y_hat = xdata_rand @ theta_hat
    y_hat_clean = xdata @ theta_hat

    # Total Least Squares
    combined = np.concatenate([xdata_rand, y])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xdata_rand[:, 1],
                y=xdata_rand[:, 2],
                z=yrand,
                mode="markers",
                name="Sample data",
            ),
            go.Scatter3d(
                x=xdata_rand[:, 1],
                y=xdata_rand[:, 2],
                z=y_hat,
                mode="lines",
                name="Standard Least Squares Best Fit",
            ),
            go.Scatter3d(
                x=xdata[:, 1],
                y=xdata[:, 2],
                z=y_hat_clean,
                mode="lines",
                name="Clean Line",
            ),
        ],
    )
    fig.show()
