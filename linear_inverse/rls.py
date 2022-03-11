import basis
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import scipy.linalg

""" Linear regression example:
Input data is x-coordinates, 1D. Outputs are y-coordinates, 1D.
A matrix is [1, X], so we can learn the intercept, as well as the slope.
We are learning theta, a 2x1 vector, mapping input to output. We will
estimate a new theta sample by sample. In this example, y0 will contain
3 samples (3 > # of degrees of freedom of our model).

Signal basis example:
Given a time series signal Y, our A matrix will be [MxN]. M is continuously
tracking the number of samples Y. It will increment as new samples appear.
N is the number of basis vectors. Our unknown X are the coefficients that best
estimate the signal Y. X is Nx1. In this example, y0 will contain at least N samples
so we always have more samples than degrees of freedom of our model.
"""

if __name__ == "__main__":
    N = 100
