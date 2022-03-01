import basis
import numpy as np
import plotly.graph_objects as go
from numpy.random import default_rng
from plotly.subplots import make_subplots

""" Demonstrates performing a deconvolution of a filter on an output signal to
recover the optimal input signal. In this problem, the A matrix will be a convolution
matrix (therefore toeplitz) incorporating both the coefficients of the filter H and
the basis vectors psi_n(t) evaluated at the filter sample times. Once again, verify
that complete signal recovery is possible only when:
1. input signal f(t) is an exact linear combination of psi_n(t) vectors.
2. There are more output observations y(t) (M) than there are basis vectors (N).
"""

if __name__ == "__main__":
    print("To Do")
