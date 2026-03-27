"""
Bessel (Matérn) kernel functions and Toeplitz matrix construction.

The Matérn covariance kernel is defined in terms of modified Bessel functions
of the second kind (K_ν).  For ν ∈ {0.5, 1.5, 2.5} the kernel simplifies to
an algebraic-exponential form; for general ν the scipy implementation is used.

References
----------
Rasmussen & Williams, "Gaussian Processes for Machine Learning", 2006,
Chapter 4.
"""

import numpy as np
from scipy.special import kv, gamma


# ──────────────────────────────────────────────────────────────────────────────
# Closed-form Matérn kernels
# ──────────────────────────────────────────────────────────────────────────────

def matern_05(r: float, length_scale: float = 1.0) -> float:
    """Matérn ν=0.5 (Ornstein–Uhlenbeck / exponential) kernel."""
    return float(np.exp(-np.abs(r) / length_scale))


def matern_15(r: float, length_scale: float = 1.0) -> float:
    """Matérn ν=1.5 kernel."""
    sqrt3_r = np.sqrt(3.0) * np.abs(r) / length_scale
    return float((1.0 + sqrt3_r) * np.exp(-sqrt3_r))


def matern_25(r: float, length_scale: float = 1.0) -> float:
    """Matérn ν=2.5 kernel."""
    sqrt5_r = np.sqrt(5.0) * np.abs(r) / length_scale
    return float((1.0 + sqrt5_r + sqrt5_r ** 2 / 3.0) * np.exp(-sqrt5_r))


def bessel_kernel(r: float, nu: float = 1.5, length_scale: float = 1.0) -> float:
    """General Matérn covariance function using the modified Bessel function K_ν.

    k(r) = (2^(1-ν) / Γ(ν)) · (√(2ν)·|r|/l)^ν · K_ν(√(2ν)·|r|/l)

    Parameters
    ----------
    r : float
        Distance (|i − j| for the Toeplitz case).
    nu : float
        Smoothness parameter.  Values 0.5, 1.5, 2.5 yield closed-form kernels.
    length_scale : float
        Characteristic length-scale ℓ.

    Returns
    -------
    float
        Kernel value k(r).
    """
    r = float(np.abs(r))

    # Use exact closed forms for the common special cases
    if nu == 0.5:
        return matern_05(r, length_scale)
    if nu == 1.5:
        return matern_15(r, length_scale)
    if nu == 2.5:
        return matern_25(r, length_scale)

    # General case via scipy Bessel functions
    if r == 0.0:
        return 1.0
    sqrt_2nu_r = np.sqrt(2.0 * nu) * r / length_scale
    return float(
        (2.0 ** (1.0 - nu) / gamma(nu))
        * (sqrt_2nu_r ** nu)
        * kv(nu, sqrt_2nu_r)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Toeplitz matrix construction
# ──────────────────────────────────────────────────────────────────────────────

def build_toeplitz_matrix(
    N: int,
    nu: float = 1.5,
    length_scale: float = 1.0,
) -> np.ndarray:
    """Build an N×N real symmetric Toeplitz matrix from the Bessel (Matérn) kernel.

    J[i, j] = k(|i − j|) where k is the Matérn-ν kernel.

    Parameters
    ----------
    N : int
        Matrix dimension.
    nu : float
        Matérn smoothness parameter.
    length_scale : float
        Kernel length-scale.

    Returns
    -------
    np.ndarray
        Shape (N, N) real symmetric positive-definite Toeplitz matrix.
    """
    first_row = np.array(
        [bessel_kernel(i, nu=nu, length_scale=length_scale) for i in range(N)],
        dtype=np.float64,
    )
    indices = np.arange(N)
    J = first_row[np.abs(indices[:, None] - indices[None, :])]
    return J
