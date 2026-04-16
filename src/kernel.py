"""
Bessel (Matérn) kernel functions and Toeplitz matrix construction.

The Matérn covariance kernel is defined in terms of modified Bessel functions
of the second kind (K_ν).  For ν ∈ {0.5, 1.5, 2.5} the kernel simplifies to
an algebraic-exponential form; for general ν the scipy implementation is used.

An additional "j0" kernel type is provided based on the zero-order Bessel
function of the first kind J₀(r/l).  Unlike the Matérn family this kernel is
oscillatory and can take negative values, so the resulting Toeplitz matrix is
not necessarily positive-definite.  Helpers ``is_positive_definite`` and
``regularize_matrix`` are provided to detect and correct indefiniteness.

References
----------
Rasmussen & Williams, "Gaussian Processes for Machine Learning", 2006,
Chapter 4.
"""

import numpy as np
from scipy.special import j0, kv, gamma


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
# Zero-order Bessel function of the first kind (J₀) kernel
# ──────────────────────────────────────────────────────────────────────────────

def bessel_j0_kernel(r: float, length_scale: float = 1.0) -> float:
    """Zero-order Bessel function of the first kind kernel.

    k(r) = J₀(|r| / l)

    J₀(0) = 1 and the function is oscillatory, taking negative values for
    |r|/l ≳ 2.4.  The resulting Toeplitz matrix is therefore **not**
    guaranteed to be positive-definite.

    Parameters
    ----------
    r : float
        Distance (|i − j| for the Toeplitz case).
    length_scale : float
        Characteristic length-scale ℓ.

    Returns
    -------
    float
        Kernel value k(r).
    """
    return float(j0(float(np.abs(r)) / length_scale))


# ──────────────────────────────────────────────────────────────────────────────
# Positive-definiteness utilities
# ──────────────────────────────────────────────────────────────────────────────

def is_positive_definite(J: np.ndarray, tol: float = 0.0) -> bool:
    """Return True if *J* is positive definite (all eigenvalues > tol).

    Parameters
    ----------
    J : np.ndarray
        Real symmetric matrix.
    tol : float
        Eigenvalue threshold.  The default of 0.0 checks strict positivity.

    Returns
    -------
    bool
    """
    eigs = np.linalg.eigvalsh(J)
    return bool(np.all(eigs > tol))


def regularize_matrix(J: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Return a regularised copy of *J* that is positive-definite.

    Computes the minimum eigenvalue and, if it is below *min_eig*, adds
    ``(min_eig - λ_min) · I`` to make all eigenvalues ≥ *min_eig*.

    Parameters
    ----------
    J : np.ndarray
        Real symmetric matrix.
    min_eig : float
        Desired minimum eigenvalue after regularization.

    Returns
    -------
    np.ndarray
        Regularised matrix (copy; original is unchanged).
    """
    lam_min = float(np.linalg.eigvalsh(J).min())
    if lam_min >= min_eig:
        return J.copy()
    shift = min_eig - lam_min
    return J + shift * np.eye(J.shape[0], dtype=J.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Toeplitz matrix construction
# ──────────────────────────────────────────────────────────────────────────────

def build_toeplitz_matrix(
    N: int,
    nu: float = 1.5,
    length_scale: float = 1.0,
    kernel_type: str = "matern",
) -> np.ndarray:
    """Build an N×N real symmetric Toeplitz matrix from a kernel function.

    J[i, j] = k(|i − j|)

    Parameters
    ----------
    N : int
        Matrix dimension.
    nu : float
        Matérn smoothness parameter (ignored when *kernel_type* is ``"j0"``).
    length_scale : float
        Kernel length-scale.
    kernel_type : {"matern", "j0"}
        Which kernel to use.  ``"matern"`` (default) produces a
        positive-definite matrix; ``"j0"`` uses J₀ and may not be PD.

    Returns
    -------
    np.ndarray
        Shape (N, N) real symmetric Toeplitz matrix.
    """
    if kernel_type == "j0":
        first_row = np.array(
            [bessel_j0_kernel(i, length_scale=length_scale) for i in range(N)],
            dtype=np.float64,
        )
    elif kernel_type == "matern":
        first_row = np.array(
            [bessel_kernel(i, nu=nu, length_scale=length_scale) for i in range(N)],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown kernel_type '{kernel_type}'. Choose 'matern' or 'j0'.")
    indices = np.arange(N)
    J = first_row[np.abs(indices[:, None] - indices[None, :])]
    return J
