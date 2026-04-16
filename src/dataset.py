"""
Dataset utilities for multi-environment generalisation training.

Provides:

``ProblemInstanceGenerator``
    Randomly sample SensorSelectionEnv instances from a configurable
    distribution over kernel type, N, nu, length_scale, sigma and eps_frac.
    Supports both "matern" and "j0" kernel types and loading from .mat files.

``CurriculumScheduler``
    Three-stage curriculum over problem difficulty.  Advances to the next
    stage when the rolling success rate exceeds a threshold.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .kernel import build_toeplitz_matrix
from .environment import SensorSelectionEnv


# ──────────────────────────────────────────────────────────────────────────────
# Problem instance generator
# ──────────────────────────────────────────────────────────────────────────────

class ProblemInstanceGenerator:
    """Sample random SensorSelectionEnv instances from a parameter distribution.

    Parameters
    ----------
    n_range        : (min_N, max_N)  inclusive range for matrix dimension
    nu_choices     : list of Matern nu values to sample from
    length_scale_range : (min_l, max_l)
    sigma_range    : (min_sigma, max_sigma)
    eps_frac_range : (min_eps, max_eps)
    kernel_types   : list of kernel type strings, e.g. ["matern", "j0"]
    mat_files      : optional list of (path, var_name) tuples for .mat files
    seed           : optional RNG seed for reproducibility
    """

    def __init__(
        self,
        n_range: Tuple[int, int] = (5, 20),
        nu_choices: Sequence[float] = (0.5, 1.5, 2.5),
        length_scale_range: Tuple[float, float] = (1.0, 5.0),
        sigma_range: Tuple[float, float] = (0.1, 1.0),
        eps_frac_range: Tuple[float, float] = (0.10, 0.40),
        kernel_types: Sequence[str] = ("matern",),
        mat_files: Optional[List[Tuple[str, str]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_range = n_range
        self.nu_choices = list(nu_choices)
        self.length_scale_range = length_scale_range
        self.sigma_range = sigma_range
        self.eps_frac_range = eps_frac_range
        self.kernel_types = list(kernel_types)
        self.mat_files = mat_files or []
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def sample(self) -> SensorSelectionEnv:
        """Sample one random problem instance.

        If ``mat_files`` is non-empty, with 50% probability a .mat file is used
        (picking a random file and random sub-matrix of the sampled N).

        Returns
        -------
        SensorSelectionEnv
        """
        N = self.rng.randint(self.n_range[0], self.n_range[1])
        sigma = float(self.np_rng.uniform(self.sigma_range[0], self.sigma_range[1]))
        eps_frac = float(self.np_rng.uniform(self.eps_frac_range[0], self.eps_frac_range[1]))

        # Decide whether to use a .mat file
        if self.mat_files and self.rng.random() < 0.5:
            path, var_name = self.rng.choice(self.mat_files)
            J = self._load_mat(path, var_name, N)
        else:
            kernel_type = self.rng.choice(self.kernel_types)
            nu = self.rng.choice(self.nu_choices)
            length_scale = float(self.np_rng.uniform(
                self.length_scale_range[0], self.length_scale_range[1]
            ))
            J = build_toeplitz_matrix(
                N, nu=nu, length_scale=length_scale, kernel_type=kernel_type
            )

        epsilon = eps_frac * float(np.trace(J))
        return SensorSelectionEnv(J, sigma=sigma, epsilon=epsilon)

    @staticmethod
    def _load_mat(path: str, var_name: str, N: int) -> np.ndarray:
        """Load a kernel matrix from a .mat file and return an N x N sub-matrix."""
        import scipy.io as sio
        mat = sio.loadmat(path)
        J_full = np.array(mat[var_name], dtype=float)
        n = min(N, J_full.shape[0])
        return J_full[:n, :n]


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum scheduler
# ──────────────────────────────────────────────────────────────────────────────

class CurriculumScheduler:
    """Three-stage difficulty curriculum, extended with a large-N J₀ stage.

    Stage 0 (easy)       : small N (≤10), large eps_frac, smooth Matérn kernels
    Stage 1 (medium)     : moderate N (≤20), moderate eps_frac, Matérn kernels
    Stage 2 (hard)       : N≤50, small eps_frac, all kernel types
    Stage 3 (large-N J₀) : N≤256, very small eps_frac, J₀ kernel only  (Plan B)

    The policy trained on stages 0-2 learns the qualitative "pick evenly spaced
    sensors" pattern; stage 3 exploits the J₀ shift-invariance to generalise
    this pattern to N=256 without relearning from scratch.

    The scheduler wraps a ``ProblemInstanceGenerator`` per stage and
    advances when the rolling success rate exceeds ``advance_threshold``.

    Parameters
    ----------
    n_min, n_max    : overall N range (hard stage maximum)
    advance_threshold : rolling success rate needed to advance stages
    window          : number of recent episodes used for the rolling rate
    """

    _STAGES = [
        # (n_max, eps_frac_min, eps_frac_max, kernel_types, nu_choices)
        (10,  0.40, 0.60, ["matern"], [1.5, 2.5]),
        (20,  0.20, 0.40, ["matern"], [0.5, 1.5, 2.5]),
        (50,  0.10, 0.20, ["matern", "j0"], [0.5, 1.5, 2.5]),
        (256, 0.05, 0.15, ["j0"], [1.5]),               # large-N J₀ stage (Plan B)
        # eps_frac in [0.05, 0.15]: tight enough to force the policy to use
        # sparse, evenly-spaced sensor patterns (which are optimal for J₀) but
        # loose enough that a feasible solution exists for typical N≤256 problems.
    ]

    def __init__(
        self,
        n_min: int = 5,
        n_max: int = 30,
        advance_threshold: float = 0.80,
        window: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self.n_min = n_min
        self.n_max = n_max
        self.advance_threshold = advance_threshold
        self.window = window
        self.current_stage = 0
        self._recent_success: List[float] = []
        self._generators = self._build_generators(seed)

    def _build_generators(
        self, seed: Optional[int]
    ) -> List[ProblemInstanceGenerator]:
        gens = []
        for i, (n_max_stage, eps_min, eps_max, ktypes, nus) in enumerate(
            self._STAGES
        ):
            n_max_eff = min(self.n_max, n_max_stage)
            n_max_eff = max(n_max_eff, self.n_min)
            s = None if seed is None else seed + i * 1000
            gens.append(
                ProblemInstanceGenerator(
                    n_range=(self.n_min, n_max_eff),
                    nu_choices=nus,
                    eps_frac_range=(eps_min, eps_max),
                    kernel_types=ktypes,
                    seed=s,
                )
            )
        return gens

    def sample(self) -> SensorSelectionEnv:
        """Sample a problem instance at the current difficulty stage."""
        return self._generators[self.current_stage].sample()

    def record(self, satisfied: bool) -> None:
        """Record whether the latest episode was satisfied.

        Automatically advances to the next stage if the rolling success rate
        exceeds ``advance_threshold`` and a harder stage is available.
        """
        self._recent_success.append(float(satisfied))
        if len(self._recent_success) > self.window:
            self._recent_success.pop(0)
        if (
            self.current_stage < len(self._STAGES) - 1
            and len(self._recent_success) >= self.window
            and float(np.mean(self._recent_success)) >= self.advance_threshold
        ):
            self.current_stage += 1
            self._recent_success.clear()

    @property
    def stage(self) -> int:
        return self.current_stage

    @property
    def rolling_success_rate(self) -> float:
        if not self._recent_success:
            return 0.0
        return float(np.mean(self._recent_success))
