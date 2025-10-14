"""
Data generation utilities for causal inference tasks.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Callable, Any
from scipy.special import erfinv, erf

from causalis.data.causaldata import CausalData



def _sigmoid(z):
    """
    Numerically stable sigmoid: 1 / (1 + exp(-z))
    Handles large positive/negative z without overflow warnings.
    Ensures outputs lie strictly within (0,1) and are strictly monotone in z
    even under floating-point saturation by using nextafter for exact 0/1 cases.

    Supports inputs of any shape; uses a flattened view to apply exact-0/1 nudges
    and reshapes back to preserve original structure.

    Parameters
    ----------
    z : array-like
        Input values of any shape

    Returns
    -------
    array-like
        Sigmoid of input values with same shape as input
    """
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)

    pos_mask = z_arr >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))
    ez = np.exp(z_arr[neg_mask])
    out[neg_mask] = ez / (1.0 + ez)

    if out.ndim == 0:
        val = float(out)
        if not (0.0 < val < 1.0):
            return float(np.nextafter(0.0, 1.0)) if val <= 0.0 else float(np.nextafter(1.0, 0.0))
        return val

    # Order-preserving nudge off exact 0/1 on a flattened view, then reshape back
    flat = out.ravel()
    zflat = z_arr.ravel()

    zero_idx = np.flatnonzero(flat <= 0.0)
    if zero_idx.size:
        order = np.argsort(zflat[zero_idx])  # ascending z
        val = 0.0
        for j in order:
            val = np.nextafter(val, 1.0)
            flat[zero_idx[j]] = val

    one_idx = np.flatnonzero(flat >= 1.0)
    if one_idx.size:
        order = np.argsort(zflat[one_idx])  # ascending z
        k = one_idx.size
        val = 1.0
        steps = []
        for _ in range(k):
            val = np.nextafter(val, 0.0)
            steps.append(val)  # descending sequence < 1
        for rank, j in enumerate(order):
            flat[one_idx[j]] = steps[k - rank - 1]

    return flat.reshape(out.shape)



def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1 - p)))


def _deterministic_ids(rng: np.random.Generator, n: int) -> List[str]:
    """Return deterministic uuid-like hex strings using the provided RNG."""
    return [rng.bytes(16).hex() for _ in range(n)]


def _sample_confounders_like_class(
    n: int,
    rng: np.random.Generator,
    *,
    confounder_specs: Optional[List[Dict[str, Any]]],
    k: int,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]],
    seed: Optional[int],
) -> Tuple[np.ndarray, List[str]]:
    """Replicates CausalDatasetGenerator._sample_X behavior (names & one-hot)."""
    # Custom sampler wins
    if x_sampler is not None:
        X = x_sampler(n, k, seed)
        names = [f"x{i+1}" for i in range(k)]
        return np.asarray(X, dtype=float), names

    # Specs provided
    if confounder_specs is not None:
        cols, names = [], []
        for spec in confounder_specs:
            name = spec.get("name") or f"x{len(names)+1}"
            dist = str(spec.get("dist", "normal")).lower()

            if dist == "normal":
                mu = float(spec.get("mu", 0.0)); sd = float(spec.get("sd", 1.0))
                col = rng.normal(mu, sd, size=n)
                cols.append(col.astype(float)); names.append(name)

            elif dist == "uniform":
                a = float(spec.get("a", 0.0)); b = float(spec.get("b", 1.0))
                col = rng.uniform(a, b, size=n)
                cols.append(col.astype(float)); names.append(name)

            elif dist == "bernoulli":
                p = float(spec.get("p", 0.5))
                col = rng.binomial(1, p, size=n).astype(float)
                cols.append(col); names.append(name)

            elif dist == "categorical":
                categories = list(spec.get("categories", [0, 1, 2]))
                probs = spec.get("probs", None)
                if probs is not None:
                    p = np.asarray(probs, dtype=float)
                    ps = p / p.sum()
                else:
                    ps = None
                col = rng.choice(categories, p=ps, size=n)
                # one-hot for all levels except the first
                rest = categories[1:]
                if len(rest) == 0:
                    # Add a zero column if only one category provided, with a collision-safe suffix
                    cols.append(np.zeros(n, dtype=float))
                    names.append(f"{name}__onlylevel")
                else:
                    for c in rest:
                        oh = (col == c).astype(float)
                        cols.append(oh)
                        names.append(f"{name}_{c}")
            else:
                raise ValueError(f"Unknown dist: {dist}")

        X = np.column_stack(cols) if cols else np.empty((n, 0))
        return X, names

    # Default: iid N(0,1) if k>0; else no X
    if k > 0:
        X = rng.normal(size=(n, k))
        names = [f"x{i+1}" for i in range(k)]
        return X, names
    else:
        return np.empty((n, 0)), []


def _gaussian_copula(
    rng: np.random.Generator,
    n: int,
    specs: List[Dict[str, Any]],
    corr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate mixed-type X with a Gaussian copula. 'specs' use the existing schema.
    For categorical, use quantile thresholds on a latent normal (via uniform U).
    """
    d = len(specs)
    if d == 0:
        return np.empty((n, 0)), []

    if corr is None:
        L = np.eye(d)
    else:
        C = np.asarray(corr, dtype=float)
        if C.shape != (d, d):
            raise ValueError(f"copula_corr must have shape {(d, d)}, got {C.shape}")
        # Ensure symmetry
        C = 0.5 * (C + C.T)
        I = np.eye(d)
        eps_list = [1e-12, 1e-10, 1e-8, 1e-6]
        L = None
        for eps in eps_list:
            try:
                L = np.linalg.cholesky(C + eps * I)
                break
            except np.linalg.LinAlgError:
                continue
        if L is None:
            # As a last resort, fall back to identity (independence)
            L = I
    # latent Z ~ N(0, corr)
    Z = rng.normal(size=(n, d)) @ L.T
    # Exact standard normal CDF via error function
    U = 0.5 * (1.0 + erf(Z / np.sqrt(2.0)))

    cols: List[np.ndarray] = []
    names: List[str] = []

    for j, spec in enumerate(specs):
        name = spec.get("name") or f"x{j+1}"
        dist = str(spec.get("dist", "normal")).lower()
        u = U[:, j]
        # keep u strictly inside (0,1) to avoid boundary issues in searchsorted
        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1.0 - eps)
        if dist == "normal":
            mu = float(spec.get("mu", 0.0)); sd = float(spec.get("sd", 1.0))
            x = mu + sd * np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
            cols.append(x.astype(float)); names.append(name)
        elif dist == "uniform":
            a = float(spec.get("a", 0.0)); b = float(spec.get("b", 1.0))
            x = a + (b - a) * u
            cols.append(x.astype(float)); names.append(name)
        elif dist == "bernoulli":
            p = float(spec.get("p", 0.5))
            x = (u < p).astype(float)
            cols.append(x); names.append(name)
        elif dist == "categorical":
            categories = list(spec.get("categories", [0, 1, 2]))
            probs = spec.get("probs", None)
            if probs is None:
                probs = [1.0 / max(len(categories), 1) for _ in categories]
            p = np.asarray(probs, dtype=float)
            if p.ndim != 1 or p.shape[0] != len(categories):
                raise ValueError("'probs' must be a 1D list matching 'categories' length")
            ps = p / p.sum()
            cum = np.cumsum(ps)
            idx = np.searchsorted(cum, u, side="right")
            # Guard boundary: if u==1 (after potential numerical saturation), clamp to last index
            if np.isscalar(idx):
                if idx >= len(categories):
                    idx = len(categories) - 1
            else:
                idx[idx == len(categories)] = len(categories) - 1
            cat_arr = np.array(categories, dtype=object)
            lab = cat_arr[idx]
            rest = categories[1:]
            if len(rest) == 0:
                cols.append(np.zeros(n, dtype=float))
                names.append(f"{name}__onlylevel")
            else:
                for c in rest:
                    cols.append((lab == c).astype(float))
                    names.append(f"{name}_{c}")
        else:
            raise ValueError(f"Unknown dist: {dist}")

    X = np.column_stack(cols) if cols else np.empty((n, 0))
    return X, names


def generate_rct(
    n: int = 20_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    target_type: str = "binary",              # {"binary","normal","poisson"}; "nonnormal" -> "poisson"
    target_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
) -> pd.DataFrame:
    """
    Generate an RCT dataset via CausalDatasetGenerator (thin wrapper), ensuring
    randomized treatment independent of X. Keeps y and d as float for ML compatibility.
    """
    # RNG for ancillary generation
    rng = np.random.default_rng(random_state)

    # Validate split
    split_f = float(split)
    if not (0.0 < split_f < 1.0):
        raise ValueError("split must be in (0,1).")

    # Normalize target_type
    ttype = target_type.lower()
    if ttype == "nonnormal":
        ttype = "poisson"
    if ttype not in {"binary", "normal", "poisson"}:
        raise ValueError("target_type must be 'binary', 'normal', or 'poisson' (or alias 'nonnormal').")

    # Default target_params
    if target_params is None:
        if ttype == "binary":
            target_params = {"p": {"A": 0.10, "B": 0.12}}
        elif ttype == "normal":
            target_params = {"mean": {"A": 0.00, "B": 0.20}, "std": 1.0}
        else:
            target_params = {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}}

    # Map to natural-scale means
    if ttype == "binary":
        pA = float(target_params["p"]["A"]); pB = float(target_params["p"]["B"])
        if not (0.0 < pA < 1.0 and 0.0 < pB < 1.0):
            raise ValueError("For binary outcomes, probabilities must be in (0,1).")
        mu0_nat, mu1_nat = pA, pB
    elif ttype == "normal":
        muA = float(target_params["mean"]["A"]); muB = float(target_params["mean"]["B"])
        sd  = float(target_params.get("std", 1.0))
        if not (sd > 0):
            raise ValueError("For normal outcomes, std must be > 0.")
        mu0_nat, mu1_nat = muA, muB
    else:  # poisson
        shape = float(target_params.get("shape", 2.0))
        scaleA = float(target_params["scale"]["A"])
        scaleB = float(target_params["scale"]["B"])
        lamA = shape * scaleA; lamB = shape * scaleB
        if not (lamA > 0 and lamB > 0):
            raise ValueError("For Poisson outcomes, implied rates must be > 0.")
        mu0_nat, mu1_nat = lamA, lamB

    # Convert to class parameters
    if ttype == "binary":
        alpha_y = _logit(mu0_nat)
        theta = _logit(mu1_nat) - _logit(mu0_nat)
        outcome_type_cls = "binary"
        sigma_y = 1.0
    elif ttype == "normal":
        alpha_y = mu0_nat
        theta = mu1_nat - mu0_nat
        outcome_type_cls = "continuous"
        sigma_y = sd
    else:  # poisson
        alpha_y = float(np.log(mu0_nat))
        theta = float(np.log(mu1_nat / mu0_nat))
        outcome_type_cls = "poisson"
        sigma_y = 1.0

    # Instantiate the unified generator with randomized treatment
    gen = CausalDatasetGenerator(
        theta=theta,
        tau=None,
        beta_y=None,
        beta_d=None,
        g_y=None,
        g_d=None,
        alpha_y=alpha_y,
        alpha_d=_logit(split_f),
        sigma_y=sigma_y,
        outcome_type=outcome_type_cls,
        confounder_specs=confounder_specs,
        k=int(k),
        x_sampler=x_sampler,
        use_copula=False,
        target_d_rate=None,
        u_strength_d=0.0,
        u_strength_y=0.0,
        seed=random_state,
    )

    df = gen.generate(n)

    # Ancillary columns (optional)
    if add_ancillary:
        y = df["y"].to_numpy()
        # Age ~ N(35 + 4*y, 8), int clipped to [18,90]
        age = rng.normal(35 + 4 * y, 8, n).round().clip(18, 90).astype(int)
        # cnt_trans ~ Poisson(max(1.5 + 2*y, 1e-6))
        lam_tx = np.maximum(1.5 + 2 * y, 1e-6)
        cnt_trans = rng.poisson(lam_tx, n).astype(int)
        # Platform: P(Android) = sigmoid(-0.4 + 0.8*y)
        p_android = _sigmoid(-0.4 + 0.8 * y)
        platform_android = rng.binomial(1, np.clip(p_android, 0.0, 1.0), n).astype(int)
        platform_ios = (1 - platform_android).astype(int)
        # Invited friend: normalized y -> [0,1]
        y_min, y_max = float(np.min(y)), float(np.max(y))
        y_norm = (y - y_min) / ((y_max - y_min) + 1e-8)
        invited_friend = rng.binomial(1, np.clip(0.05 + 0.25 * y_norm, 0.0, 1.0), n).astype(int)
        # User IDs
        if deterministic_ids:
            user_ids = _deterministic_ids(rng, n)
        else:
            user_ids = [str(uuid.uuid4()) for _ in range(n)]
        df.insert(0, "user_id", user_ids)
        df["age"] = age
        df["cnt_trans"] = cnt_trans
        df["platform_Android"] = platform_android
        df["platform_iOS"] = platform_ios
        df["invited_friend"] = invited_friend

    return df


def generate_rct_data(
    n: int = 20_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    target_type: str = "binary",
    target_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
) -> pd.DataFrame:
    """
    Backward-compatible alias for generate_rct used in documentation.
    Parameters mirror generate_rct and are forwarded directly.
    """
    return generate_rct(
        n=n,
        split=split,
        random_state=random_state,
        target_type=target_type,
        target_params=target_params,
        confounder_specs=confounder_specs,
        k=k,
        x_sampler=x_sampler,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
    )






@dataclass(slots=True)
class CausalDatasetGenerator:
    """
    Generate synthetic causal inference datasets with controllable confounding,
    treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

    **Data model (high level)**

    - Confounders X ∈ R^k are drawn from user-specified distributions.
    - Binary treatment D is assigned by a logistic model:
        D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
      where f_d(X) = X @ beta_d + g_d(X), and U ~ N(0,1) is an optional unobserved confounder.
    - Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
        outcome_type = "continuous":
            Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε,  ε ~ N(0, sigma_y^2)
        outcome_type = "binary":
            logit P(Y=1|T,X) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
        outcome_type = "poisson":
            log E[Y|T,X]     = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
      where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

    **Returned columns**
      - y: outcome
      - d: binary treatment (0/1)
      - x1..xk (or user-provided names)
      - m   : true propensity P(T=1 | X)
      - g0  : E[Y | X, T=0] on the natural outcome scale
      - g1  : E[Y | X, T=1] on the natural outcome scale
      - cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

    Notes on effect scale:
      - For "continuous", `theta` (or tau(X)) is an additive mean difference.
      - For "binary", tau acts on the *log-odds* scale (log-odds ratio).
      - For "poisson", tau acts on the *log-mean* scale (log incidence-rate ratio).

    Parameters
    ----------
    theta : float, default=1.0
        Constant treatment effect used if `tau` is None.

    tau : callable or None, default=None
        Function tau(X) -> array-like shape (n,) for heterogeneous effects. Ignored if None.

    beta_y : array-like of shape (k,), optional
        Linear coefficients of confounders in the outcome baseline f_y(X). Length must match the expanded X after one-hot encoding.

    beta_d : array-like of shape (k,), optional
        Linear coefficients of confounders in the treatment score f_t(X) (log-odds scale). Length must match the expanded X after one-hot encoding.

    g_y : callable, optional
        Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.

    g_d : callable, optional
        Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.

    alpha_y : float, default=0.0
        Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson).

    alpha_d : float, default=0.0
        Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.

    sigma_y : float, default=1.0
        Std. dev. of the Gaussian noise for continuous outcomes.

    outcome_type : {"continuous","binary","poisson"}, default="continuous"
        Outcome family and link as defined above.

    confounder_specs : list of dict, optional
        Schema for generating confounders. Each spec is one of:
          {"name": str, "dist": "normal",   "mu": float, "sd": float}
          {"name": str, "dist": "uniform",  "a": float,  "b": float}
          {"name": str, "dist": "bernoulli","p": float}
          {"name": str, "dist": "categorical", "categories": list, "probs": list}
        - For "categorical", one-hot encoding is produced for all levels except the first.

    k : int, default=5
        Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).

    x_sampler : callable, optional
        Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs` and `k`.

    target_d_rate : float in (0,1), optional
        If set, calibrates `alpha_d` via bisection so that mean propensity ≈ target_d_rate.

    u_strength_d : float, default=0.0
        Strength of the unobserved confounder U in treatment assignment.

    u_strength_y : float, default=0.0
        Strength of the unobserved confounder U in the outcome.

    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    rng : numpy.random.Generator
        Internal RNG seeded from `seed`.

    Examples
    --------
    >>> gen = CausalDatasetGenerator(
    ...     theta=2.0,
    ...     beta_y=np.array([1.0, -0.5, 0.2]),
    ...     beta_d=np.array([0.8, 1.2, -0.3]),
    ...     target_d_rate=0.35,
    ...     outcome_type="continuous",
    ...     sigma_y=1.0,
    ...     seed=42,
    ...     confounder_specs=[
    ...         {"name":"age", "dist":"normal", "mu":50, "sd":10},
    ...         {"name":"smoker", "dist":"bernoulli", "p":0.3},
    ...         {"name":"bmi", "dist":"normal", "mu":27, "sd":4},
    ...     ])
    >>> df = gen.generate(10_000)
    >>> df.columns
    Index([... 'y','d','age','smoker','bmi','propensity','mu0','mu1','cate'], dtype='object')
    """
    # Core knobs
    theta: float = 1.0                            # constant treatment effect (ATE) if tau is None
    tau: Optional[Callable[[np.ndarray], np.ndarray]] = None  # heterogeneous effect tau(X) if provided

    # Confounder -> outcome/treatment effects
    beta_y: Optional[np.ndarray] = None           # shape (k,)
    beta_d: Optional[np.ndarray] = None           # shape (k,)
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None  # nonlinear baseline outcome f_y(X)
    g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None  # nonlinear treatment score f_d(X)

    # Outcome/treatment intercepts and noise
    alpha_y: float = 0.0
    alpha_d: float = 0.0
    sigma_y: float = 1.0                          # used when outcome_type="continuous"
    outcome_type: str = "continuous"              # "continuous" | "binary" | "poisson"

    # Confounder generation
    confounder_specs: Optional[List[Dict[str, Any]]] = None   # list of {"name","dist",...}
    k: int = 5                                    # used if confounder_specs is None
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None  # custom sampler (n, k, seed)->X
    use_copula: bool = False                      # if True and confounder_specs provided, use Gaussian copula
    copula_corr: Optional[np.ndarray] = None      # correlation matrix for copula (shape dxd where d=len(specs))

    # Practical controls
    target_d_rate: Optional[float] = None         # e.g., 0.3 -> ~30% treated; solves for alpha_d
    u_strength_d: float = 0.0                     # unobserved confounder effect on treatment
    u_strength_y: float = 0.0                     # unobserved confounder effect on outcome
    propensity_sharpness: float = 1.0             # scales the X-driven treatment score to adjust positivity difficulty
    seed: Optional[int] = None

    # Internals (filled post-init)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if self.confounder_specs is not None:
            self.k = len(self.confounder_specs)

    # ---------- Confounder sampling ----------

    def _sample_X(self, n: int) -> Tuple[np.ndarray, List[str]]:
        if self.x_sampler is not None:
            X = self.x_sampler(n, self.k, self.seed)
            names = [f"x{i+1}" for i in range(self.k)]
            return X, names

        if self.confounder_specs is None:
            # Default: independent standard normals
            X = self.rng.normal(size=(n, self.k))
            names = [f"x{i+1}" for i in range(self.k)]
            return X, names

        # If specs are provided and copula is requested, use Gaussian copula
        if getattr(self, "use_copula", False):
            X, names = _gaussian_copula(self.rng, n, self.confounder_specs, getattr(self, "copula_corr", None))
            self.k = X.shape[1]
            return X, names

        cols = []
        names = []
        for spec in self.confounder_specs:
            name = spec.get("name") or f"x{len(names)+1}"
            dist = spec.get("dist", "normal").lower()
            if dist == "normal":
                mu = spec.get("mu", 0.0); sd = spec.get("sd", 1.0)
                col = self.rng.normal(mu, sd, size=n)
            elif dist == "uniform":
                a = spec.get("a", 0.0); b = spec.get("b", 1.0)
                col = self.rng.uniform(a, b, size=n)
            elif dist == "bernoulli":
                p = spec.get("p", 0.5)
                col = self.rng.binomial(1, p, size=n).astype(float)
            elif dist == "categorical":
                categories = list(spec.get("categories", [0,1,2]))
                probs = spec.get("probs", None)
                if probs is not None:
                    p = np.asarray(probs, dtype=float)
                    ps = p / p.sum()
                else:
                    ps = None
                col = self.rng.choice(categories, p=ps, size=n)
                # one-hot encode (except first level)
                rest = categories[1:]
                if len(rest) == 0:
                    cols.append(np.zeros(n, dtype=float))
                    names.append(f"{name}__onlylevel")
                    continue
                for c in rest:
                    cols.append((col == c).astype(float))
                    names.append(f"{name}_{c}")
                continue
            else:
                raise ValueError(f"Unknown dist: {dist}")
            cols.append(col.astype(float))
            names.append(name)
        X = np.column_stack(cols) if cols else np.empty((n,0))
        self.k = X.shape[1]
        return X, names

    # ---------- Helpers ----------

    def _treatment_score(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        # Ensure numeric, finite arrays
        Xf = np.asarray(X, dtype=float)
        # X-driven part of the score
        score_x = np.zeros(Xf.shape[0], dtype=float)
        if self.beta_d is not None:
            bt = np.asarray(self.beta_d, dtype=float)
            if bt.ndim != 1:
                bt = bt.reshape(-1)
            if bt.shape[0] != Xf.shape[1]:
                raise ValueError(f"beta_d shape {bt.shape} is incompatible with X shape {Xf.shape}")
            score_x += np.sum(Xf * bt, axis=1)
        if self.g_d is not None:
            score_x += np.asarray(self.g_d(Xf), dtype=float)
        # Scale sharpness to control positivity difficulty
        s = float(getattr(self, "propensity_sharpness", 1.0))
        lin = s * score_x
        # Add unobserved confounder contribution (unscaled)
        if self.u_strength_d != 0:
            lin += self.u_strength_d * np.asarray(U, dtype=float)
        return lin

    def _outcome_location(self, X: np.ndarray, D: np.ndarray, U: np.ndarray, tau_x: np.ndarray) -> np.ndarray:
        # location on natural scale for continuous; on logit/log scale for binary/poisson
        Xf = np.asarray(X, dtype=float)
        Df = np.asarray(D, dtype=float)
        Uf = np.asarray(U, dtype=float)
        taux = np.asarray(tau_x, dtype=float)
        loc = np.full(Xf.shape[0], float(self.alpha_y), dtype=float)
        if self.beta_y is not None:
            by = np.asarray(self.beta_y, dtype=float)
            if by.ndim != 1:
                by = by.reshape(-1)
            if by.shape[0] != Xf.shape[1]:
                raise ValueError(f"beta_y shape {by.shape} is incompatible with X shape {Xf.shape}")
            loc += np.sum(Xf * by, axis=1)
        if self.g_y is not None:
            loc += np.asarray(self.g_y(Xf), dtype=float)
        if self.u_strength_y != 0:
            loc += self.u_strength_y * Uf
        loc += Df * taux
        return loc

    def _calibrate_alpha_d(self, X: np.ndarray, U: np.ndarray, target: float) -> float:
        """Calibrate alpha_d so mean propensity ~= target using robust bracketing and bisection."""
        lo, hi = -50.0, 50.0
        # Define function whose root we seek
        def f(a: float) -> float:
            return float(_sigmoid(a + self._treatment_score(X, U)).mean() - target)
        flo, fhi = f(lo), f(hi)
        # If the target is not bracketed, try expanding the bracket a few times
        if flo * fhi > 0:
            for scale in (2, 5, 10):
                lo2, hi2 = -scale * 50.0, scale * 50.0
                flo2, fhi2 = f(lo2), f(hi2)
                if flo2 * fhi2 <= 0:
                    lo, hi = lo2, hi2
                    flo, fhi = flo2, fhi2
                    break
            else:
                # Fall back to the closer endpoint
                return lo if abs(flo) < abs(fhi) else hi
        # Standard bisection with tighter tolerance and bounded iterations
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if abs(fm) < 1e-6:
                return mid
            if fm > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    # ---------- Public API ----------

    def generate(self, n: int) -> pd.DataFrame:
        """
        Draw a synthetic dataset of size `n`.

        Parameters
        ----------
        n : int
            Number of observations to simulate.

        Returns
        -------
        pandas.DataFrame
        Columns:
          - y : float
          - d : {0.0, 1.0}
          - <confounder columns> : floats (and one-hot columns for categorical)
          - m   : float in (0,1), true P(T=1 | X)
          - g0  : expected outcome under control on the natural scale
          - g1  : expected outcome under treatment on the natural scale
          - cate: g1 - g0 (conditional treatment effect on the natural scale)
          (Deprecated aliases also emitted for one release: propensity, mu0, mu1)

        Notes
        -----
        - If `target_d_rate` is set, `alpha_d` is internally recalibrated (bisection)
          on the *current draw of X and U*, so repeated calls can yield slightly different
          alpha_d values even with the same seed unless X and U are fixed.
        - For binary and Poisson outcomes, `cate` is reported on the natural scale
          (probability or mean), even though the structural model is specified on
          the log-odds / log-mean scale.
        """
        X, names = self._sample_X(n)
        U = self.rng.normal(size=n)  # unobserved confounder

        # Treatment assignment
        if self.target_d_rate is not None:
            self.alpha_d = self._calibrate_alpha_d(X, U, self.target_d_rate)
        logits_t = self.alpha_d + self._treatment_score(X, U)
        m_obs = _sigmoid(logits_t)  # realized propensity including U

        # Marginal m(x) = E[D|X] (integrate out U if it affects treatment)
        if float(self.u_strength_d) != 0.0:
            gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
            gh_w = gh_w / np.sqrt(np.pi)
            base = self.alpha_d + self._treatment_score(X, np.zeros(n))
            m = np.sum(_sigmoid(base[:, None] + self.u_strength_d * np.sqrt(2.0) * gh_x[None, :]) * gh_w[None, :], axis=1)
        else:
            # Closed form when U doesn't affect D
            base = self.alpha_d + self._treatment_score(X, np.zeros(n))
            m = _sigmoid(base)

        D = self.rng.binomial(1, m_obs).astype(float)

        # Treatment effect (constant or heterogeneous)
        tau_x = (self.tau(X) if self.tau is not None else np.full(n, self.theta)).astype(float)

        # Outcome generation
        loc = self._outcome_location(X, D, U, tau_x)

        if self.outcome_type == "continuous":
            Y = loc + self.rng.normal(0, self.sigma_y, size=n)
        elif self.outcome_type == "binary":
            # logit: logit P(Y=1|D,X) = loc
            p = _sigmoid(loc)
            Y = self.rng.binomial(1, p).astype(float)
        elif self.outcome_type == "poisson":
            # log link: log E[Y|D,X] = loc; guard against overflow on link scale
            loc_c = np.clip(loc, -20, 20)
            lam = np.exp(loc_c)
            Y = self.rng.poisson(lam).astype(float)
        else:
            raise ValueError("outcome_type must be 'continuous', 'binary', or 'poisson'")

        # Compute oracle g0/g1 on the natural scale, excluding U for continuous, and
        # marginalizing over U for binary/poisson when u_strength_y != 0.
        if self.outcome_type == "continuous":
            # Oracle means exclude U (mean-zero unobserved)
            g0 = self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n))
            g1 = self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x)

        elif self.outcome_type == "binary":
            if float(self.u_strength_y) != 0.0:
                gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
                gh_w = gh_w / np.sqrt(np.pi)
                base0 = self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n))
                base1 = self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x)
                Uq = np.sqrt(2.0) * gh_x
                g0 = np.sum(_sigmoid(base0[:, None] + self.u_strength_y * Uq[None, :]) * gh_w[None, :], axis=1)
                g1 = np.sum(_sigmoid(base1[:, None] + self.u_strength_y * Uq[None, :]) * gh_w[None, :], axis=1)
            else:
                g0 = _sigmoid(self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n)))
                g1 = _sigmoid(self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x))

        elif self.outcome_type == "poisson":
            if float(self.u_strength_y) != 0.0:
                gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
                gh_w = gh_w / np.sqrt(np.pi)
                base0 = self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n))
                base1 = self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x)
                Uq = np.sqrt(2.0) * gh_x
                g0 = np.sum(np.exp(np.clip(base0[:, None] + self.u_strength_y * Uq[None, :], -20, 20)) * gh_w[None, :], axis=1)
                g1 = np.sum(np.exp(np.clip(base1[:, None] + self.u_strength_y * Uq[None, :], -20, 20)) * gh_w[None, :], axis=1)
            else:
                g0 = np.exp(np.clip(self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n)), -20, 20))
                g1 = np.exp(np.clip(self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x), -20, 20))
        else:
            raise ValueError("outcome_type must be 'continuous', 'binary', or 'poisson'")

        df = pd.DataFrame({"y": Y, "d": D})
        for j, name in enumerate(names):
            df[name] = X[:, j]
        # Useful ground-truth columns for evaluation (IRM naming)
        df["m"] = m              # marginal E[D|X]
        df["m_obs"] = m_obs      # realized sigmoid(alpha_d + score + u*U)
        df["g0"] = g0
        df["g1"] = g1
        df["cate"] = df["g1"] - df["g0"]
        # Legacy aliases retained for compatibility (no warnings)
        df["propensity"] = df["m_obs"]
        df["mu0"] = df["g0"]
        df["mu1"] = df["g1"]
        return df
    
    def to_causal_data(self, n: int, confounders: Optional[Union[str, List[str]]] = None) -> CausalData:
        """
        Generate a dataset and convert it to a CausalData object.
        
        Parameters
        ----------
        n : int
            Number of observations to simulate.
        confounders : Union[str, List[str]], optional
            Column name(s) to use as confounders. If None, all confounder columns are used.
            
        Returns
        -------
        CausalData
            A CausalData object containing the generated data.
        """
        df = self.generate(n)
        
        # Determine confounders to use
        exclude = {'y', 'd', 'm', 'm_obs', 'g0', 'g1', 'cate', 'propensity', 'mu0', 'mu1', 'user_id'}
        if confounders is None:
            # Keep original column order; exclude outcome/treatment/ground-truth columns and non-numeric
            confounder_cols = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
        elif isinstance(confounders, str):
            confounder_cols = [confounders]
        else:
            confounder_cols = [c for c in confounders if c in df.columns]

        # Create and return CausalData object
        return CausalData(df=df, treatment='d', outcome='y', confounders=confounder_cols)

    def oracle_nuisance(self, num_quad: int = 21):
        """
        Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

        NOTE: Previously documented as (e, m0, m1). The semantics are unchanged;
        only naming now matches IRM conventions.

        Behavior:
        - If both u_strength_d != 0 and u_strength_y != 0, DML is not identified; raise ValueError.
        - m(x): If u_strength_d == 0, closed form sigmoid(alpha_d + s * f_t(x)).
                 If u_strength_d != 0, approximate E_U[sigmoid(alpha_d + s * f_t(x) + u_strength_d * U)]
                 using Gauss–Hermite quadrature with `num_quad` nodes, where U ~ N(0,1).
        - g0/g1(x): closed-form mappings on the natural outcome scale evaluated with U omitted.

        Parameters
        ----------
        num_quad : int, default=21
            Number of Gauss–Hermite nodes for marginalizing over U in m(x) when u_strength_d != 0.
        """
        # Disallow unobserved confounding for DML when U affects both T and Y
        if (getattr(self, "u_strength_d", 0.0) != 0) and (getattr(self, "u_strength_y", 0.0) != 0):
            raise ValueError(
                "DML identification fails when U affects both T and Y. "
                "Use instruments (PLIV-DML) or set one of u_strength_* to 0."
            )

        # Precompute GH nodes/weights normalized for N(0,1)
        gh_x, gh_w = np.polynomial.hermite.hermgauss(int(num_quad))
        gh_w = gh_w / np.sqrt(np.pi)

        def m_of_x(x_row: np.ndarray) -> float:
            x = np.asarray(x_row, dtype=float).reshape(1, -1)
            # Base score without U contribution, matching _treatment_score(X, U=0)
            base = float(self.alpha_d)
            base += float(self._treatment_score(x, np.zeros(1, dtype=float))[0])
            ut = float(getattr(self, "u_strength_d", 0.0))
            if ut == 0.0:
                return float(_sigmoid(base))
            # Integrate over U ~ N(0,1) using Gauss–Hermite: U = sqrt(2) * gh_x
            z = base + ut * np.sqrt(2.0) * gh_x
            return float(np.sum(_sigmoid(z) * gh_w))

        def g_of_x_d(x_row: np.ndarray, d: int) -> float:
            x = np.asarray(x_row, dtype=float).reshape(1, -1)
            if self.tau is None:
                tau_val = float(self.theta)
            else:
                tau_val = float(np.asarray(self.tau(x), dtype=float).reshape(-1)[0])
            loc = float(self.alpha_y)
            if self.beta_y is not None:
                loc += float(np.sum(x * np.asarray(self.beta_y, dtype=float), axis=1))
            if self.g_y is not None:
                loc += float(np.asarray(self.g_y(x), dtype=float))
            loc += float(d) * tau_val

            if self.outcome_type == "continuous":
                return float(loc)

            uy = float(getattr(self, "u_strength_y", 0.0))
            if self.outcome_type == "binary":
                if uy == 0.0:
                    return float(_sigmoid(loc))
                z = loc + uy * np.sqrt(2.0) * gh_x
                return float(np.sum(_sigmoid(z) * gh_w))
            if self.outcome_type == "poisson":
                if uy == 0.0:
                    return float(np.exp(np.clip(loc, -20.0, 20.0)))
                z = np.clip(loc + uy * np.sqrt(2.0) * gh_x, -20.0, 20.0)
                return float(np.sum(np.exp(z) * gh_w))
            raise ValueError("outcome_type must be 'continuous','binary','poisson'.")

        return (lambda x: m_of_x(x),
                lambda x: g_of_x_d(x, 0),
                lambda x: g_of_x_d(x, 1))
