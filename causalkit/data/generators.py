"""
Data generation utilities for causal inference tasks.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Callable, Any

from causalkit.data.causaldata import CausalData


def _sigmoid(z):
    """
    Numerically stable sigmoid: 1 / (1 + exp(-z))
    Handles large positive/negative z without overflow warnings.
    
    Parameters
    ----------
    z : array-like
        Input values
        
    Returns
    -------
    array-like
        Sigmoid of input values
    """
    z = np.asarray(z, dtype=float)
    # Use a stable formulation to avoid overflow for large |z|
    # expit(z) = 0.5 * (1 + tanh(z/2)) is stable, but tanh can be slower.
    # Implement a piecewise stable computation directly.
    out = np.empty_like(z, dtype=float)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    # For positive z: 1 / (1 + exp(-z))
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    # For negative z: exp(z) / (1 + exp(z)) to avoid exp(-z) overflow
    ez = np.exp(z[neg_mask])
    out[neg_mask] = ez / (1.0 + ez)
    return out


def generate_rct_data(
    n_users: int = 20_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    target_type: str = "binary",                         # {"binary", "normal", "nonnormal"}
    target_params: Optional[Dict] = None,                # distribution specifics (see docstring)
) -> pd.DataFrame:
    """
    Create synthetic RCT data with
        • three possible outcome distributions (binary, continuous-normal,
          continuous-non-normal),
        • five covariates   ─ age, cnt_trans, platform_Android, platform_iOS,
                              invited_friend
          that are generated *conditional on the outcome* but remain independent
          of the treatment group (groups are perfectly randomised).

    Parameters
    ----------
    n_users
        Total number of users in the dataset.
    split
        Proportion of users in the treatment group. For example, 0.5 means 50% of users
        will be in the treatment group and 50% in the control group.
    random_state
        Seed for reproducibility.
    target_type
        Target distribution: "binary", "normal", or "nonnormal".
    target_params
        Distribution parameters.  If *None* sensible defaults are used:
            binary   : {"p": {"A": 0.10, "B": 0.12}}
            normal   : {"mean": {"A": 0.00, "B": 0.20}, "std": 1.0}
            nonnormal: {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}}

    Returns
    -------
    pd.DataFrame
        Columns: user_id, treatment, outcome, age, cnt_trans,
                 platform_Android, platform_iOS, invited_friend.
    """
    # ------------------------------------------------------------------ #
    # RNG & default parameters
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(random_state)

    # Calculate number of users in each group
    n_treatment = int(n_users * split)
    n_control = n_users - n_treatment

    n_samples = {"B": n_treatment, "A": n_control}  # B for treatment, A for control

    if target_params is None:
        if target_type == "binary":
            target_params = {"p": {"A": 0.10, "B": 0.12}}
        elif target_type == "normal":
            target_params = {"mean": {"A": 0.00, "B": 0.20}, "std": 1.0}
        elif target_type == "nonnormal":
            target_params = {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}}
        else:
            raise ValueError("target_type must be 'binary', 'normal', or 'nonnormal'.")

    # ------------------------------------------------------------------ #
    # Data generation loop per group
    # ------------------------------------------------------------------ #
    frames = []

    for grp, n in n_samples.items():
        # -------- outcome ------------------------------------------------
        if target_type == "binary":
            target = rng.binomial(1, target_params["p"][grp], n)
        elif target_type == "normal":
            mu, sigma = target_params["mean"][grp], target_params["std"]
            target = rng.normal(mu, sigma, n)
        else:  # non-normal (Gamma)
            k, theta = target_params["shape"], target_params["scale"][grp]
            target = rng.gamma(k, theta, n)

        # -------- covariates (all depend on `outcome`, never on `grp`) ----
        age = rng.normal(35 + 4 * target, 8, n).round().clip(18, 90).astype(int)

        cnt_trans = rng.poisson(1.5 + 2 * target, n).astype(int)

        # Android probability via simple logistic model
        p_android = 1 / (1 + np.exp(-(-0.4 + 0.8 * target)))
        platform_android = rng.binomial(1, p_android, n)
        platform_ios = 1 - platform_android

        # Invited a friend → more likely for larger outcome values
        # Normalise outcome into [0,1] to keep probabilities valid
        t_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
        invited_friend = rng.binomial(1, 0.05 + 0.25 * t_norm, n)

        # Generate UUID user_ids
        user_ids = [str(uuid.uuid4()) for _ in range(n)]

        # Convert group to treatment (1 for treatment, 0 for control)
        treatment = 1 if grp == "B" else 0

        # -------- assemble ---------------------------------------------
        df_grp = pd.DataFrame(
            {
                "user_id": user_ids,
                "treatment": [treatment] * n,
                "outcome": target,
                "age": age,
                "cnt_trans": cnt_trans,
                "platform_Android": platform_android,
                "platform_iOS": platform_ios,
                "invited_friend": invited_friend,
            }
        )
        frames.append(df_grp)

    return pd.concat(frames, ignore_index=True)


def generate_obs_data(
    n_users: int = 20_000,
    split: float = 0.1,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Create synthetic observational data where treatment assignment is influenced by covariates.

    Parameters
    ----------
    n_users
        Total number of users in the dataset.
    split
        Proportion of users in the treatment group. For example, 0.1 means 10% of users
        will be in the treatment group and 90% in the control group.
    random_state
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: user_id, treatment, age, income, education, gender, region
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Generate user_ids
    user_ids = [str(uuid.uuid4()) for _ in range(n_users)]

    # Generate covariates
    age = rng.normal(35, 10, n_users).round().clip(18, 80).astype(int)
    income = rng.normal(50000, 15000, n_users).round().clip(10000, 150000).astype(int)
    education = rng.choice(['high_school', 'bachelor', 'master', 'phd'], n_users, p=[0.3, 0.4, 0.2, 0.1])
    gender = rng.choice(['male', 'female'], n_users)
    region = rng.choice(['north', 'south', 'east', 'west', 'central'], n_users)

    # Generate propensity scores (probability of treatment) based on covariates
    # Higher income and education level increase probability of treatment
    income_norm = (income - income.min()) / (income.max() - income.min())
    education_score = np.where(education == 'high_school', 0.1,
                      np.where(education == 'bachelor', 0.3,
                      np.where(education == 'master', 0.5, 0.7)))

    # Calculate base propensity
    propensity = 0.2 * income_norm + 0.3 * education_score + 0.1 * (age / 80)

    # Adjust to match desired split ratio
    propensity = propensity * (split / propensity.mean())
    propensity = np.clip(propensity, 0, 1)

    # Assign treatment based on propensity
    treatment = rng.binomial(1, propensity)

    # Ensure exact split ratio
    current_split = treatment.mean()
    if current_split != split:
        # Adjust by randomly flipping some assignments
        if current_split < split:
            # Need to increase treatment
            n_to_flip = int((split - current_split) * n_users)
            control_indices = np.where(treatment == 0)[0]
            flip_indices = rng.choice(control_indices, n_to_flip, replace=False)
            treatment[flip_indices] = 1
        else:
            # Need to decrease treatment
            n_to_flip = int((current_split - split) * n_users)
            treatment_indices = np.where(treatment == 1)[0]
            flip_indices = rng.choice(treatment_indices, n_to_flip, replace=False)
            treatment[flip_indices] = 0

    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'treatment': treatment,
        'age': age,
        'income': income,
        'education': education,
        'gender': gender,
        'region': region
    })

    return df


@dataclass
class CausalDatasetGenerator:
    """
    Generate synthetic causal inference datasets with controllable confounding,
    treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

    **Data model (high level)**

    - Confounders X ∈ R^k are drawn from user-specified distributions.
    - Binary treatment T is assigned by a logistic model:
        T ~ Bernoulli( sigmoid(alpha_t + f_t(X) + u_strength_t * U) ),
      where f_t(X) = X @ beta_t + g_t(X), and U ~ N(0,1) is an optional unobserved confounder.
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
      - t: binary treatment (0/1)
      - x1..xk (or user-provided names)
      - propensity: P(T=1 | X) used to draw T (ground truth)
      - mu0: E[Y | T=0, X] on the natural outcome scale
      - mu1: E[Y | T=1, X] on the natural outcome scale
      - cate: mu1 - mu0 (conditional average treatment effect on the natural outcome scale)

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
        Linear coefficients of confounders in the outcome baseline f_y(X).

    beta_t : array-like of shape (k,), optional
        Linear coefficients of confounders in the treatment score f_t(X) (log-odds scale).

    g_y : callable, optional
        Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.

    g_t : callable, optional
        Nonlinear/additive function g_t(X) -> (n,) added to the treatment score.

    alpha_y : float, default=0.0
        Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson).

    alpha_t : float, default=0.0
        Treatment intercept (log-odds). If `target_t_rate` is set, `alpha_t` is auto-calibrated.

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

    target_t_rate : float in (0,1), optional
        If set, calibrates `alpha_t` via bisection so that mean propensity ≈ outcome.

    u_strength_t : float, default=0.0
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
    ...     beta_t=np.array([0.8, 1.2, -0.3]),
    ...     target_t_rate=0.35,
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
    Index([... 'y','t','age','smoker','bmi','propensity','mu0','mu1','cate'], dtype='object')
    """
    # Core knobs
    theta: float = 1.0                            # constant treatment effect (ATE) if tau is None
    tau: Optional[Callable[[np.ndarray], np.ndarray]] = None  # heterogeneous effect tau(X) if provided

    # Confounder -> outcome/treatment effects
    beta_y: Optional[np.ndarray] = None           # shape (k,)
    beta_t: Optional[np.ndarray] = None           # shape (k,)
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None  # nonlinear baseline outcome f_y(X)
    g_t: Optional[Callable[[np.ndarray], np.ndarray]] = None  # nonlinear treatment score f_t(X)

    # Outcome/treatment intercepts and noise
    alpha_y: float = 0.0
    alpha_t: float = 0.0
    sigma_y: float = 1.0                          # used when outcome_type="continuous"
    outcome_type: str = "continuous"              # "continuous" | "binary" | "poisson"

    # Confounder generation
    confounder_specs: Optional[List[Dict[str, Any]]] = None   # list of {"name","dist",...}
    k: int = 5                                    # used if confounder_specs is None
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None  # custom sampler (n, k, seed)->X

    # Practical controls
    target_t_rate: Optional[float] = None         # e.g., 0.3 -> ~30% treated; solves for alpha_t
    u_strength_t: float = 0.0                     # unobserved confounder effect on treatment
    u_strength_y: float = 0.0                     # unobserved confounder effect on outcome
    seed: Optional[int] = None

    # Internals (filled post-init)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if self.confounder_specs is not None:
            self.k = len(self.confounder_specs)

    # ---------- Confounder sampling ----------

    def _sample_X(self, n: int) -> (np.ndarray, List[str]):
        if self.x_sampler is not None:
            X = self.x_sampler(n, self.k, self.seed)
            names = [f"x{i+1}" for i in range(self.k)]
            return X, names

        if self.confounder_specs is None:
            # Default: independent standard normals
            X = self.rng.normal(size=(n, self.k))
            names = [f"x{i+1}" for i in range(self.k)]
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
                categories = spec.get("categories", [0,1,2])
                probs = spec.get("probs", None)
                col = self.rng.choice(categories, p=probs, size=n)
                # one-hot encode (except first level)
                oh = [ (col == c).astype(float) for c in categories[1:] ]
                if not oh:
                    oh = [np.zeros(n)]
                for j, c in enumerate(categories[1:]):
                    cols.append(oh[j])
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
        lin = np.zeros(Xf.shape[0], dtype=float)
        if self.beta_t is not None:
            bt = np.asarray(self.beta_t, dtype=float)
            if bt.ndim != 1:
                bt = bt.reshape(-1)
            if bt.shape[0] != Xf.shape[1]:
                raise ValueError(f"beta_t shape {bt.shape} is incompatible with X shape {Xf.shape}")
            lin += np.sum(Xf * bt, axis=1)
        if self.g_t is not None:
            lin += np.asarray(self.g_t(Xf), dtype=float)
        if self.u_strength_t != 0:
            lin += self.u_strength_t * np.asarray(U, dtype=float)
        return lin

    def _outcome_location(self, X: np.ndarray, T: np.ndarray, U: np.ndarray, tau_x: np.ndarray) -> np.ndarray:
        # location on natural scale for continuous; on logit/log scale for binary/poisson
        Xf = np.asarray(X, dtype=float)
        Tf = np.asarray(T, dtype=float)
        Uf = np.asarray(U, dtype=float)
        taux = np.asarray(tau_x, dtype=float)
        loc = float(self.alpha_y)
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
        loc += Tf * taux
        return loc

    def _calibrate_alpha_t(self, X: np.ndarray, U: np.ndarray, target: float) -> float:
        # Bisection on alpha_t so that mean propensity ~ outcome
        lo, hi = -50.0, 50.0  # Use a wider range to handle extreme cases
        for _ in range(100):  # Increase iterations for better precision
            mid = 0.5*(lo+hi)
            p = _sigmoid(mid + self._treatment_score(X, U))
            m = p.mean()
            if m > target:
                hi = mid
            else:
                lo = mid
            # Early stopping if we're close enough
            if abs(m - target) < 0.001:
                break
        return 0.5*(lo+hi)

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
              - t : {0.0, 1.0}
              - <confounder columns> : floats (and one-hot columns for categorical)
              - propensity : float in (0,1), true P(T=1 | X)
              - mu0 : expected outcome under control on the natural scale
              - mu1 : expected outcome under treatment on the natural scale
              - cate : mu1 - mu0 (conditional treatment effect on the natural scale)

        Notes
        -----
        - If `target_t_rate` is set, `alpha_t` is internally recalibrated (bisection)
          on the *current draw of X and U*, so repeated calls can yield slightly different
          alpha_t values even with the same seed unless X and U are fixed.
        - For binary and Poisson outcomes, `cate` is reported on the natural scale
          (probability or mean), even though the structural model is specified on
          the log-odds / log-mean scale.
        """
        X, names = self._sample_X(n)
        U = self.rng.normal(size=n)  # unobserved confounder

        # Treatment assignment
        if self.target_t_rate is not None:
            self.alpha_t = self._calibrate_alpha_t(X, U, self.target_t_rate)
        logits_t = self.alpha_t + self._treatment_score(X, U)
        propensity = _sigmoid(logits_t)
        T = self.rng.binomial(1, propensity).astype(float)

        # Treatment effect (constant or heterogeneous)
        tau_x = (self.tau(X) if self.tau is not None else np.full(n, self.theta)).astype(float)

        # Outcome generation
        loc = self._outcome_location(X, T, U, tau_x)

        if self.outcome_type == "continuous":
            Y = loc + self.rng.normal(0, self.sigma_y, size=n)
            mu0 = self._outcome_location(X, np.zeros(n), U, np.zeros(n))
            mu1 = self._outcome_location(X, np.ones(n),  U, tau_x)
        elif self.outcome_type == "binary":
            # logit: logit P(Y=1|T,X) = loc
            p = _sigmoid(loc)
            Y = self.rng.binomial(1, p).astype(float)
            mu0 = _sigmoid(self._outcome_location(X, np.zeros(n), U, np.zeros(n)))
            mu1 = _sigmoid(self._outcome_location(X, np.ones(n),  U, tau_x))
        elif self.outcome_type == "poisson":
            # log link: log E[Y|T,X] = loc
            lam = np.exp(loc)
            Y = self.rng.poisson(lam).astype(float)
            mu0 = np.exp(self._outcome_location(X, np.zeros(n), U, np.zeros(n)))
            mu1 = np.exp(self._outcome_location(X, np.ones(n),  U, tau_x))
        else:
            raise ValueError("outcome_type must be 'continuous', 'binary', or 'poisson'")

        df = pd.DataFrame({"y": Y, "t": T})
        for j, name in enumerate(names):
            df[name] = X[:, j]
        # Useful ground-truth columns for evaluation
        df["propensity"] = propensity
        df["mu0"] = mu0
        df["mu1"] = mu1
        df["cate"] = mu1 - mu0
        return df
    
    def to_causal_data(self, n: int, cofounders: Optional[Union[str, List[str]]] = None) -> CausalData:
        """
        Generate a dataset and convert it to a CausalData object.
        
        Parameters
        ----------
        n : int
            Number of observations to simulate.
        cofounders : Union[str, List[str]], optional
            Column name(s) to use as cofounders. If None, all confounder columns are used.
            
        Returns
        -------
        CausalData
            A CausalData object containing the generated data.
        """
        df = self.generate(n)
        
        # Determine cofounders to use
        if cofounders is None:
            # Use all confounder columns (exclude y, t, propensity, mu0, mu1, cate)
            all_cols = set(df.columns)
            exclude_cols = {'y', 't', 'propensity', 'mu0', 'mu1', 'cate'}
            cofounder_cols = list(all_cols - exclude_cols)
        else:
            cofounder_cols = cofounders
            
        # Create and return CausalData object
        return CausalData(df=df, treatment='t', outcome='y', cofounders=cofounder_cols)
