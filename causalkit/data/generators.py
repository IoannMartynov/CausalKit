"""
Data generation utilities for causal inference tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple


import numpy as np
import pandas as pd
import uuid
from typing import Dict, Optional, Union, List, Tuple


def generate_rct_data(
    n_users: int = 20_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    target_type: str = "binary",                         # {"binary", "normal", "nonnormal"}
    target_params: Optional[Dict] = None,                # distribution specifics (see docstring)
) -> pd.DataFrame:
    """
    Create synthetic RCT data with
        • three possible target distributions (binary, continuous-normal,
          continuous-non-normal),
        • five covariates   ─ age, cnt_trans, platform_Android, platform_iOS,
                              invited_friend
          that are generated *conditional on the target* but remain independent
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
        Columns: user_id, treatment, target, age, cnt_trans,
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
        # -------- target ------------------------------------------------
        if target_type == "binary":
            target = rng.binomial(1, target_params["p"][grp], n)
        elif target_type == "normal":
            mu, sigma = target_params["mean"][grp], target_params["std"]
            target = rng.normal(mu, sigma, n)
        else:  # non-normal (Gamma)
            k, theta = target_params["shape"], target_params["scale"][grp]
            target = rng.gamma(k, theta, n)

        # -------- covariates (all depend on `target`, never on `grp`) ----
        age = rng.normal(35 + 4 * target, 8, n).round().clip(18, 90).astype(int)

        cnt_trans = rng.poisson(1.5 + 2 * target, n).astype(int)

        # Android probability via simple logistic model
        p_android = 1 / (1 + np.exp(-(-0.4 + 0.8 * target)))
        platform_android = rng.binomial(1, p_android, n)
        platform_ios = 1 - platform_android

        # Invited a friend → more likely for larger target values
        # Normalise target into [0,1] to keep probabilities valid
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
                "target": target,
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
