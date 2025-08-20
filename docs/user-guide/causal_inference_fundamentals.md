# Introduction to Causal Inference

## Why causality?

We constantly reason about “what caused what.” If an ad runs and sales rise, was it the ad—or something else? Causal inference turns this intuition into a careful, testable framework so we can answer policy and product questions with data.

## The core idea

For any unit (a person, store, app session, etc.) we imagine two potential outcomes:

* $Y(1)$: what would happen **with** the treatment
* $Y(0)$: what would happen **without** the treatment

We only ever observe one of them. This “missing counterfactual” is the fundamental challenge of causal inference. Everything we do is about approximating the missing outcome in a principled way.

---

## Notation

* **Units:** $i=1,\dots,n$
* **Treatment:** $T_i\in\{0,1\}$ (can be multi-valued or continuous)
* **Outcome:** $Y_i$ (what we observe)
* **Covariates / confounders:** $X_i$ (pre-treatment features that affect both treatment and outcome)

Example dataset:

| Unit | Treatment $T$ | Outcome $Y$ | Confounder\_1 | Confounder\_2 |
| ---- | ------------- | ----------- | ------------- | ------------- |
| 1    | 1             | 100         | 1             | 13            |
| 2    | 0             | 90          | 0             | 14            |
| 3    | 1             | 110         | 1             | 51            |
| 4    | 1             | 97          | 1             | 63            |
| 5    | 0             | 80          | 0             | 34            |
| 6    | 0             | 85          | 0             | 53            |

---

## What we aim to estimate (estimands)

### ITE / CATE: individual or subgroup effect

* **Individual Treatment Effect (ITE):** $\text{ITE}_i = Y_i(1)-Y_i(0)$ (unobservable for a single unit)
* **Conditional Average Treatment Effect (CATE):**
  $\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]$
  Use when you need personalization, uplift targeting, or to study heterogeneity.

### ATT: effect on the treated

$\text{ATT}=\mathbb{E}[Y(1)-Y(0)\mid T=1]$
Use to answer “Did it work **for those who actually received it**?”—e.g., after a selective rollout.

### ATE: overall average effect

$\text{ATE}=\mathbb{E}[Y(1)-Y(0)]$
Use for headline impact or policy choices that affect the whole eligible population.

---

## When are these estimands credible?

In observational data, we typically rely on two key identification conditions:

1. **Unconfoundedness (selection on observables):** given $X$, treatment is as good as random.
   Formally, $(Y(0),Y(1)) \perp T \mid X$.
2. **Overlap (positivity):** each unit had a nonzero chance to receive either treatment: $0<\Pr(T=1\mid X)<1$.

These assumptions won’t be true by magic; we design models and diagnostics to make them as plausible as possible.

---