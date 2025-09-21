# Math notation for EDA, IRM, and refutation diagnostics (CausalKit)

This note summarizes the core notation and formulas used by CausalKit’s EDA helpers, the IRM estimator, and refutation diagnostics. The estimator and scores follow the **Double/Debiased Machine Learning (DoubleML)** formulation; we implement them on top of `CausalData`. Throughout, $\mathbb E_n[\cdot]$ denotes the sample mean.

## 1. Variables and parameters

* Observed data: $(Y_i, D_i, X_i)_{i=1}^n$.

  * $Y\in\mathbb R$ (or $\{0,1\}$) is the outcome.
  * $D\in\{0,1\}$ is a binary treatment.
  * $X\in\mathbb R^p$ are observed confounders.

* Potential outcomes: $Y(1), Y(0)$.

* Targets:

  * **ATE**: $\theta_{\mathrm{ATE}}=\mathbb E\big[Y(1)-Y(0)\big]$.
  * **ATT (a.k.a. ATET/ATTE)**: $\theta_{\mathrm{ATT}}=\mathbb E\big[Y(1)-Y(0)\mid D=1\big]$.

**Assumptions (standard):**
Unconfoundedness $(Y(1),Y(0))\perp D\mid X$; positivity $0<\Pr(D=1\mid X)<1$ a.s.; SUTVA; and regularity for cross-fitting and ML.

## 2. Nuisance functions (IRM)

* Propensity: $m(x)=\Pr(D=1\mid X=x)$.
* Outcome regressions: $g_1(x)=\mathbb E[Y\mid X=x,D=1],\quad g_0(x)=\mathbb E[Y\mid X=x,D=0]$.
* Cross-fitted predictions are denoted $\hat m,\hat g_1,\hat g_0$ (length $n$).
* **Clipping:** $\hat m\gets \mathrm{clip}(\hat m,\varepsilon,1-\varepsilon)$ with user trimming threshold $\varepsilon>0$.

**Binary outcomes.** If $Y$ is binary and the outcome learner is a classifier with `predict_proba`, CausalKit uses the class-1 probability as $\hat g_d(x)$. For numerical stability you may also clip $\hat g_d$ into $[\delta,1-\delta]$ with a tiny $\delta$ (e.g., $10^{-6}$).

## 3. Scores, EIFs, and estimators

Let $u_1=Y-\hat g_1,\; u_0=Y-\hat g_0$, and

$$
h_1=\frac{D}{\hat m},\qquad h_0=\frac{1-D}{1-\hat m}.
$$

(Optionally, CausalKit can **normalize** $h_1,h_0$ to have sample mean 1; this is a Hájek-style, second-order variance tweak. If you normalize in estimation, also use the normalized influence values in the variance formula.)

### 3.1 ATE (AIPW/DR)

Score decomposition as $\mathbb E[\psi_a\,\theta+\psi_b]=0$:

$$
\psi_a=-1,\qquad
\psi_b=(\hat g_1-\hat g_0)+u_1 h_1 - u_0 h_0.
$$

Estimator and influence function:

$$
\hat\theta_{\text{ATE}}=\mathbb E_n[\psi_b],\qquad
\hat\psi_i=\psi_{b,i}-\hat\theta_{\text{ATE}}.
$$

**Efficient influence function (truth $(g_d,m,\theta)$):**

$$
\psi_{\text{ATE}}(W)=\big(g_1-g_0-\theta\big)+\frac{D}{m}(Y-g_1)-\frac{1-D}{1-m}(Y-g_0).
$$

**Compact identity (useful in code).** With $g_D(X)=Dg_1(X)+(1-D)g_0(X)$ and $r=Y-g_D(X)$,

$$
\psi_{\text{ATE}}=(g_1-g_0-\theta)+\Big(\tfrac{D}{m}-\tfrac{1-D}{1-m}\Big)\,r.
$$

### 3.2 ATT (a.k.a. ATET/ATTE)

Let $p_1=\mathbb E[D]$ (estimated by the sample mean $\hat p_1=\mathbb E_n[D]$). Define the control reweighting factor $r_0(X)=\frac{\hat m}{1-\hat m}$. Then

$$
\psi_a=-\frac{D}{p_1},\qquad
\psi_b=\frac{D}{p_1}(\hat g_1-\hat g_0)
+\frac{D}{p_1}(Y-\hat g_1)
-\frac{1-D}{p_1}\frac{\hat m}{1-\hat m}(Y-\hat g_0).
$$

Estimator and influence function:

$$
\hat\theta_{\text{ATT}}=\mathbb E_n[\psi_b],\qquad
\hat\psi_i=\psi_{b,i}+\psi_{a,i}\hat\theta_{\text{ATT}}.
$$

Because $\mathbb E_n[D/\hat p_1]=1$, this choice centers $\hat\psi$ at zero in-sample.
If you use fold-specific $\hat p_{1,k}$, either center $\hat\psi$ per fold or re-express everything with the global $\hat p_1$.

> **Equivalent residual-weight form.** With $\bar w=\hat m/p_1$:
>
> $$
> \bar w(u_1 h_1-u_0 h_0)=\frac{D}{p_1}(Y-\hat g_1)-\frac{1-D}{p_1}\frac{\hat m}{1-\hat m}(Y-\hat g_0).
> $$

**Efficient influence function (truth $(g_d,m,\theta)$):**

$$
\psi_{\text{ATT}}(W)=\frac{D}{p_1}\big(g_1-g_0-\theta\big)
+\frac{D}{p_1}(Y-g_1)
-\frac{1-D}{p_1}\frac{m}{1-m}(Y-g_0).
$$

**Weight-sum identity (diagnostic).**

$$
\mathbb{E}\!\left[\frac{1-D}{p_1}\frac{m}{1-m}\right]=\frac{1}{p_1}\,\mathbb{E}\!\left[(1-D)\frac{m}{1-m}\right]=\frac{1}{p_1}\,\mathbb{E}[m]=\frac{p_1}{p_1}=1.
$$

Equivalently, **without** the $1/p_1$ factor:

$$
\mathbb{E}\!\left[(1-D)\frac{m}{1-m}\right]=\mathbb{E}[D]=p_1.
$$

So in-sample diagnostics can be phrased as either:

* $\sum_{i}(1-d_i)\tfrac{\hat m_i}{1-\hat m_i}\approx \sum_i d_i$ (raw factors), **or**
* $\sum_{i}(1-d_i)\tfrac{\hat m_i}{(1-\hat m_i)p_1}\approx n$ (ATT weights match treated weights $\sum_i d_i/p_1 = n$).


### 3.3 Orthogonality (Neyman)

For $\eta=(g_0,g_1,m)$,

$$
\frac{\partial}{\partial t}\,\mathbb E\big[\psi(W;\theta_0,\eta_0+t\,h)\big]\Big|_{t=0}=0
$$

for rich directions $h$. CausalKit provides OOS moment checks and numerical derivative diagnostics to assess this.

**Useful partial derivatives (ATE):**

$$
\frac{\partial\psi}{\partial g_1}=1-\frac{D}{m},\quad
\frac{\partial\psi}{\partial g_0}=-1+\frac{1-D}{1-m},\quad
\frac{\partial\psi}{\partial m}=-\frac{D}{m^2}(Y-g_1)-\frac{1-D}{(1-m)^2}(Y-g_0),
$$

whose conditional expectations given $X$ vanish at truth.

**Useful partial derivatives (ATT):**

$$
\frac{\partial\psi}{\partial g_1}=\tfrac{D}{p_1}-\tfrac{D}{p_1}=0,\quad
\frac{\partial\psi}{\partial g_0}=-\tfrac{D}{p_1}+\tfrac{1-D}{p_1}\tfrac{m}{1-m},\quad
\frac{\partial\psi}{\partial m}=-\frac{1-D}{p_1}\frac{1}{(1-m)^2}(Y-g_0),
$$

and each has zero conditional mean at truth.

## 4. Estimation (cross-fitting)

* Split into $K$ folds (stratified by $D$).
* On each train fold, fit learners for $g_0,g_1,m$; predict on the held-out fold to build cross-fitted $\hat g_0,\hat g_1,\hat m$.
* Compute $\hat\theta$ as above. Let $\hat\psi_i$ be the estimated influence function values (computed OOS per fold).

**Variance and CI (single parameter):**

$$
\widehat{\mathrm{Var}}(\hat\theta)=\frac{1}{n}\Big(\frac{1}{n}\sum_{i=1}^n \hat\psi_i^2\Big)
=\frac{1}{n^2}\sum_{i=1}^n \hat\psi_i^2,\quad
\mathrm{se}=\sqrt{\widehat{\mathrm{Var}}(\hat\theta)},\quad
\text{CI}_{1-\alpha}=\hat\theta\pm z_{1-\alpha/2}\,\mathrm{se}.
$$

**Hájek normalization (optional, ATE).** Replace $h_1=D/\hat m$ with $h_1^\star=\dfrac{D/\hat m}{\mathbb E_n[D/\hat m]}$ and similarly $h_0^\star$. This preserves asymptotics (orthogonality) and can reduce finite-sample variance; it slightly alters the finite-sample IF, so use the normalized $\hat\psi$ in variance calculations. For ATT, it’s common to normalize control weights so their sum matches the count of treated (the diagnostic above already ensures this in expectation).

## 5. Positivity (overlap) & trimming

* EDA reports the distribution of $\hat m$ and the share near 0 or 1.
* **Clipping** $\hat m$ stabilizes the IPW terms. Under positivity (true $m$ bounded away from 0 and 1) it is **asymptotically innocuous** for ATE/ATT but may introduce **small finite-sample bias**; **hard trimming** (dropping units by a propensity threshold) *changes the target population* and should be interpreted accordingly.

## 6. Refutation diagnostics

### 6.1 OOS moment checks

Compute $\hat\psi$ on each test fold using fold-specific nuisances. Verify the empirical mean of $\hat\psi$ (and conditional moments against simple basis functions of $X$) is close to zero.
APIs: `refute_irm_orthogonality`, `oos_moment_check`, `influence_summary`.

### 6.2 Orthogonality derivatives

Numerically evaluate derivatives of the moment condition w\.r.t. perturbations in $(g_0,g_1,m)$ at $\hat\theta$. Small magnitudes support Neyman orthogonality in practice.
APIs: `orthogonality_derivatives` (ATE) and ATT variants. (Derivatives summarized in §3.3.)

### 6.3 Sensitivity (heuristic bias bounds)

Following the DoubleML structure, CausalKit exposes a **Cauchy–Schwarz–style** worst-case bias bound aligned with the score:

* Outcome noise scale (pooled):

$$
\hat\sigma^2=\mathbb E_n\big[(Y-\hat g_D(X))^2\big],\quad
\hat g_D(X)=D\hat g_1(X)+(1-D)\hat g_0(X).
$$

* Score-weight norm (ATE):

$$
\hat v^2_{\text{ATE}}=\mathbb E_n\Big[\Big(\tfrac{D}{\hat m}-\tfrac{1-D}{1-\hat m}\Big)^2\Big].
$$

(Equivalently $\mathbb E_n[(D/\hat m)^2]+\mathbb E_n[((1-D)/(1-\hat m))^2]$ since $D(1-D)=0$ pointwise.)

* Score-weight norm (ATT):

$$
\hat v^2_{\text{ATT}}=\mathbb E_n\!\big[(D/p_1)^2\big]+\mathbb E_n\!\big[\big((1-D)\,\hat m/\{p_1(1-\hat m)\}\big)^2\big].
$$

For user-chosen sensitivity multipliers $c_y,c_d\ge0$ and correlation cap $\rho\in[0,1]$,

$$
\mathrm{max\_bias}(\rho)\;\le\;\rho\,\sqrt{(c_y\,\hat\sigma^2)\,(c_d\,\hat v^2)}.
$$

**Optional refinement (tighter but still heuristic).** Split outcome variance by arm:

$$
\hat\sigma_1^2=\mathbb E_n[(Y-\hat g_1)^2\mid D=1],\quad
\hat\sigma_0^2=\mathbb E_n[(Y-\hat g_0)^2\mid D=0],
$$

and use

$$
\sqrt{\hat\sigma_1^2\,\mathbb E_n[(D/\hat m)^2]+\hat\sigma_0^2\,\mathbb E_n[((1-D)/(1-\hat m))^2]}
$$

in place of $\sqrt{\hat\sigma^2\,\hat v^2}$ for ATE.
APIs: `IRM.sensitivity_analysis`, `refutation/sensitivity.py`. Bounds remain **heuristic** (not identified).

## 7. Implementation guardrails

* Abort/warn if $\hat p_1\in\{0,1\}$ (ATT undefined / division by zero).
* Enforce small clipping on $\hat m$ and (for classifiers) on $\hat g_d$ to prevent exploding residual-weight products.
* Stratify folds by $D$ when cross-fitting.
* If using fold-specific denominators (e.g., $\hat p_{1,k}$), ensure fold-wise centering of $\hat\psi$ or re-express with global $\hat p_1$.

