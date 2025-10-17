## Notation

* Observed data: $(Y_i, D_i, X_i)_{i=1}^n$.

  * $Y\in\mathbb R$ (or $\{0,1\}$) is the outcome.
  * $D\in\{0,1\}$ is a binary treatment.
  * $X\in\mathbb R^p$ are observed confounders.

* Potential outcomes: $Y(1), Y(0)$.

* Targets:

  * **ATE**: $\theta_{\mathrm{ATE}}=\mathbb E\big[Y(1)-Y(0)\big]$.
  * **ATTE**: $\theta_{\mathrm{ATTE}}=\mathbb E\big[Y(1)-Y(0)\mid D=1\big]$.

**Assumptions:**
* Unconfoundedness $(Y(1),Y(0))\perp D\mid X$
* Positivity $0<\Pr(D=1\mid X)<1$
* SUTVA
* Regularity for cross-fitting and ML.

## Scores

$$
u_1 = Y-\hat g_1,\quad
u_0 = Y-\hat g_0,\quad
h_1 = \frac{D}{\hat m},\quad
h_0 = \frac{1-D}{1-\hat m},\quad
\hat p_1=\mathbb{E}_n[D].
$$

#### ATE score (AIPW/DR)

$$
\psi_a^{\mathrm{ATE}}=-1,\qquad
\psi_b^{\mathrm{ATE}}=(\hat g_1-\hat g_0)+u_1h_1-u_0h_0.
$$

**Estimator and per-unit influence value:**

$$
\hat\theta_{\mathrm{ATE}}=\mathbb{E}n\big[\psi_b^{\mathrm{ATE}}\big],\qquad
\hat\psi_i^{\mathrm{ATE}}=\psi_{b,i}^{\mathrm{ATE}}-\hat\theta_{\mathrm{ATE}}.
$$

#### ATTE score

Let $(p_1=\mathbb{E}[D])$ (estimate with $(\hat p_1)$). Then

$$
\psi_a^{\mathrm{ATTE}}=-\frac{D}{p_1},\qquad
\psi_b^{\mathrm{ATTE}}=\frac{D}{p_1}(\hat g_1-\hat g_0)
+\frac{D}{p_1}(Y-\hat g_1)
-\frac{1-D}{p_1}\frac{\hat m}{1-\hat m}(Y-\hat g_0).
$$

**Estimator and per-unit influence value:**

$$
\hat\theta_{\mathrm{ATTE}}=\mathbb{E}n\big[\psi_b^{\mathrm{ATTE}}\big],\qquad
\hat\psi_i^{\mathrm{ATTE}}=\psi_{b,i}^{\mathrm{ATTE}}+\psi_{a,i}^{\mathrm{ATTE}}\hat\theta_{\mathrm{ATTE}}
\quad(\text{since }\mathbb{E}_n[D/\hat p_1]=1).
$$

#### Variance & CI (both ATE and ATTE)

For either target, using its corresponding $(\hat\psi_i)$:

$$
\widehat{\mathrm{Var}}(\hat\theta)=\frac{1}{n^2}\sum_{i=1}^n \hat\psi_i^2,\qquad
\mathrm{se}=\sqrt{\widehat{\mathrm{Var}}(\hat\theta)},\qquad
\mathrm{CI}_{1-\alpha}=\hat\theta\pm z_{1-\alpha/2}se
$$