“Overlap/positivity” is about **possibility**, not what actually happened. In formal terms we need
$0<P(A=1\mid X=x)<1$ for all covariate values $x$ we analyze.
In an RCT, before randomization, **every enrolled unit could have been assigned** to treatment or control (often with a known probability like 0.5), regardless of their confounders. The fact that *after* randomization one person ended up in control and “can’t get the treatment” is just the realized draw; it doesn’t violate positivity.

Why we still look at “overlap” or balance in RCTs:

* To check randomization quality (finite samples can have chance imbalances).
* To improve **precision** via covariate adjustment (not to fix confounding—randomization already did that).
* For **heterogeneous effects** (CATE/GATE), you want enough treated and control observations within each subgroup; otherwise estimates get noisy (a power issue, not an assumption failure).

When positivity can fail even in trials:

* **Deterministic assignment within a stratum** you condition on (e.g., some sites/clusters only treated or only control; if you include site fixed effects for such single-arm sites, you’ve conditioned on a variable that perfectly predicts treatment).
* **Adaptive randomization** that drives probabilities near 0 or 1 for some $X$.
* **Post-treatment conditioning** (defining subgroups by variables affected by treatment).
* **Noncompliance** if you analyze “as-treated” rather than **intention-to-treat** (ITT). ITT preserves positivity; as-treated becomes observational and needs the usual assumptions.

Practical takeaways:

* For ITT in a standard individual-level RCT, positivity holds by design; you don’t need to model a propensity score (you already know it).
* Still report covariate balance (e.g., standardized mean differences) and adjust for strong prognostic baselines to gain efficiency.
* For subgroup/CATE work, ensure each subgroup actually has both arms represented; if not, reinterpret the subgroup or pool/regularize.
