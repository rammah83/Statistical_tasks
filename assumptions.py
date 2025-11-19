from math import isfinite
import numpy as np
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.mixture import GaussianMixture


def _clean_data(data):
    x = np.array(data).flatten()
    return x[~np.isnan(x)]


def is_normal(data, method="shapiro", alpha=0.05, return_stats=False):
    # Convert to numpy array and remove NaNs
    x = _clean_data(data)
    match method:
        case "shapiro":
            result = stats.shapiro(x)
        case "jarque_bera":
            result = stats.jarque_bera(x)
        case "normaltest":
            result = stats.normaltest(x, nan_policy="omit")
        case "kolmogorov":
            result = stats.kstest(x, "norm", args=(x.mean(), x.std()))
        case "anderson":
            res = stats.anderson(x, dist="norm")
            A2 = float(res.statistic)
            sl = np.asarray(
                res.significance_level, dtype=float
            )  # e.g., [15., 10., 5., 2.5, 1.]
            cv = np.asarray(res.critical_values, dtype=float)
            target = alpha * 100.0
            idx = int(np.argmin(np.abs(sl - target)))
            crit = float(cv[idx])
            alpha_used = float(sl[idx] / 100.0)
            result = {
                    "A2": A2,
                    "crit": crit,
                    "alpha_used": alpha_used,
                    "pvalue": np.nan,
                    "normal": bool(A2 < crit),
                }
            return result['normal'] if not return_stats else result
        case _:
            result = stats.shapiro(x)
    if return_stats:
        return result
    pvalue = result.pvalue
    return bool(isfinite(pvalue) and pvalue > alpha)


def recommand_normality_test(n: int, percent_outlier: float = 0.0):
    """
    Recommend a normality test based on:
      - n: sample size (int, >=3)
      - r: ratio of outliers in [0, 1] (float), e.g., from a MAD-based detector.

    Heuristics used
    ----------------
    - If outliers are present, prefer tail-sensitive (Anderson-Darling) at small n,
      and skew/kurtosis-based (D'Agostino K^2) once n is comfortably large.
    - With negligible outliers, Shapiro-Wilk is generally most powerful for small
      to moderate n; for very large n switch to Jarque-Bera for speed/tractability.

    Implementation details / guardrails
    -----------------------------------
    - SciPy's normaltest (D'Agostino) is best used for n >= 20.
    - SciPy's shapiro warns that p-values may be inaccurate for n > 5000.
    - Anderson-Darling in SciPy returns critical values (not a p-value).
    """
    # ----- validation -----
    # refuse bools (since bool is subclass of int), non-integers, or tiny n
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n must be a plain integer (e.g., 37).")
    if n < 3:
        raise ValueError("n must be >= 3.")

    # coerce r to float and validate finite & bounded
    try:
        percent_outlier = float(percent_outlier)
    except Exception as _:
        raise ValueError("r must be a float in [0, 1].")
    if not isfinite(percent_outlier):
        raise ValueError("r must be finite.")
    # small numeric drift safety: clamp to [0,1]
    if percent_outlier < 0.0:
        percent_outlier = 0.0
    if percent_outlier > 100.0:
        percent_outlier = 100.0

    # ----- thresholds (tweak if your domain suggests different cutoffs) -----
    MANY_OUTLIERS = 10  # >=10% flagged by MAD
    SOME_OUTLIERS = 1  # 1%–10%

    MIN_N_K2 = 20  # safety for D'Agostino's K^2
    SMALL_N_AD = 50  # below this with outliers -> AD
    MED_N_AD = 200  # "many outliers" still AD if <200
    LARGE_N_SWITCH = 5000  # above this -> consider JB or AD (no p) for tails

    def pack(test, alias, why, scipy_call, notes=None, min_n=None, pvalue=True):
        return {
            "test": test,
            "alias": alias,  # common SciPy function name
            "why": why,  # one-line rationale
            "scipy_call": scipy_call,  # copy-paste hint
            "supports_pvalue": pvalue,  # AD in SciPy has no p-value
            "min_n": min_n,
            "notes": notes or "",
        }

    # ----- decision logic -----
    if n >= LARGE_N_SWITCH:
        return pack(
            test="Jarque-Bera",
            alias="jarque_bera",
            why="Very large n with outliers; JB is fast and asymptotically efficient for skew/kurt deviations.",
            scipy_call="from scipy import stats\njb, p = stats.jarque_bera(x)",
        )
    elif percent_outlier >= MANY_OUTLIERS:
        if n < MED_N_AD:
            return pack(
                test="Anderson-Darling (normal)",
                alias="anderson",
                why="Many outliers and small/medium n; AD is tail-sensitive.",
                scipy_call="from scipy import stats\nstat, crit, sig = stats.anderson(x, dist='norm')",
                notes="Interpret using critical values; SciPy does not return a p-value.",
                pvalue=False,
            )
        else:
            # prefer K^2 if n is safe; otherwise fall back to AD
            return pack(
                test="D'Agostino K^2",
                alias="normaltest",
                why="Outliers inflate skew/kurtosis; K^2 has good power at medium/large n.",
                scipy_call="from scipy import stats\nk2, p = stats.normaltest(x, nan_policy='omit')",
                min_n=MIN_N_K2,
            )
    elif percent_outlier >= SOME_OUTLIERS:
        if n < SMALL_N_AD:
            return pack(
                test="Anderson-Darling (normal)",
                alias="anderson",
                why="n too small for K^2; AD emphasizes tails.",
                scipy_call="from scipy import stats\nstat, crit, sig = stats.anderson(x, dist='norm')",
                pvalue=False,
            )
        else:
            return pack(
                test="D'Agostino K^2",
                alias="normaltest",
                why="Moderate n with some outliers; K^2 targets skewness/kurtosis.",
                scipy_call="from scipy import stats\nk2, p = stats.normaltest(x, nan_policy='omit')",
                min_n=MIN_N_K2,
            )
    else:
        # negligible outliers
        # Shapiro-Wilk up to ~5000 (SciPy warns on p for larger n)
        return pack(
            test="Shapiro-Wilk",
            alias="shapiro",
            why="Clean data; Shapiro is typically most powerful for small–moderate n.",
            scipy_call="from scipy import stats\nw, p = stats.shapiro(x)",
            notes="For n > 5000 the p-value accuracy may degrade in SciPy.",
        )


def recommand_outliers_test(data, normality, need_formal=False):
    method1, method2, method3 = None, None, None
    x = _clean_data(data)
    n = len(x)
    if n < 30:
        if normality:
            method1 = "Dixon"
            method2 = "Grubb"
        else:
            method1 = "IQR"
            method2 = "Percentile"

    elif 30 <= n < 300:
        if normality:
            if need_formal:
                method1 = "Grubb" if "ONE_OUTLIER" else "Generalized ESD"
            else:
                method1 = "z-score"
        elif abs(stats.skew(x)) >= 1:
            method1 = "Modified z-score"
            method2 = "Median +/- MAD"
            method3 = "Percentile:1-99"

        elif abs(stats.skew(x)) < 0.5:
            method1 = "IQR"
            method2 = "Percentile:1-99"
        else:
            method1 = "Modified z-score"
            method2 = "IQR"
            method3 = "Percentile:5-95"
    else:
        method1 = "IQR"
        method2 = None
    methods = filter(lambda m: m is not None, [method1, method2, method3])
    return tuple(methods)


def check_multimodality(data, max_components=5, alpha=0.01, verbose=True):
    """
    Check for multimodality in a single numeric variable.

    Parameters:
    -----------
    x : array-like
            1D array or Series of numeric data
    max_components : int, default=5
            Maximum number of Gaussian components to test in GMM
    alpha : float, default=0.05
            Significance level for statistical tests
    plot : bool, default=True
            Whether to generate diagnostic plots

    Returns:
    --------
    dict : Dictionary containing test results and diagnostics
    """

    x = _clean_data(data)
    X = x.reshape(-1, 1)

    # GAUSSIAN MIXTURE MODEL (GMM) - BIC
    bics = []
    aics = []
    models = []

    results = {}
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(X)
        models.append(gmm)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    best_k_bic = np.argmin(bics) + 1
    best_k_aic = np.argmin(aics) + 1
    if verbose:
        print()
        print("=" * 8, "GAUSSIAN MIXTURE MODEL (BIC COMPARISON)", "=" * 8)
        print("-" * 50)
        print(f"{'Components':<15} {'BIC':<15} {'AIC':>10}")
        print("-" * 50)
    for k in range(1, max_components + 1):
        bic_marker = " ← BEST" if k == best_k_bic else ""
        aic_marker = " ← BEST" if k == best_k_aic else ""
        if verbose:
            print(
                f"{k:<12} {bics[k - 1]:.2f}{bic_marker:<15} {aics[k - 1]:.2f}{aic_marker:<15}"
            )

    if best_k_bic > 1:
        results["is_unimodal"] = False
        if best_k_aic > 1:
            if verbose:
                print("-" * 50)
                print(f"✗ STRONG evidence of MULTIMODALITY ({best_k_bic} components)")
                print("=" * 56, "\n")
        else:
            if verbose:
                print("-" * 50)
                print(
                    f"✗ MODERATE evidence of MULTIMODALITY ({best_k_bic} components).Check histogram"
                )
                print("=" * 56, "\n")
    else:
        results["is_unimodal"] = True
        if verbose:
            print("-" * 50)
            print("✓ Data appear UNIMODAL (1 component)")
            print("=" * 56, "\n")

    results["best_k_bic"] = best_k_bic
    results["best_k_aic"] = best_k_aic
    results["bics"] = bics
    results["aics"] = aics
    results["best_gmm"] = models[best_k_bic - 1]

    return results["is_unimodal"]


if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.normal(10, 2, 59)
    # Example 2: Bimodal data
    data = np.concatenate(
        [
            np.random.normal(loc=70, scale=5, size=200),
            np.random.normal(loc=30, scale=5, size=300),
        ]
    )

    n = len(data)
    method = recommand_normality_test(n, percent_outlier=2.0)['alias']
    normality = is_normal(data, method=method, return_stats=False)
    unimodality = check_multimodality(data, verbose=False)
    outlier_test_method = recommand_outliers_test(data, normality)
    print(f"Normality test '{method}':\n\t{is_normal(data, method=method, return_stats=True)}")
    print(f"Is Normal: {is_normal(data, method=method, return_stats=False)}")
    print(f"Outliers Methods : {outlier_test_method}")
    print(f"Unimodality: {unimodality}")
