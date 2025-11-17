import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.mixture import GaussianMixture


def _clean_data(data):
    x = np.array(data).flatten()
    return x[~np.isnan(x)]


def is_normal(data, alpha=0.05, return_stats=False):
    # Convert to numpy array and remove NaNs
    x = _clean_data(data)
    result = stats.shapiro(x)
    pvalue = result.pvalue
    if return_stats:
        return result
    return pvalue > alpha


def check_multimodality(data, max_components=5, alpha=0.05, verbose=True):
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
                print(
                    f"✗ STRONG evidence of MULTIMODALITY ({best_k_bic} components)"
                )
                print("=" * 56, '\n')
        else:
            if verbose:
                print("-" * 50)
                print(
                    f"✗ MODERATE evidence of MULTIMODALITY ({best_k_bic} components).Check histogram"
                )
                print("=" * 56, '\n')
    else:
        results["is_unimodal"] = True
        if verbose:
            print("-" * 50)
            print("✓ Data appear UNIMODAL (1 component)")
            print("=" * 56, '\n')

    results["best_k_bic"] = best_k_bic
    results["best_k_aic"] = best_k_aic
    results["bics"] = bics
    results["aics"] = aics
    results["best_gmm"] = models[best_k_bic - 1]
    
    return results['is_unimodal']


def global_outliers(data, need_formal=False):
    method1, method2, method3 = None, None, None
    x = _clean_data(data)
    n = len(x)
    normality = is_normal(x, return_stats=False)
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
    methods = filter(lambda m: m is not None , [method1, method2, method3])
    return tuple(methods)


if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.normal(10, 2, 5000)
    # Example 2: Bimodal data
    bimodal_data = np.concatenate(
        [
            np.random.normal(loc=30, scale=5, size=500),
            np.random.normal(loc=70, scale=5, size=500),
        ]
    )
    print(f"Is Normal: {is_normal(data, return_stats=False)}")
    print(f"Outliers Methods : {global_outliers(data)}")
    results = check_multimodality(data, verbose=False)
    print(f"Unimodality: {results}")
