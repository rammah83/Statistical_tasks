import numpy as np
import scipy.stats as stats
import pingouin as pg
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from assumptions import _clean_data


def recommand_outliers_test(
    x, normality, unimodal=True, need_formal=False, one_outlier=False
):
    """
    Recommend an outlier test based on:
    - Sample size (n)
    - Normality of data
    - Need for formal testing
    """
    dixon = "Dixon"
    grubb = "Grubb"
    iqr = "IQR"
    percentile = "Percentile"
    z_score = "z-score"
    modified_z_score = "Modified z-score"
    generalised_esd = "Generalized ESD"
    median_kmad = "Median +/-k MAD"
    percentile_1_99 = "Percentile:1-99"
    percentile_5_95 = "Percentile:5-95"
    lof = "LOF"
    dbscan = "DBSCAN"
    isolation_forest = "Isolation Forest"

    n = len(x)
    method1, method2, method3 = None, None, None
    if not unimodal:
        if n <= 10_000:
            method1 = lof
            method2 = dbscan
            print("Multimodal data")
        else:
            method1 = isolation_forest
            print("Multimodal data")
    else:
        if n < 30:
            if normality:
                if one_outlier:
                    method1 = dixon
                    method2 = grubb
                else:
                    method1 = generalised_esd
            else:
                method1 = iqr
                method2 = percentile

        elif 30 <= n < 300:
            if normality:
                if need_formal:
                    method1 = grubb if one_outlier else generalised_esd
                else:
                    method1 = z_score
                    method2 = iqr
            elif abs(stats.skew(x)) >= 1:
                method1 = modified_z_score
                method2 = median_kmad
                method3 = percentile_1_99

            elif abs(stats.skew(x)) < 0.5:
                # If data is heavy-tailed (Leptokurtic), standard deviation is not reliable
                if stats.kurtosis(x) > 1.0:
                    method1 = modified_z_score
                    method2 = median_kmad
                else:
                    method1 = iqr
                    method2 = percentile_1_99
            else:
                method1 = modified_z_score
                method2 = iqr
                method3 = percentile_5_95
        else:
            method1 = iqr
            method2 = None
    methods = filter(lambda m: m is not None, [method1, method2, method3])
    return tuple(methods)


def _format_result(X, is_outlier):
    """Helper to format the output dictionary."""
    n_outliers = np.sum(is_outlier)
    return {
        "n_outliers": int(n_outliers),
        "ratio_outliers": n_outliers / len(X) if len(X) > 0 else 0.0,
        "outlier_mask": is_outlier,
        "outlier_values": X[is_outlier],
    }


def iqr_method(X, k=1.5):
    """Interquartile Range Method."""
    if len(X) == 0:
        return _format_result(X, np.array([], dtype=bool))

    q1 = np.percentile(X, 25)
    q3 = np.percentile(X, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    is_outlier = (X < lower_bound) | (X > upper_bound)
    return _format_result(X, is_outlier)


def z_score_method(X, threshold=2):  #  (2, 3) are common choices
    """Standard Z-Score Method."""
    if len(X) < 2:
        return _format_result(X, np.zeros(len(X), dtype=bool))

    mean = np.mean(X)
    std = np.std(X)

    if std == 0:
        z_scores = np.zeros(len(X))
    else:
        z_scores = (X - mean) / std  # alternative : stats.zscore
    is_outlier = np.abs(z_scores) > threshold
    return _format_result(X, is_outlier)


def modified_z_score_method(
    X, q_threshold=0.975
):  # Common thresholds: 2.24 (95%), 3.5 (99%)
    """
    Modified Z-Score using Median and MAD.
    Outlier detection based on the MAD Median Rule, often used in robust statistics.
    This method identifies outliers as points that deviate significantly from the median,
    scaled by the Median Absolute Deviation (MAD).
    it's same as using pg.madmedianrule(X)
    """
    if len(X) == 0:
        return _format_result(X, np.array([], dtype=bool))

    median = np.median(X)
    # Median Absolute Deviation
    mad = np.median(np.abs(X - median))

    if mad == 0:
        # If MAD is 0, any value different from median is technically an outlier
        # in this framework, or we return 0 to avoid division by zero.
        # Here we treat non-median values as outliers if MAD is 0.
        is_outlier = X != median
    else:
        # 0.6745 is the consistency constant for normal distribution
        modified_z = stats.norm.ppf(3 / 4.0) * (X - median) / mad
        chi2_value = stats.chi2.ppf(q=q_threshold, df=1)
        threshold = np.sqrt(chi2_value)
        is_outlier = np.abs(modified_z) > threshold
    return _format_result(X, is_outlier)


def percentile_method(
    X, lower_percentile=1, upper_percentile=99
):  # 1, 5 and 95, 99 are common choices
    """Percentile Method."""
    if len(X) == 0:
        return _format_result(X, np.array([], dtype=bool))

    lower = np.percentile(X, lower_percentile)
    upper = np.percentile(X, upper_percentile)

    is_outlier = (X < lower) | (X > upper)
    return _format_result(X, is_outlier)


def mad_method(X, k=3):  # 3 is a common choice
    """Median +/- k * MAD Method."""
    if len(X) == 0:
        return _format_result(X, np.array([], dtype=bool))

    median = np.median(X)
    mad = np.median(np.abs(X - median))

    lower = median - k * mad
    upper = median + k * mad

    is_outlier = (X < lower) | (X > upper)
    return _format_result(X, is_outlier)


def grubb_test(X, alpha=0.05):
    """
    Grubbs' Test for outliers (detects one outlier at a time).
    This implementation runs iteratively to find multiple outliers.
    """
    X_temp = X.copy()
    outlier_indices = []

    while len(X_temp) > 2:
        mean = np.mean(X_temp)
        std = np.std(X_temp, ddof=1)
        if std == 0:
            break

        abs_diff = np.abs(X_temp - mean)
        max_dev_idx = np.argmax(abs_diff)
        G = abs_diff[max_dev_idx] / std

        # Critical value calculation
        N = len(X_temp)
        t_crit = stats.t.ppf(1 - alpha / (2 * N), N - 2)
        G_crit = ((N - 1) / np.sqrt(N)) * np.sqrt(t_crit**2 / (N - 2 + t_crit**2))

        if G > G_crit:
            # Find the value in the original array to mark it
            val = X_temp[max_dev_idx]
            # We need to be careful with duplicate values, this is a simple approach
            # In a robust system, we'd track original indices.
            # Here we just remove the value from temp and mark it.
            X_temp = np.delete(X_temp, max_dev_idx)
        else:
            break

    # Reconstruct mask based on values remaining
    # Note: This simple reconstruction assumes unique values or removes all instances of the outlier value
    # For strict index tracking, a different approach is needed, but this fits "simple".
    is_outlier = ~np.isin(X, X_temp)
    return _format_result(X, is_outlier)


def generalized_esd_test(X, max_outliers=None, alpha=0.05):
    """Generalized Extreme Studentized Deviate (ESD) Test."""
    n = len(X)
    if n < 3:
        return _format_result(X, np.zeros(n, dtype=bool))

    # Default max_outliers to 10% of data if not specified
    if max_outliers is None:
        max_outliers = max(int(n * 0.1), 1)

    # We must track indices to mask correctly
    indices = np.arange(n)
    current_indices = indices.copy()
    current_data = X.copy()

    outliers_found = []

    # Calculate critical values and stats for k iterations
    for i in range(max_outliers):
        if len(current_data) < 3:
            break

        mean = np.mean(current_data)
        std = np.std(current_data, ddof=1)
        if std == 0:
            break

        residuals = np.abs(current_data - mean)
        max_idx_in_current = np.argmax(residuals)
        R = residuals[max_idx_in_current] / std

        # Critical Value
        curr_n = len(current_data)
        p = (
            1 - alpha / (2 * (n - i))
        )  # Note: uses original N in denominator usually, or curr_n depending on variant
        # Standard definition uses n - i - 1 degrees of freedom logic
        t_crit = stats.t.ppf(p, curr_n - 2)
        lambda_i = ((curr_n - 1) * t_crit) / np.sqrt((curr_n - 2 + t_crit**2) * curr_n)

        if R > lambda_i:
            outliers_found.append(current_indices[max_idx_in_current])

        # Always remove the max observation for the next iteration
        current_data = np.delete(current_data, max_idx_in_current)
        current_indices = np.delete(current_indices, max_idx_in_current)

    is_outlier = np.zeros(n, dtype=bool)
    is_outlier[outliers_found] = True
    return _format_result(X, is_outlier)


def dixon_test(X):
    """
    Dixon's Q Test.
    Note: Strictly valid for small sample sizes (3 <= n <= 30).
    This is a simplified implementation for the most common Q-test (r10).
    """
    n = len(X)
    if n < 3:
        return _format_result(X, np.zeros(n, dtype=bool))

    # Critical values for Q (alpha=0.05) for N=3 to 30
    # Source: Rorabacher (1991) or standard tables
    q_crit_95 = {
        3: 0.941,
        4: 0.765,
        5: 0.642,
        6: 0.560,
        7: 0.507,
        8: 0.468,
        9: 0.437,
        10: 0.412,
        11: 0.392,
        12: 0.376,
        13: 0.361,
        14: 0.349,
        15: 0.338,
        16: 0.329,
        17: 0.320,
        18: 0.313,
        19: 0.306,
        20: 0.300,
        21: 0.295,
        22: 0.290,
        23: 0.285,
        24: 0.281,
        25: 0.277,
        26: 0.273,
        27: 0.269,
        28: 0.266,
        29: 0.263,
        30: 0.260,
    }

    # If N is too large for Dixon, fallback to Z-score or return empty
    if n > 30:
        # Fallback or warning
        return z_score_method(X)

    sorted_indices = np.argsort(X)
    sorted_X = X[sorted_indices]

    gap_low = sorted_X[1] - sorted_X[0]
    gap_high = sorted_X[-1] - sorted_X[-2]
    data_range = sorted_X[-1] - sorted_X[0]

    if data_range == 0:
        return _format_result(X, np.zeros(n, dtype=bool))

    is_outlier = np.zeros(n, dtype=bool)
    critical_value = q_crit_95.get(n, 0.260)  # Default to 30's value if missing

    # Check low end
    Q_low = gap_low / data_range
    if Q_low > critical_value:
        is_outlier[sorted_indices[0]] = True

    # Check high end
    Q_high = gap_high / data_range
    if Q_high > critical_value:
        is_outlier[sorted_indices[-1]] = True

    return _format_result(X, is_outlier)


def lof_test(X, n_neighbors=20):
    """Local Outlier Factor (LOF) Method."""
    if len(X) <= n_neighbors:
        # Adjust neighbors if data is smaller than default
        n_neighbors = max(1, len(X) - 1)

    # LOF expects 2D array
    X_reshaped = X.reshape(-1, 1)

    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    # fit_predict returns -1 for outliers, 1 for inliers
    y_pred = clf.fit_predict(X_reshaped)

    is_outlier = y_pred == -1
    return _format_result(X, is_outlier)


def isoforest_test(X, contamination="auto", random_state=42):
    """Isolation Forest Method."""
    if len(X) == 0:
        return _format_result(X, np.array([], dtype=bool))

    X_reshaped = X.reshape(-1, 1)

    clf = IsolationForest(
        n_estimators=500, contamination=contamination, random_state=random_state
    )
    y_pred = clf.fit_predict(X_reshaped)

    is_outlier = y_pred == -1
    return _format_result(X, is_outlier)


def dbscan_test(X, eps=0.5, min_samples=5):
    """
    DBSCAN Method.
    Note: DBSCAN is sensitive to scale. We standardize the data internally
    so the default eps=0.5 works reasonably well for normal-ish distributions.
    """
    if len(X) == 0:
        return _format_result(X, np.array([], dtype=bool))

    X_reshaped = X.reshape(-1, 1)

    # Standardize data to make eps parameter meaningful across different scales
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    # -1 indicates noise (outliers) in DBSCAN
    is_outlier = labels == -1
    return _format_result(X, is_outlier)


# Updated Wrapper Function
def outliers_info(data, method, return_outliers=False):
    X = _clean_data(data)
    match method:
        # --- Statistical Methods --- for global outliers
        case "Dixon":
            results = dixon_test(X)
        case "Grubb":
            results = grubb_test(X)
        case "IQR":
            results = iqr_method(X)
        case "Percentile1-99":
            results = percentile_method(X, lower_percentile=1, upper_percentile=99)
        case "Percentile5-95":
            results = percentile_method(X, lower_percentile=5, upper_percentile=95)
        case "z-score":
            results = z_score_method(X)
        case "Modified z-score":
            results = modified_z_score_method(X)
        case "MADmedianrule":
            results = modified_z_score_method(X)
        case "generalized ESD":
            results = generalized_esd_test(X)
        case "Median +/- k MAD":
            results = mad_method(X)

        # --- Machine Learning Methods --- for local and large datasets
        case "LOF":
            results = lof_test(X)
        case "DBSCAN":
            results = dbscan_test(X)
        case "Isolation Forest":
            results = isoforest_test(X)

        case _:
            raise ValueError(f"Unknown outlier detection method: {method}")
    if not return_outliers:
        results.pop("outlier_values", None)
        results.pop("outlier_mask", None)
    return results


if __name__ == "__main__":
    np.random.seed(42)
    data1 = np.random.normal(10, 2, 10000)
    # Example 2: Bimodal data
    data2 = np.concatenate(
        [
            np.random.normal(loc=70, scale=10, size=500),
            np.random.normal(loc=30, scale=10, size=300),
        ]
    )
    data3 = [
        -1.09,
        1.0,
        0.28,
        -1.51,
        -0.58,
        6.61,
        -2.43,
        -0.43,
        -5.24,
        0.19,
    ]

    robustness_methods = [
        "Dixon",
        "Grubb",
        "IQR",
        "Percentile1-99",
        "Percentile5-95",
        "z-score",
        "Modified z-score",
        "MADmedianrule",
        "generalized ESD",
        "Median +/- k MAD",
        "LOF",
        "DBSCAN",
        "Isolation Forest",
    ]
    print("=" * 56)
    print(f"{'Method/Test':<25} {'N outliers':>5}")
    print("-" * 50)
    for method in robustness_methods[:]:
        robustness_result = outliers_info(data1, method=method)
        print(f"{method:<25} {robustness_result['n_outliers'] * 1:>5}")
    print("-" * 50)
    print("=" * 56)
