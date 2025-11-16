import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture


_SIZE = 250
_MEAN = 10
_STD = 2
np.random.seed(42)
data = np.random.normal(_MEAN, _STD, _SIZE)
n = len(data)

def _clean_data(data):
	x = np.array(data).flatten()
	return x[~np.isnan(x)]


def is_normal(data, alpha=.05):
	# Convert to numpy array and remove NaNs
	x = _clean_data(data)
	results = stats.shapiro(x)
	pvalue = results.pvalue
	print(f"p-value : {pvalue :.3g}")
	return results.pvalue > alpha


def global_outliers(data, need_formal=False):
	method1, method2, method3 = None, None, None
	x = _clean_data(data)
	if n < 30:
		if is_normal(x):
			method1 = 'Dixon'
			method2 = 'Grubb'
		else:
			method1 = 'IQR'
			method2 = 'Percentile'

	elif 30 <= n < 300:
		if is_normal(x):
			if need_formal:
				method1 = 'Grubb' if 'ONE_OUTLIER' else 'Generalized ESD'
			else:
				method1 = 'z-score'
		elif abs(stats.skew(x)) >= 1:
			method1 = 'Modified z-score'
			method2 = 'Median +/- MAD'
			method3 = 'Percentile:1-99'

		elif abs(stats.skew(x))< 0.5:
			method1 = 'IQR'
			method2 = 'Percentile:1-99'
		else:
			method1 = 'Modified z-score'
			method2 = 'IQR'
			method3 = 'Percentile:5-95'
	else:
		method1 = 'IQR'
		method2 = None
	return method1, method2, method3

def check_multimodality(data, max_components=5, alpha=0.05):
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
		n = len(x)
		
		# GAUSSIAN MIXTURE MODEL (GMM) - BIC
		
		X = x.reshape(-1, 1)
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
	
		print("-" * 60)
		print("2. GAUSSIAN MIXTURE MODEL (BIC COMPARISON)")
		print("-" * 60)
		print(f"{'Components':<12} {'BIC':<15} {'AIC':<15}")
		print("-" * 42)
		for k in range(1, max_components + 1):
			bic_marker = " ← BEST" if k == best_k_bic else ""
			aic_marker = " ← BEST" if k == best_k_aic else ""
			print(f"{k:<12} {bics[k-1]:.2f}{bic_marker:<15} {aics[k-1]:.2f}{aic_marker:<15}")
		
		print()
		print(f"Best number of components (BIC): {best_k_bic}")
		print(f"Best number of components (AIC): {best_k_aic}")
		
		if best_k_bic > 1:
			print(f"✓ GMM suggests MULTIMODALITY ({best_k_bic} components)")
			results['gmm_conclusion'] = 'multimodal'
		else:
			print("✗ GMM suggests UNIMODALITY (1 component)")
			results['gmm_conclusion'] = 'unimodal'
		
		results['best_k_bic'] = best_k_bic
		results['best_k_aic'] = best_k_aic
		results['bics'] = bics
		results['aics'] = aics
		results['best_gmm'] = models[best_k_bic - 1]
		
		print()
		
		# ========================================
		# 3. OVERALL CONCLUSION
		# ========================================
		print("=" * 60)
		print("OVERALL CONCLUSION : ", end='\t')

		
		evidence_count = 0
		if results.get('gmm_conclusion') == 'multimodal':
			evidence_count += 1
		
		if evidence_count >= 2:
			overall = "STRONG evidence of MULTIMODALITY"
		elif evidence_count == 1:
			overall = "MODERATE evidence of multimodality (check plots)"
		else:
			overall = "Data appear UNIMODAL"
		
		print(overall)
		results['overall_conclusion'] = overall
		print("=" * 60)
		print()
		
		return results


if __name__ == '__main__':
	# global_outliers(data)
	global_outliers(data)
	

	# Example 2: Bimodal data
	bimodal_data = np.concatenate([
		np.random.normal(loc=30, scale=5, size=500),
		np.random.normal(loc=70, scale=5, size=500)
	])
	results = check_multimodality(bimodal_data)

	# Access results programmatically
	print(f"Best k: {results['best_k_bic']}")