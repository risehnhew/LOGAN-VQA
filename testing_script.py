import numpy as np

# After both runs (or in a separate script):
a = np.load("scores_logit_restriction.npy")
b = np.load("scores_baseline.npy")

from scipy.stats import kendalltau, spearmanr
print("corr (kendall):", kendalltau(a, b)[0])
print("corr (spearman):", spearmanr(a, b)[0])
print("allclose:", np.allclose(a, b))