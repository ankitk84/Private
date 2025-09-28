from scipy import stats
import numpy as np

def pval(a1, a2):
    pval = stats.wilcoxon(a1,a2)  #wilcoxon p-value
    return pval.pvalue

def mean(a):
    return np.mean(a)  #mean

def sd(a):
    return np.std(a)  #standard deviation

