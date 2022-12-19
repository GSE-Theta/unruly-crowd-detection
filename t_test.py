from scipy.stats import ttest_1samp

samples = [0.9922, 0.9839, 0.9735, 0.9742, 0.9942]
n_sample = len(samples)
print("samples:", samples)

t_test = ttest_1samp(samples, popmean=0.95, alternative="less")
print("t-statistic:", t_test.statistic)
print("p-value:", t_test.pvalue)
