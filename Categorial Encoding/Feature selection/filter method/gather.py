import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

df = 2
alpha = 0.05
c_val = stats.chi2.ppf(1 - alpha, df)
cal_chi_s = 3.747

x = np.linspace(0, 10, 1000)
y = stats.chi2.pdf(x, df)

plt.plot(x, y, label='Chi-Square Distribution (df=2)')
plt.fill_between(x, y, where=(x > c_val), color='red', alpha=0.5, label='Critical Region')
plt.axvline(cal_chi_s, color='blue', linestyle='dashed', label='Calculated Chi-Square')
plt.axvline(c_val, color='green', linestyle='dashed', label='Critical Value')
plt.title('Chi-Square Distribution and Critical Region')
plt.xlabel('Chi-Square Value')
plt.ylabel('Probability Density Function')
plt.legend()
plt.show()