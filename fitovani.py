# Fitovani dat

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#merena data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.7, 7.4, 20.1, 55.9, 150.3])
dy = np.array([0.2, 0.5, 1.0, 2.0, 5.0])  # nejistoty merenych dat
# modelova funkce
def funkce(x, a, b, c):
    return a * np.exp(b * x) + c

# fitovani modelove funkce
popt, pcov = curve_fit(funkce, x, y,sigma=dy,absolute_sigma=True)

# ziskani fitovanych parametru
a, b, c = popt
a_std = np.sqrt(pcov[0, 0])  # nejistota a
b_std = np.sqrt(pcov[1, 1])  # nejistota b
c_std = np.sqrt(pcov[2, 2])  # nejistota c

print(f"Parametry:")
print(f"a = {a:.4f} ± {a_std:.4f}")
print(f"b = {b:.4f} ± {b_std:.4f}")
print(f"c = {c:.4f} ± {c_std:.4f}")


# korelacni macite
# spocti standardni odchylky jednotlivych parametru
std = np.sqrt(np.diag(pcov))

# normalizuj kovariancni matici
korel_matice = pcov / np.outer(std, std)   

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


print(f"Fitted Parameters:\n a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")

# ziskani fitovane funkce
x_fit = np.linspace(min(x), max(x), 500)
y_fit = funkce(x_fit, *popt)

# vykresleni dat
#plt.scatter(x, y, color='red', label='Mereno')  # Original data points
plt.errorbar(x, y, yerr=dy, fmt='o', color='red', label='Mereno')
plt.plot(x_fit, y_fit, color='blue', label='Fit')  # Fitted curve
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Fitovani')
plt.grid(True)
plt.show()
