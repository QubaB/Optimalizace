# dipoly

from Dipol import Dipol
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

dipol=Dipol(np.array([0,0]),
            10,
            0
            )


# Zobrazani pole
MAX=0.1
xmin, xmax = -MAX, MAX
ymin, ymax = -MAX, MAX
x = np.linspace(xmin, xmax, 20)  
y = np.linspace(ymin, ymax, 20)  
X, Y = np.meshgrid(x, y)         

def f(x,y):
    Bx=np.zeros(X.shape)
    By=np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = np.array([X[i, j], Y[i, j]]) 
            B=dipol.getDipol_B(r)
            Bx[i,j]=B[0]
            By[i,j]=B[1]
    return([Bx,By])

# Compute the function values on the grid
Bx,By = f(X, Y)


# Plot the function using a contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Bx, 50, cmap='viridis')  # Filled contours
#plt.colorbar(contour)                                # Add a color bar
plt.quiver(X, Y, Bx, By, scale=100, scale_units='xy', angles='xy')


silo_x,silo_y=dipol.getDipol_Silocara(
    np.array([0.02,0.02]),
    0.2)
plt.plot(silo_x,silo_y)

plt.title('2D Plot of f(x, y)')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


def minfun(x):
    dipol.setDipol_ma(x[0])

    r = np.array([0,0.1]) 
    B=dipol.getDipol_B(r)
    return(B[0]-0.0)

x=[0]
result = least_squares(minfun, x, verbose=2, method='trf',gtol=1e-100)

print(resul.x[0])

