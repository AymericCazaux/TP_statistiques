import numpy as np 
import matplotlib.pyplot as plt

data3 = np.load("../data/data3.npy")
data3 = data3.T

#data3
#construire X et y pour la régression
n3 = 20
lam = 0.1
X3 = np.ones((n3, 2))
y3 = np.zeros((n3,1))
for i in range(n3):
    X3[i, 1] = data3[i, 0]
    y3[i] = data3[i, 1]

beta3 = np.linalg.inv((X3.T @ X3) + n3*lam*np.eye(2)) @ X3.T @ y3

f3 = beta3[0] + beta3[1]*data3[:, 0]


plt.figure(1)
plt.scatter(data3[:, 0], data3[:, 1], s=1)
plt.plot(data3[:, 0], f3, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("data1 (nuages de points et regression (linéaire))")
plt.show()