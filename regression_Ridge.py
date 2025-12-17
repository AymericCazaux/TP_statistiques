import numpy as np 
import matplotlib.pyplot as plt

data3 = np.load("../data/data3.npy")
data3 = data3.T

#data3
#construire X et y pour la régression
n3 = 20
lam = 1
q = 10
X3 = np.ones((n3, q+1))
y3 = np.zeros((n3,1))
for i in range(n3):
    for j in range(q):
        X3[i, j+1] = data3[i, 0]**(j+1)
    y3[i] = data3[i, 1]

beta3 = np.linalg.inv((X3.T @ X3) + n3*lam*np.eye(q+1)) @ X3.T @ y3

#f3 = beta3[0] + beta3[1]*data3[:, 0]
f3 = np.zeros((n3, 1))
for i in range(q+1):
    for j in range(n3):
        f3[j] += beta3[i]*X3[j,i]

x3 = data3[:, 0]
indices3 = np.argsort(x3)

# trier x et f
x3 = x3[indices3]
f3 = f3[indices3]

plt.figure(1)
plt.scatter(data3[:, 0], data3[:, 1], s=1)
plt.plot(x3, f3, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("data3 (nuages de points et regression (Ridge / polynomiale))")
plt.show()