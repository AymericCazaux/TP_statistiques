import numpy as np 
import matplotlib.pyplot as plt 

data = np.load("../data/data1.npy")
data2 = np.load("../data/data2.npy")
data = data.T
data2 = data2.T
print(data2.shape)

#data1
#construire X et y pour la régression
X = np.ones((100, 2))
y = np.zeros((100,1))
for i in range(100):
    X[i, 1] = data[i, 0]
    y[i] = data[i, 1]

beta = np.linalg.inv(X.T @ X) @ X.T @ y

f = beta[0] + beta[1]*data[:, 0]


plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], s=1)
plt.plot(data[:, 0], f, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("data1 (nuages de points et regression (linéaire))")

#data2
#construire X et y pour la régression
X2 = np.ones((100, 2))
y2 = np.zeros((100,1))
for i in range(100):
    X2[i, 1] = data2[i, 0]
    y2[i] = data2[i, 1]

beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ y2

f2 = beta2[0] + beta2[1]*data2[:, 0]







plt.figure(2)
plt.scatter(data2[:, 0], data2[:, 1], s=1)
plt.plot(data2[:, 0], f2, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("data2 (nuages de points et regression (linéaire))")
plt.show()