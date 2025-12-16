import numpy as np 
import matplotlib.pyplot as plt

data2 = np.load("../data/data2.npy")
data2 = data2.T 

#data2
#construire X et y pour la régression
q = 10
n2 = 100
X2 = np.ones((n2, q+1))
y2 = np.zeros((n2,1))
for i in range(n2):
    for j in range(q):
        X2[i, j+1] = (data2[i, 0])**(j+1)
    y2[i] = data2[i, 1]

beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ y2

f2 = np.zeros((n2, 1))
for i in range(q):
    for j in range(n2):
        f2[j] += beta2[i+1]*X2[j,i+1]

for i in range(n2):
    f2[i] += beta2[0]

x2 = data2[:, 0]
indices2 = np.argsort(x2)

# trier x et f
x2 = x2[indices2]
f2 = f2[indices2]


plt.figure(1)
plt.scatter(data2[:, 0], data2[:, 1], s=1)
plt.plot(x2, f2, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("data2 (nuages de points et regression (polynomiale))")

data3 = np.load("../data/data3.npy")
data3 = data3.T 
#print(data3.shape)

#data3
#construire X et y pour la régression
n3 = 20
X3 = np.zeros((n3, q))
y3 = np.zeros((n3,1))
for i in range(n3):
    for j in range(q):
        X3[i, j] = (data3[i, 0])**(j+1)
    y3[i] = data3[i, 1]

beta3 = np.linalg.inv(X3.T @ X3) @ X3.T @ y3

f3 = np.zeros((n3, 1))
for i in range(q):
    for j in range(n3):
        f3[j] += beta3[i]*X3[j,i]

x3 = data3[:, 0]
indices3 = np.argsort(x3)

# trier x et f
x3 = x3[indices3]
f3 = f3[indices3]


plt.figure(2)
plt.scatter(data3[:, 0], data3[:, 1], s=1)
plt.plot(x3, f3, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("data3 (nuages de points et regression (polynomiale))")
plt.show()
