import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

data3 = np.load("../data/data3.npy").T

n3 = 20
q = 10
lam = 0.001   

X = np.ones((n3, q+1))
y = np.zeros((n3,1))
for i in range(n3):
    for j in range(q):
        X[i, j+1] = data3[i, 0]**(j+1)
    y[i] = data3[i, 1]


X_mean = X[:, 1:].mean(axis=0)
X_std = X[:, 1:].std(axis=0)

X_norm = X.copy()
X_norm[:, 1:] = (X[:, 1:] - X_mean) / X_std

lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
lasso.fit(X_norm, y)

beta_lasso = lasso.coef_

f = X_norm @ beta_lasso

x = data3[:, 0]
idx = np.argsort(x)

plt.figure()
plt.scatter(x, y, s=10)
plt.plot(x[idx], f[idx], color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Régression LASSO polynomiale")
plt.show()