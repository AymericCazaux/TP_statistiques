###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
###############################################################################
# LOAD MNIST
###############################################################################
# Download MNIST
mnist = fetch_openml(data_id=554, parser='auto')
# copy mnist.data (type is pandas DataFrame)
data = mnist.data
# array (70000,784) collecting all the 28x28 vectorized images
img = data.to_numpy()
# array (70000,) containing the label of each image
lb = np.array(mnist.target,dtype=int)
# Splitting the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    img, lb, 
    test_size=0.25, 
    random_state=0)
# Number of classes
k = len(np.unique(lb))
# Sample sizes and dimension
(n,p) = img.shape
n_train = y_train.size
n_test = y_test.size 
###############################################################################
# DISPLAY A SAMPLE
###############################################################################
m=16
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(img[i,:].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  #lt.title("Label = %i" % lb[i])



# CODE AJOUTÉ POUR LA QUESTION 4.1

# Normalisation des données (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Entraînement de la régression logistique en cours... (cela peut prendre quelques secondes)")

# Entraînement du classifieur 
# les solvers sont: {l2: lbfgs et l1: saga}
#clf = LogisticRegression(penalty='l1', C=0.5, solver='saga', tol=0.1, max_iter=1000, multi_class='multinomial')
clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
clf.fit(X_train_scaled, y_train)

print("Entraînement terminé. Calcul des prédictions...")

# Prédiction sur l'ensemble de test
y_pred = clf.predict(X_test_scaled)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title("Matrice de Confusion - Régression Logistique (l2)")

# CODE AJOUTÉ POUR LA QUESTION 4.2

# On récupère les coefficients (matrice 10 x 784)
coefs = clf.coef_

# Pour que la comparaison soit juste, on fixe l'échelle des couleurs
# Le 0 sera blanc, le max positif sera Bleu, le max négatif sera Rouge
scale = np.max(np.abs(coefs))

plt.figure(figsize=(15, 6))

for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    
    # 1. On récupère les 784 poids de la classe i
    # 2. On les remet en forme carrée 28x28
    beta_img = coefs[i, :].reshape(28, 28)
    
    # 3. Affichage avec la colormap 'RdBu' (Red-Blue)
    img_plot = ax.imshow(beta_img, cmap='RdBu', vmin=-scale, vmax=scale)
    
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f'Coeffs digit {i}')

# Ajout d'une barre de couleur commune à droite
plt.subplots_adjust(right=0.8)
cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.02, 0.7])
plt.colorbar(img_plot, cax=cbar_ax, label='Valeur du coefficient beta')
plt.suptitle("Visualisation des poids (Beta) pour chaque classe - Régularisation L2")
plt.show()