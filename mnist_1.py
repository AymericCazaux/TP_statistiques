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
###############################################################################

# 1. Normalisation des données (StandardScaler)
# On centre et réduit les données (moyenne = 0, variance = 1)
scaler = StandardScaler()

# On "fit" (calcule moyenne/écart-type) sur le train et on transforme le train
X_train_scaled = scaler.fit_transform(X_train)

# On applique la MÊME transformation sur le test (sans recalculer les moyennes)
X_test_scaled = scaler.transform(X_test)

print("Entraînement de la régression logistique en cours... (cela peut prendre quelques secondes)")

# 2. Entraînement du classifieur (Régression Logistique l2)
# penalty='l2' est le défaut, C=1.0 est l'inverse du lambda (régularisation standard)
# max_iter est augmenté car sur MNIST l'algo par défaut (100) ne converge pas toujours
clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
clf.fit(X_train_scaled, y_train)

print("Entraînement terminé. Calcul des prédictions...")

# Prédiction sur l'ensemble de test
y_pred = clf.predict(X_test_scaled)

# 3. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage graphique
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', values_format='d') # 'd' pour afficher des entiers
ax.set_title("Matrice de Confusion - Régression Logistique (l2)")
plt.show()