"""
on lit le chemin du dataset chemin = "dataset/UTKFace"
    ça contient les images du dataset UTKFace télécharggé sur Kaggle
    son format est âge_genre_ethnie_date.jpg
    
    et age = int(fichier.split("_")[0]) récupère l'age.

met la taille des images à 64 pour plus rapidité et moins de mémoire

Puis on parcours de toutes les images dans dataset/UTKFace/
    pour extraire l'age et l'image img = cv2.imread(img_chemin)
    et avec la normalisation img = img / 255.0 :
    les pixels seront entre 0 et 1 au lieu de 255 pour un apprentissage plus stable
et on ajoute dans les listes :
    images.append(img)
    ages.append(age)

Convertir ensuite en tableaux NumPy
    X = np.array(images)
    y = np.array(ages)

et à la fin on sauvegarde les données
np.save("X.npy", X) = X.npy pour les images
np.save("y.npy", y) = y.npy pour les âges

"""


import os
import cv2
import numpy as np

chemin = "dataset/UTKFace"
taille = 64

images = []
ages = []

for fichier in os.listdir(chemin):
    try:
        age = int(fichier.split("_")[0])

        img_chemin = os.path.join(chemin, fichier)
        img = cv2.imread(img_chemin)

        if img is None:
            continue

        img = cv2.resize(img, (taille, taille))
        img = img / 255.0

        images.append(img)
        ages.append(age)

    except:
        continue

X = np.array(images)
y = np.array(ages)

print("Images:", X.shape)
print("Ages:", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)
