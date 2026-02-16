"""
on importe les bibliothèque 
    os pour manipuler les chemins et dossiers
    cv2  pour lire et traiter les images
    numpy pour manipuler les tableaux numériques
    to_categorical depuis tensorflow.keras.utils pour convertir les labels en format "one-hot" pour la classification

On déclare les 2 chemins vers les images d'entraînement et test
    train_chemin = "dataset_emotion/train" 
    test_chemin = "dataset_emotion/test"

Liste des classes d'émotions qu'on a
    emotions = ["Colère", "Dégoût", "Peur", "Joie", "Tristesse", "Surprise", "Neutre"]

Taille pour redimensionner les images (48x48 pixels)
    taille = 48

Listes pour stocker les images et labels avant de les convertir en tableaux NumPy
    _X_test = []
    _y_test = []
    _X_train = []
    _y_train = []

Chargement et traitement des images
    for i, emotion in enumerate(emotions):  (i est index de l'émotion de 0 à 6)
        chemin = os.path.join(train_chemin, emotion)  (Chemin vers le dossier de cette émotion
        for fichier in os.listdir(chemin):       puis on parcourt toute les images
            chemin_image = os.path.join(chemin, fichier)  ===== Chemin complet
            img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)  pour lire l'image en niveau de gris
            if img is None:   si l'image  est non lisible on  passe
                continue
            img = cv2.resize(img, (taille, taille))  sinon on va redimensionner à 48x48
            img = img / 255.0                    puis normaliser les pixels entre 0 et 1
            _X_train.append(img)  et on a joute l'image à la liste d'entraînement
            _y_train.append(i)    et enfin on ajoute l'index de l'émotion comme label

    on refait la même chose pour les images de test
        chemin = os.path.join(test_chemin, emotion) 
        for fichier in os.listdir(chemin):
            ...

On convertit les listes en tableaux NumPy
    X_test_emo = np.array(_X_test).reshape(-1, taille, taille, 1)
    X_train_emo = np.array(_X_train).reshape(-1, taille, taille, 1)
.reshape(-1, 48, 48, 1) ajoute une dimension pour le canal unique (niveaux de gris)
car Keras attend (batch_size, height, width, channels).

Conversion des labels en one-hot pour pouvoir classifier
y_train_emo = to_categorical(np.array(_y_train), num_classes=7)
y_test_emo = to_categorical(np.array(_y_test), num_classes=7)

On affiche les formes pour vérifier que les données ont bien la bonne dimension.
    print("X_test_emo:", X_test_emo.shape)   # (nombre_images_test, 48, 48, 1)
    print("y_test_emo:", y_test_emo.shape)   # (nombre_images_test, 7)
    print("X_train_emo:",X_train_emo.shape)  # (nombre_images_train, 48, 48, 1)
    print("y_train_emo:",y_train_emo.shape)  # (nombre_images_train, 7)


Sauvegarder les tableaux pour pouvoir les charger directement plus tard (np.load) sans retraiter toutes les images
np.save("X_test_emo.npy", X_test_emo)
np.save("y_test_emo.npy", y_test_emo)
np.save("X_train_emo.npy", X_train_emo)
np.save("y_train_emo.npy", y_train_emo)


"""


import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical


train_chemin = "dataset_emotion/train" 
test_chemin = "dataset_emotion/test"


emotions = ["colere", "degout", "peur", "joie", "tristesse", "surprise", "neutre"]

taille = 48

_X_test = []
_y_test = []
_X_train = []
_y_train = []


for i, emotion in enumerate(emotions):
    chemin = os.path.join(train_chemin, emotion)
    for fichier in os.listdir(chemin):
        chemin_image = os.path.join(chemin, fichier)
        img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (taille, taille))
        img = img / 255.0 
        _X_train.append(img)
        _y_train.append(i)
        
    chemin = os.path.join(test_chemin, emotion)
    for fichier in os.listdir(chemin):
        chemin_image = os.path.join(chemin, fichier)
        img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (taille, taille))
        img = img / 255.0 
        _X_test.append(img)
        _y_test.append(i)


X_test_emo = np.array(_X_test).reshape(-1, taille, taille, 1)
X_train_emo = np.array(_X_train).reshape(-1, taille, taille, 1)

y_train_emo = to_categorical(np.array(_y_train) , num_classes=7)
y_test_emo = to_categorical(np.array(_y_test) , num_classes=7)

print("X_test_emo:", X_test_emo.shape)
print("y_test_emo:", y_test_emo.shape)
print("X_train_emo:",X_train_emo.shape)
print("y_train_emo:",y_train_emo.shape)

np.save("X_test_emo.npy", X_test_emo)
np.save("y_test_emo.npy", y_test_emo)
np.save("X_train_emo.npy", X_train_emo)
np.save("y_train_emo.npy", y_train_emo)
