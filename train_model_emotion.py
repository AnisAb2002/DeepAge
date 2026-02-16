"""

importer les bibliothèque 
    numpy pour manipuler les tableaux numériques (images, labels…)
    tensorflow est Framework de deep learning pour créer et entraîner le réseau de neurones
    from tensorflow.keras.models import Sequential pour de créer un modèle couche par couche
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout pour utiliser
    différentes couches du réseau de neurones
    from tensorflow.keras.optimizers import Adam pour optimiseur utilisé pour entraîner le modèle

on charge les données sauvegardés précédemment
    X_train = np.load("X_train_emo.npy")
    y_train = np.load("y_train_emo.npy")
    X_test = np.load("X_test_emo.npy")
    y_test = np.load("y_test_emo.npy")

Création du modèle CNN
    model = Sequential([     On crée un modèle séquentiel : les couches seront exécutées dans cet ordre :
        Couche 1 — elle détecte contours et textures
            Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
                32 filtres
                taille 3x3
                input_shape = taille d'image
            MaxPooling2D((2,2)),  réduit l'image de moitié et garde seulement les infos importantes

        Couche 2 — Extraction de formes
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
                64 filtres pour détecter les yeux, bouche, nez...

        Couche 3 — Extraction avancée
            Conv2D(128, (3,3), activation='relu'),
                MaxPooling2D((2,2)),
                128 filtres pour détecter les caractéristiques complexes du visage

        Flatten() transforme les matrices en vecteur 1D pour les couches finales

        Couche dense
            Dense(128, activation='relu') est un réseau de neurone qui combine les features extraites

        Régularisation
            Dropout(0.5) désactiver aléatoirement 50 % des neurones à chaque batch pour éviter le surapprentissage

        Couche de sortie
            Dense(7, activation='softmax') on a 7 neurones, une probabilité pour chaque émotion
            softmax transforme les sorties en probabilités dont la somme = 1

Compilation du modèle
    model.compile(.... )
        Adam : optimiseur 
        loss : fonction d'erreur pour classification
        accuracy : métrique de performance.

Résumé du modèle
    model.summary() pour afficher :
        couches, dimensions, nombre de paramètres

Entraînement
    history = model.fit(... )
        epochs=25 nombre de passages sur les données
        batch_size=32 : nombre d'images traitées à la fois
        validation_data : données test pour vérifier la performance pendant l'entraînement

Sauvegarde du modèle
    model.save("emotion_model.h5")

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Charger les données
X_train = np.load("X_train_emo.npy")
y_train = np.load("y_train_emo.npy")
X_test = np.load("X_test_emo.npy")
y_test = np.load("y_test_emo.npy")

# Vérifier les shapes
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Définir le modèle CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(7, activation='softmax')  # 7 classes pour les émotions
])

# Compiler le modèle
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Résumé du modèle
model.summary()

# Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32
)

# Sauvegarde du modèle
model.save("emotion_model.h5")
print("Modèle sauvegardé : emotion_model.h5")