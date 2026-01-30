"""
Dans ce code on charge les données X et y
puis on les sépare en données train et test
puis on entraine le modèle :
 - Sequential permet de créer un modèle couche par couche
 - Conv2D, MaxPooling2D, Flatten, Dense, Dropout : servent à 
    construire un réseau de neurones convolutif (CNN)
        
        Couche 1
            32 filtres, taille filtre 3x3
            détecter les contours, coins et textures
            MaxPooling réduit la taille de l'image /2 (32*32)

        Couche 2
            64 filtres , meme taille
            détecter les yeux, nez et la bouche

        Couche 3
            128 filtres
            détecter les rides et formes du visage

        Flatten() transforme les 8192 neurones en vecteur
            pour pouvoir entrer dans les couches denses.
        
        Dense(128, activation='relu')
            réseau neuronal classique qui 
            apprend les relations complexes
            et combine les caractéristiques extraites

        Dropout(0.5)
            éteint la moitié des neurones aléatoirement
            ça évite le surapprentissage (overfitting)

        
        Dense(1, activation='linear')
            une seule couche de sortie
            on prédit un nombre réel
            c'est pas une classe c'est un nombre donc
            régression, pas classification.

    - Compiler le modèle
        model.compile(...)
        Optimiser avec Adam car c'est rapide et très utilisé en deep learning

       Fonction de perte : mean_absolute_error (moyenne absolue = |age réel - age prédit|)


    - model.summary()
        Afficher des détails comme les couches, les tailles et les paramètres

    - Entraînement avec model.fit()
        epochs c'est le nombre de passages donc 25
        batch_size est le nombre de images traitées à la fois
        validation_data	données test du modèle
    
    - Sauvegarde du modèle avec model.save("age_model.h5")
    et ça crée le fichier age_model.h5

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Charger les données
X = np.load("X.npy")
y = np.load("y.npy")

# Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""
        
    """
# Modèle CNN
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_absolute_error",
    metrics=["mae"]
)

model.summary()

# Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32
)

# Sauvegarde
model.save("age_model.h5")

print("Modèle sauvegardé : age_model.h5")
