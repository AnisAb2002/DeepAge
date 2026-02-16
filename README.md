# DeepAge – Analyse intelligente du visage

DeepAge est une application Python qui prédit l’âge, le genre et l'émotion en temps réel à partir d’une caméra. Elle utilise :
  - Tkinter pour l’interface graphique
  - OpenCV pour la détection de visages
  - TensorFlow / Keras pour le modèle CNN
  - Pillow pour l’affichage des images dans Tkinter

## Fonctionnalités

- Détection de visage en temps réel
- Estimation de l’âge à partir du visage
- Prédiction du genre à partir du visage
- Reconnaissance d’émotions (7 classes)
- Résultat final calculé après analyse de plusieurs frames
- Affichage direct dans une interface graphique

## Prérequis

- Python 3.10+
- Bibliothèques Python (numpy opencv-python tensorflow pillow matplotlib scikit-learn)

## Model de l'age et genre

- Le projet utilise le dataset UTKFace pour entraîner le modèle age_genre_model.h5 :

Télécharger : UTKFace Dataset 
https://www.kaggle.com/datasets/jangedoo/utkface-new

Extraire le contenu dans le dossier dataset/UTKFace dans le projet.
La structure doit ressembler à :

DeepAge/dataset/UTKFace/

### Générer X.npy et y_age.npy et y_genre.npy

Pour préparer les données pour le modèle :

Assurez-vous que le dataset est bien placé.

Exécuter le script preparer_donnees.py : python prepare_data.py

Ce script crée : X.npy c'est les images prétraitées et y_age.npy c'est le label d’âge et y_aenre.npy c'est le label du genre

## Model de l'émotion

- Le projet utilise le dataset FER2013 pour entraîner le modèle emotion_model.h5 :

Télécharger : FER2013 Dataset 

https://www.kaggle.com/datasets/msambare/fer2013

Extraire le contenu dans le dossier dataset_emotion dans le projet.
La structure doit ressembler à :

DeepAge/dataset_emotion/

## Générer X_test_emo.npy, X_train_emo.npy, y_test_emo.npy et y_train_emo.npy

Pour préparer les données pour le modèle :

Assurez-vous que le dataset est bien placé.

Exécuter le script preparer_donnees_emotion.py : python preparer_donnees_emotion.py

Ce script crée : X_test_emo.npy, X_train_emo.npy c'est les images prétraitées test et train, puis y_test_emo.npy et y_train_emo.npy c'est les labels d’émotion test et train

# Entraîner les modèles

Vérifiez que toutes les données sont présents.

Exécuter le script train_model.py : python train_model.py

Exécuter le script train_model_emotion.py : python train_model_emotion.py

Le modèle CNN sera entraîné sur les données.

Après entraînement, 2 fichiers age_genre_model.h5 et emotion_model.h5 seront créé dans le dossier du projet.

# Lancer l’application

Assurez-vous d’avoir :

age_genre_model.h5 et emotion_model.h5

haarcascade_frontalface_default.xml

Exécuter le script principal : python deepage.py


Une interface Tkinter s’ouvre avec les boutons :

- Démarrer caméra
- Arrêter caméra
- Recalculer
  
La caméra détecte les visages et affiche l’âge, le genre et l'émotion estimé en temps réel, et prend quelques frames pour faire une moyenne et l'afficher au bout de 5sec. En cliquant sur le bouton Recalculer, les résultats seront recalculés une nouvelle fois.


# Auteur

Anis ABDAT

étudiant en Licence Informatique ISEI Université Paris 8
