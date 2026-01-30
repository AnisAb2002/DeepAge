# DeepAge – Prédiction d'age

DeepAge est une application Python qui prédit l’âge en temps réel à partir d’une caméra. Elle utilise :
  - Tkinter pour l’interface graphique
  - OpenCV pour la détection de visages
  - TensorFlow / Keras pour le modèle CNN
  - Pillow pour l’affichage des images dans Tkinter

# Fonctionnalités

- Détection de visage en temps réel
- Estimation de l’âge à partir du visage
- Affichage direct dans une interface graphique

# Prérequis

- Python 3.10+
- Bibliothèques Python (numpy opencv-python tensorflow pillow matplotlib scikit-learn)

# Télécharger le dataset

Le projet utilise le dataset UTKFace pour entraîner le modèle :

Télécharger : UTKFace Dataset 
https://www.kaggle.com/datasets/jangedoo/utkface-new

Extraire le contenu dans le dossier dataset/UTKFace dans le projet.
La structure doit ressembler à :

DeepAge/dataset/UTKFace/

# Générer X.npy et y.npy

Pour préparer les données pour le modèle :

Assurez-vous que le dataset est bien placé.

Exécuter le script preparer_donnees.py : python prepare_data.py


Ce script crée : X.npy c'est les images prétraitées et y.npy c'est le label d’âge

# Entraîner le modèle

Vérifiez que X.npy et y.npy sont présents.

Exécuter le script train_model.py : python train_model.py


Le modèle CNN sera entraîné sur les données.

Après entraînement, un fichier age_model.h5 sera créé dans le dossier du projet.

# Lancer l’application

Assurez-vous d’avoir :

age_model.h5

haarcascade_frontalface_default.xml

Exécuter le script principal : python deepage.py


Une interface Tkinter s’ouvre avec les boutons :

Démarrer caméra

Arrêter caméra

La caméra détecte les visages et affiche l’âge estimé en temps réel.



# Auteur

Anis ABDAT

étudiant en Licence Informatique ISEI Université Paris 8
