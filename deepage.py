"""
L'application DeepAge charge un modèle entrainé pour prédire les ages
et capte avec la caméra en temps ton visage et prédit ton age

- Chargement du modèle IA
    model = tf.keras.models.load_model("age_model.h5")
    déjà entrainé gràce à train_model.py et aux données dans dataset

- Détecteur de visage :
    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )
    le modèle OpenCV est pré-entraîné à détecter les visages humains 
    en temps réel et il est très rapide

L'application est une classe.
    - Le constructeur def __init__(self, root)
    - fenêtre principale
        self.root.title(...) met le titre
        self.root.geometry() pour la taille de la fenetre
        Le flux caméra sera affiché dans
            self.label = tk.Label(root)
            self.label.pack()
        
    - Bouton démarer et arrêter
        pour activer ou désactive la caméra et 
        cap = caméra
        marche = état actif ou non

    update() tourne en boucle
        lire dans la caméra ret, frame = self.cap.read()
        frame = image actuelle
        ret = succès ou échec
    
        puis elle convertit en gris
            grris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            car la détection de visage fonctionne mieux en gris.

            on détecte le visage dans cette image (frame)
                faces = face_cascade.detectMultiScale(
                    gris,
                    1.3,  facteur d’échelle
                    5,    précision
                    minSize=(80, 80)  taille minimale du visage
                )
            
            on extrait le visage avec ses coordonnées x, y (position) w largeur et h hauteur
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]

            puis on fait un prétraitement comme celui de l'entraînement
                cv.resize(face, (64,64))
                face / 255.0
                np.reshape(face, (1,64,64,3))

            Prédiction 
                age = model.predict(face)[0][0]


            on affiche un rectangle vert avec l'age prédit
                cv2.rectangle(...)
                cv2.putText(
                    ...
                   f"Age: {int(age)}"
                   ...)

            pour afficher sur tkinter on transforme l'image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) de bleu vert rouge à rouge vert bleu
                img = Image.fromarray(frame)    transforme le tableau frame en image 
                img = img.resize((850, 550))    on adapte la taille

                imgtk = ImageTk.PhotoImage(img) format qu'on va afficher

                puis l'affichage dans la fenêtre
                    self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

        boucle temps réel
            self.root.after(40, self.update)
        Relancer toutes les 40 millisecondes.

"""


import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

# Charger modèle IA
model = tf.keras.models.load_model("age_model.h5")

# Détecteur visage
mon_visage = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

taille = 64


class DeepAgeApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DeepAge - Prédiction de l'age")
        self.root.geometry("1080x720")

        self.btn_demarrer = tk.Button(
            root, text="Démarrer caméra",
            width=25, command=self.demarrer
        )
        self.btn_demarrer.pack(pady=10)

        self.btn_arreter = tk.Button(
            root, text="Arrêter caméra",
            width=25, command=self.arreter
        )
        self.btn_arreter.pack()

        self.label = tk.Label(root)
        self.label.pack()
        
        self.cap = None
        self.marche = False

    def demarrer(self):
        if not self.marche:
            self.cap = cv2.VideoCapture(0)
            self.marche = True
            self.update()

    def arreter(self):
        self.marche = False
        if self.cap:
            self.cap.release()
        self.label.config(image="")

    def update(self):
        if not self.marche:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = mon_visage.detectMultiScale(
            gray,
            1.3,
            5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            face = cv2.resize(face, (taille, taille))
            face = face / 255.0
            face = np.reshape(face, (1, taille, taille, 3))

            age = model.predict(face, verbose=0)[0][0]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.putText(
                frame,
                f"Age: {int(age)}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((900, 650))

        imgtk = ImageTk.PhotoImage(img)

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self.root.after(40, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = DeepAgeApp(root)
    root.mainloop()
