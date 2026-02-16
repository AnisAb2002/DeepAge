"""
    on importe les bibliothèque 
      OpenCV pour traitement d'image et vidéo
    numpy pour la manipulation de tableaux numériques
     tensorflow pour le chargement et utilisation des modèles IA
    tkinter pour créer les interface graphiques
    Image et ImageTk pour la conversion d'images OpenCV en format Tkinter
     time pour avoir le temps présent et mesurer le temps écoulé

     on charge les deux modèles :
    model = tf.keras.models.load_model("age_genre_model.h5")
    emotion_model = tf.keras.models.load_model("emotion_model.h5")

    age_genre_model prédit l'âge et le genre et emotion_model prédit l'émotion.
    
    mon_visage = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") pour détecter les visages dans une image

    On définit le titre et la taille et la couleur de la fenêtre principale
        self.root = root
        self.root.title("DeepAge - Analyse intelligente du visage")
        self.root.geometry("1200x750")
        self.root.configure(bg="#1e1e1e")

    Afficher un titre :
        titre = tk.Label(root, text="DeepAge\nAnalyse IA du visage ...", font=("Helvetica",17,"bold"), fg="white", bg="#1e1e1e")
        titre.pack(pady=10)

    Frame principale (conteneur principal pour les frames de vidéo et résultat)
        main_frame = tk.Frame(root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)
        main_frame.pack_propagate(False)
        pack_propagate(False) bloque la taille

    Frame vidéo à gauche pour afficher la vidéo de la caméra
        video_frame = ...
        self.label affichera la vidéo transformée en image Tkinter.


    Frame résultats pour afficher les résultats danalyse.
        resultat_frame = tk.Frame(main_frame, bg="#2b2b2b", bd=2, relief="ridge", width=380, height=650)
        ...
        resultat_frame.pack_propagate(False)
    on affiche le titre et le texte des résultats :
        titre_resultat = tk.Label(resultat_frame, text="Résultat de l'étude", font=("Helvetica",18,"bold"), fg="white", bg="#2b2b2b")
        titre_resultat.pack(pady=15)
        self.resultat_text = tk.Label(resultat_frame, text="Aucune analyse pour le moment", font=("Helvetica",16), fg="white", bg="#2b2b2b", justify="left")
        self.resultat_text.pack(padx=20, pady=20)

    Frame des boutons pour les boutons sous la fenêtre :
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=15)

        style = {"font": ("Helvetica",12,"bold"), "width":18, "height":2, "bd":0}
        est un style pour tous les boutons

    on crée 3 boutons démarrer et arrêter et recalculer
        self.btn_demarrer = tk.Button(btn_frame, text="Démarrer caméra", bg="#45ff5d", fg="#062906", command=self.demarrer, **style)
        self.btn_demarrer.grid(row=0, column=0, padx=10)
        self.btn_arreter = tk.Button(btn_frame, text="Arrêter caméra", bg="#FF4646", fg="#300808", command=self.arreter, **style)
        self.btn_arreter.grid(row=0, column=1, padx=10)
        self.btn_recalculer = tk.Button(btn_frame, text="Recalculer", bg="#419cf1", fg="#052836", command=self.recalculer, **style)
        self.btn_recalculer.grid(row=0, column=2, padx=10)

    Variables pour la capture de vidéo et stockage de ces données temporairement
        self.cap = None   ests l'objet VideoCapture pour la caméra
        self.marche = False : iIndique si la caméra est active
        self.age_donnees = [] pour stocker les âges pendant l'analyse
        self.genre_donnees = []  pour stocker les prédictions de genre
        self.emotion_donnees = []  pour tocker les prédictions d'émotion
        self.debut = None initaliser le début de l'analyse
        self.analyse_terminee = False  booléen pour voir si le résultat final a été calculé

    la méthode demarrer() sert à activer la caméra et initialise les variables ou vide les anciennes données
        puis appelle update()

    la méthode arreter() désactive la caméra et réinitialiser le texte des résultats

    la méthode recalculer() redémarrer l'analyse 

    la méthode update() est appelée en boucle avec self.root.after pour traiter les image capturées

        en détectant les visages avec OpenCV
            face_rgb = frame[y:y+h, x:x+w]
            face_rgb = cv2.resize(face_rgb, (64,64))
            face_rgb = face_rgb / 255.0
            face_rgb = np.reshape(face_rgb, (1,64,64,3))
        Puis on prédit l'age et le genre :
            age, genre = model.predict(face_rgb, verbose=0)

        transformer en niveau de gris pour l'émotion
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (48,48))
            face_gray = face_gray / 255.0
            face_gray = np.reshape(face_gray, (1,48,48,1))
            

        puis enfin on trouve l'émotion :
            emotion = emotion_model.predict(face_gray, verbose=0)
            emotion_label = emotions[np.argmax(emotion)]

        On Stocke des valeurs pour calculer la moyenne après 5secondes :
            moyenne_age = int(np.mean(self.age_donnees))
            moyenne_genre = np.mean(self.genre_donnees)
            gennre_label_final = "M" if moyenne_genre < 0.5 else "F"
            moyenne_emotion = np.mean(self.emotion_donnees, axis=0)
            emotion_label_final = emotions[np.argmax(moyenne_emotion)]

    
        afficher l'image OpenCV en format Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((850, 620))
            imgtk = ImageTk.PhotoImage(img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

  
    Bloc principal crée la fenêtre principale lance l'application
        if __name__ == "__main__":
        root = tk.Tk()
        app = DeepAgeApp(root)
        root.mainloop()

"""


import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import time

# Charger modèle IA
age_genre_model = tf.keras.models.load_model("age_genre_model.h5")
emotion_model = tf.keras.models.load_model("emotion_model.h5")

emotions = ["colere", "degout", "peur", "joie", "tristesse", "surprise", "neutre"]

# Détecteur visage
mon_visage = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


class DeepAgeApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DeepAge - Analyse intelligente du visage")
        self.root.geometry("1200x750")
        self.root.configure(bg="#1e1e1e")

        # ---------- TITRE ----------
        titre = tk.Label(
            root,
            text="DeepAge\nAnalyse IA du visage et prédiction de l'age, du genre et de l'émotion",
            font=("Helvetica", 17, "bold"),
            fg="white",
            bg="#1e1e1e"
        )
        titre.pack(pady=10)


        main_frame = tk.Frame(root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)
        main_frame.pack_propagate(False)
        




        video_frame = tk.Frame(main_frame, bg="#1e1e1e", width=850, height=620)
        video_frame.pack(side="left", padx=20, pady=20)
        video_frame.pack_propagate(False)

        self.label = tk.Label(video_frame, bg="black")
        self.label.pack(fill="both", expand=True)




        resultat_frame = tk.Frame(
            main_frame,
            bg="#2b2b2b",
            bd=2,
            relief="ridge",
            width=380, height=650
        )
        resultat_frame.pack(side="right", padx=5, pady=5)
        resultat_frame.pack_propagate(False)


        titre_resultat = tk.Label(
            resultat_frame,
            text="Résultat de l'étude",
            font=("Helvetica", 18, "bold"),
            fg="white",
            bg="#2b2b2b"
        )
        titre_resultat.pack(pady=15)

        self.resultat_text = tk.Label(
            resultat_frame,
            text="Aucune analyse pour le moment",
            font=("Helvetica", 16),
            fg= "white",
            bg="#2b2b2b",
            justify="left"
        )
        self.resultat_text.pack(padx=20, pady=20)

        # ---------- BOUTONS (BAS) ----------
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=15)

        style = {
            "font": ("Helvetica", 12, "bold"),
            "width": 18,
            "height": 2,
            "bd": 0
        }


        # Boutons
        self.btn_demarrer = tk.Button(
            btn_frame,
            text="Démarrer caméra",
            bg="#45ff5d",
            fg="#062906",
            command=self.demarrer,
            **style
        )
        self.btn_demarrer.grid(row=0, column=0, padx=10)

        self.btn_arreter = tk.Button(
            btn_frame,
            text="Arrêter caméra",
            bg="#FF4646",
            fg="#300808",
            command=self.arreter,
            **style
        )
        self.btn_arreter.grid(row=0, column=1, padx=10)

        self.btn_recalculer = tk.Button(
            btn_frame,
            text="Recalculer",
            bg="#419cf1",
            fg="#052836",
            command=self.recalculer,
            **style
        )
        self.btn_recalculer.grid(row=0, column=2, padx=10)
    

        self.cap = None
        self.marche = False

        # Stockage des prédictions pour moyenne
        self.age_donnees = []
        self.genre_donnees = []
        self.emotion_donnees = []

        self.debut = None
        self.analyse_terminee = False


    def demarrer(self):
        if not self.marche:
            self.cap = cv2.VideoCapture(0)
            self.marche = True
            self.debut = time.time()
            self.analyse_terminee = False
            self.age_donnees.clear()
            self.genre_donnees.clear()
            self.emotion_donnees.clear()
            self.update()
            self.resultat_text.config(text="Analyse en cours...")

    def arreter(self):
        self.marche = False
        if self.cap:
            self.cap.release()
        self.label.config(image="")
        self.resultat_text.config(text="Aucune analyse pour le moment")

    def recalculer(self):
        if self.marche:
            self.debut = time.time()
            self.analyse_terminee = False
            self.age_donnees.clear()
            self.genre_donnees.clear()
            self.emotion_donnees.clear()
            self.resultat_text.config(text="Analyse en cours...")

    def update(self):
        if not self.marche:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = mon_visage.detectMultiScale(
            gray, 1.3, 5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_rgb = frame[y:y+h, x:x+w]

            face_rgb = cv2.resize(face_rgb, (64, 64))
            face_rgb = face_rgb / 255.0
            face_rgb = np.reshape(face_rgb, (1, 64, 64, 3))


            # prédiction de l'age et genre
            age, genre = age_genre_model.predict(face_rgb, verbose=0)
            age = age[0][0]
            genre = genre[0][0]

            genre_label = "M" if genre < 0.5 else "F"


            face_gray = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (48, 48))
            face_gray = face_gray / 255.0
            face_gray = np.reshape(face_gray, (1, 48, 48, 1))

            # prédiction de l'emotion
            emotion = emotion_model.predict(face_gray, verbose=0)
            emotion_label = emotions[np.argmax(emotion)]

            if not self.analyse_terminee:
                self.age_donnees.append(age)
                self.genre_donnees.append(genre)
                self.emotion_donnees.append(emotion)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

            cv2.putText(
                frame,
                f"Age : {int(age)} | Genre : {genre_label} | Emotion: {emotion_label}",
                (x-100, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

            
            # --- Calcul résultat final après 10 secondes ---
            if not self.analyse_terminee and self.debut and (time.time() - self.debut) > 5:
                moyenne_age = int(np.mean(self.age_donnees))
                moyenne_genre = np.mean(self.genre_donnees)
                gennre_label_final = "M" if moyenne_genre < 0.5 else "F"
                moyenne_emotion = np.mean(self.emotion_donnees, axis=0)
                emotion_label_final = emotions[np.argmax(moyenne_emotion)]

                self.resultat_text.config(
                    text=f"Résultat de l'étude de votre visage :\n"
                        f"- Age     : {moyenne_age}\n"
                        f"- Genre   : {gennre_label_final}\n"
                        f"- Emotion : {emotion_label_final}"
                )
                self.analyse_terminee = True

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((850, 620))

        imgtk = ImageTk.PhotoImage(img)

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self.root.after(30, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = DeepAgeApp(root)
    root.mainloop()
