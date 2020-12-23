import tkinter as tk
import tensorflow as tf
import numpy as np
import PIL
import cv2

from tkinter import ttk
from tensorflow import keras

# Classe pour la création du canvas
class Sketchpad():
    def __init__(self, **kwargs):
        # Fenêtre de l'application
        self.root = tk.Tk()
        self.root.title("Number Prediction")
        self.root.geometry('700x700')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.configure(bg='#ACD0FF')

        # Frame pour le dessin
        self.draw_frame = tk.Frame(self.root, width=700, height=600, bg='#ACD0FF')
        self.draw_frame.grid(row=0)
        # Frame pour les boutons et l'affichage de la prédiction
        self.container_frame = tk.Frame(self.root, width=700, height=200, bg='#ACD0FF')
        self.container_frame.grid(row=1)
        self.container_frame.columnconfigure(0, weight=1)
        self.container_frame.columnconfigure(1, weight=1)
        self.container_frame.rowconfigure(0, weight=1)
        # Frame des boutons
        self.btn_frame = tk.Frame(self.container_frame, width=400, height=200, bg='#ACD0FF')
        self.btn_frame.grid(row=0, column=0)
        self.btn_frame.rowconfigure(0, weight=1)
        self.btn_frame.rowconfigure(1, weight=1)
        # Frame pour l'affichage de la prédiction
        self.pred_frame = tk.Frame(self.container_frame, width=400, height=200, bg='#ACD0FF')
        self.pred_frame.grid(row=0, column=1)

        # Canvas pour le dessin
        self.sketch = tk.Canvas(self.draw_frame, width=560, height=560, bg='white')
        self.sketch.grid(column=0, row=0, sticky=('N', 'W', 'E', 'S'))

        # Boutons pour l'interaction avec l'utilisateur
        self.pred_btn = tk.Button(self.btn_frame, text='PREDICTION', width=10, font=('Helvetica', 20), bg='#48ae4c', fg='white', relief='flat', command=self.pred)
        self.pred_btn.grid(row=0, pady=5)
        self.clear_btn = tk.Button(self.btn_frame, text='CLEAR', width=10, font=('Helvetica', 20), bg='#e13b3b', fg='white', relief='flat', command=self.clear_canvas)
        self.clear_btn.grid(row=1, pady=5)

        # Affichage de la prédiction
        self.pred_show = tk.Label(self.pred_frame, text=' ', font=('Helvetica', 60), bg='#ACD0FF', fg='white')
        self.pred_show.pack(padx=50)

        self.launch()
        self.root.mainloop()

    def launch(self):
        self.lastx = None
        self.lasty = None
        self.line_width = 32
        self.sketch.bind("<Button-1>", self.save_posn)
        self.sketch.bind("<B1-Motion>", self.add_line)
        # Chargement du modèle
        self.model = tf.keras.models.load_model('mnist_model')

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        self.sketch.create_line(self.lastx, self.lasty, event.x, event.y, width=self.line_width, capstyle='round')
        self.save_posn(event)
    
    def clear_canvas(self):
        self.sketch.delete('all')
        self.pred_show.config(text=' ')

    def get_img(self):
        self.sketch.postscript(file='nb_to_pred.eps') 
        self.img = PIL.Image.open('nb_to_pred.eps') 
        self.img.save('nb_to_pred.png', 'png')

    def pred(self):
        self.get_img()
        self.pred = cv2.imread('nb_to_pred.png', 0)
        self.pred = cv2.bitwise_not(self.pred)
        self.pred = cv2.resize(self.pred, (28, 28))
        self.pred = self.pred.reshape(-1, 28, 28, 1)
        self.prediction = self.model.predict(self.pred)
        self.prediction = np.argmax(self.prediction, axis=1)[0]
        self.pred_show.config(text=self.prediction)

Sketchpad()