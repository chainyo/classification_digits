import tkinter as tk
import tensorflow as tf
import numpy as np
import PIL
import cv2

from tkinter import ttk
from tensorflow import keras

# Classe pour la création du canvas
class Sketchpad(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.line_width = 40
        self.bind("<Button-1>", self.save_posn)
        self.bind("<B1-Motion>", self.add_line)

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        self.create_line(self.lastx, self.lasty, event.x, event.y, width=self.line_width)
        self.save_posn(event)
    
    def clear_canvas(self):
        self.delete('all')

    def get_img(self):
        self.postscript(file='nb_to_pred.eps') 
        self.img = PIL.Image.open('nb_to_pred.eps') 
        self.img.save('nb_to_pred.png', 'png')

    def pred(self):
        self.pred = cv2.imread('nb_to_pred.png', 0)
        self.pred = cv2.bitwise_not(self.pred)
        self.pred = cv2.resize(self.pred, (28, 28))
        self.pred = self.pred.reshape(-1, 28, 28, 1)
        self.prediction = model.predict(self.pred)
        print(self.prediction)
        self.prediction = np.argmax(self.prediction, axis=1)[0]
        pred_show.config(text=self.prediction)

# Chargement du modèle
model = tf.keras.models.load_model('mnist_model')

# Fenêtre de l'application
root = tk.Tk()
root.geometry('800x800')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Frame pour le dessin
draw_frame = tk.Frame(root, width=800, height=600)
draw_frame.grid(row=0)
# Frame pour les bouton et l'affichage de la prédiction
container_frame = tk.Frame(root, width=800, height=200)
container_frame.grid(row=1)
container_frame.columnconfigure(0, weight=1)
container_frame.columnconfigure(1, weight=1)
container_frame.rowconfigure(0, weight=1)
# Frame des boutons
btn_frame = tk.Frame(container_frame, width=400, height=200, bg='blue')
btn_frame.grid(row=0, column=0)
btn_frame.rowconfigure(0, weight=1)
btn_frame.rowconfigure(1, weight=1)
# Frame pour l'affichage de la prédiction
pred_frame = tk.Frame(container_frame, width=400, height=200, bg='green')
pred_frame.grid(row=0, column=1)

# Canvas pour le dessin
sketch = Sketchpad(draw_frame, width=560, height=560, bg='white')
sketch.grid(column=0, row=0, sticky=('N', 'W', 'E', 'S'))

# Boutons pour l'interaction avec l'utilisateur
pred_btn = tk.Button(btn_frame, text='PREDICTION', width=10, command=lambda:[sketch.get_img(), sketch.pred(), sketch.clear_canvas()])
pred_btn.grid(row=0)
clear_btn = tk.Button(btn_frame, text='CLEAR', width=10, command=sketch.clear_canvas)
clear_btn.grid(row=1)

# Affichage de la prédiction
pred_show = tk.Label(pred_frame, text=' ', font=('Helvetica', 45))
pred_show.pack()


root.mainloop()