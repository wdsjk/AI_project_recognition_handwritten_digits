from keras import models
from tkinter import *
import tkinter as tk
import numpy as np
from PIL import ImageGrab

# Load model
model = models.load_model('../saved_models/model')

def predict(img):
    # Bring images to the desired format
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)

    img = img.reshape(-1, 28, 28, 1)
    img = img.astype('float32')
    img /= 255

    # Digit prediction
    res = model.predict([img])[0]
    
    # Return predicted digit and confidence percentage
    return np.argmax(res), max(res)

# Graphics
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title('Handwritten digits recognition with AI')

        # Start coordinates
        self.x = 0
        self.y = 0

        # Creat vidgets
        self.canvas = Canvas(self, width=300, height=300, bg='black', cursor='cross')
        self.label = Label(self, text='Draw a digit', font=('Roboto', 38))
        self.rec_button = Button(self, text='Recognize', command=self.recognize_dg)
        self.clear_button = Button(self, text='Clear', command=self.delete_dg)

        # Arrange elements on the screen
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.rec_button.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button.grid(row=1, column=0, pady=2)

        # Calling the drawing function when moving mouse with LMB pressed
        self.canvas.bind('<Button-1>', self.get_coords)
        self.canvas.bind('<B1-Motion>', self.draw)

    def delete_dg(self):
        self.canvas.delete('all')

    def recognize_dg(self):
        x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
        width, height = (self.canvas.winfo_width(), self.canvas.winfo_height())

        a, b, c, d = (x, y, x+width, y+height)
        img = ImageGrab.grab((a, b, c, d))
        
        dg, acc = predict(img)
        self.label.configure(text=f'{dg}, {round(acc*100, 2)}%')

    # Cursor coordinates
    def get_coords(self, event):
        self.x = event.x
        self.y = event.y

    def draw(self, event):
        # Create "brush"
        self.canvas.create_line((self.x, self.y, event.x, event.y), fill='white', width=10)
        self.x = event.x
        self.y = event.y


app = App()
mainloop()
