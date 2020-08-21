import random
from tkinter import Tk, Label, Frame, BOTTOM,LEFT,RIGHT, ttk,messagebox
import cv2
from PIL import Image as PImage
from PIL import ImageTk
import numpy as np
import os
from playsound import playsound
import threading
from fastai.imports import *
from fastai.vision import *

path="./INPUT/rps/"
learn = load_learner(path)
winSound= "/home/maria/GIT/rps-fastai/INPUT/win.mp3"
tieSound= "/home/maria/GIT/rps-fastai/INPUT/tie.wav"
loseSound= "/home/maria/GIT/rps-fastai/INPUT/lose.mp3"

def play(yourChoice):
    options={0:"PAPER",1:"ROCK" ,2:"SCISSORS"}
    pcOption=random.randint(0,2)
    result =f"PC:{options[pcOption]} vs YOU: {options[yourChoice]}"
    if(yourChoice == pcOption):
        return (result+"\n\n\n    TIE!!",tieSound)
    elif yourChoice==0 and pcOption==1 or yourChoice==1 and pcOption==2 or yourChoice==2 and pcOption==0:
        return (result+"\n\n\n    YOU WIN!!!",winSound)
    else:
        return (result+"\n\n\n    YOU LOSE!!",loseSound)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window = Tk()
window.title("PAPER, ROCK OR SCISSORS")
window.config(background="black")
lmain = Label(window)
lmain.pack()

frame = Frame(window)
frame.pack()

def show():

    ret, frame = video.read()
    image = cv2.flip(frame, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image[28:452, 183:457] 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pil_img = PImage.fromarray(cv2.resize(img,(68,106)).astype('uint8'), 'L')
    pil_img = pil2tensor(pil_img,np.float32)
    inf_img = Image(pil_img.div_(255))
    pred = learn.predict(inf_img)
    cv2.rectangle(image,(180,25),(460,455),(255,0,55),6)
    cv2.putText(image,str(pred[0]),(250,430), font, 1.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,"[PAPER, ROCK, SCISSORS] = "+str(pred[2]),(1,15), font, 0.5,(0,255,0),1,cv2.LINE_AA)

    img = PImage.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show)


def predict():
    ret, frame = video.read()
    image = cv2.flip(frame, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image[28:452, 183:457] 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pil_img = PImage.fromarray(cv2.resize(img,(68,106)).astype('uint8'), 'L')
    pil_img = pil2tensor(pil_img,np.float32)
    inf_img = Image(pil_img.div_(255))
    pred = learn.predict(inf_img)
    print("Predicted:",str(pred[0]))

    if str(pred[0]) == "paper":
        yourChoice=0
        x=play(yourChoice)
        threading.Thread(target=playsound, args=(x[1],)).start()
        messagebox.showinfo(message=x[0], title="Game results")
    elif str(pred[0]) == "rock":
        yourChoice=1
        x=play(yourChoice)
        threading.Thread(target=playsound, args=(x[1],)).start()
        messagebox.showinfo(message=x[0], title="Game results")
    elif str(pred[0]) == "scissors":
        yourChoice=2
        x=play(yourChoice)
        threading.Thread(target=playsound, args=(x[1],)).start()
        messagebox.showinfo(message=x[0], title="Game results")
    else:
        print("Please, try again")
        
     
    
style = ttk.Style()
style.theme_use('alt')
style.configure('TButton', background = 'gray', foreground = 'white', width = 20, borderwidth=3, focusthickness=3, focuscolor='none')
style.map('TButton', background=[('active','red')])

button = ttk.Button(frame, text="QUIT", command=window.destroy)
button.pack(side=RIGHT)
playbutton = ttk.Button(frame,text="PLAY",command=predict)
playbutton.pack(side=LEFT)

show()

window.mainloop()