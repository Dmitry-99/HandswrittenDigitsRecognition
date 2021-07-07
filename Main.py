# trkinter для создания формы
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from functools import partial

# для создания и обучения нейросети
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchsummary import summary

# cv2 для обратки изображений и распознавания объектов
import os
import cv2
import numpy as np
import imutils

from Net import Net

def recognize(path):

    #читаем изображение
    img = cv2.imread(path)
    img = imutils.resize(img,width=300)
    
    #преобразуем изображение из RGB в Gray 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #создаем матрицу
    kernel = np.ones((40,40),np.uint8)

    #создаем границу, с помощью которой будем отделять цифру от всего остального
    blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
    ret, thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    print(cv2.__version__)
    print(imutils.__version__)
    #находим с помощью границы контуры изображения
    _,cnts,hie = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #загружаем обученную нейросеть
    net = Net()
    state = torch.load("results/model.pth")
    net.load_state_dict(state)
    net.eval()
    
    # для каждого найденного контура цифры
    for c in cnts:
        #создаем маску
        mask = np.zeros(gray.shape,dtype="uint8")
        # получаем границы контура
        (x,y,w,h) = cv2.boundingRect(c)

        # находим многоугольник внутри которого расположена цифра
        hull = cv2.convexHull(c) 
        cv2.drawContours(mask,[hull],-1,255,-1) 
        # побитовое умножение изображения на фильтр (получаем ч/б изображение)
        mask = cv2.bitwise_and(thresh,thresh,mask=mask)

        #находим Region of interest
        roi = mask[y-7:y+h+7,x-7:x+w+7] 
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        # изменяем размер изображение на 28х28
        roi = cv2.resize(roi,(28,28))
        roi = np.array(roi)
        #изменяем размер матрицы до того размера, который может принять модель
        roi = torch.Tensor(roi)

        #делаем предсказание
        prediction = net(roi.reshape(1,1,28,28)).detach().numpy()

        # рисуем область изображения (зеленый прямоугольник) и рисуем цифру
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img,str(int(np.argmax(prediction))),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)


    img = imutils.resize(img,width=500)
    cv2.imwrite('result.jpg',img)
    
    # выводим изображение на форму
    img = Image.open('result.jpg')
    img = img.resize((300, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.place(x=400, y=90)


# функция для открытия изображения из проводника
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

# функция, которая загружает изображение и выводит на форму
def open_img(btn_rec):
    path = openfn()
    img = Image.open(path)
    img = img.resize((300, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.place(x=20, y=90)
    #return path
    btn_rec['command'] = partial(recognize, path)
    
    
# создаем форму
root = Tk() 
  
# задаем размер формы и заголовок            
root.geometry('740x400')     
root.resizable(width='false', height='false')
root.title("HandswrittenDigitsRecognition") 
  
# создаем необходимые элементы формы
btn_recognize = Button(root, text='Recognize digits')
btn_load_img = Button(root, text = 'Load image', command = partial(open_img, btn_recognize))
label = Label(root, text="Welcome!").place(x=165,y=10)
  
# размещаем кнопки на форме
btn_load_img.place(x=225, y=50)
btn_recognize.place(x=420, y = 50)

  
root.mainloop()