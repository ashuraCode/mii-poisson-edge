#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from QtImageViewer import *

import numpy as np
import cv2
from math import factorial
from PIL import Image

OPTIONS = """conversion.py <option> file
Skrypt korzysta z rozkładu Poissona do generacji obrazów. Korzystając z pikseli jako parametr alfa generuje sekwencje zdjęć {alfa, alfa/2, alfa/4, alfa/8, alfa/16, alfa/32, alfa/64, alfa/128}.
option:
HSL - korzysta z średniego światła białego w przestrzeni HSL
HSV - korzysta z mocy światła białego w przestrzeni HSV
GRAY - konwersja do odcieni szarości
RGB - użycie wartości R, G, B
YCbCr - korzysta z składowej luminancji w przestrzeni HSV
"""

def poisson(narr):
    return np.random.poisson(narr)#np.divide(np.multiply(np.power(narr, k), np.exp(-narr)), factorial(k))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    if len(sys.argv) < 2:
        print("Nie podano parametrów.")
        print(OPTIONS)
        sys.exit(-1)

    if sys.argv[1] != "HSL" and sys.argv[1] != "HSV" and sys.argv[1] != "GRAY" and sys.argv[1] != "RGB" and sys.argv[1] != "YCbCr": 
        print("Nieprawidłowy parametr.")
        print(OPTIONS)
        sys.exit(-1)

    image = cv2.imread(sys.argv[2])
    viewer = QtImageViewer()
    viewer.show()
    viewer.setGeometry(100,100,400,400)
 
    if sys.argv[1] == "HSL":
        im_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        del image
        
        for it in [1,2,4,8,16,32,64,128]:
            tmp = np.divide(im_hls, it)
            tmp = np.uint8(poisson(tmp))

            image = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)
            img = Image.fromarray(image, 'RGB')
            img.save('tmp'+str(it)+'.jpg')
            print("Job "+str(it)+" done")

            viewer.loadImageFromFile('tmp'+str(it)+'.jpg')  # Pops up file dialog.
            app.processEvents()

    elif sys.argv[1] == "HSV":
        im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        del image
        
        for it in [1,2,4,8,16,32,64,128]:
            tmp = np.divide(im_hsv, it)
            tmp = np.uint8(poisson(tmp))

            image = cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB)
            img = Image.fromarray(image, 'RGB')
            img.save('tmp'+str(it)+'.jpg')
            print("Job "+str(it)+" done")

            viewer.loadImageFromFile('tmp'+str(it)+'.jpg')  # Pops up file dialog.
            app.processEvents()

    elif sys.argv[1] == "GRAY":
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        del image
        
        for it in [1,2,4,8,16,32,64,128]:
            tmp = np.divide(im_gray, it)
            tmp = np.uint8(poisson(tmp))

            img = Image.fromarray(tmp, 'L')
            img.save('tmp'+str(it)+'.jpg')
            print("Job "+str(it)+" done")

            viewer.loadImageFromFile('tmp'+str(it)+'.jpg')  # Pops up file dialog.
            app.processEvents()

    elif sys.argv[1] == "RGB":
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        del image
        
        for it in [1,2,4,8,16,32,64,128]:
            tmp = np.divide(im_rgb, it)
            tmp = np.uint8( poisson(tmp) )

            img = Image.fromarray(tmp, 'RGB')
            img.save('tmp'+str(it)+'.jpg')
            print("Job "+str(it)+" done")

            viewer.loadImageFromFile('tmp'+str(it)+'.jpg')  # Pops up file dialog.
            app.processEvents()

    elif sys.argv[1] == "YCbCr":
        im_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        del image
        
        for it in [1,2,4,8,16,32,64,128]:
            tmp = np.divide(im_ycrcb, it)
            tmp = np.uint8( poisson(tmp) )

            image = cv2.cvtColor(tmp, cv2.COLOR_YCrCb2RGB)
            img = Image.fromarray(image, 'RGB')
            img.save('tmp'+str(it)+'.jpg')
            print("Job "+str(it)+" done")

            viewer.loadImageFromFile('tmp'+str(it)+'.jpg')  # Pops up file dialog.
            app.processEvents()
        

    sys.exit(app.exec_())