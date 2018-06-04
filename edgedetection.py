#!/usr/bin/python3
# -*- coding: utf-8 -*-

# import gc
import numpy as np
from scipy import ndimage
import cv2
from shutil import copyfile
import datetime as dtim
import gc

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout, 
    QLineEdit, QPushButton, QHBoxLayout, QMessageBox, 
    QTabWidget, QSlider, QFileDialog, QLineEdit, 
    QListWidget, QRadioButton, QGroupBox, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QCheckBox,
    QSpacerItem, QSizePolicy )

from QtImageViewer import *


class AppForMiI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.interfejs()
        self.deserialize()

    def interfejs(self):
        lay1d = QVBoxLayout()
        tabs = QTabWidget()
        lay1d.addWidget(tabs)

        wid2d1 = QWidget()
        lay2d1 = QGridLayout()
        
        imageBtn = QPushButton("Wybierz..")
        lay2d1.addWidget(imageBtn, 0, 0)
        imageBtn.clicked.connect(self.wybierzObraz)
        
        self.filePath = QLineEdit()
        lay2d1.addWidget(self.filePath, 0, 1)
        self.filePath.textChanged.connect(self.filePathChanged)

        self.lastFiles = QListWidget()
        lay2d1.addWidget(self.lastFiles, 1, 0, 1, 2)
        self.lastFiles.doubleClicked.connect(self.wybierzZHistorii)

        self.grayCheck = QCheckBox("Obraz w odcieniach szarości")
        lay2d1.addWidget(self.grayCheck, 2, 0, 1, 2)
        self.grayCheck.stateChanged.connect(self.grayscaleChanged)

        wid2d1.setLayout(lay2d1)
        tabs.addTab(wid2d1, "Wybierz obraz")
        
        wid2d2 = QWidget()
        lay2d2 = QGridLayout()
        
        sobelUpBtn = QPushButton("Sobel")
        lay2d2.addWidget(sobelUpBtn, 0, 0, 1, 3)
        sobelUpBtn.clicked.connect(self.sobelUpdate)
        
        sobelModeLabel = QLabel("Wykrywanie krawędzi:")
        lay2d2.addWidget(sobelModeLabel, 1, 0)

        self.sobelMode1 = QRadioButton("po x")
        self.sobelMode2 = QRadioButton("po y")
        self.sobelMode3 = QRadioButton("jako średnia ważona x,y")
        sobelLay = QHBoxLayout()
        sobelLay.addWidget(self.sobelMode1)
        sobelLay.addWidget(self.sobelMode2)
        sobelLay.addWidget(self.sobelMode3)
        sobelGr = QGroupBox()
        sobelGr.setLayout(sobelLay)
        self.sobelMode1.setChecked(True)
        
        lay2d2.addWidget(sobelGr, 1, 1, 1, 2)

        self.sobelXtoYWeight = QSlider()
        self.sobelXtoYWeight.setMinimum(1)
        self.sobelXtoYWeight.setMaximum(99)
        self.sobelXtoYWeight.setOrientation(Qt.Horizontal)
        self.sobelXtoYWeight.setValue(50)

        sobelWeightLabel = QLabel("Waga x do y")
        self.sobelWeightValue = QLabel("0.50")
        
        lay2d2.addWidget(sobelWeightLabel, 2, 0)
        lay2d2.addWidget(self.sobelXtoYWeight, 2, 1)
        lay2d2.addWidget(self.sobelWeightValue, 2, 2)
        self.sobelXtoYWeight.valueChanged.connect(self.sobelSlider)

        vSpacer = QSpacerItem(5, 5, QSizePolicy.Minimum, QSizePolicy.Expanding)
        lay2d2.addItem(vSpacer, 3,0)

        wid2d2.setLayout(lay2d2)
        tabs.addTab(wid2d2, "Sobel")
        
        wid2d3 = QWidget()
        lay2d3 = QGridLayout()
        
        cannyUpBtn = QPushButton("Canny")
        lay2d3.addWidget(cannyUpBtn, 0, 0, 1, 3)
        cannyUpBtn.clicked.connect(self.cannyUpdate)

        self.cannyThr1 = QSlider()
        self.cannyThr1.setMinimum(0)
        self.cannyThr1.setMaximum(255)
        self.cannyThr1.setOrientation(Qt.Horizontal)
        self.cannyThr1.setValue(100)
        self.cannyThr1.valueChanged.connect(self.cannyThreshold1)
        
        cannyThr1Label = QLabel("Pierwszy próg")
        self.cannyThr1Value = QLabel("100")

        lay2d3.addWidget(cannyThr1Label, 1, 0)
        lay2d3.addWidget(self.cannyThr1, 1, 1)
        lay2d3.addWidget(self.cannyThr1Value, 1, 2)
        
        self.cannyThr2 = QSlider()
        self.cannyThr2.setMinimum(0)
        self.cannyThr2.setMaximum(255)
        self.cannyThr2.setOrientation(Qt.Horizontal)
        self.cannyThr2.setValue(200)
        self.cannyThr2.valueChanged.connect(self.cannyThreshold2)
        
        cannyThr2Label = QLabel("Drugi próg")
        self.cannyThr2Value = QLabel("200")
        
        lay2d3.addWidget(cannyThr2Label, 2, 0)
        lay2d3.addWidget(self.cannyThr2, 2, 1)
        lay2d3.addWidget(self.cannyThr2Value, 2, 2)

        vSpacer = QSpacerItem(5, 5, QSizePolicy.Minimum, QSizePolicy.Expanding)
        lay2d3.addItem(vSpacer, 3,0)

        wid2d3.setLayout(lay2d3)
        tabs.addTab(wid2d3, "Canny")
        
        wid2d4 = QWidget()
        lay2d4 = QGridLayout()
        
        prewittUpBtn = QPushButton("Prewitt")
        lay2d4.addWidget(prewittUpBtn, 0, 0, 1, 2)
        prewittUpBtn.clicked.connect(self.prewittUpdate)
        
        prewittModeLabel = QLabel("Wykrywanie krawędzi:")
        lay2d4.addWidget(prewittModeLabel, 1, 0)

        self.prewittMode1 = QRadioButton("po x")
        self.prewittMode2 = QRadioButton("po y")
        self.prewittMode3 = QRadioButton("jako suma x,y")
        prewittLay = QHBoxLayout()
        prewittLay.addWidget(self.prewittMode1)
        prewittLay.addWidget(self.prewittMode2)
        prewittLay.addWidget(self.prewittMode3)
        prewittGr = QGroupBox()
        prewittGr.setLayout(prewittLay)
        self.prewittMode1.setChecked(True)
        
        lay2d4.addWidget(prewittGr, 1, 1)
        
        prewittXMatLabel = QLabel("kernel x:")
        prewittYMatLabel = QLabel("kernel y:")
        
        lay2d4.addWidget(prewittXMatLabel, 2, 0)
        lay2d4.addWidget(prewittYMatLabel, 2, 1)

        self.prewittXMat = QTableWidget()
        self.prewittYMat = QTableWidget()

        self.prewittXMat.setColumnCount(3)
        self.prewittXMat.setRowCount(3)
        self.prewittYMat.setColumnCount(3)
        self.prewittYMat.setRowCount(3)
        
        self.prewittXMat.setColumnWidth(0, 50)
        self.prewittXMat.setColumnWidth(1, 50)
        self.prewittXMat.setColumnWidth(2, 50)
        self.prewittYMat.setColumnWidth(0, 50)
        self.prewittYMat.setColumnWidth(1, 50)
        self.prewittYMat.setColumnWidth(2, 50)

        self.prewittXMat.setItem(0, 0, QTableWidgetItem("1"))
        self.prewittXMat.setItem(0, 1, QTableWidgetItem("1"))
        self.prewittXMat.setItem(0, 2, QTableWidgetItem("1"))
        self.prewittXMat.setItem(1, 0, QTableWidgetItem("0"))
        self.prewittXMat.setItem(1, 1, QTableWidgetItem("0"))
        self.prewittXMat.setItem(1, 2, QTableWidgetItem("0"))
        self.prewittXMat.setItem(2, 0, QTableWidgetItem("-1"))
        self.prewittXMat.setItem(2, 1, QTableWidgetItem("-1"))
        self.prewittXMat.setItem(2, 2, QTableWidgetItem("-1"))
        
        self.prewittYMat.setItem(0, 0, QTableWidgetItem("-1"))
        self.prewittYMat.setItem(0, 1, QTableWidgetItem("0"))
        self.prewittYMat.setItem(0, 2, QTableWidgetItem("1"))
        self.prewittYMat.setItem(1, 0, QTableWidgetItem("-1"))
        self.prewittYMat.setItem(1, 1, QTableWidgetItem("0"))
        self.prewittYMat.setItem(1, 2, QTableWidgetItem("1"))
        self.prewittYMat.setItem(2, 0, QTableWidgetItem("-1"))
        self.prewittYMat.setItem(2, 1, QTableWidgetItem("0"))
        self.prewittYMat.setItem(2, 2, QTableWidgetItem("1"))
        
        lay2d4.addWidget(self.prewittXMat, 3, 0)
        lay2d4.addWidget(self.prewittYMat, 3, 1)

        wid2d4.setLayout(lay2d4)
        tabs.addTab(wid2d4, "Prewitt")
        
        wid2d5 = QWidget()
        lay2d5 = QGridLayout()
        
        sharrUpBtn = QPushButton("Scharr")
        lay2d5.addWidget(sharrUpBtn, 0, 0, 1, 3)
        sharrUpBtn.clicked.connect(self.scharrUpdate)
        
        sharrModeLabel = QLabel("Wykrywanie krawędzi:")
        lay2d5.addWidget(sharrModeLabel, 1, 0)

        self.scharrMode1 = QRadioButton("po x")
        self.scharrMode2 = QRadioButton("po y")
        self.scharrMode3 = QRadioButton("jako średnia ważona x,y")
        sharrLay = QHBoxLayout()
        sharrLay.addWidget(self.scharrMode1)
        sharrLay.addWidget(self.scharrMode2)
        sharrLay.addWidget(self.scharrMode3)
        sharrGr = QGroupBox()
        sharrGr.setLayout(sharrLay)
        self.scharrMode1.setChecked(True)
        
        lay2d5.addWidget(sharrGr, 1, 1, 1, 2)

        self.scharrXtoYWeight = QSlider()
        self.scharrXtoYWeight.setMinimum(1)
        self.scharrXtoYWeight.setMaximum(99)
        self.scharrXtoYWeight.setOrientation(Qt.Horizontal)
        self.scharrXtoYWeight.setValue(50)

        scharrWeightLabel = QLabel("Waga x do y")
        self.scharrWeightValue = QLabel("0.50")
        
        lay2d5.addWidget(scharrWeightLabel, 2, 0)
        lay2d5.addWidget(self.scharrXtoYWeight, 2, 1)
        lay2d5.addWidget(self.scharrWeightValue, 2, 2)

        self.scharrXtoYWeight.valueChanged.connect(self.scharrSlider)

        vSpacer = QSpacerItem(5, 5, QSizePolicy.Minimum, QSizePolicy.Expanding)
        lay2d5.addItem(vSpacer, 3,0)

        wid2d5.setLayout(lay2d5)
        tabs.addTab(wid2d5, "Scharr")

        wid1d2 = QWidget()
        lay1d2 = QHBoxLayout()

        imageBtnSave = QPushButton("Zapisz jako")
        lay1d2.addWidget(imageBtnSave)
        imageBtnSave.clicked.connect(self.zapiszObraz)
        
        self.filePathSave = QLineEdit()
        lay1d2.addWidget(self.filePathSave)
        self.filePathSave.setText("out")

        wid1d2.setLayout(lay1d2)

        lay1d.addWidget(wid1d2)

        self.setLayout(lay1d)
        self.resize(512, 400)
        self.setWindowTitle("Algorytmy wykrywania krawędzi: Sobel, Canny, Prewitt, Scharr")
        self.show()
        
        self.viewer = QtImageViewer()

    def grayscaleChanged(self):
        if isinstance(self.image, np.ndarray):
            if self.grayCheck.isChecked():
                if len(np.shape(self.image)) == 3:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                self.image = cv2.imread(self.filePath.text())
                
            cv2.imwrite("tmp.png", self.image)

            self.viewer.loadImageFromFile("tmp.png")
            self.viewer.show()

    def serialize(self):
        with open("config.txt", "w") as fin:
            for item in range(0, self.lastFiles.count()):
                fin.write(self.lastFiles.item(item).text()+"\r\n") 
                print(self.lastFiles.item(item).text())

    def deserialize(self):
        try:
            with open("config.txt", "r") as fin:
                for line in fin:
                    if len(line.strip()) > 0:
                        self.lastFiles.addItem(line.strip())
        except IOError:
            print("cannot open config.txt")
        
    def closeEvent(self, event):
        self.serialize()
        event.accept()
        
    def wybierzZHistorii(self):
        selItem = self.lastFiles.selectedItems()[0].text()
        if len(selItem) != 0:
            self.filePath.setText(selItem)
        
    def wybierzObraz(self):
        fd = QFileDialog.getOpenFileName(None, "Wybierz obraz", "", "Obraz (*.jpg *.png)")
        if len(fd) != 0:
            self.filePath.setText(fd[0])
        
    def zapiszObraz(self):
        copyfile("tmp.png", self.filePathSave.text()+
            '{:%Y-%m-%d_%H%M%S}'.format(dtim.datetime.now())+".png")
        
    def filePathChanged(self):
        if isinstance(self.image, np.ndarray):
            del self.image
            gc.collect()

        self.image = cv2.imread(self.filePath.text())
        
        if self.lastFiles.count() == 0 or self.lastFiles.item(0).text() != self.filePath.text():
            self.lastFiles.insertItem(0, self.filePath.text())

        if self.grayCheck.isChecked():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("tmp.png", self.image)

        self.viewer.loadImageFromFile("tmp.png")
        self.viewer.show()

        print( self.image.shape )
        
    def sobelSlider(self):
        self.sobelWeightValue.setText(str(self.sobelXtoYWeight.value()/100))
        
    def sobelUpdate(self):
        if isinstance(self.image, np.ndarray):
            if not self.sobelMode2.isChecked():
                sobelx64 = cv2.Sobel(self.image,cv2.CV_64F,1,0,ksize=5)
            if not self.sobelMode1.isChecked():
                sobely64 = cv2.Sobel(self.image,cv2.CV_64F,0,1,ksize=5)

            if self.sobelMode1.isChecked():
                edge_sobel = sobelx64
            elif self.sobelMode2.isChecked():
                edge_sobel = sobely64
            elif self.sobelMode3.isChecked():
                sobelx = np.uint8(np.absolute(sobelx64))
                sobely = np.uint8(np.absolute(sobely64))

                xweight = self.sobelXtoYWeight.value()/100
                yweight = 1 - xweight

                edge_sobel = cv2.addWeighted( sobelx, xweight, sobely, yweight, 0)

            cv2.imwrite("tmp.png", edge_sobel)
            self.viewer.loadImageFromFile("tmp.png")
            self.viewer.show()
            
    def cannyThreshold1(self):
        if self.cannyThr1.value() >= self.cannyThr2.value():
            self.cannyThr1.setValue(self.cannyThr2.value()-1)
        self.cannyThr1Value.setText(str(self.cannyThr1.value()))
            
    def cannyThreshold2(self):
        if self.cannyThr2.value() <= self.cannyThr1.value():
            self.cannyThr2.setValue(self.cannyThr1.value()+1)
        self.cannyThr2Value.setText(str(self.cannyThr2.value()))
        
    def cannyUpdate(self):
        if isinstance(self.image, np.ndarray):
            edge_canny = cv2.Canny(self.image, self.cannyThr1.value(), self.cannyThr2.value())
            cv2.imwrite("tmp.png", edge_canny)
            self.viewer.loadImageFromFile("tmp.png")
            self.viewer.show()
        
    def prewittUpdate(self):
        if isinstance(self.image, np.ndarray):
            kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            
            for i in range(0,3):
                for j in range(0,3):
                    kernelx[i,j] = float(self.prewittXMat.item(i,j).text())
                    kernely[i,j] = float(self.prewittYMat.item(i,j).text())

            if not self.prewittMode2.isChecked():    
                img_prewittx = cv2.filter2D(self.image, -1, kernelx)
            if not self.prewittMode1.isChecked():    
                img_prewitty = cv2.filter2D(self.image, -1, kernely)

            if self.prewittMode1.isChecked():
                edge_prewitt = img_prewittx
            elif self.prewittMode2.isChecked():
                edge_prewitt = img_prewitty
            elif self.prewittMode3.isChecked():
                edge_prewitt = img_prewittx + img_prewitty

            cv2.imwrite("tmp.png", edge_prewitt)
            self.viewer.loadImageFromFile("tmp.png")
            self.viewer.show()
        
    def robertsUpdate(self):
        if isinstance(self.image, np.ndarray):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            roberts_cross_v = np.array( [[ 0, 0, 0 ],  [ 0, 1, 0 ], [ 0, 0,-1 ]] )
            roberts_cross_h = np.array( [[ 0, 0, 0 ], [ 0, 0, 1 ], [ 0,-1, 0 ]] )

            vertical = ndimage.convolve( gray_image, roberts_cross_v )
            horizontal = ndimage.convolve( gray_image, roberts_cross_h )
            edge_roberts = np.sqrt( np.square(horizontal) + np.square(vertical))
            
            cv2.imwrite("tmp.png", edge_roberts)
            self.viewer.loadImageFromFile("tmp.png")
            self.viewer.show()
        
    def scharrSlider(self):
        self.scharrWeightValue.setText(str(self.scharrXtoYWeight.value()/100))
        
    def scharrUpdate(self):
        if isinstance(self.image, np.ndarray):
            if not self.scharrMode2.isChecked():
                scharrx = cv2.Scharr(self.image, cv2.CV_64F, 1, 0)
            if not self.scharrMode1.isChecked():
                scharry = cv2.Scharr(self.image, cv2.CV_64F, 0, 1)

            if self.scharrMode1.isChecked():
                edge_scharr = scharrx
            elif self.scharrMode2.isChecked():
                edge_scharr = scharry
            elif self.scharrMode3.isChecked():
                abs_grad_x = cv2.convertScaleAbs(scharrx)
                abs_grad_y = cv2.convertScaleAbs(scharry)
                
                xweight = self.scharrXtoYWeight.value()/100
                yweight = 1 - xweight

                edge_scharr = cv2.addWeighted(abs_grad_x, xweight, abs_grad_y, yweight, 0)
            
            cv2.imwrite("tmp.png", edge_scharr)
            self.viewer.loadImageFromFile("tmp.png")
            self.viewer.show()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    okno = AppForMiI()
    sys.exit(app.exec_())

