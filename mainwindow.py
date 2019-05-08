import sys
from PyQt5 import QtWidgets , QtCore ,QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QLCDNumber, QTextEdit, QLabel, QProgressBar,QComboBox
import numpy
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy
from scipy import signal , fftpack
from PIL import Image, ImageDraw
import imageio
import math
import  scipy.fftpack as fftim
import math
from math import sqrt, pi, cos, sin, atan2
from collections import defaultdict
from scipy import ndimage
import scipy.ndimage.filters as filters
import random
import time



class Task2(QMainWindow):

    def __init__(self):
        super(Task2, self).__init__()
        loadUi('mainwindow.ui', self)
        self.pushButton_filters_load.clicked.connect(self.load)
        self.pushButton_histograms_load.clicked.connect(self.load_histo)
        self.pushButton_histograms_load_target.clicked.connect(self.load_Matchhisto)
        self.pushButton_lines_load.clicked.connect(self.linesLoad)
        self.pushButton_circles_load.clicked.connect(self.circlesLoad)
        self.segmentationPushButton.clicked.connect(self.load_segment)
        self.pushButton_corners_load.clicked.connect(self.load_corner)
        self.comboBox.addItems([' ','Prewitt','Sobel','Laplacian','LoG','DoG','box','Gaussian','Median',
                                'Sharpening','HIGH PASS FILTER','LOW_PASS_FILTER'])
        self.comboBox2.addItems([' ', 'Region Growing','Kmeans','Mean Shift','Snake'])
        self.comboBox.activated[str].connect(self.change_text)
        self.comboBox2.activated[str].connect(self.change_text2)
        self.radioButton.toggled.connect(lambda: self.btnstate(self.radioButton))
        self.radioButton_2.toggled.connect(lambda: self.btnstate(self.radioButton_2))

        self.label_seg_input.mousePressEvent = self.doSomething

        global Sx , Sy
        Sx,Sy=160,120



    def doSomething(self, event):
        global Sx, Sy
        print(event.x(),event.y())

        img = mpimg.imread(SegmentImage)


        Sx = int((event.x() * img.shape[1]) / self.label_seg_input.width())
        Sy = int((event.y() * img.shape[0]) / self.label_seg_input.height())
        print(Sx, Sy)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

        ax.scatter(Sx, Sy, marker='x', c='r', s=100)
        plt.axis('off')
        plt.savefig('Figure.png', bbox_inches='tight', pad_inches=0)
        Segoutpixmap = QtGui.QPixmap("Figure.png")  # Setup pixmap with the provided image
        Segoutpixmap = Segoutpixmap.scaled(self.label_seg_input.width(), self.label_seg_input.height(),
                                           QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_seg_input.setPixmap(Segoutpixmap)  # Set the pixmap onto the label
        self.label_seg_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""

    def change_text(self):
        current_value = self.comboBox.currentText()
        if(current_value=='Prewitt'):
            img=mpimg.imread(fileName)
            prewitt=self.prewitt_filter(img)
            scipy.misc.imsave("Filtered_Image.png", prewitt)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value=='Sobel'):
            img=mpimg.imread(fileName)
            Sobel = self.sobel_filter(img)
            scipy.misc.imsave("Filtered_Image.png", Sobel)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value=='Laplacian'):
            img = mpimg.imread(fileName)
            laplace = self.laplacian_filter(img)
            scipy.misc.imsave("Filtered_Image.png", laplace)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'LoG'):
            img = mpimg.imread(fileName)
            LoG = self.laplacian_filter(img)
            scipy.misc.imsave("Filtered_Image.png", LoG)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'DoG'):
            img = mpimg.imread(fileName)
            DoG = self.DoG_filter(img)
            scipy.misc.imsave("Filtered_Image.png", DoG)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'box'):
            img = mpimg.imread(fileName)
            box = self.box_filter(img)
            scipy.misc.imsave("Filtered_Image.png", box)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'Gaussian'):
            img = mpimg.imread(fileName)
            Gaussian = self.gaussian_filter(img)
            scipy.misc.imsave("Filtered_Image.png", Gaussian)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'Median'):
            img = mpimg.imread(fileName)
            Median = self.Median_filter(img)
            scipy.misc.imsave("Filtered_Image.png", Median)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'Sharpening'):
            img = mpimg.imread(fileName)
            Sharpening = self.sharpening_filter(img)
            scipy.misc.imsave("Filtered_Image.png", Sharpening)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'HIGH PASS FILTER'):
            img = mpimg.imread(fileName)
            HIGH_PASS_FILTER = self.HIGH_PASS_FILTER(img)
            scipy.misc.imsave("Filtered_Image.png", HIGH_PASS_FILTER)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'LOW_PASS_FILTER'):
            img = mpimg.imread(fileName)
            LOW_PASS_FILTER = self.LOW_PASS_FILTER(img)
            imageio.imwrite("Filtered_Image.png", LOW_PASS_FILTER)
            HMpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            HMpixmap = HMpixmap.scaled(self.label_filters_output.width(), self.label_filters_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_filters_output.setPixmap(HMpixmap)  # Set the pixmap onto the label
            self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""

    def change_text2(self):
        current_value = self.comboBox2.currentText()
        if (current_value == 'Mean Shift'):
            img = mpimg.imread(SegmentImage)
            threshold = self.plainTextEdit.document().toPlainText()
            R = self.Mean_Shift(img,int(threshold))
            scipy.misc.imsave("Filtered_Image.png", R)
            Segoutpixmap = QtGui.QPixmap("Filtered_Image.png")  # Setup pixmap with the provided image
            Segoutpixmap = Segoutpixmap.scaled(self.label_seg_output.width(), self.label_seg_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_seg_output.setPixmap(Segoutpixmap)  # Set the pixmap onto the label
            self.label_seg_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""

        elif (current_value == 'Kmeans'):
            img = mpimg.imread(SegmentImage)
            iterationNum = self.plainTextEdit_2.document().toPlainText()
            clusterNum = self.plainTextEdit_3.document().toPlainText()

            if iterationNum =='':
                iterationNum='6'
            if clusterNum=='':
                clusterNum='3'

            R = self.Kmeans(img,int(clusterNum),int(iterationNum))

            mpimg.imsave('temp.png', R)

            Segoutpixmap = QtGui.QPixmap('temp.png')  # Setup pixmap with the provided image
            Segoutpixmap = Segoutpixmap.scaled(self.label_seg_output.width(), self.label_seg_output.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_seg_output.setPixmap(Segoutpixmap)  # Set the pixmap onto the label
            self.label_seg_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif(current_value == 'Snake'):
            img = mpimg.imread(SegmentImage)
            img = self.rgb2gray(img)
            alpha = self.plainTextEdit_4.document().toPlainText()
            beta = self.plainTextEdit_5.document().toPlainText()
            gamma = self.plainTextEdit_6.document().toPlainText()
            iterations = self.plainTextEdit_7.document().toPlainText()
            r = self.plainTextEdit_8.document().toPlainText()

            if alpha =='':
                alpha='0.005'
            if beta=='':
                beta='0.005'
            if gamma =='':
                gamma='60'
            if iterations=='':
                iterations='200'
            if r =='':
                r='160'


            self.Snake_seg(img,float(alpha),float(beta),float(gamma),int(iterations),float(r))
            Segoutpixmap = QtGui.QPixmap('Figure.png')  # Setup pixmap with the provided image
            Segoutpixmap = Segoutpixmap.scaled(self.label_seg_output.width(), self.label_seg_output.height(),
                                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_seg_output.setPixmap(Segoutpixmap)  # Set the pixmap onto the label
            self.label_seg_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        elif (current_value == 'Region Growing'):
            img = mpimg.imread(SegmentImage)
            img = self.rgb2gray(img)


            self.find_region(Sx, Sy, SegmentImage)
            Segoutpixmap = QtGui.QPixmap('regiongrowing.jpg')  # Setup pixmap with the provided image
            Segoutpixmap = Segoutpixmap.scaled(self.label_seg_output.width(), self.label_seg_output.height(),
                                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_seg_output.setPixmap(Segoutpixmap)  # Set the pixmap onto the label
            self.label_seg_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""



    def load(self):
            global fileName
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "",
                                                                "Image Files (*.png *.jpg *jpeg *.bmp)")
            if fileName:  # If the user gives a file
                pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
                pixmap = pixmap.scaled(self.label_filters_input.width(), self.label_filters_input.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
                self.label_filters_input.setPixmap(pixmap)  # Set the pixmap onto the label
                self.label_filters_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
            return fileName

    def load_histo(self):
        global HistoImage
        HistoImage, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "",
                                                            "Image Files (*.png *.jpg *jpeg *.bmp)")
        if HistoImage:  # If the user gives a file
            Npixmap = QtGui.QPixmap(HistoImage)  # Setup pixmap with the provided image
            Npixmap = Npixmap.scaled(self.label_histograms_input.width(), self.label_histograms_input.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_histograms_input.setPixmap(Npixmap)  # Set the pixmap onto the label
            self.label_histograms_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
            return HistoImage

    def load_Matchhisto(self):
        global MatchImage
        MatchImage, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "",
                                                                  "Image Files (*.png *.jpg *jpeg *.bmp)")
        if MatchImage:  # If the user gives a file
            Mpixmap = QtGui.QPixmap(MatchImage)  # Setup pixmap with the provided image
            Mpixmap = Mpixmap.scaled(self.label_histograms_output.width(), self.label_histograms_output.height(),
                                         QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_histograms_output.setPixmap(Mpixmap)  # Set the pixmap onto the label
            self.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
            return MatchImage

    def btnstate(self, b):
        if b.text() == "Equalization":
            if b.isChecked() == True:
                image = mpimg.imread(HistoImage)
                Histo = self.DRAW_HISTO(image)
                scipy.misc.imsave("Histogram_Equalization_output.png", Histo)
                Hpixmap = QtGui.QPixmap("Histogram_Equalization_output.png")  # Setup pixmap with the provided image
                Hpixmap = Hpixmap.scaled(self.label_histograms_output.width(), self.label_histograms_output.height(),
                                           QtCore.Qt.KeepAspectRatio)  # Scale pixmap
                self.label_histograms_output.setPixmap(Hpixmap)  # Set the pixmap onto the label
                self.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""

        if b.text() == "Matching":
            if b.isChecked() == True:
                HImage = mpimg.imread(HistoImage)
                Mimage = mpimg.imread(MatchImage)
                HImage = np.dot(HImage[..., :3], [0.299, 0.587, 0.114])
                Mimage = np.dot(Mimage[..., :3], [0.299, 0.587, 0.114])
                Matched_image = self.HISTOGRAM_MATCHING(HImage,Mimage)
                scipy.misc.imsave("Histogram_matching_output.png", Matched_image)
                Hpixmap = QtGui.QPixmap("Histogram_matching_output.png")  # Setup pixmap with the provided image
                Hpixmap = Hpixmap.scaled(self.label_histograms_input.width(), self.label_histograms_input.height(),
                                         QtCore.Qt.KeepAspectRatio)  # Scale pixmap
                self.label_histograms_input.setPixmap(Hpixmap)  # Set the pixmap onto the label
                self.label_histograms_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""

    def linesLoad(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                            "Image Files (*.png *.jpg *jpeg *.bmp)")  # Ask for file
        if fileName:  # If the user gives a file

            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.label_lines_input.width(), self.label_lines_input.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_lines_input.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label_lines_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

            # self.print11()
            img = mpimg.imread(fileName)
            cannyImg = self.cannyEdge(img)

            accumulator, thetas, rhos = self.houghLine(cannyImg)

            theta, rs = self.getMaximumHoughLine(accumulator, 30, 50)
            mpimg.imsave('temp.jpg', accumulator)

            plt.imshow(accumulator, origin='lower', aspect='auto')
            # plt.show()

            pixmap = QtGui.QPixmap('temp.jpg')  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.label_lines_hough.width(), self.label_lines_hough.height())  # Scale pixmap
            self.label_lines_hough.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label_lines_hough.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

            plt.autoscale(False)
            plt.plot(theta, rs, 'ro')

            # plt.show()

            self.drawHoughLines(img, theta, rs)

            pixmap = QtGui.QPixmap('temp.jpg')  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.label_lines_input_2.width(), self.label_lines_input_2.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_lines_input_2.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label_lines_input_2.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

        return fileName

    def circlesLoad(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                            "Image Files (*.png *.jpg *jpeg *.bmp)")  # Ask for file
        if fileName:  # If the user gives a file
            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.label_circles_input.width(), self.label_circles_input.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_circles_input.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label_circles_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

            input_image = Image.open(fileName)

            # Output image:
            output_image = Image.new("RGB", input_image.size)
            output_image.paste(input_image)
            draw_result = ImageDraw.Draw(output_image)

            # Find circles
            rmin = 18
            rmax = 100
            steps = 100
            threshold = 0.35

            points = []
            for r in range(rmin, rmax + 1):
                for t in range(steps):
                    points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

            acc = defaultdict(int)
            for x, y in self.canny_edge_detector(input_image):
                for r, dx, dy in points:
                    a = x - dx
                    b = y - dy
                    acc[(a, b, r)] += 1

            circles = []
            for k, v in sorted(acc.items(), key=lambda i: -i[1]):
                x, y, r = k
                if v / steps >= threshold and all(
                        (x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                    print(v / steps, x, y, r)
                    circles.append((x, y, r))

            for x, y, r in circles:
                draw_result.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))

            imgplot = plt.imshow(output_image)
            plt.axis('off')
            plt.savefig('temp.jpg', bbox_inches='tight')

            pixmap = QtGui.QPixmap('temp.jpg')  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.label_circles_output.width(), self.label_circles_output.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_circles_output.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label_circles_output.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

        return fileName

    def load_segment(self):
            global SegmentImage
            global segpixmap
            SegmentImage, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "",
                                                                "Image Files (*.png *.jpg *jpeg *.bmp)")
            if SegmentImage:  # If the user gives a file
                segpixmap = QtGui.QPixmap(SegmentImage)  # Setup pixmap with the provided image
                segpixmap = segpixmap.scaled(self.label_seg_input.width(), self.label_seg_input.height(),
                                       QtCore.Qt.KeepAspectRatio)  # Scale pixmap
                self.label_seg_input.setPixmap(segpixmap)  # Set the pixmap onto the label
                self.label_seg_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
            return SegmentImage

    def load_corner(self):
        global CornerImage
        CornerImage, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "",
                                                               "Image Files (*.png *.jpg *jpeg *.bmp)")
        if CornerImage:  # If the user gives a file
            corpixmap = QtGui.QPixmap(CornerImage)  # Setup pixmap with the provided image
            corpixmap = corpixmap.scaled(self.label_corners_input.width(), self.label_corners_input.height(),
                                         QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_corners_input.setPixmap(corpixmap)  # Set the pixmap onto the label
            self.label_corners_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

        self.Hessian_corners(CornerImage)
        coroutpixmap = QtGui.QPixmap("corenerss.png")  # Setup pixmap with the provided image
        coroutpixmap = coroutpixmap.scaled(self.label_corners_corners.width(),self.label_corners_corners.height())  # Scale pixmap
        self.label_corners_corners.setPixmap(coroutpixmap)  # Set the pixmap onto the label
        self.label_corners_corners.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
        return CornerImage



    def gaussian_kernel(self, kernlen, std):
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def gaussianFilter(self, img):
        gKernel = self.gaussian_kernel(5, 1)
        FilteredImg = ndimage.filters.convolve(img, gKernel)

        return FilteredImg

    def sobelGrad(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)
        G = np.hypot(Ix, Iy)
        theta = np.arctan2(Iy, Ix)

        return (G, theta)

    def rgb2gray(self, rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    def threshold(self, img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        highThreshold = img.max() * highThresholdRatio;
        lowThreshold = highThreshold * lowThresholdRatio;

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res, weak, strong)

    def hysteresis(self, img, weak, strong=255):
        M, N = img.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (img[i, j] == weak):
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                        img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img

    def cannyEdge(self, img):
        grayImg = self.rgb2gray(img)
        smoothImg = self.gaussianFilter(grayImg)
        (grad, theta) = self.sobelGrad(smoothImg)
        afterNonMaxSuppression = self.non_max_suppression(grad, theta)
        (afterDoubleThreshold, weak, strong) = self.threshold(afterNonMaxSuppression)
        cannyImg = self.hysteresis(afterDoubleThreshold, weak)

        #  plt.subplot(2, 2, 1), plt.imshow(grayImg, cmap='gray')
        #  plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
        #  plt.subplot(2, 2, 2), plt.imshow(smoothImg, cmap='gray')
        #  plt.title('GaussianFilter'), plt.xticks([]), plt.yticks([])
        #  plt.subplot(2, 2, 3), plt.imshow(grad, cmap='gray')
        #  plt.title('sobelGrad'), plt.xticks([]), plt.yticks([])
        #  plt.subplot(2, 2, 4), plt.imshow(cannyImg, cmap='gray')
        # plt.title('Canny'), plt.xticks([]), plt.yticks([])
        # plt.show()

        return cannyImg

    def houghLine(self, image):
        # Get image dimensions
        # y for rows and x for columns
        Ny = image.shape[0]
        Nx = image.shape[1]

        # Max diatance is diagonal one
        Maxdist = int(np.round(np.sqrt(Nx ** 2 + Ny ** 2)))
        print(Maxdist)
        # Theta in range from -90 to 90 degrees
        thetas = np.deg2rad(np.arange(-90, 90))
        # Range of radius
        rs = np.linspace(-Maxdist, Maxdist, 2 * Maxdist)
        accumulator = np.zeros((2 * Maxdist, len(thetas)))

        for y in range(Ny):
            for x in range(Nx):
                if image[y, x] > 0:
                    # Map edge pixel to hough space
                    for k in range(len(thetas)):
                        # Calculate space parameter
                        r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                        # Update the accumulator
                        # N.B: r has value -max to max
                        # map r to its idx 0 : 2*max
                        accumulator[int(r) + Maxdist, k] += 1
        return accumulator, thetas, rs

    def getMaximumHoughLine(self, houghImg, neighborhood_size, threshold):
        neighborhood_size = neighborhood_size
        threshold = threshold

        hough_space = houghImg
        data_max = filters.maximum_filter(hough_space, neighborhood_size)
        maxima = (hough_space == data_max)

        data_min = filters.minimum_filter(hough_space, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)

        theta, rs = [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) / 2
            theta.append(x_center)
            y_center = (dy.start + dy.stop - 1) / 2
            rs.append(y_center)

        print(theta)
        print(rs)
        return theta, rs

    def drawHoughLines(self, originalImg, theta, rs):
        img_shape = originalImg.shape
        y_max = img_shape[0]
        x_max = img_shape[1]
        Maxdist = int(np.round(np.sqrt(x_max ** 2 + y_max ** 2)))
        fig, ax = plt.subplots()
        ax.imshow(originalImg)
        for i in range(len(theta)):
            t = (theta[i] - 90) * np.pi / 180
            r = rs[i] - Maxdist
            x = np.linspace(0, x_max, x_max)
            y = (-np.cos(t) * x / np.sin(t)) + (r / np.sin(t))
            ax.plot(x, y);

        plt.ylim(y_max, 0);
        plt.xlim(0, x_max);
        plt.axis('off')
        plt.savefig('temp.jpg', bbox_inches='tight')

    def canny_edge_detector(self, input_image):
        input_pixels = input_image.load()
        width = input_image.width
        height = input_image.height

        # Transform the image to grayscale
        grayscaled = np.empty((width, height))
        for x in range(width):
            for y in range(height):
                pixel = input_pixels[x, y]
                grayscaled[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3

        # Blur it to remove noise

        clip = lambda x, l, u: l if x < l else u if x > u else x

        kernel = np.array([
            [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256]
        ])

        offset = len(kernel) // 2

        blurred = np.empty((width, height))
        for x in range(width):
            for y in range(height):
                acc = 0
                for a in range(len(kernel)):
                    for b in range(len(kernel)):
                        xn = clip(x + a - offset, 0, width - 1)
                        yn = clip(y + b - offset, 0, height - 1)
                        acc += grayscaled[xn, yn] * kernel[a, b]
                blurred[x, y] = int(acc)

        # Compute the gradient
        gradient = np.zeros((width, height))
        direction = np.zeros((width, height))
        for x in range(width):
            for y in range(height):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    magx = blurred[x + 1, y] - blurred[x - 1, y]
                    magy = blurred[x, y + 1] - blurred[x, y - 1]
                    gradient[x, y] = sqrt(magx ** 2 + magy ** 2)
                    direction[x, y] = atan2(magy, magx)

            # Non-maximum suppression
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
                    rangle = round(angle / (pi / 4))
                    mag = gradient[x, y]
                    if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                            or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                            or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                            or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                        gradient[x, y] = 0

        # Filter out some edges

        # Keep strong edges
        keep = set()
        for x in range(width):
            for y in range(height):
                if gradient[x, y] > 25:
                    keep.add((x, y))

        # Keep weak edges next to a pixel to keep
        lastiter = keep
        while lastiter:
            newkeep = set()
            for x, y in lastiter:
                for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    if gradient[x + a, y + b] > 20 and (x + a, y + b) not in keep:
                        newkeep.add((x + a, y + b))
            keep.update(newkeep)
            lastiter = newkeep

        keep = list(keep)

        return keep

    def prewitt_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        height = img.shape[0]
        width = img.shape[1]

        x_prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        y_prewitt = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        global newgradientImage
        newgradientImage = np.zeros((height, width))
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                Gx = (x_prewitt[0, 0] * img[i - 1, j - 1]) + \
                     (x_prewitt[0, 1] * img[i - 1, j]) + \
                     (x_prewitt[0, 2] * img[i - 1, j + 1]) + \
                     (x_prewitt[1, 0] * img[i, j - 1]) + \
                     (x_prewitt[1, 1] * img[i, j]) + \
                     (x_prewitt[1, 2] * img[i, j + 1]) + \
                     (x_prewitt[2, 0] * img[i + 1, j - 1]) + \
                     (x_prewitt[2, 1] * img[i + 1, j]) + \
                     (x_prewitt[2, 2] * img[i + 1, j + 1])

                Gy = (y_prewitt[0, 0] * img[i - 1, j - 1]) + \
                     (y_prewitt[0, 1] * img[i - 1, j]) + \
                     (y_prewitt[0, 2] * img[i - 1, j + 1]) + \
                     (y_prewitt[1, 0] * img[i, j - 1]) + \
                     (y_prewitt[1, 1] * img[i, j]) + \
                     (y_prewitt[1, 2] * img[i, j + 1]) + \
                     (y_prewitt[2, 0] * img[i + 1, j - 1]) + \
                     (y_prewitt[2, 1] * img[i + 1, j]) + \
                     (y_prewitt[2, 2] * img[i + 1, j + 1])
                Magnitude = np.sqrt(pow(Gx, 2.0) + pow(Gy, 2.0))
                newgradientImage[i - 1][j - 1] = Magnitude
        return newgradientImage

    def sobel_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        height = img.shape[0]
        width = img.shape[1]
        x_sob = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)], dtype=np.float)
        y_sob = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)], dtype=np.float)
        X_image = np.zeros((height, width))
        y_image = np.zeros((height, width))
        Grad_image = np.zeros((height, width))

        for i in range(0, height - 1):
            for j in range(0, width - 1):
                gx = (x_sob[0][0] * img[i - 1][j - 1]) + (x_sob[0][1] * img[i - 1][j]) + \
                     (x_sob[0][2] * img[i - 1][j + 1]) + (x_sob[1][0] * img[i][j - 1]) + \
                     (x_sob[1][1] * img[i][j]) + (x_sob[1][2] * img[i][j + 1]) + \
                     (x_sob[2][0] * img[i + 1][j - 1]) + (x_sob[2][1] * img[i + 1][j]) + \
                     (x_sob[2][2] * img[i + 1][j + 1])

                gy = (y_sob[0][0] * img[i - 1][j - 1]) + (y_sob[0][1] * img[i - 1][j]) + \
                     (y_sob[0][2] * img[i - 1][j + 1]) + (y_sob[1][0] * img[i][j - 1]) + \
                     (y_sob[1][1] * img[i][j]) + (y_sob[1][2] * img[i][j + 1]) + \
                     (y_sob[2][0] * img[i + 1][j - 1]) + (y_sob[2][1] * img[i + 1][j]) + \
                     (y_sob[2][2] * img[i + 1][j + 1])
                g = np.sqrt(gx * gx + gy * gy)

                X_image[i - 1][j - 1] = gx
                y_image[i - 1][j - 1] = gy
                Grad_image[i - 1][j - 1] = g
        return Grad_image

    def laplacian_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        G = np.absolute(scipy.signal.convolve2d(img, lap_kernel, 'same'))
        return G

    def LoG_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        Gaussian_Filter33 = np.array([[0.102059, 0.115349, 0.102059]
                                     , [0.115349, 0.130371, 0.115349],
                                  [0.102059, 0.115349, 0.102059]])
        Gaussian_Filter = np.absolute(scipy.signal.convolve2d(img, Gaussian_Filter33, boundary='symm', mode='same'))
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        G = np.absolute(scipy.signal.convolve2d(Gaussian_Filter, lap_kernel, 'same'))
        return G

    def DoG_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        Gaussian_Filter33 = np.array([[0.102059, 0.115349, 0.102059]
                                         , [0.115349, 0.130371, 0.115349],
                                      [0.102059, 0.115349, 0.102059]])
        Gaussian_Filter3 = scipy.signal.convolve2d(img, Gaussian_Filter33, boundary='symm', mode='same')

        Gaussian_Filter55 = np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                      [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                      [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                      [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                      [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        Gaussian_Filter5 = scipy.signal.convolve2d(img, Gaussian_Filter55, boundary='symm', mode='same')

        Diff = np.absolute(Gaussian_Filter3 - Gaussian_Filter5)
        return Diff

    def box_filter(self,img):
        w = 4
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        weight = 2
        noisy = img + weight * img.std() * np.random.random(img.shape)
        imageio.imwrite("noisy.png", noisy)
        pixmap = QtGui.QPixmap("noisy.png")  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_filters_input.width(), self.label_filters_input.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_filters_input.setPixmap(pixmap)  # Set the pixmap onto the label
        self.label_filters_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
        Box_filter = np.ones((w, w)) / (w * w)
        filtered_img_box = signal.convolve2d(img, Box_filter, 'same')
        return filtered_img_box

    def gaussian_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        weight = 2
        noisy = img + weight * img.std() * np.random.random(img.shape)
        imageio.imwrite("noisy.png", noisy)
        pixmap = QtGui.QPixmap("noisy.png")  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_filters_input.width(), self.label_filters_input.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_filters_input.setPixmap(pixmap)  # Set the pixmap onto the label
        self.label_filters_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
        Gaussian_Filter33 = np.array([[0.102059, 0.115349, 0.102059]
                                         , [0.115349, 0.130371, 0.115349],
                                      [0.102059, 0.115349, 0.102059]])
        Gaussian_Filter3 = scipy.signal.convolve2d(img, Gaussian_Filter33, boundary='symm', mode='same')
        return Gaussian_Filter3

    def Median_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        salt_pepper = np.random.random(img.shape) * 255

        pepper = salt_pepper < 30;
        salt = salt_pepper > 225;

        img[pepper] = 0
        img[salt] = 255
        imageio.imwrite("noisy.png", img)
        pixmap = QtGui.QPixmap("noisy.png")  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_filters_input.width(), self.label_filters_input.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_filters_input.setPixmap(pixmap)  # Set the pixmap onto the label
        self.label_filters_input.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
        height = img.shape[0]
        width = img.shape[1]
        image = np.zeros((height, width))
        members = [(0, 0)] * 9
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                members[0] = img[i - 1][j - 1]
                members[1] = img[i - 1][j]
                members[2] = img[i - 1][j + 1]
                members[3] = img[i][j - 1]
                members[4] = img[i][j]
                members[5] = img[i][j + 1]
                members[6] = img[i + 1][j - 1]
                members[7] = img[i + 1][j]
                members[8] = img[i + 1][j + 1]
                members.sort()
                image[i][j] = members[4]
        return image

    def sharpening_filter(self,img):
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        sharp_kernel = np.array([[0, -.5, 0], [-.5, 3, -.5], [0, -.5, 0]])
        sharpe_image = scipy.signal.convolve2d(img, sharp_kernel, boundary='symm', mode='same')
        return sharpe_image

    def HIGH_PASS_FILTER(self,image):
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        image = numpy.asarray(image)
        image = fftim.fftshift(fftim.fft2(image))
        # variables for conv function
        M = image.shape[0]
        N = image.shape[1]
        H = numpy.ones((M, N))
        center1 = M / 2
        center2 = N / 2
        d_0 = 30.0  # cut off radius
        t1 = 2 * d_0
        # low pass filter conv function
        for i in range(1, M):
            for j in range(1, N):
                r1 = (i - center1) ** 2 + (j - center2) ** 2
                r = math.sqrt(r1)
                # usong cut off radius to eliminate low freq
                if 0 < r < d_0:
                    H[i, j] = 1 - math.exp(-r ** 2 / t1 ** 2)
        # convert H to an image
        H = Image.fromarray(H)
        # perform conv
        conv = image * H
        # compute mag of the inverse fft
        e = abs(fftim.ifft2(conv))
        # convert e from array to image
        img_back = Image.fromarray(e)
        return img_back

    def LOW_PASS_FILTER(self, image):
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        image = fftim.fftshift(fftim.fft2(image))
        # variables for conv function
        M = image.shape[0]
        N = image.shape[1]
        H = numpy.ones((M, N))
        center1 = M / 2
        center2 = N / 2
        d_0 = 30.0  # cut off radius
        # low pass filter conv function
        for i in range(1, M):
            for j in range(1, N):
                r1 = (i - center1) ** 2 + (j - center2) ** 2
                r = math.sqrt(r1)
                # usong cut off radius to eliminate high freq
                if r > d_0:
                    H[i, j] = 0.0
        # convert H to an image
        H = Image.fromarray(H)
        # perform conv
        conv = image * H
        # compute mag of the inverse fft
        e = abs(fftim.ifft2(conv))
        # convert e from array to image
        filtered = Image.fromarray(e)
        return filtered

    def DRAW_HISTO(self,image):
            img1 = np.asarray(image)
            # put pixels in a 1D array by flattening out img array
            flat = img1.flatten()
            # execute our histogram function
            hist = self.get_histogram(flat)
            # execute the fn
            cs = self.cumsum(hist)
            # get the value from cumulative sum for every index in flat, and set that as img_new
            img_new = cs[flat]
            # put array back into original shape since we flattened it
            img_new = np.reshape(img_new, image.shape)
            imags = self.get_histogram(img_new)
            fig2, ax2 = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
            ax2.plot(hist)
            plt.savefig("Hisogram1.png")
            plt.close(fig2)  # close the figure
            Hpixmap1 = QtGui.QPixmap("Hisogram1.png")  # Setup pixmap with the provided image
            Hpixmap1 = Hpixmap1.scaled(self.label_histograms_hinput.width(), self.label_histograms_hinput.height(),
                                     QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_histograms_hinput.setPixmap(Hpixmap1)  # Set the pixmap onto the label
            self.label_histograms_hinput.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
            return img_new

        # create our own histogram function

    def HISTOGRAM_MATCHING(self,template, image):
        matched = self.hist_match(image, template)
        HImage = mpimg.imread(HistoImage)
        Mimage = mpimg.imread(MatchImage)
        HImage = np.dot(HImage[..., :3], [0.299, 0.587, 0.114])
        Mimage = np.dot(Mimage[..., :3], [0.299, 0.587, 0.114])
        Matched_histo1=self.get_histogram(HImage)
        Matched_histo2=self.get_histogram(Mimage)
        Matched_histo3=self.get_histogram(matched)

        fig1, ax1 = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax1.plot(Matched_histo1)
        plt.savefig("Hisogram.png")
        plt.close(fig1)  # close the figure
        Hpixmap2 = QtGui.QPixmap("Hisogram.png")  # Setup pixmap with the provided image
        Hpixmap2 = Hpixmap2.scaled(self.label_histograms_houtput.width(), self.label_histograms_houtput.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_histograms_houtput.setPixmap(Hpixmap2)  # Set the pixmap onto the label
        self.label_histograms_houtput.setAlignment(
            QtCore.Qt.AlignCenter)  # Align the label to center""

        fig2, ax2 = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax2.plot(Matched_histo2)
        plt.savefig("Matched_histo2.png")
        plt.close(fig2)  # close the figure
        Hpixmap4 = QtGui.QPixmap("Matched_histo2.png")  # Setup pixmap with the provided image
        Hpixmap4 = Hpixmap4.scaled(self.label_histograms_hinput.width(), self.label_histograms_hinput.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_histograms_hinput.setPixmap(Hpixmap4)  # Set the pixmap onto the label
        self.label_histograms_hinput.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""

        fig3, ax3 = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax3.plot(Matched_histo3)
        plt.savefig("Matched_histo3.png")
        plt.close(fig3)  # close the figure
        Hpixmap3 = QtGui.QPixmap("Matched_histo3.png")  # Setup pixmap with the provided image
        Hpixmap3 = Hpixmap3.scaled(self.label_histograms_hinput.width(), self.label_histograms_hinput.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_histograms_hinput.setPixmap(Hpixmap3)  # Set the pixmap onto the label
        self.label_histograms_hinput.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center""
        return matched

    def get_histogram(self,img):
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])
        fig1, ax1 = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax1.plot(hist)
        plt.savefig("Hisogram.png")
        plt.close(fig1)  # close the figure
        Hpixmap2 = QtGui.QPixmap("Hisogram.png")  # Setup pixmap with the provided image
        Hpixmap2 = Hpixmap2.scaled(self.label_histograms_houtput.width(), self.label_histograms_houtput.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_histograms_houtput.setPixmap(Hpixmap2)  # Set the pixmap onto the label
        self.label_histograms_houtput.setAlignment(
            QtCore.Qt.AlignCenter)  # Align the label to center""        return hist
        return hist

    # create our cumulative sum function
    def cumsum(self,a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def hist_match(self,image, template):
        oldshape = image.shape
        source = image.ravel()
        template = template.ravel()
        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        return interp_t_values[bin_idx].reshape(oldshape)

    def ecdf(self,x):
        # """convenience function for computing the empirical CDF"""
        vals, counts = np.unique(x, return_counts=True)
        ecdf = np.cumsum(counts).astype(np.float64)
        ecdf /= ecdf[-1]
        return vals, ecdf

    def Mean_Shift(self,K, threshold):
        row = K.shape[0]
        col = K.shape[1]

        J = row * col
        Size = row, col, 3
        R = np.zeros(Size, dtype=np.uint8)
        D = np.zeros((J, 5))
        arr = np.array((1, 3))

        counter = 0
        iter = 1.0
        current_mean_random = True
        current_mean_arr = np.zeros((1, 5))
        below_threshold_arr = []

        # converted the image K[rows][col] into a feature space D. The dimensions of D are [rows*col][5]
        for i in range(0, row):
            for j in range(0, col):
                arr = K[i][j]

                for k in range(0, 5):
                    if (k >= 0) & (k <= 2):
                        D[counter][k] = arr[k]
                    else:
                        if (k == 3):
                            D[counter][k] = i
                        else:
                            D[counter][k] = j
                counter += 1

        while (len(D) > 0):
            # selecting a random row from the feature space and assigning it as the current mean
            if (current_mean_random):
                current_mean = random.randint(0, len(D) - 1)
                for i in range(0, 5):
                    current_mean_arr[0][i] = D[current_mean][i]
            below_threshold_arr = []
            for i in range(0, len(D)):
                # print "Entered here"
                ecl_dist = 0
                color_total_current = 0
                color_total_new = 0
                # Finding the eucledian distance of the randomly selected row i.e. current mean with all the other rows
                for j in range(0, 5):
                    ecl_dist += ((current_mean_arr[0][j] - D[i][j]) ** 2)

                ecl_dist = ecl_dist ** 0.5

                # Checking if the distance calculated is within the threshold. If yes taking those rows and adding
                # them to a list below_threshold_arr

                if (ecl_dist < threshold):
                    below_threshold_arr.append(i)
                    # print "came here"

            mean_R = 0
            mean_G = 0
            mean_B = 0
            mean_i = 0
            mean_j = 0
            current_mean = 0
            mean_col = 0

            # For all the rows found and placed in below_threshold_arr list, calculating the average of
            # Red, Green, Blue and index positions.

            for i in range(0, len(below_threshold_arr)):
                mean_R += D[below_threshold_arr[i]][0]
                mean_G += D[below_threshold_arr[i]][1]
                mean_B += D[below_threshold_arr[i]][2]
                mean_i += D[below_threshold_arr[i]][3]
                mean_j += D[below_threshold_arr[i]][4]

            mean_R = mean_R / len(below_threshold_arr)
            mean_G = mean_G / len(below_threshold_arr)
            mean_B = mean_B / len(below_threshold_arr)
            mean_i = mean_i / len(below_threshold_arr)
            mean_j = mean_j / len(below_threshold_arr)

            # Finding the distance of these average values with the current mean and comparing it with iter

            mean_e_distance = ((mean_R - current_mean_arr[0][0]) ** 2 + (mean_G - current_mean_arr[0][1]) ** 2 + (
                    mean_B - current_mean_arr[0][2]) ** 2 + (mean_i - current_mean_arr[0][3]) ** 2 + (
                                       mean_j - current_mean_arr[0][4]) ** 2)

            mean_e_distance = mean_e_distance ** 0.5

            nearest_i = 0
            min_e_dist = 0
            counter_threshold = 0
            # If less than iter, find the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
            # This is because mean_i and mean_j could be decimal values which do not correspond
            # to actual pixel in the Image array.

            if (mean_e_distance < iter):

                new_arr = np.zeros((1, 3))
                new_arr[0][0] = mean_R
                new_arr[0][1] = mean_G
                new_arr[0][2] = mean_B

                # When found, color all the rows in below_threshold_arr with
                # the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
                for i in range(0, len(below_threshold_arr)):
                    R[int(D[below_threshold_arr[i]][3])][int(D[below_threshold_arr[i]][4])] = new_arr

                    # Also now don't use those rows that have been colored once.

                    D[below_threshold_arr[i]][0] = -1
                current_mean_random = True
                new_D = np.zeros((len(D), 5))
                counter_i = 0
                for i in range(0, len(D)):
                    if (D[i][0] != -1):
                        new_D[counter_i][0] = D[i][0]
                        new_D[counter_i][1] = D[i][1]
                        new_D[counter_i][2] = D[i][2]
                        new_D[counter_i][3] = D[i][3]
                        new_D[counter_i][4] = D[i][4]
                        counter_i += 1

                D = np.zeros((counter_i, 5))

                counter_i -= 1
                for i in range(0, counter_i):
                    D[i][0] = new_D[i][0]
                    D[i][1] = new_D[i][1]
                    D[i][2] = new_D[i][2]
                    D[i][3] = new_D[i][3]
                    D[i][4] = new_D[i][4]

            else:
                current_mean_random = False

                current_mean_arr[0][0] = mean_R
                current_mean_arr[0][1] = mean_G
                current_mean_arr[0][2] = mean_B
                current_mean_arr[0][3] = mean_i
                current_mean_arr[0][4] = mean_j
        return R

    def getFeatureSpace(self,Img):

        featureSpace = np.ones((256, 256))  # 2d array of r and g componenets
        for y in range(Img.shape[0]):
            for x in range(Img.shape[1]):
                featureSpace[Img[y, x, 0], Img[y, x, 1]] = 0
        return featureSpace

    def getRandomMeans(self,k, featureSpace):

        means = []  # 2d array of mean values (y,x)
        for i in range(k):
            means.append([np.random.randint(0, featureSpace.shape[0]), np.random.randint(0, featureSpace.shape[1])])

        return means

    def Kmeans(self,originalImg, k, iterationNum):

        featureSpaceImg = self.getFeatureSpace(originalImg)
        means = self.getRandomMeans(k, featureSpaceImg)
        distances = np.zeros(k)

        for i in range(iterationNum):
            clusterColors = []
            clusterSum = []
            clusterCount = []
            # plt.imshow(featureSpaceImg, cmap='gray')
            for clusterNum in range(k):
                clusterSum.append((0, 0))
                clusterColors.append([abs(np.random.rand()), abs(np.random.rand()), abs(np.random.rand())])
                clusterCount.append((0, 0))
                # print(clusterColors[clusterNum])
                # plt.scatter(means[clusterNum][0], means[clusterNum][1], marker='x', c='b', s=100)

            # plt.show()
            # time.sleep(0.5)

            for y in range(featureSpaceImg.shape[0]):
                for x in range(featureSpaceImg.shape[1]):
                    if featureSpaceImg[y, x] == 0:

                        for clusterNum in range(k):
                            distances[clusterNum] = np.round(
                                np.sqrt((y - means[clusterNum][0]) ** 2 + (x - means[clusterNum][1]) ** 2))
                        # print(means)
                        # print(means[clusterNum][0])
                        pixelClusterNum = np.argmin(distances, axis=0)
                        clusterSum[pixelClusterNum] = np.add(clusterSum[pixelClusterNum], (y, x))
                        # print(clusterSum)
                        clusterCount[pixelClusterNum] = np.add(clusterCount[pixelClusterNum], (1, 1))
                        # print(clusterCount[pixelClusterNum])
                        # print(clusterSum[pixelClusterNum])
                        # plt.scatter(y, x, marker='.', c=clusterColors[pixelClusterNum], s=1)
            '''
            print("*********************************")
            print(means)
            print(clusterSum)
            print(clusterCount)
            print("*********************************")
            '''
            for clusterNum in range(k):

                if (clusterCount[clusterNum][0] > 0):
                    means[clusterNum] = clusterSum[clusterNum] / clusterCount[clusterNum]

            # means=int(np.divide(clusterSum,clusterCount))
            # plt.show()
        print('sdasds')
        img = np.zeros((originalImg.shape[0], originalImg.shape[1], 3), 'int8')

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for clusterNum in range(k):
                    distances[clusterNum] = np.round(np.sqrt((originalImg[y, x, 0] - means[clusterNum][0]) ** 2 + (
                                originalImg[y, x, 1] - means[clusterNum][1]) ** 2))
                pixelClusterNum = np.argmin(distances, axis=0)

                # img[y,x]= clusterColors[pixelClusterNum]
                img[y, x] = np.multiply(clusterColors[pixelClusterNum], 255)
            #    plt.scatter(y, x, marker='.', c=clusterColors[pixelClusterNum], s=1)

        #plt.imshow(featureSpaceImg, cmap='gray')
        #for clusterNum in range(k):
         #   plt.scatter(means[clusterNum][0], means[clusterNum][1], marker='x', c='r', s=100)
        #plt.show()

        #print(np.multiply(clusterColors, 255))
        return img

    def Snake_seg(self,img,alpha,beta,gamma,iterations,r):

        t = np.arange(0, 2 * np.pi, 0.1)
        x = Sx + r * np.cos(t)
        y = Sy + r * np.sin(t)

        # fx and fy are callable functions
        fx, fy = self.create_external_edge_force_gradients_from_img(img, sigma=5)

        snakes = self.iterate_snake(
            x=x,
            y=y,
            a=alpha,
            b=beta,
            fx=fx,
            fy=fy,
            gamma=gamma,
            n_iters=iterations,
            return_all=True
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], c=(0, 1, 0), lw=2)

        for i, snake in enumerate(snakes):
            if i % 10 == 0:
                ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0, 0, 1), lw=2)

        # Plot the last one a different color.
        ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1, 0, 0), lw=2)
        plt.axis('off')
        plt.savefig('Figure.png', bbox_inches='tight',pad_inches=0)

    def create_A(self,a, b, N):
        """
        a: float
        alpha parameter

        b: float
        beta parameter

        N: int
        N is the number of points sampled on the snake curve: (x(p_i), y(p_i)), i=0,...,N-1
        """
        row = np.r_[
            -2 * a - 6 * b,
            a + 4 * b,
            -b,
            np.zeros(N - 5),
            -b,
            a + 4 * b
        ]
        A = np.zeros((N, N))
        for i in range(N):
            A[i] = np.roll(row, i)
        return A

    def create_external_edge_force_gradients_from_img(self,img, sigma=30.):
        """
        Given an image, returns 2 functions, fx & fy, that compute
        the gradient of the external edge force in the x and y directions.

        img: ndarray
            The image.
        """
        # Gaussian smoothing.
        smoothed = filters.gaussian_filter((img - img.min()) / (img.max() - img.min()), sigma)
        # Gradient of the image in x and y directions.
        giy, gix = np.gradient(smoothed)
        # Gradient magnitude of the image.
        gmi = (gix ** 2 + giy ** 2) ** (0.5)
        # Normalize. This is crucial (empirical observation).
        gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())

        # Gradient of gradient magnitude of the image in x and y directions.
        ggmiy, ggmix = np.gradient(gmi)

        def fx(x, y):
            """
            Return external edge force in the x direction.

            x: ndarray
                numpy array of floats.
            y: ndarray:
                numpy array of floats.
            """
            # Check bounds.
            x[x < 0] = 0.
            y[y < 0] = 0.

            x[x > img.shape[1] - 1] = img.shape[1] - 1
            y[y > img.shape[0] - 1] = img.shape[0] - 1

            return ggmix[(y.round().astype(int), x.round().astype(int))]

        def fy(x, y):
            """
            Return external edge force in the y direction.

            x: ndarray
                numpy array of floats.
            y: ndarray:
                numpy array of floats.
            """
            # Check bounds.
            x[x < 0] = 0.
            y[y < 0] = 0.

            x[x > img.shape[1] - 1] = img.shape[1] - 1
            y[y > img.shape[0] - 1] = img.shape[0] - 1

            return ggmiy[(y.round().astype(int), x.round().astype(int))]

        return fx, fy

    def iterate_snake(self,x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
        """
        x: ndarray
            intial x coordinates of the snake

        y: ndarray
            initial y coordinates of the snake

        a: float
            alpha parameter

        b: float
            beta parameter

        fx: callable
            partial derivative of first coordinate of external energy function. This is the first element of the gradient of the external energy.

        fy: callable
            see fx.

        gamma: float
            step size of the iteration

        n_iters: int
            number of times to iterate the snake

        return_all: bool
            if True, a list of (x,y) coords are returned corresponding to each iteration.
            if False, the (x,y) coords of the last iteration are returned.
        """
        A = self.create_A(a, b, x.shape[0])
        B = np.linalg.inv(np.eye(x.shape[0]) - gamma * A)
        if return_all:
            snakes = []

        for i in range(n_iters):
            x_ = np.dot(B, x + gamma * fx(x, y))
            y_ = np.dot(B, y + gamma * fy(x, y))
            x, y = x_.copy(), y_.copy()
            if return_all:
                snakes.append((x_.copy(), y_.copy()))

        if return_all:
            return snakes
        else:
            return (x, y)

    def Hessian_corners(self, image):
        # ---------------- sobel --------------------------#
        image = mpimg.imread(image)
        img = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        height = img.shape[0]
        width = img.shape[1]
        x_sob = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)], dtype=np.float)
        y_sob = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)], dtype=np.float)
        Ix = signal.convolve2d(img, x_sob, 'same')
        Iy = signal.convolve2d(img, y_sob, 'same')
        # ---------------- Hessian matrix -----------------#
        Ixx = np.multiply(Ix, Iy)
        Iyy = np.multiply(Iy, Iy)
        Ixy = np.multiply(Ix, Iy)
        Ixx_hat = signal.convolve2d(Ixx, self.boxx_filter(1), 'same')
        Iyy_hat = signal.convolve2d(Iyy, self.boxx_filter(1), 'same')
        Ixy_hat = signal.convolve2d(Ixy, self.boxx_filter(1), 'same')
        K = 0.05
        detM = np.multiply(Ixx_hat, Iyy_hat) - np.multiply(Ixy_hat, Ixy_hat)
        trM = Ixx_hat + Iyy_hat
        R = detM - K * trM
        corners = np.abs(R) > np.quantile(np.abs(R), 0.999)
        pos = np.argwhere(corners)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap=plt.cm.gray)
        ax.scatter(pos[:, 1], pos[:, 0], c='b', marker='x')
        plt.axis('off')
        plt.savefig('corenerss.png', bbox_inches='tight', pad_inches=0)
        return corners

    def boxx_filter(self, w):
        box_filter = np.ones((w, w)) / (w * w)
        return box_filter

    def find_region(self, Sx, Sy, image):
        image = mpimg.imread(image)
        arr = np.asarray(image)

        rows, columns = np.shape(arr)
        seed_pixel = []
        seed_pixel.append(Sx)
        seed_pixel.append(Sy)
        img_rg = np.zeros((rows + 1, columns + 1))
        img_display = np.zeros((rows, columns))
        region_points = []
        region_points.append([Sx, Sy])
        count = 0
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]

        while (len(region_points) > 0):

            if count == 0:
                point = region_points.pop(0)
                i = point[0]
                j = point[1]
            val = arr[i][j]
            lt = val - 8
            ht = val + 8
            for k in range(8):
                # print '\ncomparison val:',val, 'ht',ht,'lt',lt
                if img_rg[i + x[k]][j + y[k]] != 1:
                    try:
                        if arr[i + x[k]][j + y[k]] > lt and arr[i + x[k]][j + y[k]] < ht:
                            img_rg[i + x[k]][j + y[k]] = 1
                            p = [0, 0]
                            p[0] = i + x[k]
                            p[1] = j + y[k]
                            if p not in region_points:
                                if 0 < p[0] < rows and 0 < p[1] < columns:
                                    ''' adding points to the region '''
                                    region_points.append([i + x[k], j + y[k]])
                        else:
                            img_rg[i + x[k]][j + y[k]] = 0
                    except IndexError:
                        continue

            point = region_points.pop(0)
            i = point[0]
            j = point[1]
            count = count + 1
        plt.figure()
        plt.imshow(img_rg, cmap=plt.cm.gray)
        plt.gray()
        plt.imsave('regiongrowing.jpg', img_rg)





if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    app = QApplication(sys.argv)
    widget = Task2()
    widget.show()
    sys.exit(app.exec_())