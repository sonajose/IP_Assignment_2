import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import cv2
import numpy as np
import Full_inv as fi
import weiner_inv as wi
import trunc_inv as ti
import clsf as cl

from CuteUI import Ui_MainWindow

class Firstwindow(QMainWindow):

    image_in = np.array([0])
    im_prev = np.array([0])
    blur_psf = np.array([0])
    original =np.array([0])
    psnr = 0
    SSIM = 0

    def __init__(self):
        super(Firstwindow, self).__init__()
        self.gui = Ui_MainWindow()
        self.gui.setupUi(self)
        self.gui.chooseimage.clicked.connect(lambda: self.open())
        self.gui.psnr.clicked.connect(lambda: self.compute_psnr())
        self.gui.ssim.clicked.connect(lambda: self.compute_ssim())
        self.gui.fullinverse.clicked.connect(lambda: self.full_inv())
        self.gui.truncinv.clicked.connect(lambda: self.trunc_inv())
        self.gui.weiner.clicked.connect(lambda: self.weiner())
        self.gui.constls.clicked.connect(lambda: self.const_ls())
        self.gui.save.clicked.connect(lambda: self.save_im())
        self.gui.undoall.clicked.connect(lambda: self.fullundo())

    
    def compute_psnr(self):
	self.gui.psnrdispaly.setText(str(psnr))
    
        

    def compute_ssim(self):
        self.gui.ssimdisp.setText(str(SSIM))
    def full_inv(self):
	 out, psnr, SSIM = fi.full_inverse(image_in, blur_psf, original)

    def trunc_inv(self):
         out, psnr, SSIM=ti.trunc(image_in, blur_psf,original,R)
    def weiner(self):
         out, psnr, SSIM=wi.weiner_inv(image_in, blur_psf,original,k)
    def const_ls(self):
        out, psnr, SSIM = cl.constrained_least_square(image_in, blur_psf,original,gamma)
    def save_im(self): #function to save
        savewindow = QFileDialog()
        savewindow.setDefaultSuffix('jpg')
        savewindow.setAcceptMode(QFileDialog.AcceptSave)

        if savewindow.exec_() ==QDialog.Accepted: #if save button is pressed
            save = savewindow.selectedFiles()[0]
            cv2.imwrite(save, cv2.cvtColor(self.image_in, cv2.COLOR_HSV2BGR)) #write the image to desired location

    def fullundo(self):
        self.image_in = self.im.copy() #Copies the current image to im_prev
        self.display() #Display the image



    def open(self): #For opening the image
        file_name = QFileDialog.getOpenFileName(QFileDialog(), 'Browse', '/') #Selecting the file

        if file_name: #If an image is chosen
            self.image_in = cv2.imread(file_name, 1) #Reading the image
            self.image_in = cv2.cvtColor(self.image_in, cv2.COLOR_BGR2HSV) #Converts to HSV
            self.im = self.image_in.copy() #copies the image to im
            self.im_prev = self.image_in.copy() #Copies the current image to im_prev
            self.display() #Display the image

            self.displayOriginalImage() #display the image in 'Original Image' box

    def display(self): #function to display the image
        size = self.gui.editedimage.size() #Finding the size of the edited image
        image = cv2.cvtColor(self.image_in, cv2.COLOR_HSV2RGB) #converting HSV to BGR
        qim = QImage(image, len(image[0]), len(image), len(image[0]) * 3, QImage.Format_RGB888)
        pixmap = QPixmap() #Function for colour images
        QPixmap.convertFromImage(pixmap, qim)
        pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.gui.editedimage.setPixmap(pixmap)

    def displayOriginalImage(self): #Function to display original image
        size = self.gui.editedimage.size()
        image = cv2.cvtColor(self.im, cv2.COLOR_HSV2RGB)
        qim = QImage(image, len(image[0]), len(image), len(image[0]) * 3, QImage.Format_RGB888)
        pixmap = QPixmap()
        QPixmap.convertFromImage(pixmap, qim)
        pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.gui.originalimage.setPixmap(pixmap)

if __name__=="__main__":
    app = QApplication(sys.argv)
    myapp = Firstwindow()
    myapp.show()
    sys.exit(app.exec_())
