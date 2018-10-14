import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


#Naming
#Blurred input image = image_in
#PSF = blur_psf
#Number of rows in PSF = psf_row
#Number of columns in PSF = psf_column
#Zero padded psf = psf_zeropaddedhttps://in.mathworks.com/help/matlab/ref/fftshift.html
image_in = cv2.imread('/home/sona/0_IP_Assignment2/blur4.bmp', 1) #To read blurred input image
A = image_in.shape[0]
B = image_in.shape[1]
blur_psf = cv2.imread('/home/sona/0_IP_Assignment2/Kernel4G_new.png', 0) #To read the point spread function


#-----------------------------------------INVERSE FILTERING------------------------------------

#Deblurred image = inv_deblur
#DFT of psf = psf_dft
#DFT of input image = image_in_dft

psf_row = blur_psf.shape[0] #Finding the dimension of PSF array
psf_column = blur_psf.shape[1]
psf_zeropadded = np.zeros_like(image_in[:, :, 0]) #Creating a zero matrix with same dimension as that of input image
psf_zeropadded[0:psf_row, 0:psf_column] = blur_psf #Inserting the original PSF in the zero matrix
psf_zeropadded = psf_zeropadded / np.sum(psf_zeropadded)  #Normalizing

inv_deblur= np.zeros_like(image_in)
psf_dft = np.fft.fft2(psf_zeropadded) #Finding DFT
psf_dft[psf_dft == 0] = 0.000005

for i in range(0, 3):
    image_in_dft = np.fft.fft2(image_in[:, :, i]) #Finding the dft of input image considering the 3 channels
    inv_deblur[:, :, i] = np.abs(np.fft.ifft2(np.divide(image_in_dft, psf_dft))).astype(np.uint8) 
    # Finding inverse DFT of ratio of DFTs of input image and PSF
plt.imshow(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(inv_deblur, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite('blur4_inv_deblur.jpg', inv_deblur)

