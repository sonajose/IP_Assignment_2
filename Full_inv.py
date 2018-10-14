import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_in = cv2.imread('/home/sona/0_IP_Assignment2/myblurhouse.bmp', 1) #To read blurred input image

blur_psf = cv2.imread('/home/sona/0_IP_Assignment2/myblurkernel.png', 0) #To read the point spread function




#Naming
#Blurred input image = image_in
#PSF = blur_psf
#Number of rows in PSF = psf_row
#Number of columns in PSF = psf_column
#Zero padded psf = psf_zeropaddedhttps://in.mathworks.com/help/matlab/ref/fftshift.html

A = image_in.shape[0]
B = image_in.shape[1]


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

cv2.imwrite('deblur.bmp', inv_deblur)




 #PSNR        
#max_sq = 255*255
#mse = np.sum(np.square(original - inv_deblur))/(A*B)
#Mean square error
#psnr = 10*np.log10(max_sq/mse)

#SSIM
#u1 = np.sum(original)/(A*B)
#u2 = np.sum(inv_deblur)/(A*B)
#s1 = np.power(np.sum(np.square(original - u1*np.ones_like(original))/(A*B)), 0.5)
#s2 = np.power(np.sum(np.square(inv_deblur - u2*np.ones_like(inv_deblur))/(A*B)), 0.5)
#s12 = np.sum(np.multiply(original - u1*np.ones_like(original),inv_deblur - u2*np.ones_like(inv_deblur)))/(A*B)
#L = (2*u1*u2 + 10)/(np.square(u1)+np.square(u2) + 10)
#C = (2*s1*s2 + 10)/(np.square(s1) + np.square(s2) + 10)
#S = (s12 + 10)/(s1*s2 + 10)
#SSIM = L*C*S
#print(psnr)
#print(SSIM)




