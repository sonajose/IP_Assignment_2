import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

 def weiner_inv(image_in, blur_psf,original, k)
	#Naming
	#Blurred input image = image_in
	#PSF = blur_psf
	#Number of rows in PSF = psf_row
	#Number of columns in PSF = psf_column
	#Zero padded psf = psf_zeropaddedhttps://in.mathworks.com/help/matlab/ref/fftshift.html
	#image_in = cv2.imread('/home/sona/0_IP_Assignment2/blur1_noise_amt5_str20.bmp', 1) #To read blurred input image
	A = image_in.shape[0]
	B = image_in.shape[1]
	#blur_psf = cv2.imread('/home/sona/0_IP_Assignment2/Kernel1G_new.png', 0) #To read the point spread function

	#Initialising K to a value where K is the ratio of power spectrum of noise to that of undegraded image
	#Deblurred image = weiner_deblur
	#Square of magnitude of PSF's DFT = mag_square_psf_dft
	psf_row = blur_psf.shape[0] #Finding the dimension of PSF array
	psf_column = blur_psf.shape[1]
	psf_zeropadded = np.zeros_like(image_in[:, :, 0]) #Creating a zero matrix with same dimension as that of input image
	psf_zeropadded[0:psf_row, 0:psf_column] = blur_psf #Inserting the original PSF in the zero matrix
	psf_zeropadded = psf_zeropadded / np.sum(psf_zeropadded)  #Normalizing

	weiner_deblur = np.zeros_like(image_in)
	psf_dft = np.fft.fft2(psf_zeropadded) #Finding DFT of PSF
	psf_dft[psf_dft == 0] = 0.000005

	for i in range(0, 3):
	    image_in_dft = np.fft.fft2(image_in[:, :, i])
	    mag_square_psf_dft = np.square(np.abs(psf_dft))
	    const = K * np.ones_like(image_in_dft)
	    F1= np.divide(mag_square_psf_dft, mag_square_psf_dft + const )
	    F2= np.divide(image_in_dft, psf_dft)
	    F1F2 = np.abs(np.fft.ifft2(np.multiply(F1, F2)))
	    weiner_deblur[:, :, i] = F1F2.astype(np.uint8)
	#plt.imshow(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
	#plt.show()
	#plt.imshow(cv2.cvtColor(weiner_deblur, cv2.COLOR_BGR2RGB))
	#plt.show()
        
        #PSNR        
	max_sq = 255*255
	mse = np.sum(np.square(original - weiner_deblur))/(A*B)
	#Mean square error
	psnr = 10*np.log10(max_sq/mse)
	
	#SSIM
	u1 = np.sum(original)/(A*B)
	u2 = np.sum(weiner_deblur)/(A*B)
	s1 = np.power(np.sum(np.square(original - u1*np.ones_like(original))/(A*B)), 0.5)
	s2 = np.power(np.sum(np.square(weiner_deblur - u2*np.ones_like(weiner_deblur))/(A*B)), 0.5)
	s12 = np.sum(np.multiply(original - u1*np.ones_like(original),weiner_deblur - u2*np.ones_like(weiner_deblur)))/(A*B)
	L = (2*u1*u2 + 10)/(np.square(u1)+np.square(u2) + 10)
	C = (2*s1*s2 + 10)/(np.square(s1) + np.square(s2) + 10)
	S = (s12 + 10)/(s1*s2 + 10)
	SSIM = L*C*S

	#cv2.imwrite('blur1_noise_k0.5_weiner_deblur.jpg', weiner_deblur)
	return weiner_deblur, psnr, SSIM

