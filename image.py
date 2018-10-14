import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

#-----------------------------------------PSNR and SSIM----------------------------------------------------------
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

#Naming
#Blurred input image = image_in
#PSF = blur_psf
#Number of rows in PSF = psf_row
#Number of columns in PSF = psf_column
#Zero padded psf = psf_zeropaddedhttps://in.mathworks.com/help/matlab/ref/fftshift.html
image_in = cv2.imread('imgs/blur_1_church_noise.jpg', 1) #To read blurred input image
A = image_in.shape[0]
B = image_in.shape[1]
blur_psf = cv2.imread('imgs/kernel_1.bmp', 0) #To read the point spread function


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


#----------------------------------TRUNCATED INVERSE FILTER---------------------------------------------------------
#Deblurred image = trunc_deblur

psf_row = blur_psf.shape[0] #Finding the dimension of PSF array
psf_column = blur_psf.shape[1]
psf_zeropadded = np.zeros_like(image_in[:, :, 0]) #Creating a zero matrix with same dimension as that of input image
psf_zeropadded[0:psf_row, 0:psf_column] = blur_psf #Inserting the original PSF in the zero matrix
psf_zeropadded = psf_zeropadded / np.sum(psf_zeropadded)  #Normalizing

trunc_deblur= np.zeros_like(image_in)
psf_dft = np.fft.fft2(psf_zeropadded) #Finding DFT of PSF
psf_dft[psf_dft == 0] = 0.000005

def butterworth(order, radius):
     a = range(0, A)
     a0 = int(A / 2) * np.ones(A)
     b = range(0, B)
     b0 = int(B / 2) * np.ones(B)

     r2 = radius ** 2

     row = np.tile((np.power(m - m0, 2 * np.ones(M)) / r2).reshape(M, 1), (1, N))
     column = np.tile((np.power(n - n0, 2 * np.ones(N)) / r2).reshape(1, N), (M, 1))

     butterworth_lpf = np.divide(np.ones_like(row), np.power(row + column, order * np.ones_like(row)) + np.ones_like(row))

   return butterworth_lpf

lpf = np.fft.fftshift(butter(12, R)) #Butterworth low pass filter of order 12


for i in range(0, 3):
    image_in_dft = np.fft.fft2(image_in[:, :, i]) #Finding the DFT of input image considering the 3 channels
    image_in_dft = np.multiply(image_in_dft, lpf) #Truncating the DFT of input image by a low pass filter
    trunc_deblur[:, :, i] = np.abs(np.fft.ifft2(np.divide(image_in_dft, psf_dft))).astype(np.uint8) 
   

#-------------------------------------------------WEINER FILTER----------------------------------------------------------
K = 0.2 #Initialising K to a value where K is the ratio of power spectrum of noise to that of undegraded image
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
    F1*F2 = np.abs(np.fft.ifft2(np.multiply(F1, F2)))
    weiner_deblur[:, :, i] = F1*F2.astype(np.uint8)

#------------------------------------------------CONSTRAINED LEAST SQUARE FILTER---------------------------------------







#
# # image = image[0:42, 0:42, :]
#
# psf_M = psf.shape[0]
# psf_N = psf.shape[1]
# psf_padded = np.zeros_like(image[:, :, 0])
# psf_padded[0:psf_M, 0:psf_N] = psf
# psf_padded = psf_padded / np.sum(psf_padded)
#
result = np.zeros_like(image)
# psf_dft = np.fft.fft2(psf_padded)
# # psf_dft[psf_dft == 0] = 0.0001
#
# for i in range(0, 3):
#     image_dft = np.fft.fft2(image[:, :, i])
#     # temp = np.uint8(np.abs(np.fft.ifft2(image_dft)))
#     temp = np.abs(np.fft.ifft2(np.divide(image_dft, psf_dft)))# * np.max(psf_dft))
#     result[:, :, i] = temp.astype(np.uint8)

# ---------------------------------------------- BUTTERWORTH AND TRUNCATED INVERSE ----------------------------------------------------------------

# Butterworth LPF
#
# M = result.shape[0]
# N = result.shape[1]
#
# order = 2
#
# m = range(0, M)
# m0 = int(M/2) * np.ones(M)
# n = range(0, N)
# n0 = int(N/2) * np.ones(N)
#
# R = 100
# R2 = R*R
#
# row = np.tile((np.power(m-m0, 2 * np.ones(M)) / R2).reshape(M,1), (1, N))
# column = np.tile((np.power(n-n0, 2 * np.ones(N)) / R2).reshape(1, N), (M,1))
#
# lpf = np.divide(np.ones_like(row), np.power(row + column, order * np.ones_like(row)) + np.ones_like(row)) * 255
#
# print(lpf[399:401, 399:401])
# # lpf = np.zeros_like(result[:, :, 0])
# #
# # lpf =
#
# plt.imshow(lpf, cmap='gray')
# plt.show()

# --------------------------------------------------------------------------------------------------------------

#
# def get_butterworth_lpf(M, N, order, radius):
#     m = range(0, M)
#     m0 = int(M / 2) * np.ones(M)
#     n = range(0, N)
#     n0 = int(N / 2) * np.ones(N)
#
#     r2 = radius ** 2
#
#     row = np.tile((np.power(m - m0, 2 * np.ones(M)) / r2).reshape(M, 1), (1, N))
#     column = np.tile((np.power(n - n0, 2 * np.ones(N)) / r2).reshape(1, N), (M, 1))
#
#     butterworth_lpf = np.divide(np.ones_like(row), np.power(row + column, order * np.ones_like(row)) + np.ones_like(row))
#
#     return butterworth_lpf
#
#
# def truncated_inverse_filter(image, psf, R):
#     psf_M = psf.shape[0]
#     psf_N = psf.shape[1]
#     psf_padded = np.zeros_like(image_in[:, :, 0])
#     psf_padded[0:psf_M, 0:psf_N] = psf
#     psf_padded = psf_padded / np.sum(psf_padded)
#
#     result = np.zeros_like(image_in)
#     psf_dft = np.fft.fft2(psf_padded)
#     psf_dft[psf_dft == 0] = 0.00001
#
#     lpf = np.fft.fftshift(get_butterworth_lpf(image.shape[0], image.shape[1], 10, R))
#     Y = fftshift(X) rearranges a Fourier transform X by shifting the zero-frequency component to the center of the array.
#
#     for i in range(0, 3):
#         image_dft = np.fft.fft2(image[:, :, i])
#         temp = np.multiply(np.divide(image_dft, psf_dft), lpf)
#         temp = np.abs(np.fft.ifft2(temp))
#         # temp = np.abs(np.fft.ifft2(np.multiply(np.divide(image_dft, psf_dft), lpf)))
#         result[:, :, i] = temp.astype(np.uint8)
#         # plt.imshow(20 * np.log10(np.abs(np.fft.fftshift(np.multiply(image_dft, lpf)))), cmap='gray')
#         # plt.show()
#
#     return result, lpf
#
#
# result, lpf = truncated_inverse_filter(image, psf, 100)
#
# print(np.min(result))
# print(np.max(result))
#
# fig=plt.figure(figsize=(1, 2))
#
# fig.add_subplot(1, 2, 1)
# plt.imshow(255 * lpf, cmap='gray')
#
# fig.add_subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#
# plt.show()


# psnr_value = psnr(result, image_og)
# print(psnr_value)
#
# fig=plt.figure(figsize=(1, 2))
#
# fig.add_subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
# fig.add_subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#
# plt.show()
#
# # cv2.imwrite('imgs/Out1.bmp', result)
# --------------------------------------------------------------------------------------------------------------

