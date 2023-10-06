import cv2
import sklearn
import skimage.feature as sk
import numpy as np
im1=cv2.imread('D:\Ringworm\database\ISIC-images\ISIC_MSK-2_1\ISIC_0009869.jpg')
b,g,r=cv2.split(im1)
nk1,nk2,nk3,fet=[],[],[],[]
im2=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

im3=cv2.cvtColor(im1,cv2.COLOR_BGR2HSV)
#imp= cv2.adaptiveThreshold(im2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
glcm1 = sk.greycomatrix(b,[1],[0],symmetric=False,normed=True)
glcm2 = sk.greycomatrix(r,[1],[0],symmetric=False,normed=True)
glcm3 = sk.greycomatrix(g,[1],[0],symmetric=False,normed=True)
glcm1=np.float32(glcm1)
glcm2=np.float32(glcm2)
glcm3=np.float32(glcm3)




nk1.append(sk.greycoprops(glcm1,"contrast")[0][0])
nk1.append(sk.greycoprops(glcm1,"ASM")[0][0])
nk1.append(sk.greycoprops(glcm1,"energy")[0][0])
nk1.append(sk.greycoprops(glcm1,"homogeneity")[0][0])
nk1.append(sk.greycoprops(glcm1,"dissimilarity")[0][0])
nk1.append(sk.greycoprops(glcm1,"correlation")[0][0])
fet.append(nk1)

nk2.append(sk.greycoprops(glcm2,"contrast")[0][0])
nk2.append(sk.greycoprops(glcm2,"ASM")[0][0])
nk2.append(sk.greycoprops(glcm2,"energy")[0][0])
nk2.append(sk.greycoprops(glcm2,"homogeneity")[0][0])
nk2.append(sk.greycoprops(glcm2,"dissimilarity")[0][0])
nk2.append(sk.greycoprops(glcm2,"correlation")[0][0])
fet.append(nk2)

nk3.append(sk.greycoprops(glcm3,"contrast")[0][0])
nk3.append(sk.greycoprops(glcm3,"ASM")[0][0])
nk3.append(sk.greycoprops(glcm3,"energy")[0][0])
nk3.append(sk.greycoprops(glcm3,"homogeneity")[0][0])
nk3.append(sk.greycoprops(glcm3,"dissimilarity")[0][0])
nk3.append(sk.greycoprops(glcm3,"correlation")[0][0])
fet.append(nk3)                      

print nk1
print nk2
print nk3