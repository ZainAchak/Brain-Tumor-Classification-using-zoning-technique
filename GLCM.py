# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:22:59 2019

@author: Zain
"""

import skimage.io
from skimage.feature import greycomatrix, greycoprops
import matplotlib.image as mt
import numpy as np
import matplotlib.pyplot as plt
im = skimage.io.imread('0Tumor.png', as_grey=True)
Img = mt.imread('0Tumor.png');
#plt.imshow(Img)
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


I= Img 

#http://i425.photobucket.com/albums/pp337/jewarmy/python.jpg” 
I = skimage.img_as_ubyte(I)
#plt.imshow(I)
GLCM2 = greycomatrix(I, distances = [1], angles = [4], levels = 255, symmetric=False,normed=False)

Contrast = greycoprops(GLCM2, 'contrast')
Energy = greycoprops(GLCM2, 'energy')
Homogeneity = greycoprops(GLCM2, 'homogeneity')
Correlation = greycoprops(GLCM2, 'correlation')
Dissimilarity = greycoprops(GLCM2, 'dissimilarity')
ASM = greycoprops(GLCM2, 'ASM')

temp = GLCM2
glcm = GLCM2[:,:,0,0].ravel()
#plt.imshow(GLCM2)

print("\n\nContrast:      {}\n\nEnergy:        {}\n\nHomogeneity:   {}\n\nCorrelation:   {}\n\nDissimilarity: {}\n\nASM:           {}\n\n".format(Contrast,Energy,Homogeneity,Correlation,Dissimilarity,ASM))