import os
import h5py as h5
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cv2 #For Median Filter
import pywt  #For Discrete Wavelet transform
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.color import rgb2gray
import imageio
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from segmentation import segement
import pandas as pd
from skimage import io, color
from skimage.feature import greycomatrix, greycoprops
import matplotlib.image as mt
import skimage.io

class Patient(object):
    PID = ""
    image=""
    label=""
    tumorBorder=""
    tumorMask=""
    
    def __init__(self, PID, image, label,tumorBorder,tumorMask):
        self.PID = PID
        self.image = image
        self.label = label
        self.tumorBorder=tumorBorder
        self.tumorMask=tumorMask


file_path="C:/Users/Zain/Desktop/A Thesis/Data/"
file_path1="C:/Users/Zain/Desktop/A Thesis/Retrieved_Images/"

f=[]
finalList = []
classLabels = []
Zones = []
countArray = []
z = 0
nList = []
zoneAppend = []
zoneAppendFinal = []

for i in range(3064):
     f.append(h5.File(os.path.join(file_path,str(i+1)+".mat"),'a'))

for i in range(3064):
    list(f[i].items()) 
    list(f[i]['/cjdata'].keys())
    p=Patient('','','','','')
    p.image=np.mat(f[i]['/cjdata/image'])
    p.PID=np.array(f[i]['/cjdata/PID'])
    p.label=np.array(f[i]['/cjdata/label'])
    p.tumorBorder=np.mat(f[i]['/cjdata/tumorBorder'])
    p.tumorMask=np.mat(f[i]['/cjdata/tumorMask'])
    imageio.imwrite('Retrieved_Images/'+str(z)+'Orig.png', p.image )
    
    #+++++++++++++++++++++  MEDIAN and DWT FILTER  ++++++++++++++++++++++++++
    p.image = cv2.medianBlur(p.image, 5)
    imageio.imwrite('Retrieved_Images/'+str(z)+'Median.png', p.image )
    cA, cD = pywt.dwt(p.image, 'db1')
    p.image = cD
    imageio.imwrite('Retrieved_Images/'+str(z)+'DWT.png', p.image )
    size = 512, 512
    im = Image.open('Retrieved_Images/'+str(z)+'DWT.png')
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save('Retrieved_Images/'+str(z)+'DWT.png', "PNG")
    imgarr = np.array(p.image) 
    coeffs = pywt.dwt2(imgarr, 'haar')
    p.image = pywt.idwt2(coeffs, 'haar')
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    #+++++++++++++++++  K MEAN SEGMENTATION +++++++++++++++++++++++++++++++++
    image = cv2.imread('Retrieved_Images/'+str(z)+'DWT.png')
    (h1, w1) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = 3)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    frr = quant
    quant = quant.reshape((h1, w1, 3))
    image = image.reshape((h1, w1, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    imageio.imwrite('Retrieved_Images/'+str(z)+'KMean.png', quant)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    #+++++++++++++++++++++++++  GABOR FILTER   ++++++++++++++++++++++++++++++
    """g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 1.0, 0.2, 0, ktype=cv2.CV_32F)
    img = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(filtered_img, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(str(i)+".png", filtered_img)
    imageio.imwrite('Retrieved_Images/'+str(i)+'Gabor.png', quant)"""
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    #+++++++++++++++++++++++++  ZOINING   +++++++++++++++++++++++++++++++++++
    try:
        im_gray = cv2.imread('Retrieved_Images/'+str(z)+'Orig.png')
        #plt.imshow(im_gray)
        segmentImage = segement.getSeg(im_gray,p.tumorMask)
        tDimage = color.rgb2gray(segmentImage)
        
        #++++++++++++++++++++++  Cropping Tumor +++++++++++++++++++++++++++++
        
        listy = p.tumorBorder.tolist()
        maxi = int(max(listy[0]))
        mini = int(min(listy[0]))
        crop_img = tDimage[mini:maxi, mini:maxi]
        #plt.imshow(tDimage)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        imageio.imwrite('Retrieved_Images/'+str(z)+'Tumor.png', crop_img)
        size = 200, 200
        im = Image.open('Retrieved_Images/'+str(z)+'Tumor.png')
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save('Retrieved_Images/'+str(z)+'Tumor.png', "PNG")
    
        tDimage = cv2.imread('Retrieved_Images/'+str(z)+'Tumor.png')
        
        #+++++++  Zoning of Single image to 8 Zones  ++++++++++++++++++++++++
        zoneAppend = []
        r = 0
        zoneImage = None
        old_descriptors = None
        
        for l in range(4):
            sZone = tDimage[r:r+50]
            r = r + 50
            #sift = cv2.xfeatures2d.SIFT_create(nfeatures=200)
            
            #keypoints_sift, descriptors = sift.detectAndCompute(sZone, None)
            #plt.imshow(img)
            
            #New Addition
            #ngcm= greycomatrix(im, [1], [0], 256, symmetric=False, normed=True)
            
            imageio.imwrite('Retrieved_Images/'+str(z)+'sZone.png', sZone)
            I = cv2.imread('Retrieved_Images/'+str(z)+'sZone.png')
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            #I = im#mt.imread('Retrieved_Images/'+str(z)+'sZone.png');
            os.remove('Retrieved_Images/'+str(z)+'sZone.png')
            
            I = skimage.img_as_ubyte(I)
            #plt.imshow(I)
            GLCM2 = greycomatrix(I, distances = [1], angles = [4], levels = 255, symmetric=False,normed=False)
            
            Contrast = greycoprops(GLCM2, 'contrast')
            Energy = greycoprops(GLCM2, 'energy')
            Homogeneity = greycoprops(GLCM2, 'homogeneity')
            Correlation = greycoprops(GLCM2, 'correlation')
            Dissimilarity = greycoprops(GLCM2, 'dissimilarity')
            ASM = greycoprops(GLCM2, 'ASM')
            newList = []
            newList.append(Contrast[0][0])
            newList.append(Energy[0][0])
            newList.append(Homogeneity[0][0])
            newList.append(Correlation[0][0])
            newList.append(Dissimilarity[0][0])
            newList.append(ASM[0][0])
            descriptors = newList
            #End Here
            
            #img = cv2.drawKeypoints(sZone, keypoints_sift, None)
            
            if zoneImage is None:
                if descriptors is not None:
                    zoneImage = descriptors
                    zoneAppend.append(descriptors)
                else:
                    zoneAppend.append(None)
            else:
                if descriptors is not None:
                    zoneImage = np.concatenate((zoneImage, descriptors), axis=0)
                    zoneAppend.append(descriptors)
                else:
                    zoneAppend.append(None)
            
        if zoneImage is not None:
            zoneAppendFinal.append(zoneAppend)
            b = np.reshape(zoneImage, (1,np.product(zoneImage.shape)))
            z = z + 1
            print("++++   "+str(z)+"    ++++" )
            finalList.append(b[0])
            classLabels.append(p.label[0][0])
            #imageio.imwrite('Retrieved_Images/Features_Extracted/'+str(z)+'FeaturedEX.png', zoneImage)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
        else:
            nList.append(p.label[0][0])
            print("N List: {}".format(nList))
            os.remove('Retrieved_Images/'+str(z)+'Tumor.png')
            os.remove('Retrieved_Images/'+str(z)+'Orig.png')
            os.remove('Retrieved_Images/'+str(z)+'Median.png')
            os.remove('Retrieved_Images/'+str(z)+'DWT.png')
            os.remove('Retrieved_Images/'+str(z)+'KMean.png')
        
    except ValueError as ve:
        print("Error: {}".format(ve))
        nList.append(p.label[0][0])
        print("N List: {}".format(nList))
        os.remove('Retrieved_Images/'+str(z)+'Orig.png')
        os.remove('Retrieved_Images/'+str(z)+'Median.png')
        os.remove('Retrieved_Images/'+str(z)+'DWT.png')
        os.remove('Retrieved_Images/'+str(z)+'KMean.png')


#+++++++++++    This is CSV DATA for understanding  +++++++++++++++++++++++
cList = pd.DataFrame(classLabels)
Fappend = pd.DataFrame(zoneAppendFinal)
Fappend.columns = ['Zone1','Zone2','Zone3','Zone4']    
#Fappend.columns = ['Zone1','Zone2','Zone3','Zone4','Zone5','Zone6','Zone7','Zone8']  
Fappend.insert(0,"Labels",cList,True)
Fappend = Fappend.fillna(0)
Fappend.to_csv("CSV_DATA/GLCM_4Z_DatatoShow.csv") 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#+++++++++++++++   This Data is for Training and Testing   ++++++++++++++++   
fList = pd.DataFrame(finalList)
fList.insert(0,"Labels",cList,True)
fList = fList.fillna(0)
#df = finalList
#df = df.loc[:, (df != 0).any(axis=0)]
fList.to_csv("CSV_DATA/GLCM_4Z_BrainTumor_Data.csv")
#df.to_csv("CSV_DATA/BrainTumor_DataOutZeros.csv")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 


        