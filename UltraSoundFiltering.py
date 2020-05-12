# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:46:13 2020

@author: Richard Schultheis
"""

#%%imports
import sys
import os
import time
from random import randrange
import numpy as np
from numpy import ndarray
import scipy as sp
from scipy import ndimage
from scipy import signal
import medpy as mp
from medpy import filter
import matplotlib 
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
import skimage as si
from skimage import segmentation
from pylab import imshow
from myshow import myshow,myshow3d

#%%Section Functions
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        #ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap= 'gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def inputUS(image,view):
    img=sitk.GetArrayFromImage(image)
    viewDict={'axial':(1,2,0),'sagittal':(0,1,2),'coronal':(0,2,1)}
    orientation=viewDict[view]
    original=img
    original = np.transpose(original, orientation)
    if(view=='coronal' or view=='sagittal'):
        original = np.rot90(original, k=-1)
    return (original,view)

def plot3d(image,view):
    img=image
    viewDict={'axial':(1,2,0),'sagittal':(0,1,2),'coronal':(0,2,1)}
    orientation=viewDict[view]
    original=img
    original = np.transpose(original, orientation)
    if(view=='coronal'):
        original = np.rot90(original, k=-1)
    #original = np.rot90(original, k=-1)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(view)
    tracker = IndexTracker(ax, original)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

def plot2d(image):
    original=((image[::2,::2])[::2,::2])[::2,::2]
    #original = np.rot90(original, k=-1)
    #fig, ax = plt.subplots(1, 1)
    plt.imshow(sitk.GetArrayViewFromImage(original))
    
#def changeContrast(image):
        
def contrast(image, c):
    #array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    shape = array.shape
    ntotpixel = shape[0] * shape[1] * shape[2]
    IOD = np.sum(array)
    luminanza = int(IOD / ntotpixel)

    max = 255
    

    d = array - luminanza
    dc = d * abs(c) / 100

    if c >= 0:
        J = array + dc
        J[J >= max] = max
        J[J <= 0] = 0
    else:
        J = array - dc
        J[J >= max] = max
        J[J <= 0] = 0

    J = J.astype(int)

    img = sitk.GetImageFromArray(J)
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    return img


def aeh(image,tup=tuple):
#Standard Implementation: aeh(image,0.9,0.6.255,0)

    adapt = sitk.AdaptiveHistogramEqualizationImageFilter()
    adapt.SetAlpha(tup[0])
    adapt.SetBeta(tup[1])
    image = adapt.Execute(image)  # set mean and std deviation

    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(tup[2])
    resacleFilter.SetOutputMinimum(tup[3])
    image = resacleFilter.Execute(image)  # set intensity 0-255
    image = normalizeFilter.Execute(image)  # set mean and std deviation
    return image

def setROI(image,size:tuple,index:tuple):
    # size origin and index need to be coordinate pixel information, x,y,z as tuple (x,z,y)
    #image needs to be SimpleITK.SimpleITK.Image.
    roiF=sitk.RegionOfInterestImageFilter()
    roiF.SetSize(size)
    roiF.SetIndex(index)
    imageROI=roiF.Execute(image)
    return imageROI

figList,axList=[],[]
def plotSlices(image,view):
    img=inputUS(image,view)
    fig, ax = plt.subplots(1, 1)
    figList.append(fig)
    axList.append(ax)
    
    axList[-1].set_title(view)
    tracker=IndexTracker(axList[-1], img[0])
    figList[-1].canvas.mpl_connect('scroll_event', tracker.onscroll)
    #plt.show(figList[-1],blocking=False)
    #plt.show(figList[-1])
    plt.show(blocking=False)
    plt.tight_layout()
    
def addSITK(a,b):
    a=sitk.GetArrayFromImage(a)
    b=sitk.GetArrayFromImage(b)
    c=a+b
    return sitk.GetImageFromArray(c)

def subtractSITK(a,b):
    a=sitk.GetArrayFromImage(a)
    b=sitk.GetArrayFromImage(b)
    a=a.astype(float)
    b=b.astype(float)
    c=a-b
    c[c<0]=0
    c=c.astype(int)  
    return sitk.GetImageFromArray(c)

def subtractSITK_Number(a,b):
    a=sitk.GetArrayFromImage(a)
    b=b
    a=a.astype(float)
    c=a-b
    c[c<0]=0
    c=c.astype(int)  
    return sitk.GetImageFromArray(c)

def multiplySITK(a,b):
    a=sitk.GetArrayFromImage(a)
    b=sitk.GetArrayFromImage(b)
    a=a.astype(float)
    b=b.astype(float)
    c=a*b
    c[c<0]=0
    c=c.astype(int)  
    return sitk.GetImageFromArray(c)

def waveletFilter(a):
    a=sitk.GetArrayFromImage(a)
    a.astype(float)
    w=np.arange(1,31)
    a=signal.cwt(a,signal.ricker,w)
    a.astype(int)
    return sitk.GetImageFromArray(a)

def intToFloat(a):
    a=sitk.GetArrayFromImage(a)
    a=a.astype(float)
    return sitk.GetImageFromArray(a)

def floatToInt(a):
    a=sitk.GetArrayFromImage(a)
    a=a.astype(int)
    return sitk.GetImageFromArray(a)

def setIntensity(a,b):
    intensityStatFilter=sitk.StatisticsImageFilter()
    intensityStatFilter.Execute(a)
    maxA=intensityStatFilter.GetMaximum()
    a=sitk.GetArrayFromImage(a)
    a=(a/maxA)*b
    return sitk.GetImageFromArray(a)

def SPinSITK(a,fun):
    a=sitk.GetArrayFromImage(a)
    a=fun(a)
    return sitk.GetImageFromArray(a.astype(float))
#Take cleaner image which is the outside area extracted
def cleanImg(image,cleaner):
    arr1=sitk.GetArrayFromImage(image)
    arr2=sitk.GetArrayFromImage(cleaner)
    arrCleaned=arr1*arr2
    return sitk.GetImageFromArray(arrCleaned)

def findBladderCenter(image,edgeImage,borderImage):
    image=sitk.GetArrayFromImage(image)
    edgeImage=sitk.GetArrayFromImage(edgeImage)
    borderImage=sitk.GetArrayFromImage(borderImage)==0
    #since borderImage tLarge has the inside of the area marked 1, the outside 0, we take the negative
    out=image+edgeImage
    out=out+(borderImage*255)
    return sitk.GetImageFromArray(out)
    
def s2a(s):
    return sitk.GetArrayFromImage(s)
def a2s(a):
    return sitk.GetImageFromArray(a)
def ps(im):
    plotSlices(im,'sagittal')
    
def glob(im,seed,side):
    im=s2a(im)
    imArea=im[seed[0]:(seed[0]+side),seed[1]:(seed[1]+side),seed[2]:(seed[2]+side)]
    return (seed, np.mean(imArea))

#%%setup SITK tools
gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
median=sitk.MedianImageFilter()
sob1=sitk.SobelEdgeDetectionImageFilter()
statF=sitk.StatisticsImageFilter()

#%% Open File
dataPath=''
im3d=sitk.ReadImage('')
sitkImages=[]
for f in os.listdir(dataPath):
    sitkImages.append(sitk.ReadImage(dataPath+f))
#%%Extract image area
#t=im3d>1
#tClean1 = sitk.BinaryOpeningByReconstruction(t, [10, 10, 10])
#tClean2= sitk.BinaryClosingByReconstruction(tClean1, [10, 10, 10])
#statF.Execute(im3d)
#arrIm3d=sitk.GetArrayFromImage(im3d)
#arrExcl=sitk.GetArrayFromImage(tClean2)
#imgExtract=sitk.GetImageFromArray((arrIm3d*arrExcl)+(arrE))
#plotSlices(im3d,'sagittal')
#%%Extract area outside image

t=sitkImages[10]<1
tClean1 = sitk.BinaryOpeningByReconstruction(t, [10, 10, 10])
tClean2= sitk.BinaryClosingByReconstruction(tClean1, [10, 10, 10])
tClean2=tClean2<1
gaussian.SetSigma(0.1)
tLarge=gaussian.Execute(tClean2)
tLarge=tLarge==1
tLargeClean1 = sitk.BinaryOpeningByReconstruction(tLarge, [10, 10, 10])
tLargeClean2= sitk.BinaryClosingByReconstruction(tLargeClean1, [10, 10, 10])
#tLargeClean2=tLargeClean2<1
#imgExtract=addSITK(multiplySITK(im3d, tLargeClean2),(tLargeClean2<1)*255)
#plotSlices(tLargeClean2,'sagittal')

#%%Extract edges 
'''
median.SetRadius(3)
imMedian=median.Execute(imgExtract)
for i in range(5):
    imMedian=median.Execute(imMedian)
subMedian=subtractSITK(imgExtract,imMedian/10)
sob1=sitk.SobelEdgeDetectionImageFilter()
sobMedian=sob1.Execute(intToFloat(imMedian))  
sobMedian255=setIntensity(sobMedian,255)
sobMedExcl=median.Execute(sobMedian255>5)
for i in range(2):
    sobMedExcl=median.Execute(sobMedExcl)
imEdgeEnhanced=addSITK(imgExtract, sobMedExcl*50)
#plotSlices(imMedian,'sagittal')
'''

#%%anisotropic filtering and sobel filtering
image=sitk.GetArrayFromImage(sitkImages[10])
anisotropicSmoothed=mp.filter.smoothing.anisotropic_diffusion(image,10,25,0.15,None,option=2)
anisotropicExtract=addSITK(multiplySITK(sitk.GetImageFromArray(anisotropicSmoothed), tLargeClean2),(tLargeClean2<1)*255)
anisotropicSmoothed=sitk.GetImageFromArray(anisotropicSmoothed)
anisotropicEdges=setIntensity(sob1.Execute(intToFloat(anisotropicExtract)),255)
anisotropicEdges2=sitk.GetImageFromArray(mp.filter.smoothing.anisotropic_diffusion(sitk.GetArrayFromImage(anisotropicEdges),10,25,0.15,None,option=2))
#plotSlices(anisotropicExtract,'sagittal')

#%%region growing segmentation on edge filtered anisotropic diffusion filtered image
#Ok segmentation all inside target area, but very fragmented
seedList=((100,60,96),)
fm=sitk.ConnectedThreshold(anisotropicEdges,seedList,0,3)
gaussian.SetSigma(5)
fmGauss=gaussian.Execute(setIntensity(fm,255))

median.SetRadius(1)
fmMedian=median.Execute(fm)

fmAnisotropic=sitk.GetImageFromArray(mp.filter.smoothing.anisotropic_diffusion(sitk.GetArrayFromImage(fm),10,20,0.15,None,option=2))
#plotSlices(fmAnisotropic>0.25,'sagittal')

#%%Confidence connected thresholding
#Useless
'''
seedList=((100,60,96),)
anisotropicSource=addSITK(anisotropicEdges2,anisotropicExtract*1)
SegconfidenceThreshold=sitk.ConfidenceConnected(anisotropicSource, seedList,numberOfIterations=2,multiplier=2,initialNeighborhoodRadius=2,replaceValue=1)
plotSlices(SegconfidenceThreshold,'sagittal')
'''
#%%
'''
anisotropEdgeEnhanced=setIntensity(addSITK(anisotropicExtract, 4*anisotropicEdges),255)
seedList=((100,60,96),)
fmAnisotropicEnhanced=sitk.ConnectedThreshold(anisotropEdgeEnhanced,seedList,0,13)
fmAnisotropicEnhanced2=sitk.GetImageFromArray(mp.filter.smoothing.anisotropic_diffusion(sitk.GetArrayFromImage(fmAnisotropicEnhanced),10,20,0.15,None,option=2))
#plotSlices(fmAnisotropic>0, 'sagittal')
'''
#%%
fmLol=sitk.LabelOverlay(floatToInt(anisotropicEdges2), floatToInt(fm))
#myshow(fmLol)
size=fmLol.GetSize()
#myshow3d(fmLol,yslices=range(50,size[1]-int(size[1]/8),30), zslices=range(50,size[2]-int(size[2]/16),20))

#%%
#superVol=segmentation.slic(s2a(im3d)[:,:,96],n_segments=2000,compactness=0.004)
#imshow(superVol)
gradientMagnitude=sitk.GradientMagnitude(anisotropicExtract)
gradientMagnitudeGauss=sitk.GradientMagnitudeRecursiveGaussian(anisotropicExtract,sigma=1)
#ps(gradientMagnitudeGauss)
#%%
for el in sitkImages[5:10]:
    extractor=s2a(el[30:180,:,:])
    meanIntensityList=[]
    meanIntensity=np.mean(extractor)
    stdIntensity=np.std(extractor)
    for i in range(1,len(extractor[:,1,1])):
        meanIntensityList.append(np.mean(extractor[i,:,:]))
    plt.plot(range(1,len(extractor[:,1,1])),(meanIntensityList>meanIntensity))
    #plt.hlines(np.mean(meanIntensityList)+stdIntensity/10,0,len(extractor[:,1,1]))
plt.show
#ps(anisotropicExtract[:,:,:])

#%%
ws=sitk.MorphologicalWatershed(anisotropicExtract[30:180,:,:], level=5, markWatershedLine=False, fullyConnected=False)
#ws1=(ws==0)
#ps(ws)
c=(np.bincount(ndarray.flatten(s2a(ws))))
#c.argmax()
wsArr=s2a(ws)
wsList=np.nonzero(wsArr==c.argmax())
randomList=[]
for i in range(500):
    randomList.append(np.random.randint(0,len(wsList[0])))
globList=[]
for el in randomList:
    seedLoc=[wsList[0][el],wsList[1][el],wsList[2][el]]
    globList.append(glob(anisotropicExtract,seedLoc,30))
    c=c+1
globMeans=[]
for el in globList:
    globMeans.append(el[1])
seedLocation=globList[globMeans.index(np.min(globMeans))][0]
anisotropicExtractA=s2a(anisotropicExtract)
area=(anisotropicExtractA[seedLocation[0]:(seedLocation[0]+30),seedLocation[1]:(seedLocation[1]+30),seedLocation[2]:(seedLocation[2]+30)])
anisotropicExtractA[seedLocation[0]:(seedLocation[0]+30),seedLocation[1]:(seedLocation[1]+30),seedLocation[2]:(seedLocation[2]+30)]=150
anisotropicExtractA=a2s(anisotropicExtractA)#ps(anisotropicExtractA)
ps(anisotropicExtract<np.max(area))
#%%

seed=(tuple(seedLocation),)

anisotropicSource=addSITK(anisotropicEdges2,anisotropicExtract*1)
SegconfidenceThreshold=sitk.ConfidenceConnected(anisotropicSource, seed,numberOfIterations=2,multiplier=1.5,initialNeighborhoodRadius=1,replaceValue=1)
plotSlices(SegconfidenceThreshold,'sagittal')
#%%
'''
imFourier=np.fft.fftn(sitk.GetArrayFromImage(im3d))
#imshow(np.log(np.abs(imFourier[100,:,:])))
img=np.log(np.abs(imFourier))
fig, ax = plt.subplots(1, 1)    
ax.set_title('view')
tracker=IndexTracker(ax, img)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#plt.show(figList[-1],blocking=False)
#plt.show(figList[-1])
plt.show(blocking=False)
plt.tight_layout()
'''
#%%
'''
imFourier=np.fft.fftn(sitk.GetArrayFromImage(im3d))
imFourier[np.abs(imFourier)>750000000]=0
imFourier[np.abs(imFourier)<500000]=0
imRev=sitk.GetImageFromArray(np.log(np.abs(np.fft.ifftn(imFourier))))
imRev=setIntensity(imRev,255)
imRevR=multiplySITK(imRev, tLargeClean2)
#plotSlices(imRevR, 'sagittal')
'''
