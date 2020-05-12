# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:47:24 2020

@author: Richard Schultheis
"""
#%%imports
import sys
import os
import time
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import signal
import matplotlib 
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
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
    
def plotAgainstLowRes(image):
        im1LowRes=image[::16,::16]
        
        plt.subplot(2,1,1)
        plt.imshow(sitk.GetArrayViewFromImage(image))
        plt.axis('off')
        
        plt.subplot(2,1,2)
        plt.imshow(sitk.GetArrayViewFromImage(im1LowRes))
        plt.axis('off')
        
        
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

'''
    os.chdir('')
    im1=sitk.ReadImage('')
    #im1Transposed=np.transpose(im1,(1,0))    
    plot2d(im1) 
    plotAgainstLowRes(im1)    
'''
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
    
#%%Section Read Files
    
#im3d=sitk.ReadImage('')'

'''
#im3d.GetSize()
#im3dLowRes=im3d[::7,::7,::7]
#im3d.GetSize()
#im3dLowRes.GetSize()
#im3dCut=im3d[0:100,0:100,0:100]
#im3dCut.GetSize()
#np3d=sitk.GetArrayFromImage(im3dCut)
#type(im3dCut)
#type(contrast(im3dCut,0))
#aeh(image,0.9,0.6.255,0)
#plot3d(sitk.GetArrayFromImage(im3dCut)[:,:,:])
#plot3d(sitk.GetArrayFromImage(contrast(aeh(im3dCut),1))[:,:,:])
'''
#US1=inputUS(im3d,'coronal')

#aeh_settings=(1.4,0.8,255,0)
#imAeh=aeh(im3d,aeh_settings)
#imAeh=(aeh(im3d,aeh_settings))wha
#contrastAehImg=sitk.GetArrayFromImage(contrast((aeh(im3d,aeh_settings)),100))
#contrastAehImg=(aeh(contrast(im3d,10),aeh_settings))
#contrastImg=contrast(im3d,-6)
#%%
gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
   
#%%
t=im3d<1
tClean1 = sitk.BinaryOpeningByReconstruction(t, [10, 10, 10])
tClean2= sitk.BinaryClosingByReconstruction(tClean1, [10, 10, 10])
#t_inv=sitk.InvertIntensity(cleaned_thresh_img,255)
#myshow(sitk.LabelOverlay(im3d,sitk.LabelContour(cleaned_thresh_img)),title='sitk.LOL cleaned')
#size=im3d.GetSize()
#myshow3d(sitk.LabelOverlay(im3d,sitk.LabelContour(cleaned_thresh_img)),yslices=range(50,size[1]-50,20),zslices=range(50, size[2] - 50, 20),dpi=100)
#myshow(sitk.LabelOverlay(im3d,t),title='sitk.LOL')
t=im3d>1
tClean1 = sitk.BinaryOpeningByReconstruction(t, [10, 10, 10])
tClean2= sitk.BinaryClosingByReconstruction(tClean1, [10, 10, 10])
arrIm3d=sitk.GetArrayFromImage(im3d)
arrExcl=sitk.GetArrayFromImage(tClean2)
imgResult=sitk.GetImageFromArray(arrIm3d*arrExcl)
#imgResult=sitk.GetImageFromArray(arrIm3d-arrExcl)
#arrResult2=arrIm3d[arrExcl==1]=0
#imgResult2=sitk.GetImageFromArray(arrResult2)
#plotSlices(imgResult,'sagittal')
#%%
median=sitk.MedianImageFilter()
median.SetRadius(3)
imMedian=median.Execute(im3d)
for i in range(15):
    imMedian=median.Execute(imMedian)
#subMedian=subtractSITK(imgResult,imMedian/10)

sob1=sitk.SobelEdgeDetectionImageFilter()
sobMedian=sob1.Execute(intToFloat(imMedian))  
sobMedian255=setIntensity(sobMedian,255)
#plotSlices(sobMedian255,'sagittal')
#%%
gaussian.SetSigma(0.6)
tLarge=gaussian.Execute(tClean2)
tLarge=tLarge==1
tLargeClean1 = sitk.BinaryOpeningByReconstruction(tLarge, [10, 10, 10])
tLargeClean2= sitk.BinaryClosingByReconstruction(tLargeClean1, [10, 10, 10])
#plotSlices(tLargeClean2,'sagittal')
#%%
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
#%%
aehFilter=sitk.AdaptiveHistogramEqualizationImageFilter()
aehFilter.SetAlpha(1.3)
aehFilter.SetBeta(1)
aehFilter.SetRadius(5)
aehImg=aehFilter.Execute(imgResult)
arrAeh=sitk.GetArrayFromImage(aehImg)
arrAehCleaned=arrAeh*arrExcl
imgAehCleaned=sitk.GetImageFromArray(arrAehCleaned)
#plotSlices(imgAehCleaned,'sagittal')
#%%
gaussian.SetSigma(3)
imOut=gaussian.Execute(imgAehCleaned)
#plotSlices(imOut,'sagittal')

#%%Array Magic
statF2=sitk.StatisticsImageFilter()
statF2.Execute(imOut)
#addF=sitk.AddImageFilter()
#imAdd=addF.Execute(imOut,imOut)
divF=sitk.DivideImageFilter()
imDiv=divF.Execute(imOut,statF2.GetMean())
multF=sitk.MultiplyImageFilter()
imMult=multF.Execute(imDiv,imDiv)
imMultR=multF.Execute(imMult,statF2.GetMean())

imMultCleaned=cleanImg(imMultR,tLargeClean2)
#plotSlices(imMultCleaned,'sagittal')

#%%edge detector
sob=sitk.SobelEdgeDetectionImageFilter()
sobImg=sob.Execute(imMultCleaned)
#sobIm3d=sob.Execute(imAeh)
gaussian.SetSigma(1.2)
sobGauss=gaussian.Execute(sobImg)
#plotSlices(sobImg,'sagittal')

#%%
addF=sitk.AddImageFilter()
imEdgDiv=divF.Execute(sobGauss,10)
gaussian.SetSigma(0.0001)
imGauss=gaussian.Execute(imMultCleaned)
imAdd=addF.Execute(imGauss,imEdgDiv)
imFin=cleanImg(imAdd,tLargeClean2)
#plotSlices(divF.Execute(imFin,1),'sagittal')

#%%
statF=sitk.StatisticsImageFilter()
statF.Execute(sobGauss)
sobGauss2=sobGauss>statF.GetMean()
sobGauss2Cleaned=cleanImg(sobGauss2,tLargeClean2)
sobGauss2CleanedM=multF.Execute(sobGauss2Cleaned,statF.GetMaximum()/statF.GetMean())
#plotSlices(sobGauss2,'sagittal')
imFin2=addSITK(sobGauss2CleanedM,imFin)
imFin2Cleaned=cleanImg(imFin2,tLargeClean2)
#plotSlices(imFin2Cleaned,'sagittal')
#%%Region Growing Segmentation
seed=((87,64,90),)
seedImage=imFin
fact=0.0
if statF.GetSigma()>statF.GetMean():
    fact=statF.GetSigma()/statF.GetMean()
else:
    fact=statF.GetMean()/statF.GetSigma()
#RS=sitk.ConnectedThreshold(imFin2Cleaned, seedList=seed,lower=np.abs(statF.GetMinimum()),upper=statF.GetMean()/(np.abs(statF.GetMinimum()*(statF.GetSigma()/statF.GetMean()))))
RS=sitk.ConnectedThreshold(seedImage, seedList=seed,lower=np.sqrt(np.abs(statF.GetMinimum())),upper=seedImage[seed[0]]*fact+np.abs(statF.GetMinimum()))
RS=cleanImg(RS,tLargeClean2)
RSClean1 = sitk.BinaryOpeningByReconstruction(RS, [10, 10, 10])
RSClean2= sitk.BinaryClosingByReconstruction(RSClean1, [10, 10, 10])
plotSlices(RSClean2,'sagittal')
#%%
RSCleanInt=floatToInt(RSClean2)
size=imgResult.GetSize()
myshow3d(sitk.LabelOverlay(imgResult,RSCleanInt),yslices=range(50,size[1]-50,30), zslices=range(50,size[2]-50,20), dpi=30)
#%%

x=findBladderCenter(imFin2Cleaned,sobGauss2Cleaned,tLarge)
gaussian.SetSigma(3)
xy=gaussian.Execute(x)
statF3=sitk.StatisticsImageFilter()
statF3.Execute(xy)
ndimage.measurements.center_of_mass(sitk.GetArrayFromImage(xy))
###Need to be numpy Arrays, are sitk imgs
xz=xy==-1000
xy[102,89,87]=1000
plotSlices(xy<np.square(np.abs(statF3.GetMinimum())),'sagittal')
plotSlices(xy,'sagittal')

#%%Save slices of unsegmented medical images in a loop
os.chdir('')
p=('')
superList=os.walk('')
for root, dirs, files in superList:
    for file in files:
        path=os.path.join(root,file)
        if path.endswith('.nrrd'):
            img=sitk.ReadImage(path)
            #myshow(img)
            size=img.GetSize()
            myshow3d(img,yslices=range(50,size[1]-50,30), zslices=range(50,size[2]-50,20), dpi=30)
            plt.savefig(p+'\\'+file+'.png')
            plt.close()
#%%Save slices of segmented medical images in a loop
p=('')
superList=os.walk('')
for root, dirs, files in superList:
    for file in files:
        path=os.path.join(root,file)
        if path.endswith('.mhd'):
            if file.endswith('segmentation.mhd'):
                imgLabel=sitk.ReadImage(path)
                img=sitk.ReadImage(path.replace('_segmentation.mhd','.mhd'))
                size=img.GetSize()
                myshow(sitk.LabelOverlay(floatToInt(setIntensity(img,255)),floatToInt(setIntensity(imgLabel,255))))
                plt.savefig(p+'\\'+file+'.png')
                plt.close()
            

