# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:29:36 2024

@author: melikianlab
"""
import numpy as np
import pandas as pd
import trackpy as tp  # trackpy
import tkinter as tk  # tkinter
from tkinter import filedialog
from tkinter.filedialog import askdirectory
import os
import matplotlib.pyplot as plt  # matplotlib
from scipy.optimize import curve_fit
import pims  # pims
import random



''' 1. Experimental parameters'''

RedCh = 1 # Red channel
FreCh = 1 # FRET channel
GreCh = 1 # Green channel
AddSap = 10 # The frame number when saponin was added
sec_per_f = 5 # how many sec per frame


''' 2. Tracking parameters '''
Diameter = 11     # in pixels. Represents the minimum diameter of the the feature in microns.
Min_mass = 2     # in total brightness. This is the minimum integrated brightness of the feature.
Separation = 6      # in pixels. The minimum separation between two features in microns.
Percentile = 90     # in percent. Features must have a peak brightness higher than pixels in this percentile to eliminate spurious points.
Max_distance = 3    # in pixels. Maximum distance in microns that features can move between frames to be considered the same feature
Memory = 5         # in frames. Maximum number of frames in which a feature can disappear, then reappear nearby, and be considered the same particle
Threshold = 10      # in frames. Minimum number of points for a track to be considered.


# Read_video  = r"M:\Gokul\Facility Line\2024-01-04-Ctrl_TM1_TM3_0.3_0.5ug trans_FlipperTR\Data split\Ctrl 0.3ug\ctrl 0.3 merged"  # Lab Video
Read_video = filedialog.askopenfilename(filetypes=(("Image files", ("*.jpg","*.jpeg","*.tif","*.tiff","*.png","*.bmp")), ("All files", "*.*")), title='Open Video File')
reader = pims.open(Read_video)
totCh = max([GreCh, FreCh, RedCh]) # number of channels
red = np.array(reader[(RedCh-1)::totCh])    # mCherry in the first channel. Red channel repeats every 3 frames.
fret = np.array(reader[(FreCh-1)::totCh])   # FRET in the second channel. Repeats every 3 frames.
green = np.array(reader[(GreCh-1)::totCh])  # YFP in the third channel.
both = np.array(reader)
#%%

video_name = Read_video.split('/')[-1].split('.')[0]
folder = askdirectory(title='Save files in')
new_folder = folder+"/" +video_name
# Creat folders for classification
try: 
    os.mkdir(os.path.join(new_folder)) 
except:
    pass
os.chdir(new_folder)

a = np.zeros(both.shape[0])
for i in range(both.shape[0]):
    a[i]=np.mean(green[i])
    
Peak_F = a.argmax()
# print(int(a.argmax()))


def contrast_img(img,min_,max_):
    img1 = img.copy()
    img1[img1>max_]=max_*1000
    img1[img1<(min_)]=min_
    img1 -= min_
    return img1

rand1 = random.randrange(1,both.shape[0],totCh)
img = contrast_img(green[Peak_F], np.min(green[Peak_F]), np.mean(green[Peak_F])*80)
img1 = contrast_img(reader[rand1], np.min(reader[rand1]), np.mean(reader[rand1])*150)
img

# Do not edit any of the below code
# Diameter = 2*math.floor(Diameter/Dimension*green[0].shape[0]/2)+1   # Converts diameter to nearest odd pixel value
# Separation = Separation/Dimension*green[0].shape[0]                 # Converts separation to pixel
# Max_distance = Max_distance/Dimension*green[0].shape[0]             # Converts max_distance to pixel

particles0 = tp.locate(green[Peak_F], Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile)
# particles1 = tp.locate(green[int(rand1/totCh)], Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile)

img3=tp.preprocessing.bandpass(reader[rand1], lshort=3, llong=11, threshold=1, truncate=4)
img4 = contrast_img(img3, np.min(reader[rand1]), np.mean(reader[rand1])*2)

with plt.rc_context({'figure.facecolor':'white'}):
    plt.figure(figsize=(9,9), dpi=300)
    print(tp.annotate(particles0, img, plot_style={'markersize': 9}))
    plt.figure(figsize=(9,9), dpi=300)
    print(tp.annotate(particles0, img1, plot_style={'markersize': 9}))
    # find frequency of pixels in range 0-255
plt.hist(img.ravel(), 100, label='img')
# plt.ylim(0, len(reader[3].ravel())/2)
plt.yscale('log', base = 10)
plt.legend()
plt.show()



# plot time vs intensity for each particle and export the data
def expo(x, a, t, c):
    return a*np.exp(-x/t)+c
def biexp(x, a1, t1, a2, t2, c):
        return a1*np.exp(-x/t1)+a2*np.exp(-x/t2)+c


data = particles0
data = data.T.reset_index().rename(columns={"index":"Particle"})
data=data[:2] # drop rows

y, x = data.iloc[0].copy(), data.iloc[1].copy()
data.iloc[0],data.iloc[1] = x,y		# exchange x and y position
data=data[:2]
data.loc[len(data)] = pd.Series(dtype='float64') #add an empty row

Int_info = pd.DataFrame({'Frame': range(both.shape[0])})
fit_result = pd.DataFrame()

for j in range(particles0.shape[0]):
    Mean_Int = np.zeros(both.shape[0])
    PN = 'Particle '+str(j)
    for i in range(both.shape[0]):
        try:
            Intensity = green[i, int(particles0['y'][j]-(Diameter-1)/2):int(particles0['y'][j]+(Diameter-1)/2), int(particles0['x'][j]-(Diameter-1)/2):int(particles0['x'][j]+(Diameter-1)/2)]
            Mean_Int[i] = np.mean(Intensity)
        except:
            pass
    x = np.arange(both.shape[0])
    plt.plot(x, Mean_Int, label = 'Intensity')
    
    ## One exponential
    # popt1, pcov1 = curve_fit(expo, x[:-(Peak_F)], Mean_Int[(Peak_F):])
    # plt.plot(x[(Peak_F):], expo(x[:-(Peak_F)], *popt1))
    ## Bi-exponential
    try:
        popt2, pcov2 = curve_fit(biexp, x[:-(Peak_F)], Mean_Int[(Peak_F):])
        plt.plot(x[(Peak_F):], biexp(x[:-(Peak_F)], *popt2), linestyle='dashed', label = 'biexponential')
    except:
        pass    
    plt.legend()

    plt.savefig(PN, bbox_inches="tight", dpi=200)
    plt.close() # figures won't be displayed. This will save time.
    
    
    Int_info[j] = Mean_Int
    fit_result[j] = pd.DataFrame({PN:[popt2[0], popt2[1], popt2[2], popt2[3], popt2[4]]})
    

# Sort lifetime
fit_result2 = fit_result.copy() 

for i in range(fit_result.shape[1]):
    if fit_result[i][1]>fit_result[i][3]:
        fit_result2[i][0], fit_result2[i][1],fit_result2[i][2], fit_result2[i][3] = fit_result2[i][2], fit_result2[i][3], fit_result2[i][0], fit_result2[i][1]

idx = pd.DataFrame(data= {'Particle':['A1', 't1(frame)', 'A2', 't2(frame)', 'y0']})
fit_result2=pd.concat([idx, fit_result2], axis=1)

# Save analyzed data
final = pd.concat([data, Int_info], ignore_index=True)
final.to_csv(new_folder+'.csv', index=False)
fit_result2.to_csv(new_folder+'_biexpo_fitting.csv', index=False)    
        
