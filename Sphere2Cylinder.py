# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:10:59 2023

@author: cvkelly
Email: cvkelly@wayne.edu
Web: https://cvkelly.wayne.edu

Please cite us upon use:

"Methods for making and observing model lipid droplets." 
Sonali Gandhi1, Shahnaz Parveen1, Munirah AlDuhailan1, Ramesh Tripathi1, 
Nasser Junedi1, Mohammad Saqallah1, Matthew A. Sanders2, Peter M. Hoffmann3, 
Katherine Truex4, James G. Granneman3,5, Christopher V Kelly1,5,*
1 Department of Physics and Astronomy, Wayne State University, Detroit, MI, USA 48201 
2 Center for Molecular Medicine and Genetics, School of Medicine, Wayne State University, Detroit, MI, USA 40201 
3 Physical Sciences Department, Embry-Riddle Aeronautical University, Daytona Beach, FL, USA 32114 
4 Department of Physics, United States Naval Academy, Annapolis, MD, USA 21402 
5 Center for Integrative Metabolic and Endocrine Research, School of Medicine, Wayne State University, Detroit, MI USA 48201 
*cvkelly@wayne.edu

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

def calc_area(x,y):
    mx = (x[:-1]+x[1:])/2
    dy = y[:-1]-y[1:]
    dx = x[:-1]-x[1:]
    dr = np.sqrt(dx**2+dy**2)
    circ = np.pi*2*mx
    area = np.sum(circ*dr)
    return area

def calc_vol(x,y):
    dy = np.abs(y[1:]-y[:-1])
    mx = (x[1:]+x[:-1])/2
    a = np.pi * mx**2
    vol = np.sum(a*dy)
    return vol

def get_colors(num_cols):
# num_cols = 4
    var = np.linspace(0,1,num_cols)
    cols = np.zeros((num_cols,4))
    for i in range(num_cols):
        cols[i,:] = matplotlib.cm.jet(var[i])
    cols = cols.astype(np.float16)
    return(cols)

def import_npz(npz_file,allow_pickle=False):
    # This doesn't work as part of a Python module
    Data = np.load(npz_file,allow_pickle=allow_pickle)
    for varName in Data:
        globals()[varName] = Data[varName]

# %% calculate n vs h

rad = 2 # radius of the initial hemisphere
numn = 10**5

h_start = rad
h_end = 0.9*rad

vol_hemi = 4/3*np.pi*rad**3 / 2

x = np.linspace(0,rad,10**3)
rad2list = np.linspace(h_start,h_end,1000)

nlist = np.logspace(np.log10(2),np.log10(8),numn)

volarray=np.zeros((len(rad2list),len(nlist)))
bestdata = np.zeros((len(nlist),3))
per = np.round(len(nlist)/20)
rold = rad

for ni,n in enumerate(nlist):
    if ni/per == np.round(ni/per):
        print(int(ni/len(nlist)*100),'% done with line 54')
    # print(ni)
    for ri,r2 in enumerate(rad2list):
        y = ((1-(x/rad)**n)*r2**n)**(1/n)
        vol = calc_vol(x,y)/vol_hemi
        volarray[ri,ni] = vol
        if vol<1:
            bestdata[ni,:] = [r2,n,vol] # np.mean([rold,r2])
            rold = r2.copy()
            break
    if ri > 1:
        rad2list = rad2list[(ri-2):]

bestdata = bestdata[bestdata[:,0]>0,:]
hlist = bestdata[:,0] #/rad
nlist = bestdata[:,1]

# %% shape vs n
## USED PANEL B

arealist = np.zeros(len(nlist))

fig = plt.figure(dpi=600,figsize=(3.5,2.8))
fs = 12
fslegend = 7

x = np.linspace(0,rad,1000)
nlistnow = nlist[hlist>(0.9*rad)]
cols = get_colors(len(nlistnow))

freqplot = 10
freqplot2 = int(len(nlistnow)/freqplot)
for i,n in enumerate(nlistnow):
    h = hlist[i]
    y = ((1-(x/rad)**n)*h**n)**(1/n)
    # k = (np.isnan(y) == False)
    arealist[i] = calc_area(x,y)
    if i/freqplot2 == int(i/freqplot2):
        plt.plot(x/rad,
                 y/rad,
                 color=cols[i],
                 label=str(n)[:4])

plt.xlabel('$x / w$',fontsize=fs)
plt.ylabel('$z / w$',fontsize=fs)
plt.legend(fontsize=fslegend)
plt.xlim([0,1.33])
plt.ylim([0,1.03])
plt.show()

# %% USED - SUPPLEMENTAL
# n vs h/w

fig = plt.figure(dpi=600,figsize=(3.5,2.8))
fs = 12
fslegend = 7
xn = np.linspace(0.9,1,100)
plt.plot(hlist/rad,
         nlist,
         '-k',
         linewidth=2)
plt.xlabel('$h / w$',fontsize=fs)
plt.ylabel('$n$',fontsize=fs)
plt.xlim([0.9,1])
plt.ylim([1.95,2.45])

# %% USED - PANEL C
# area vs h

fig = plt.figure(dpi=600,figsize=(3.5,2.8))
fs = 12
fslegend = 7

k = np.all([np.abs(bestdata[:,2]-1)<0.00003,
            hlist>(0.9*rad)],axis=0)
x = hlist[k]/rad
y = arealist[k]/rad**2/2/np.pi
f = lambda x,a,b,c,d,e : 0 + \
        a + b*(x-1)+c*(x-1)**2 + \
        d*(x-1)**3 + e*(x-1)**4
fres = curve_fit(f,x,y)

xn = np.linspace(0.9,1,100)

areaNorm = f(xn,*fres[0])
plt.plot(xn,areaNorm,
         '-k',
         linewidth=2)
plt.xlabel('$h / w$',fontsize=fs)
plt.ylabel('Area / (2$\pi w^2$)',fontsize=fs)
plt.xlim([0.9,1])

# %% USED - Panel D

fig = plt.figure(dpi=600,figsize=(3.5,2.8))
fs = 12
fslegend = 7

firsth = 0.995
lasth = 1

firsthplot = 0.90

xn = np.linspace(firsthplot,1,1000) # firsth
dx = xn[1:]-xn[:-1]
mx = (xn[1:]+xn[:-1])/2

area = f(xn,*fres[0])*2* np.pi *rad**2
da = area[1:]-area[:-1]
force = -da/(dx*rad)

forceNorm = force/rad
hNorm = mx.copy()

arg1 = np.argmin(np.abs(mx-firsth))
arg2 = np.argmin(np.abs(mx-lasth))
xfit = hNorm[arg1:arg2]
yfit = forceNorm[arg1:arg2]
linfit = lambda x,m,b : m*x+b
linfitres = curve_fit(linfit,xfit,yfit,p0 = (-1,1))
slope = linfitres[0][0]

plt.plot(hNorm,forceNorm,
         '-k',
         linewidth=2,
         label='Numerical result',zorder = 1)

plt.plot(xfit,linfit(xfit,*linfitres[0]),
         'r-',label='Linear fit',linewidth=5,zorder = 0.5)

plt.xlabel('$h / w$',fontsize=fs)
plt.ylabel('Force / ($\gamma w$)',fontsize=fs)
plt.xlim([firsthplot,1])

slope = np.round(slope*100)/100
uncert = np.round(np.sqrt(linfitres[1][0,0])*100)/100
if uncert == 0:
    uncert = 0.01

plt.text(firsthplot+0.001,force[-1],'Slope = '+str(slope)[:6]+' $\pm$ '+str(uncert)[:5],
         fontsize = fslegend+2)
plt.legend(fontsize=fslegend)
plt.show()
