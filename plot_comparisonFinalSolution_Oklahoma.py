#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 01 00:00:00 2020

@author: Jianlong Yuan (yuan_jianlong@126.com)
    
    Supervisors: Honn Kao & Jiashun Yu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
barWidth = 0.3

#-- load data
idata1='./Example2_2014-10-07-mw40-Oklahoma/results/LocatingResults.csv'
idata2='./Example3_2014-10-10-mw43_Oklahoma/results/LocatingResults.csv'

data1 = pd.read_csv( '{0}'.format(idata1) )
data2 = pd.read_csv( '{0}'.format(idata2) )

solutionsMw40 = data1['Loc(km)']
solutionsMw43 = data2['Loc(km)']
rmsMw40 = data1['MinRms(s)']
rmsMw43 = data2['MinRms(s)']
print("rmsMw40 =\n", rmsMw40)
print("rmsMw43 =\n", rmsMw43)
print("Event Mw=4.0: {0} well-behaved stations".format( len(solutionsMw40)) )
print("Event Mw=4.3: {0} well-behaved stations".format( len(solutionsMw43)) )     
        
# set figure layout
fig = plt.figure( constrained_layout=True, figsize=(8.25,2.5))
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.25)
gs0 = fig.add_gridspec(1, 1  )
gs00 = gs0[0].subgridspec(1,2)

ax0 = fig.add_subplot(gs00[0, 0])
ax1 = fig.add_subplot(gs00[0, 1])
 
#-- DSA results
medianRmsMw40   = np.mean( rmsMw40 )
medianRmsMw43   = np.mean( rmsMw43 )
medianDepthMw40 = np.median( solutionsMw40 )
medianDepthMw43 = np.median( solutionsMw43 )

stdRmsMw40   = np.std(rmsMw40, ddof=0)
stdRmsMw43   = np.std(rmsMw43, ddof=0)
stdDepthMw40 = np.std(solutionsMw40, ddof=0)
stdDepthMw43 = np.std(solutionsMw43, ddof=0)

print( "medianRmsMw40=", medianRmsMw40, "stdRmsMw40=", stdRmsMw40 )
print( "medianRmsMw43=", medianRmsMw43, "stdRmsMw43=", stdRmsMw43 )
print( "medianDepthMw40=", medianDepthMw40, "stdDepthMw40=", stdDepthMw40 )
print( "medianDepthMw43=", medianDepthMw43, "stdDepthMw43=", stdDepthMw43 )

ax0.errorbar( medianDepthMw40, medianRmsMw40,  xerr=stdDepthMw40, yerr=stdRmsMw40,
              fmt='o', mfc='black', mec='black', ms=10, mew=1,
              ecolor='black', capthick=1, capsize=4,
              label='DSA')


ax1.errorbar( medianDepthMw43, medianRmsMw43,  xerr=stdDepthMw43, yerr=stdRmsMw43,
              fmt='o', mfc='black', mec='black', ms=10, mew=1,
              ecolor='black', capthick=1, capsize=4,
              label='DSA')


#-- Results of McNamara et al.(2015)
rmsMw40 = [0]
rmsMw43 = [0]
solutionsMw40=[ 6.43 ]
solutionsMw43=[ 6.49 ]

medianDepthMw40 = np.mean( solutionsMw40 )
medianDepthMw43 = np.mean( solutionsMw43 )

stdRmsMw40   = 0
stdRmsMw43   = 0
stdDepthMw40  = 0.7
stdDepthMw43  = 1.0

ax0.axvline( medianDepthMw40-stdDepthMw40,
             linewidth=1.25, color='red', linestyle='--')
ax0.axvline( medianDepthMw40,
             linewidth=1.25, color='red', linestyle='-')
ax0.axvline( medianDepthMw40+stdDepthMw40,
             linewidth=1.25, color='red', linestyle='--')
ax1.axvline( medianDepthMw43-stdDepthMw43,
             linewidth=1.25, color='red', linestyle='--')
ax1.axvline( medianDepthMw43,
             linewidth=1.25, color='red', linestyle='-')
ax1.axvline( medianDepthMw43+stdDepthMw43,
             linewidth=1.25, color='red', linestyle='--')    
    
ax0.set_title( "Mw 4.0 (Oklahoma)", fontsize=14, color='black' )
ax0.set_xlabel('Depth (km)', fontsize=12)
ax0.set_ylabel('RMS (s)', fontsize=12)
ax0.set_xlim( 4.0, 9.0 )
ax0.set_ylim( 0.0, 0.6 )
ax1.set_title( "Mw 4.3 (Oklahoma)", fontsize=14, color='black' )
ax1.set_xlabel('Depth (km)', fontsize=12)
ax1.set_ylabel('RMS (s)', fontsize=12)
ax1.set_xlim( 4.0, 9.0 )
ax1.set_ylim( 0.0, 0.6 )

plt.tight_layout()
plt.show()
