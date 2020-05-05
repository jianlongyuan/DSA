#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 01:33:57 2019

@author: jianlongyuan
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from obspy import read
from obspy.core import UTCDateTime 
plt.rcParams["font.family"] = "Times New Roman"

SNR = 10000  # SNR defined by the maximum amplitude ratio
inputPath = './'
com = [ 'R', 'T', 'Z' ]
#%%-- get wanted station list
filenameList = []
for iFile in sorted(os.listdir("{0}".format( inputPath ))):
    if 'A' in iFile:
        filenameList.append( iFile )
        print( iFile )
        
numFiles = len(filenameList)

#%%
# create waveform with wanted time length
recordTimeLength = 1 # minite
samplingRate = 400 # Hz

  
for ifile in np.arange( 0, numFiles, 1 ):
    filename1 = open( "{0}/{1}".format( inputPath, filenameList[ ifile ] )) 
    st1 = read( filename1.name, debug_headers=True)

    
    st1.resample(samplingRate)
    nSamples = len(st1[0].data)
    
    
    #-- create headers
    st1[0].stats.network = 'PAPER'
    st1[0].stats.station = 'ST'+str( np.int( np.floor(ifile/3) ) )
    st1[0].stats.channel = com[ ifile%3 ]
    st1[0].stats.npts = nSamples
    st1[0].stats.sac.npts = nSamples
    st1[0].stats.sac.evla = 0
    st1[0].stats.sac.evlo = 0
    st1[0].stats.sac.stla = 0
    st1[0].stats.sac.stlo = 0.8997
    st1[0].stats.sac.baz  = 0

    
    print( 'st1[0].stats:\n', st1[0].stats )    
    #-- output as mseed
    st1.write( '{0}/{1}_Table2_Strike-slip.{2}..{3}.sac'.format(
                inputPath,
                st1[0].stats.network,
                st1[0].stats.station,
                st1[0].stats.channel ),
                format='sac')