#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 01:33:57 2019

@author: jianlongyuan

Function:
    set wanted event's longitude and latitude

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from obspy import read
from obspy.core import UTCDateTime 
from obspy.geodetics import gps2dist_azimuth
plt.rcParams["font.family"] = "Times New Roman"

inputPath = './'

#%%-- get wanted station list
filenameList = []
for iFile in sorted(os.listdir("{0}".format( inputPath ))):
    if '.SAC' in iFile:
        filenameList.append( iFile )
        print( iFile )
        
numFiles = len(filenameList)

#%%
#-- changed evla and evlo
for ifile in np.arange( 0, numFiles, 1 ):
    filename1 = open( "{0}/{1}".format( inputPath, filenameList[ ifile ] )) 
    st1 = read( filename1.name, debug_headers=True)

    eventLat =  33.961    # event's latitude.  unit: degree
    eventLon = -117.892   # event's longitude. unit: degree
    
    epi_dist, az, baz = gps2dist_azimuth(eventLat, eventLon,
                                         st1[0].stats.sac.stla,
                                         st1[0].stats.sac.stlo)
    epi_dist = epi_dist / 1000.0 # km   

    #-- create headers
    st1[0].stats.sac.evla = eventLat
    st1[0].stats.sac.evlo = eventLon
    st1[0].stats.sac.dist = epi_dist
    st1[0].stats.sac.az   = az
    st1[0].stats.sac.baz  = baz

    
    print( 'st1[0].stats:\n', st1[0].stats )    
    #-- output
    st1.write( '{0}/{1}'.format(
                inputPath,
                filenameList[ ifile ] ),
                format='sac')