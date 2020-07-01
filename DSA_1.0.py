#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 01 00:00:00 2020

@author: Jianlong Yuan (yuan_jianlong@126.com)
    
    Supervisors: Honn Kao & Jiashun Yu


Algorithm name: 
        Depth-Scanning Algorithm (DSA)


Framework:
 1. Automatic generation of synthetic waveforms for all possible depth phases.    
 2. Match-filtering of all possible depth phases.
 3. Preliminary determination of the focal depth.
 4. Final solution based on travel time residuals.
    

Input:
  1. Three-component waveforms.
      Notice: SAC format. Header at least should has corrected 'dist' and 'baz'.
  2. Velocity model.
      Notice: TauP Toolkit format (see Section 5 in
              https://www.seis.sc.edu/downloads/TauP/taup.pdf )

Output:
  Focal depth (median) 
  
  
For more details please refer to our paper below:
  Yuan, J., Kao, H., & Yu, J. (2020). Depth-Scanning Algorithm: Accurate, Automatic, and
  Efficient Determination of Focal Depths for Local and Regional Earthquakes.
  Journal of Geophysical Research: Solid Earth, 125, e2020JB019430.
  https://doi.org/10.1029/2020JB019430


Any questions or advices? Please contact:
  yuan_jianlong@126.com
  honn.kao@canada.ca
  j.yu@cdut.edu.cn
     
"""

from obspy.taup import TauPyModel, taup_create
import matplotlib.pyplot as plt
from obspy.geodetics.base import kilometer2degrees
from obspy.core import UTCDateTime
import matplotlib.pyplot as pltDebug
import math
import numpy as np
from scipy.signal import hilbert, find_peaks
from obspy import read, read_inventory
import pandas as pd
from scipy.stats import kurtosis as kurt
import scipy.stats as stats
from scipy import signal
import os, fnmatch, sys
import timeit
start = timeit.default_timer()
import shutil
import csv 
plt.rcParams["font.family"] = "Times New Roman"



#%%-- subroutine: load input parameters from 'DSA_SETTINGS.txt'
def load_settings():
    '''
     PARAMETER          DESCRIPTION
     
     par1    data directory, including wavefroms and velocity model 
     par2    velocity model name (this string should not include '.nd')    
     par3    tolerance between the observed and predicted differential travel times (second)
     par4    cross-correlation coefficient threshold
     par5    minimal frequency used for band-pass filter (Hz)
     par6    maximal frequency used for band-pass filter (Hz)
     par7    minimal scanning depth candidate (interger, km)
     par8    maximal scanning depth candidate (interger, km)
     par9    for monitoring: 1 -> active,  0 -> inactive
     par10   plot Steps 1 and 2 of DSA: 1 -> active,  0 -> inactive
    '''
    
    try:
        SETTINGS = pd.read_csv('./DSA_SETTINGS.txt', delim_whitespace=True, index_col='PARAMETER')
        
        par1 = SETTINGS.VALUE.loc['dataPath'] 
        par2 = SETTINGS.VALUE.loc['velModel']   
        par3 = float( SETTINGS.VALUE.loc['arrTimeDiffTole'] ) 
        par4 = float( SETTINGS.VALUE.loc['ccThreshold']  )
        par5 = float( SETTINGS.VALUE.loc['frequencyFrom']  )
        par6 = float( SETTINGS.VALUE.loc['frequencyTo']  )
        par7 =   int( SETTINGS.VALUE.loc['scanDepthFrom']  )
        par8 =   int( SETTINGS.VALUE.loc['scanDepthTo']  )
        par9 =   int( SETTINGS.VALUE.loc['verboseFlag']  )
        par10=   int( SETTINGS.VALUE.loc['plotSteps1n2Flag']  )
        
        return par1, par2, par3, par4, par5, par6, par7, par8, par9, par10    
    
    except:
        sys.exit("Errors in 'DSA_SETTINGS.txt' !\n")
        



#%%-- subroutine: Extract top depths of the crust interfaces from velocity model
def GetCrustInterfaceDepths( velModel ):
    crustInterfaceDepths = []
    
    velMod = pd.read_csv(str(dataPath)+str(velModel)+'.nd',
                         delim_whitespace=True, header=None,
                         names=['TopDepth', 'Vp', 'Vs', 'Rho', 'Qp', 'Qs'])
    
    for irow in range( len(velMod) ):
        ival = velMod.loc[irow]['TopDepth']
    
        if ival == 'mantle':
            break  
        
        try:
            ival = np.float( ival )
            if ival > 0.0:
                crustInterfaceDepths.append(ival)
        except:
            continue

    crustInterfaceDepths = sorted(set(crustInterfaceDepths))
    return crustInterfaceDepths


#%%-- subroutine: cross-correlation
def xcorrssl( scanTimeBeg, scanTimeEnd, tem, tra ):
    
    temLeng = len(tem)
    traLeng = len(tra)
    time_lags= traLeng - temLeng + 1
    
    #-- demean for the template
    b = tem - np.mean(tem)
    corr_norm_idx = []
    corr_norm_val = []
    
    for k in range( time_lags ):
        if ( k >= scanTimeBeg and k <=scanTimeEnd ):
            # demean for the trace
            a = tra[k:(k+temLeng)] - np.mean(tra[k:(k+temLeng)])
            stdev = (np.sum(a**2)) ** 0.5 * (np.sum(b**2)) ** 0.5
            if stdev != 0:
                corr = np.sum(a*b)/stdev
            else:
                corr = 0
            corr_norm_idx.append(k)
            corr_norm_val.append(corr)
        else:
            corr_norm_idx.append(k)
            corr_norm_val.append(0.)
            
    return corr_norm_val

#-- subroutine: arrival time forward modelling kernel
def subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth ):
    model = TauPyModel(model= velModel )
    
    try:
        arrivals = model.get_travel_times(source_depth_in_km=srcDepth,
                                           distance_in_degree=recDisInDeg,
                                           phase_list= phaList,
                                           receiver_depth_in_km=recDepth)
        
        rays = model.get_ray_paths(source_depth_in_km=srcDepth,
                                   distance_in_degree=recDisInDeg,
                                   phase_list= phaList,
                                   receiver_depth_in_km=recDepth)
    except: # avoid TauP error
        srcDepth += 0.1
        arrivals = model.get_travel_times(source_depth_in_km=srcDepth,
                                           distance_in_degree=recDisInDeg,
                                           phase_list= phaList,
                                           receiver_depth_in_km=recDepth)
        
        rays = model.get_ray_paths(source_depth_in_km=srcDepth,
                                   distance_in_degree=recDisInDeg,
                                   phase_list= phaList,
                                   receiver_depth_in_km=recDepth)
        
        
    # correct phase name (e.g., PvmP is code name, PmP is academic name)
    for i in range(len(arrivals)):
        if arrivals[i].name == 'PvmP':
            arrivals[i].name = 'PmP'
        if arrivals[i].name == 'pPvmP':
            arrivals[i].name = 'pPmP'
        if arrivals[i].name == 'sPvmP':
            arrivals[i].name = 'sPmP'
        if arrivals[i].name == 'SvmS':
            arrivals[i].name = 'SmS'
        if arrivals[i].name == 'sSvmS':
            arrivals[i].name = 'sSmS'    
    
    return  arrivals, rays

#-- subroutine: filter out some strange rays that do not arrive at the staion,
#   and delete some refracted waves.
def subDeleteRefractedWave( crustInterfaceDepths, srcDepth, arrivals, rays ):        
    removeIdx = []
    nRays = len(rays)
    
    ###################################################################
    #--  filter out some strange rays that do not arrive at the staion:
    # 1. get distance value of the last point of each ray
    # 2. get the median value of the distances
    # 3. find out the distance that is different with the median
    # 4. if the ratio (the different distance value / median) > 10 %, then
    #    delete this distance
    ###################################################################
    dist  = []
    for iRay in range( nRays ):
        nPointsRay = np.shape( rays[iRay].path )
        lastPtIdx = nPointsRay[0]
        dist.append( rays[iRay].path[ lastPtIdx-1 ][2] )
    if( len(dist) > 0 ): # avoid unstable situation
        median = np.median( dist )
        #print( "median=", median )
        
        for iRay in range( nRays ):
            disRatio = np.fabs( ( dist[ iRay ] - median ) / median )
            #print( "disRatio={0}".format( format(disRatio,".3f" )))
            if disRatio > 0.1:
                removeIdx.append(iRay)
    
    
    #%%-- delete some refracted waves, which are related to 
     # the interfaces by using:
     # 1) find out the maximum depth of each ray
     # 2) delete the ray whose maximum depth is not located at the interface    
    for iRay in range( nRays ):
        depthData = []
        nPointsRay = np.shape( rays[iRay].path )
        for i in np.arange( 1, nPointsRay[0], 1 ):
            depthData.append( rays[iRay].path[i][3] )
        maxRayDepth = np.max( depthData )
        #print(  maxRayDepth )
             
        if( arrivals[iRay].name != "p" and arrivals[iRay].name != "s" ):
            rayIsRefractedWave = 1
            for iDep in crustInterfaceDepths:
                if maxRayDepth == iDep  or maxRayDepth == srcDepth:
                    rayIsRefractedWave = 0
            if rayIsRefractedWave == 1:
                removeIdx.append( iRay )
        

        for i in range( nPointsRay[0] - 2):
            i1 = i
            i2 = i+1
            i3 = i+2
            # rays[iRay].path[i][3], in which 3 is the index of "depth"
            if rays[iRay].path[i1][3] == rays[iRay].path[i3][3]:
                if ((rays[iRay].path[i1][3] < rays[iRay].path[i2][3]) and 
                    (rays[iRay].path[i2][3] > rays[iRay].path[i3][3]) ):
                       err12 = rays[iRay].path[i1][3] - rays[iRay].path[i2][3]
                       err23 = rays[iRay].path[i2][3] - rays[iRay].path[i3][3]
                       if np.fabs(err12) < 0.5 or np.fabs(err23) < 0.5:
                           removeIdx.append(iRay)
                           break
                
    #-- deleting using reverse order
    if len(removeIdx) > 0:
        removeIdxInv = sorted( set( list(removeIdx) ), reverse=True )

        for i in removeIdxInv:
            #print(i)
            arrivals.remove( arrivals[i] )
            rays.remove( rays[i] )
            
    return  arrivals, rays






#%%
###################################################
# Input parameters
###################################################
'''
 PARAMETER          DESCRIPTION
 
 dataPath           data directory, including wavefroms and velocity model 
 velModel           velocity model name (this string should not include '.nd')    
 arrTimeDiffTole    tolerance between the observed and predicted differential travel times (second)
 ccThreshold        cross-correlation coefficient threshold
 frequencyFrom      minimal frequency used for band-pass filter (Hz)
 frequencyTo        maximal frequency used for band-pass filter (Hz)
 scanDepthFrom      minimal scanning depth candidate (interger, km)
 scanDepthTo        maximal scanning depth candidate (interger, km)
 verboseFlag        for monitoring: 1 -> active,  0 -> inactive
 plotSteps1n2Flag   plot Steps 1 and 2 of DSA: 1 -> active,  0 -> inactive
'''

#%%-- load input parameters
dataPath, velModel, arrTimeDiffTole, ccThreshold, frequencyFrom, frequencyTo,\
scanDepthFrom, scanDepthTo, verboseFlag, plotSteps1n2Flag = load_settings()
    
#%%-- get the number of waveform files (HH* components)
wfFiles = fnmatch.filter( sorted(os.listdir(dataPath)), '*.SAC')
numSt = np.int( len(wfFiles)/3 ) # 3 -> three components
for i in range( numSt):
    print( '\t SAC files in the directory:')
    print( wfFiles[i*3], wfFiles[i*3+1], wfFiles[i*3+2] )

#%%-- create output file directory
if not os.path.exists(str(dataPath)+'results'):
    os.mkdir(str(dataPath)+'results')
else:
    print( '"resluts" already exists!')
    shutil.rmtree(str(dataPath)+'results')
    os.mkdir(str(dataPath)+'results')
outfilePath = str(dataPath)+'results'

#%%-- output file, here to create newfile    
outPath = str(outfilePath)+'/LocatingResults.csv'
with open( '{0}'.format( outPath ), mode='w', newline=''  ) as resultsFile:
    writer = csv.writer( resultsFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow( [ 'StName', 'Az(deg)', 'EpDis(deg)', 'NumMatPha',
                       'Loc(km)', 'SubDivDep(km)',' Rms(s)', 'MinRms(s)' ] )

    
#%%-- print key information
print( '\n==========- INPUT PARAMETERS -==========\n')
print( 'dataPath           =', dataPath)
print( 'velModel           =', velModel)
print( 'arrTimeDiffTole    =', arrTimeDiffTole)
print( 'ccThreshold        =', ccThreshold )
print( 'frequencyFrom      =', frequencyFrom )
print( 'frequencyTo        =', frequencyTo )
print( 'scanDepthFrom      =', scanDepthFrom )
print( 'scanDepthTo        =', scanDepthTo )
print( 'verboseFlag        =', verboseFlag )
print( 'plotSteps1n2Flag   =', plotSteps1n2Flag )
print( 'Number of stations =', numSt )
print( 'outfilePath        =', outfilePath)
print( '\n======================================\n')    
    
    

#%%-- velocity for tauP
taup_create.build_taup_model( str(dataPath)+str(velModel)+'.nd' )
    
#%%-- Allocate memory
numScanDepth = np.int(scanDepthTo-scanDepthFrom)
idxMaxNumPhaEachStation = np.zeros((numSt))
idxMaxNumPhaEachStation.fill(9999) # initial array with a high value
azimuthEachStation= np.zeros((numSt))
epiDisEachStation = np.zeros((numSt))
onsetPEachStation = np.zeros((numSt))
onsetSEachStation = np.zeros((numSt))
totNumMatPhaGlobal  = np.zeros((numSt, numScanDepth))
totNumCalPhaGlobal  = np.zeros((numSt, numScanDepth))
percNumMatPhaGlobal = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationZ = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationZ.fill(9999) # initial array with a high value
avgArrTimeDiffResEachStationR = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationR.fill(9999) # initial array with a high value
avgArrTimeDiffResEachStationT = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationT.fill(9999) # initial array with a high value
avgArrTimeDiffResEachStationSum = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationSum.fill(9999) # initial array with a high value
sumAvgArrTimeDiffResGlobal = np.zeros((numScanDepth))
sumAvgArrTimeDiffResGlobal.fill(9999) # initial array with a high value
    
nameEachStation        = [[] for i in range(numSt)]
histogramGlobalZ       = [[] for i in range(numSt)]
histogramGlobalR       = [[] for i in range(numSt)]
histogramGlobalT       = [[] for i in range(numSt)]
template0GlobalZ       = [[] for i in range(numSt)]
template0GlobalR       = [[] for i in range(numSt)]
template0GlobalT       = [[] for i in range(numSt)]
template60GlobalZ      = [[] for i in range(numSt)]
template60GlobalR      = [[] for i in range(numSt)]
template60GlobalT      = [[] for i in range(numSt)]
template120GlobalZ     = [[] for i in range(numSt)]
template120GlobalR     = [[] for i in range(numSt)]
template120GlobalT     = [[] for i in range(numSt)]
template170GlobalZ     = [[] for i in range(numSt)]
template170GlobalR     = [[] for i in range(numSt)]
template170GlobalT     = [[] for i in range(numSt)]
waveformGlobalZ        = [[] for i in range(numSt)]
waveformGlobalR        = [[] for i in range(numSt)]
waveformGlobalT        = [[] for i in range(numSt)]
normWaveformGlobalZ    = [[] for i in range(numSt)]
normWaveformGlobalR    = [[] for i in range(numSt)]
normWaveformGlobalT    = [[] for i in range(numSt)]
bigAmpGlobalZ          = [[] for i in range(numSt)]
bigAmpGlobalR          = [[] for i in range(numSt)]
bigAmpGlobalT          = [[] for i in range(numSt)]
bigAmpTimeGlobalZ      = [[] for i in range(numSt)]
bigAmpTimeGlobalR      = [[] for i in range(numSt)]
bigAmpTimeGlobalT      = [[] for i in range(numSt)]
peaksCurveGlobalZ      = [[] for i in range(numSt)]
peaksCurveGlobalR      = [[] for i in range(numSt)]
peaksCurveGlobalT      = [[] for i in range(numSt)]
finalpeaksPtsGlobalZ   = [[] for i in range(numSt)]
finalpeaksPtsGlobalR   = [[] for i in range(numSt)]
finalpeaksPtsGlobalT   = [[] for i in range(numSt)]
finalCcGlobalZ         = [[] for i in range(numSt)]
finalCcGlobalR         = [[] for i in range(numSt)]
finalCcGlobalT         = [[] for i in range(numSt)]
phaShiftAngGlobalZ     = [[] for i in range(numSt)]
phaShiftAngGlobalR     = [[] for i in range(numSt)]
phaShiftAngGlobalT     = [[] for i in range(numSt)]
phaShiftAngTimeGlobalZ = [[] for i in range(numSt)]
phaShiftAngTimeGlobalR = [[] for i in range(numSt)]
phaShiftAngTimeGlobalT = [[] for i in range(numSt)]
leftBoundry1GlobalZ    = [[] for i in range(numSt)]
leftBoundry1GlobalR    = [[] for i in range(numSt)]
leftBoundry1GlobalT    = [[] for i in range(numSt)]
rightBoundry1GlobalZ   = [[] for i in range(numSt)]
rightBoundry1GlobalR   = [[] for i in range(numSt)]
rightBoundry1GlobalT   = [[] for i in range(numSt)]
DTGlobalZ              = [[] for i in range(numSt)]
DTGlobalR              = [[] for i in range(numSt)]
DTGlobalT              = [[] for i in range(numSt)]
st_begin_timeNorGlobalZ= [[] for i in range(numSt)]
st_begin_timeNorGlobalR= [[] for i in range(numSt)]
st_begin_timeNorGlobalT= [[] for i in range(numSt)]
wantedTimeLengGlobalZ  = [[] for i in range(numSt)]
wantedTimeLengGlobalR  = [[] for i in range(numSt)]
wantedTimeLengGlobalT  = [[] for i in range(numSt)]
temBegNorGlobalZ       = [[] for i in range(numSt)]
temBegNorGlobalR       = [[] for i in range(numSt)]
temBegNorGlobalT       = [[] for i in range(numSt)]
corrLengGlobalZ        = [[] for i in range(numSt)]
corrLengGlobalR        = [[] for i in range(numSt)]
corrLengGlobalT        = [[] for i in range(numSt)]

depthCandidateArrGlobalZ        = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrGlobalR        = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrGlobalT        = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrDiffGlobalZ    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrDiffGlobalR    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrDiffGlobalT    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaDigNameGlobalZ = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaDigNameGlobalR = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaDigNameGlobalT = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaOrgNameGlobalZ = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaOrgNameGlobalR = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaOrgNameGlobalT = [[[] for i in range(numScanDepth)] for j in range(numSt)]
  

#%%-- Step 1 to 4 of DSA
for ist in range( numSt ):    
   
    #%%########################################################
    # Step 1: Automatic generation of synthetic waveforms for #
    #         all possible depth phases                       #
    ###########################################################
    
    #%%-- Read waveform data                  
    infileE = open('{0}{1}'.format( dataPath, wfFiles[ist*3+0] ) )
    infileN = open('{0}{1}'.format( dataPath, wfFiles[ist*3+1] ) )
    infileZ = open('{0}{1}'.format( dataPath, wfFiles[ist*3+2] ) )
    
    #%%--
    stRawE = read(infileE.name, debug_headers=True)
    stRawN = read(infileN.name, debug_headers=True)
    stRawZ = read(infileZ.name, debug_headers=True)
    
    #%%-- Get some key infomation
    evLa = stRawZ[0].stats.sac.evla # event's latitude.  unit: degree
    evLo = stRawZ[0].stats.sac.evlo # event's longitude. unit: degree    
    stLa = stRawZ[0].stats.sac.stla # station's latitude.  unit: degree
    stLo = stRawZ[0].stats.sac.stlo # station's latitude.  unit: degree
    recDisInKm = stRawZ[0].stats.sac.dist
    recDisInDeg = kilometer2degrees(recDisInKm)
    nameEachStation[ist].append( stRawZ[0].stats.sac.kstnm.strip() )
    epiDisEachStation[ist] = recDisInDeg
    azimuthEachStation[ist] = stRawZ[0].stats.sac.az
    baz = stRawZ[0].stats.sac.baz
    DT = stRawZ[0].stats.sac.delta
    st_begin_timeNor = stRawZ[0].stats.sac.o
    st_begin_timeUTC = UTCDateTime(stRawZ[0].stats.starttime+st_begin_timeNor)
    
    print("\n\n\n========================================================")
    print('Now processing:')
    print(infileE.name)
    print(infileN.name)
    print(infileZ.name)
    print("Station = ", stRawZ[0].stats.sac.kstnm )
    print("Scanning station id=", ist)
    print("Az, Baz = ", stRawZ[0].stats.sac.az, baz )
    print("evLa, evLo = ", evLa, evLo )
    print("stLa, stLo = ", stLa, stLo )
    print("recDisInKm  = ",recDisInKm)
    print("recDisInDeg = ", recDisInDeg)
    print("Dt = ", DT)
    print("stRawZ[0].stats.sac.dist = ", stRawZ[0].stats.sac.dist )
    print("stRawZ[0].stats.starttime", stRawZ[0].stats.starttime)
    print("st_begin_timeNor = ", st_begin_timeNor)
    print("st_begin_timeUTC = ", st_begin_timeUTC)
    print("--------------------------------------------------------\n")
    
    #%%-- Waveform scanning window used for DSA in second
    if recDisInKm < 80:
        wantedTimeLengZ = 40
    else:
        wantedTimeLengZ = 70
        
    
    #-- This commend is only for the synthetic example in Section 3.2 of DSA paper
    if velModel == "ak135_Section3.2":
        wantedTimeLengZ = 22
        
    #-- This commend is only for the synthetic example in Section3.3 of DSA paper
    if velModel == "ak135_Section3.3":
        wantedTimeLengZ = 30
        
    #%%-- These two commends are only for test diffetent azimuthal coverages
    '''
    if stRawZ[0].stats.sac.az < 180 or stRawZ[0].stats.sac.az >= 270:
           continue
    '''      
    
    #%%-- Extract wavefroms within wanted time window
    stWantedE = stRawE.trim(st_begin_timeUTC, st_begin_timeUTC+wantedTimeLengZ)
    stWantedN = stRawN.trim(st_begin_timeUTC, st_begin_timeUTC+wantedTimeLengZ)
    stWantedZ = stRawZ.trim(st_begin_timeUTC, st_begin_timeUTC+wantedTimeLengZ)

    
    #%%-- Remove mean value and trend
    stWantedE0 = stWantedE.copy()
    stWantedN0 = stWantedN.copy()
    stWantedZ0 = stWantedZ.copy()
    stWantedE0[0].detrend( type='demean')
    stWantedN0[0].detrend( type='demean')
    stWantedZ0[0].detrend( type='demean')
    stWantedE0[0].detrend( type='simple')
    stWantedN0[0].detrend( type='simple')
    stWantedZ0[0].detrend( type='simple')   

    
    #%%-- Remove response
    try:
        inv = read_inventory( "{0}/{1}.{2}.xml".format( dataPath, stRawZ[0].stats.network, stRawZ[0].stats.station) )
        pre_filt = (0.005, 0.006, 30.0, 35.0)
        stWantedE0[0].remove_response(inventory=inv, output="DISP", pre_filt=pre_filt)
        stWantedN0[0].remove_response(inventory=inv, output="DISP", pre_filt=pre_filt)
        stWantedZ0[0].remove_response(inventory=inv, output="DISP", pre_filt=pre_filt)
    
        #%% -- do ratation from Z12 to ZNE
        if stRawE[0].stats.channel == 'HH1' or stRawE[0].stats.channel == 'BH1':
            stZ12 = stWantedZ0 + stWantedE0 + stWantedN0
            print(stZ12)
            stZ12.rotate( method='->ZNE', inventory=inv )
            stWantedZ0 = stZ12.select(component="Z")
            stWantedN0 = stZ12.select(component="N")
            stWantedE0 = stZ12.select(component="E")
        
        #%%-- do ratation: NE to RT using back-azimuth angle
        stNE = stWantedN0 + stWantedE0
        stNE.rotate( method='NE->RT', back_azimuth=baz )
        stR0 = stNE.select(component="R")
        stT0 = stNE.select(component="T")        
        stZ0 = stWantedZ0.copy()
#        print(stNE)
#        print(stR0)
#        print(stT0)    
        
        #%%-- taper before filtering
        stZ0[0] = stZ0[0].taper(max_percentage=0.1, side='left')
        stR0[0] = stR0[0].taper(max_percentage=0.1, side='left')
        stT0[0] = stT0[0].taper(max_percentage=0.1, side='left')

        print( 'Real data, remove response done!\n')

    except:
        print( 'No response file! Maybe synthetic data?\n')
        stR0 = stWantedE0.copy()
        stT0 = stWantedN0.copy()        
        stZ0 = stWantedZ0.copy()
        
        
    #%%-- frequency filtering
    stZ0[0] = stZ0[0].filter('bandpass', freqmin=frequencyFrom, freqmax=frequencyTo,
                                         corners=4, zerophase=False)
    stR0[0] = stR0[0].filter('bandpass', freqmin=frequencyFrom, freqmax=frequencyTo,
                                         corners=4, zerophase=False)
    stT0[0] = stT0[0].filter('bandpass', freqmin=frequencyFrom, freqmax=frequencyTo,
                                         corners=4, zerophase=False)

    #%%-- check waveform
    print('\n Check Z, R, and T waveforms: \n')
    stZ0.plot()
    stR0.plot()
    stT0.plot()


    #%%-- Get P and S onset using kurtosis of scipy 
    #-- Z
    findOnsetIdxFlagZ = 0
    dfZ = pd.DataFrame()   
    dfZ['stZ0[0]'] = stZ0[0].data
    dfZ['kurtosisZ'] = dfZ['stZ0[0]'].rolling(200).apply(kurt, raw=True)       
    kurtosisZ = dfZ[ 'kurtosisZ' ]
    maxKurZ = np.max( np.fabs( kurtosisZ ))
    for i in range( len(kurtosisZ) ):
        if kurtosisZ[i] > (maxKurZ*0.99):
            onsetBegIdxZ = i
            findOnsetIdxFlagZ = 1
            break
    if findOnsetIdxFlagZ == 0:
        onsetBegIdxZ = 0
    onsetZ = onsetBegIdxZ*DT
    onsetZNor = onsetZ
    onsetZUTC = UTCDateTime( stRawZ[0].stats.starttime + onsetZNor)
    print( "onsetZNor=", onsetZNor)
    print( "onsetZUTC=", onsetZUTC)
    
    #-- R
    findOnsetIdxFlagR = 0
    dfR = pd.DataFrame()   
    dfR['stR0[0]'] = stR0[0].data
    dfR['kurtosisR'] = dfR['stR0[0]'].rolling(200).apply(kurt, raw=True)       
    kurtosisR = dfR[ 'kurtosisR' ]
    maxKurR = np.max( np.fabs( kurtosisR ))
    for i in range( len(kurtosisR) ):
        if kurtosisR[i] > (maxKurR*0.99):
            onsetBegIdxR = i
            findOnsetIdxFlagR = 1
            break
    if findOnsetIdxFlagR == 0:
        onsetBegIdxR = 0
    onsetR = onsetBegIdxR*DT
    onsetRNor = onsetR
    onsetRUTC = UTCDateTime( stRawZ[0].stats.starttime + onsetRNor)
    print( "onsetRNor=", onsetRNor)
    print( "onsetRUTC=", onsetRUTC)
    
    # The scanning time of S-wave starts at a time = onset of P-wave 
    scanBegTimeT = onsetR
    scanBegTimeIdxT = np.int(scanBegTimeT/DT)
    findOnsetIdxFlagT = 0
    dfT = pd.DataFrame()   
    dfT['stT0[0]'] = stT0[0].data[scanBegTimeIdxT:]
    dfT['kurtosisT'] = dfT['stT0[0]'].rolling(500).apply(kurt, raw=True)       
    kurtosisT = dfT[ 'kurtosisT' ]
    maxKurT = np.max( np.fabs( kurtosisT ))
    #-- scan from 3 s after the onset time of P-wave
    for i in range( len(kurtosisT) ):
        if kurtosisT[i] > (maxKurT*0.99):
            onsetBegIdxT = i + scanBegTimeIdxT
            findOnsetIdxFlagT = 1
            break
    if findOnsetIdxFlagT == 0:
        onsetBegIdxT = 0
    onsetT = onsetBegIdxT*DT
    onsetTNor = onsetT
    onsetTUTC = UTCDateTime( stRawZ[0].stats.starttime + onsetTNor)
    print( "onsetTNor=", onsetTNor)
    print( "onsetTUTC=", onsetTUTC)

    #%%-- here we will use two conditions to evaluate current station:
    # Condition 1: when onset time of Z/R/T cannot match its theorectical
    # arrival time, meaning this station is unreliable, then skip it
    recDepth = 0  # station's depth( default 0 km)
    refDepth = 15 # default depth for calculte referenced onset time of direct wave (km)
    crustInterfaceDepths = GetCrustInterfaceDepths( velModel )
    print( 'crustInterfaceDepths = ', crustInterfaceDepths )
    phaList = [ "p", "Pg" ]
    arrivals, rays = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
    arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
    calOnsetP = arrivals[0].time
    print( arrivals )
    print( "calOnsetP=", calOnsetP )            

    
    phaList = [ "s", "Sg" ]
    arrivals, rays = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
    arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
    calOnsetS = arrivals[0].time
    print( arrivals )
    print( "calOnsetS=", calOnsetS ) 
    
    if np.fabs( onsetZNor+stRawZ[0].stats.sac.b-calOnsetP ) > 5 or\
       np.fabs( onsetRNor+stRawZ[0].stats.sac.b-calOnsetP ) > 5 or\
       np.fabs( onsetTNor+stRawZ[0].stats.sac.b-calOnsetS ) > 5:
        print("\n Onset time match failed, skip station=", stRawZ[0].stats.sac.kstnm, 
                  "EpiDis:", recDisInKm, "km \n")
        continue
        
    
    #-- Condition 2: when all kurtosis values of Z/R/T cannot meet preset 
    # threshold, meaning this station is with low-quality S/N, then skip it
    if findOnsetIdxFlagZ == 0 or\
       findOnsetIdxFlagR == 0 or\
       findOnsetIdxFlagT == 0:
        print("\n Cannot find onset time, skip station=",
                  stRawZ[0].stats.sac.kstnm, 
                 "EpiDis:",
                 recDisInKm, "km\n\n\n")
        continue

    
        
    #%%-- plot kurtosis function
    if verboseFlag == 1:
        print('\n Check kurtosis picking: \n')
        kurNorZ = kurtosisZ / np.max(np.fabs(kurtosisZ))
        stNorZ0 = stZ0[0].data / np.max(np.fabs(stZ0[0].data))
        tKur = np.arange( 0, len(kurtosisZ), 1)*DT+stRawZ[0].stats.sac.b
        tT0  = np.arange( 0, len(stNorZ0), 1)*DT+stRawZ[0].stats.sac.b
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'Z: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorZ )
        pltDebug.plot( tT0, stNorZ0*1.0+1, color='lightgray' )
        pltDebug.scatter( onsetZ+stRawZ[0].stats.sac.b, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetP, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper right')

        kurNorR = kurtosisR / np.max(np.fabs(kurtosisR))
        stNorR0 = stR0[0].data / np.max(np.fabs(stR0[0].data))
        tKur = np.arange( 0, len(kurtosisR), 1)*DT+stRawZ[0].stats.sac.b
        tT0  = np.arange( 0, len(stNorR0), 1)*DT+stRawZ[0].stats.sac.b
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'R: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorR )
        pltDebug.plot( tT0, stNorR0*1.0+1, color='lightgray' )
        pltDebug.scatter( onsetR+stRawZ[0].stats.sac.b, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetP, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper right')
        
        kurNorT = kurtosisT / np.max(np.fabs(kurtosisT))
        stNorT0 = stT0[0].data / np.max(np.fabs(stT0[0].data))                
        tKur = np.arange( 0, len(kurtosisT), 1)*DT+scanBegTimeT+stRawZ[0].stats.sac.b
        tT0  = np.arange( 0, len(stNorT0), 1)*DT+stRawZ[0].stats.sac.b
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'T: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorT )
        pltDebug.plot( tT0, stNorT0*1.0+1, color='lightgray' )
        pltDebug.scatter( onsetT+stRawZ[0].stats.sac.b, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetS, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper right')
        
        plt.show()
                
    #%%-- first time to roughly chose direct wave
    stZ1 = stZ0.copy()
    temZtBegNor = onsetZNor - 0.5
    temZtEndNor = onsetZNor + 0.5
    temZtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + temZtBegNor)
    temZtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + temZtEndNor )
    print("temZtBegUTC=", temZtBegUTC)
    print("temZtEndUTC=", temZtEndUTC)
    templateZ=stZ1.trim( temZtBegUTC, temZtEndUTC)
    maxAmpTemZ = np.max(templateZ[0].data)
    print("maxAmpTemZ=", maxAmpTemZ )
    
    stR1 = stR0.copy()
    zeroPtArrTempR = []
    zeroPtArrTempT = []
    zeroPtArrTempZ = []
    temRtBegNor = onsetRNor - 0.5
    temRtEndNor = onsetRNor + 0.5
    temRtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + temRtBegNor )
    temRtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + temRtEndNor )
    print("temRtBegUTC", temRtBegUTC)
    templateR=stR1.trim( temRtBegUTC, temRtEndUTC)
    maxAmpTemR = np.max(templateR[0].data)
    print("maxAmpTemR=", maxAmpTemR )
    
    stT1 = stT0.copy()
    temTtBegNor = onsetTNor - 0.5
    temTtEndNor = onsetTNor + 0.5
    temTtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + temTtBegNor )
    temTtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + temTtEndNor )
    templateT=stT1.trim( temTtBegUTC, temTtEndUTC)
    maxAmpTemT = np.max(templateT[0].data)
    print("maxAmpTemT=", maxAmpTemT )

    
    #-- find the minimum and maximum amplitudes and their arrival times
    minAmpTemValZ = np.min(templateZ[0].data)
    minAmpTemIdxZ = np.argmin(templateZ[0].data)
    minAmpTemTimeZ = minAmpTemIdxZ * DT + temZtBegNor
    maxAmpTemValZ = np.max(templateZ[0].data)
    maxAmpTemIdxZ = np.argmax(templateZ[0].data)
    maxAmpTemTimeZ = maxAmpTemIdxZ * DT + temZtBegNor
    
    minAmpTemValT = np.min(templateT[0].data)
    minAmpTemIdxT = np.argmin(templateT[0].data)
    minAmpTemTimeT = minAmpTemIdxT * DT + temTtBegNor
    maxAmpTemValT = np.max(templateT[0].data)
    maxAmpTemIdxT = np.argmax(templateT[0].data)
    maxAmpTemTimeT =  maxAmpTemIdxT * DT + temTtBegNor
    
    halfCircleTimeZ = np.fabs( minAmpTemTimeZ - maxAmpTemTimeZ )
    halfCircleTimeT = np.fabs( minAmpTemTimeT - maxAmpTemTimeT )
    print("halfCircleTimeZ = ", halfCircleTimeZ, "sec")
    print("halfCircleTimeT = ", halfCircleTimeT, "sec")
          
    if np.fabs(minAmpTemValZ) > np.fabs(maxAmpTemValZ):
        t0P = minAmpTemTimeZ
        t1P = minAmpTemTimeZ
        idxP = minAmpTemIdxZ
    else:
        t0P = maxAmpTemTimeZ
        t1P = maxAmpTemTimeZ
        idxP = maxAmpTemIdxZ
    
    if np.fabs(minAmpTemValT) > np.fabs(maxAmpTemValT):
        t0S = minAmpTemTimeT
        t1S = minAmpTemTimeT
        idxS = minAmpTemIdxT
    else:
        t0S = maxAmpTemTimeT
        t1S = maxAmpTemTimeT
        idxS = maxAmpTemIdxT

    #%%-- using 2.5 times periods as the time length of direct wave     
    t0P = t0P - 2.5 * halfCircleTimeZ    
    t1P = t1P + 2.5 * halfCircleTimeZ    
    t0S = t0S - 2.5 * halfCircleTimeT 
    t1S = t1S + 2.5 * halfCircleTimeT
    print("t0P, t1P=", t0P, t1P)
    print("t0S, t1S=", t0S, t1S)
    onsetPEachStation[ist] = t0P
    onsetSEachStation[ist] = t0S

    stZ1 = stZ0.copy()
    temZtBegNor = t0P
    temZtEndNor = t1P
    temZtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + t0P )
    temZtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + t1P )
    templateZ=stZ1.trim( temZtBegUTC, temZtEndUTC)
    maxAmpTemZ = np.max(templateZ[0].data)
    print("maxAmpTemZ=", maxAmpTemZ )  
    
    stR1 = stR0.copy()
    temRtBegNor = t0P
    temRtEndNor = t1P
    temRtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + t0P )
    temRtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + t1P )
    print("temRtBegUTC", temRtBegUTC)
    templateR=stR1.trim( temRtBegUTC, temRtEndUTC)
    maxAmpTemR = np.max(templateR[0].data)
    print("maxAmpTemR=", maxAmpTemR )
    
    stT1 = stT0.copy()
    temTtBegNor = t0S
    temTtEndNor = t1S
    temTtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + t0S )
    temTtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + t1S )
    templateT=stT1.trim( temTtBegUTC, temTtEndUTC)
    maxAmpTemT = np.max(templateT[0].data)
    print("maxAmpTemT=", maxAmpTemT )
        
    #%%-- plot direct-wave templates
    if verboseFlag == 1:       
        print('\n Check the selected direct phases: \n')
        pltDebug.figure(figsize=(5,2))
        t = np.arange( 0, len( templateZ[0] ), 1 )*DT+temZtBegNor+stRawZ[0].stats.sac.b
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateZ[0])
        pltDebug.title( 'P template (Z)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        
        pltDebug.figure( figsize=(5,2))
        t = np.arange( 0, len( templateR[0] ), 1 )*DT+temRtBegNor+stRawZ[0].stats.sac.b
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateR[0] )
        pltDebug.title( 'P template (R)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        
        pltDebug.figure( figsize=(5,2) )
        t = np.arange( 0, len( templateT[0] ), 1 )*DT+temTtBegNor+stRawZ[0].stats.sac.b
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateT[0] )
        pltDebug.title( 'S template (T)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        pltDebug.show()


    #%%##############################################################
    # Step 2: Match-filtering of all possible depth phases by using #
    #         1) phase shifting and 2) match-flitering              #
    #################################################################
    
    phaseShiftStart = -180
    phaseShiftEnd   = 180
    PhaseShittInc   = 10
    numPhase = int( (phaseShiftEnd-phaseShiftStart)/PhaseShittInc )
    print("Number of phase shift=", numPhase)
    scanTimeBegP = (int) ( temZtBegNor / DT)
    scanTimeEndP = len( stZ0[0] )
    scanTimeBegS = (int) ( temTtBegNor / DT)
    scanTimeEndS = len( stZ0[0] )
    print("P scanning from",scanTimeBegP*DT, "to", scanTimeEndP*DT, "sec")
    print("S scanning from",scanTimeBegS*DT, "to", scanTimeEndS*DT, "sec")

    #-- calculate cross-correlation coefficient (CC) on Z component
    count = 0
    temLengZ = len(templateZ[0])
    traLengZ = len(stZ0[0])
    corrLengZ= traLengZ - temLengZ + 1
    corrValZ = np.zeros((numPhase, corrLengZ))
    
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShittInc):
        #-- Phase shift using Hilbert transform
        st2 = hilbert(templateZ[0])
        st2 = np.real(np.abs(st2) * np.exp((np.angle(st2) +\
                        (phaseShift)/180.0 * np.pi) * 1j))      
        #-- cross-corelation
        corrValZ[count] = xcorrssl( scanTimeBegP, scanTimeEndP, st2, stZ0[0])
        count += 1
        
        #-- plotting for paper
        if phaseShift == 60:
            templateZ60 = st2
        if phaseShift == 120:
            templateZ120 = st2
        if phaseShift == 170:
            templateZ170 = st2

    #-- calculate cross-correlation coefficient (CC) on R component
    count = 0
    temLengR = len(templateR[0])
    traLengR = len(stR0[0])
    corrLengR= traLengR - temLengR + 1
    corrValR = np.zeros((numPhase, corrLengR))    
 
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShittInc):
        #-- Phase shift using Hilbert transform
        st2 = hilbert(templateR[0])
        st2 = np.real(np.abs(st2) * np.exp((np.angle(st2) +\
                        (phaseShift)/180.0 * np.pi) * 1j))          
        #-- cross-corelation
        corrValR[count] = xcorrssl( scanTimeBegP, scanTimeEndP, st2, stR0[0])
        count += 1
        
        #-- plotting for paper
        if phaseShift == 60:
            templateR60 = st2
        if phaseShift == 120:
            templateR120 = st2
        if phaseShift == 170:
            templateR170 = st2
    
    #-- calculate cross-correlation coefficient (CC) on T component        
    count = 0
    temLengT = len(templateT[0])
    traLengT = len(stT0[0])
    corrLengT= traLengT - temLengT + 1
    corrValT = np.zeros((numPhase, corrLengT))
   
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShittInc):
        #-- Phase shift using Hilbert transform
        st2 = hilbert(templateT[0])
        st2 = np.real(np.abs(st2) * np.exp((np.angle(st2) +\
                        (phaseShift)/180.0 * np.pi) * 1j))          
        #-- cross-corelation
        corrValT[count] = xcorrssl( scanTimeBegS, scanTimeEndS, st2, stT0[0])
        count += 1
        
        #-- plotting for paper
        if phaseShift == 60:
            templateT60 = st2
        if phaseShift == 120:
            templateT120 = st2
        if phaseShift == 170:
            templateT170 = st2
    

        
    #%%-- Get the maximum cross-correlation value of each template
    PickCorrR = np.amax( corrValR, axis=0 )
    PickCorrT = np.amax( corrValT, axis=0 )
    PickCorrZ = np.amax( corrValZ, axis=0 )
    
    #%%-- Get the time lag showing the peak value of cross-correlation  
    peaksR0, _ = find_peaks(PickCorrR, height=ccThreshold, distance=50)
    peaksT0, _ = find_peaks(PickCorrT, height=ccThreshold, distance=50)
    peaksZ0, _ = find_peaks(PickCorrZ, height=ccThreshold, distance=50)
    pickedPhaseTimeR0 = peaksR0 * DT
    pickedPhaseTimeT0 = peaksT0 * DT
    pickedPhaseTimeZ0 = peaksZ0 * DT
    

    peaksCurveR = np.zeros( len(PickCorrR) )
    for i in range( len(peaksR0) ):
        peaksCurveR[ peaksR0[i] ] = PickCorrR[ peaksR0[i] ]
    peaksCurveT = np.zeros( len(PickCorrT) )        
    for i in range( len(peaksT0) ):
        peaksCurveT[ peaksT0[i] ] = PickCorrT[ peaksT0[i] ]
    peaksCurveZ = np.zeros( len(PickCorrZ) )
    for i in range( len(peaksZ0) ):
        peaksCurveZ[ peaksZ0[i] ] = PickCorrZ[ peaksZ0[i] ]        
                         
    #%%-- Select phase with large amplitude using the distribution
    # of amplitude peak and drop
    stNorZ = stZ0[0].data / max( np.fabs(stZ0[0].data))
    stNorR = stR0[0].data / max( np.fabs(stR0[0].data))
    stNorT = stT0[0].data / max( np.fabs(stT0[0].data))
    maxAmpTemZ = np.max( np.fabs(templateZ[0].data) ) / max( np.fabs(stZ0[0].data))
    maxAmpTemR = np.max( np.fabs(templateR[0].data) ) / max( np.fabs(stR0[0].data))
    maxAmpTemT = np.max( np.fabs(templateT[0].data) ) / max( np.fabs(stT0[0].data))
    tGlobal = np.arange( 0, wantedTimeLengZ, DT)

    extremaMinIdxZ = signal.argrelextrema( np.array( stNorZ ), np.less)
    extremaMinIdxR = signal.argrelextrema( np.array( stNorR ), np.less)
    extremaMinIdxT = signal.argrelextrema( np.array( stNorT ), np.less)
    extremaMaxIdxZ = signal.argrelextrema( np.array( stNorZ ), np.greater)
    extremaMaxIdxR = signal.argrelextrema( np.array( stNorR ), np.greater)
    extremaMaxIdxT = signal.argrelextrema( np.array( stNorT ), np.greater)      
    extremaIdxZ = np.concatenate( (extremaMinIdxZ, extremaMaxIdxZ), axis=1 )
    extremaIdxR = np.concatenate( (extremaMinIdxR, extremaMaxIdxR), axis=1 )
    extremaIdxT = np.concatenate( (extremaMinIdxT, extremaMaxIdxT), axis=1 )
    extremaAmpTimeZ = extremaIdxZ * DT        
    extremaAmpTimeR = extremaIdxR * DT 
    extremaAmpTimeT = extremaIdxT * DT 
    
    histZ = sorted(  stNorZ[extremaIdxZ][0] )
    histR = sorted(  stNorR[extremaIdxR][0] )
    histT = sorted(  stNorT[extremaIdxT][0] )
    
    meanZ = np.mean(histZ)
    meanR = np.mean(histR)
    meanT = np.mean(histT)
    stdZ  = np.std(histZ)
    stdR  = np.std(histR)
    stdT  = np.std(histT)
    
    ratioStd1 = 1.0
    leftBoundry1Z  = meanZ - stdZ * ratioStd1
    leftBoundry1R  = meanR - stdR * ratioStd1
    leftBoundry1T  = meanT - stdT * ratioStd1
    rightBoundry1Z = meanZ + stdZ * ratioStd1
    rightBoundry1R = meanR + stdR * ratioStd1
    rightBoundry1T = meanT + stdT * ratioStd1
    
    fitHistZ = stats.norm.pdf( histZ, meanZ, stdZ )
    fitHistR = stats.norm.pdf( histR, meanR, stdR )
    fitHistT = stats.norm.pdf( histT, meanT, stdT )

    bigAmpZ0 = np.zeros( len(stNorZ) )
    bigAmpR0 = np.zeros( len(stNorR) )
    bigAmpT0 = np.zeros( len(stNorT) )
    
    #%%-- To show the deep-phase waveform corresponding to the peak/troughthat
    # meets the CC threshold, we keep the waveform within a time-window 
    # centering on the peak/trough amplitude. 
    for i in range (len(stNorZ) ):
        if stNorZ[i] <= leftBoundry1Z or stNorZ[i] >= rightBoundry1Z:
            for j in range( np.int(temLengZ/2) ):
                if (i-j) >=0 and (i+j) < len(stNorZ):
                    bigAmpZ0[ i-j ] = 1.
                   
    for i in range (len(stNorR) ):
        if stNorR[i] <= leftBoundry1R or stNorR[i] >= rightBoundry1R:
            for j in range( np.int(temLengR/2) ):
                if (i-j) >=0 and (i+j) < len(stNorR):
                    bigAmpR0[ i-j ] = 1.
                    
    for i in range (len(stNorT) ):
        if stNorT[i] <= leftBoundry1T or stNorT[i] >= rightBoundry1T:
            for j in range( np.int(temLengT/2) ):
                if (i-j) >=0 and (i+j) < len(stNorT):
                    bigAmpT0[ i-j ] = 1.
                             
        
    #%%-- Select CC for the phase with large amplitude
    PickCorrZ_span = np.zeros( len(PickCorrZ) )      
    for i in range( len(PickCorrZ) ):
        PickCorrZ_span[i] = bigAmpZ0[i] * peaksCurveZ[i]
    peaksZ, _ = find_peaks(PickCorrZ_span, height=ccThreshold, distance=1)
    pickedPhaseTimeZ = peaksZ * DT
    
    PickCorrR_span = np.zeros( len(PickCorrR) )
    for i in range( len(PickCorrR) ):
        PickCorrR_span[i] = bigAmpR0[i] * peaksCurveR[i]
    peaksR, _ = find_peaks(PickCorrR_span, height=ccThreshold, distance=1)
    pickedPhaseTimeR = peaksR * DT    
 
    PickCorrT_span = np.zeros( len(PickCorrT) )
    for i in range( len(PickCorrT) ):
        PickCorrT_span[i] = bigAmpT0[i] * peaksCurveT[i] 
    peaksT, _ = find_peaks(PickCorrT_span, height=ccThreshold, distance=1)
    pickedPhaseTimeT = peaksT * DT
           
    
    #%%-- Evaluate the quality of data by using a condition:
    # If only the direct phase has CC, then skip the current station.
    if (len(pickedPhaseTimeZ) < 2) or\
       (len(pickedPhaseTimeR) < 2) or\
       (len(pickedPhaseTimeT) < 2):
        print( "\n\n\n Only the direct phase has CC, skip station =",
               stRawZ[0].stats.sac.kstnm, 
               "EpiDis:", recDisInKm, "km\n\n\n" )
        continue

    
    #%%-- calculate arrival time differences between the selected phases and
    # the direct waves
    pickedPhaseTimeDiffR = pickedPhaseTimeR - pickedPhaseTimeR0[0]
    print('pickedPhaseTimeDiffR =', pickedPhaseTimeDiffR)
    pickedPhaseTimeDiffT = pickedPhaseTimeT - pickedPhaseTimeT0[0]
    print('pickedPhaseTimeDiffT =', pickedPhaseTimeDiffT)
    pickedPhaseTimeDiffZ = pickedPhaseTimeZ - pickedPhaseTimeZ0[0]
    print('pickedPhaseTimeDiffZ =', pickedPhaseTimeDiffZ)  
    
    #%%-- save the phase-shifting angle of the picked CC
    phaShiftAngR = []
    phaShiftAngIdxR = []
    phaShiftAngTimeR = []
    for i in range( len(peaksR) ):
        for j in range(numPhase):
            if PickCorrR[ peaksR[i] ] == corrValR[j][ peaksR[i] ]:
                phaShiftAngR.append( j*PhaseShittInc+phaseShiftStart )
                phaShiftAngIdxR.append( peaksR[i] )
                phaShiftAngTimeR.append( peaksR[i]*DT )

    phaShiftAngT = []
    phaShiftAngIdxT = []
    phaShiftAngTimeT = []
    for i in range( len(peaksT) ):
        for j in range(numPhase):
            if PickCorrT[ peaksT[i] ] == corrValT[j][ peaksT[i] ]:
                phaShiftAngT.append( j*PhaseShittInc+phaseShiftStart )
                phaShiftAngIdxT.append( peaksT[i] )
                phaShiftAngTimeT.append( peaksT[i]*DT )

    phaShiftAngZ = []
    phaShiftAngIdxZ = []
    phaShiftAngTimeZ = []
    for i in range( len(peaksZ) ):
        for j in range(numPhase):
            if PickCorrZ[ peaksZ[i] ] == corrValZ[j][ peaksZ[i] ]:
                phaShiftAngZ.append( j*PhaseShittInc+phaseShiftStart )
                phaShiftAngIdxZ.append( peaksZ[i] )
                phaShiftAngTimeZ.append( peaksZ[i]*DT )



    #%%#####################################################
    # Step 3: Preliminary determination of the focal depth #
    ########################################################
    #-- search for focal depth using TauP
    scanDepthMembers = []

    DepthCandidateArrR = [[] for i in range(100)]
    DepthCandidateArrDiffR = [[] for i in range(100)]
    DepthCandidatePhaDigNameR= [[] for i in range(100)]
    DepthCandidatePhaOrgNameR= [[] for i in range(100)]
    DepthCandidateArrT = [[] for i in range(100)]
    DepthCandidateArrDiffT = [[] for i in range(100)]
    DepthCandidatePhaDigNameT= [[] for i in range(100)]
    DepthCandidatePhaOrgNameT= [[] for i in range(100)]
    DepthCandidateArrZ = [[] for i in range(100)]
    DepthCandidateArrDiffZ = [[] for i in range(100)]
    DepthCandidatePhaDigNameZ= [[] for i in range(100)]
    DepthCandidatePhaOrgNameZ= [[] for i in range(100)]
 
    srcDepthScanBeg = scanDepthFrom
    srcDepthScanEnd = scanDepthTo
    srcDepthScanInc = 1
    indexCounter = 0
  
    for tmp_depth in np.arange( srcDepthScanBeg, srcDepthScanEnd, srcDepthScanInc):
        taupOriginalName = []
        arrivalsTimeDiffP = []
        arrivalsTimeDiffS = []
        srcDepth= tmp_depth  # km
        scanDepthMembers.append( tmp_depth )
        

        phaList = [ "s", "Sg" ]
        arrivals, rays = subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth )
        arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
        while (len(arrivals) < 1): #avoid showing that there are no direct s phase
            arrivals, rays = subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth )
            arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
            srcDepth += 0.1
            #print("No direct s phase, trying srcDepth=", srcDepth)
        onsetCalS = arrivals[0].time
    
        #-- first class phases, which is related to the free-surface and Moho
        phaList = [  "p", "Pg", "pPg", "sPg", "PvmP", "pPvmP", "sPvmP", "Sg", "sSg", "SvmS", "sSvmS" ]
        arrivals, rays = subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth )
#            ax = rays.plot_rays(plot_type="cartesian",legend=False,
#                    fig=plt.figure(figsize=(4,3), linewidth=0.01, dpi=200 ))
        arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )           
#        ax = rays.plot_rays(plot_type="cartesian",legend=False,
#                            fig=plt.figure(figsize=(1,1), linewidth=0.01, dpi=200 ))
  
        #%% calculate time differences of P-wave and S-wave
        for i in range(len(arrivals)):
            taupOriginalName.append( arrivals[i].name )
            arrivalsTimeDiffP.append( arrivals[i].time - arrivals[0].time) # P-wave
            arrivalsTimeDiffS.append( arrivals[i].time - onsetCalS) # S-wave
            #because Taup will give several same phase names, here use number to identify each phase
            arrivals[i].name = i
        
        #%%-- matching arrival time differences of observed data and that of
        # synthetic data calculated by using TauP pakadge.
        matchedTaupPhaseOrgNameR  = []
        matchedTaupPhaseDigNameR  = []
        matchedTaupTimeDiffR   = []
        matchedPickedTimeR = []
        matchedPickedTimeDiffR = []
        
        matchedTaupPhaseOrgNameT  = []
        matchedTaupPhaseDigNameT  = []
        matchedTaupTimeDiffT   = []
        matchedPickedTimeT = []
        matchedPickedTimeDiffT = []

        
        matchedTaupPhaseOrgNameZ  = []
        matchedTaupPhaseDigNameZ  = []
        matchedTaupTimeDiffZ   = []
        matchedPickedTimeZ = []
        matchedPickedTimeDiffZ = []

        #%%-- get the number of matched phases (Z component)
        tmpArrivalsTimeDiffP    = arrivalsTimeDiffP
        tmpPickedPhaseTimeDiffZ = list(pickedPhaseTimeDiffZ)
        tmpPickedPhaseTimeZ     = list( pickedPhaseTimeZ )
        loopFlag = 1
        while( loopFlag == 1 ):    
            for i in range(len(tmpArrivalsTimeDiffP)):
                for x in range(len(tmpPickedPhaseTimeDiffZ)):
                    if( tmpArrivalsTimeDiffP[i]>0.0 and tmpPickedPhaseTimeDiffZ[x] >0.0 ): # match phases except direct p
                        if math.fabs( tmpArrivalsTimeDiffP[i] - tmpPickedPhaseTimeDiffZ[x] ) <= arrTimeDiffTole:                
                            matchedPickedTimeDiffZ.append(tmpPickedPhaseTimeDiffZ[x])
                            matchedTaupTimeDiffZ.append(tmpArrivalsTimeDiffP[i])
                            matchedTaupPhaseDigNameZ.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameZ.append(taupOriginalName[i])
                            matchedPickedTimeZ.append(tmpPickedPhaseTimeZ[x])
                            tmpPickedPhaseTimeDiffZ.remove( tmpPickedPhaseTimeDiffZ[x] )
                            tmpPickedPhaseTimeZ.remove( tmpPickedPhaseTimeZ[x] )
                            break
            loopFlag = 0
            
        #%%-- get the number of matched phases (R component)    
        tmpArrivalsTimeDiffP    = arrivalsTimeDiffP
        tmpPickedPhaseTimeDiffR = list( pickedPhaseTimeDiffR )
        tmpPickedPhaseTimeR     = list( pickedPhaseTimeR )
        loopFlag = 1
        while( loopFlag == 1 ):    
            for i in range(len(tmpArrivalsTimeDiffP)):
                for x in range(len(tmpPickedPhaseTimeDiffR)):
                    if( tmpArrivalsTimeDiffP[i]>0.0 and tmpPickedPhaseTimeDiffR[x] >0.0 ): # match phases except direct p
                        if math.fabs( tmpArrivalsTimeDiffP[i] - tmpPickedPhaseTimeDiffR[x] ) <= arrTimeDiffTole:                
                            matchedPickedTimeDiffR.append(tmpPickedPhaseTimeDiffR[x])
                            matchedTaupTimeDiffR.append(tmpArrivalsTimeDiffP[i])
                            matchedTaupPhaseDigNameR.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameR.append(taupOriginalName[i])
                            matchedPickedTimeR.append(tmpPickedPhaseTimeR[x])
                            tmpPickedPhaseTimeDiffR.remove( tmpPickedPhaseTimeDiffR[x] )
                            tmpPickedPhaseTimeR.remove( tmpPickedPhaseTimeR[x] )
                            break
            loopFlag = 0
            
        #%%-- get the number of matched phases (T component)
        tmpArrivalsTimeDiffS    = arrivalsTimeDiffS
        tmpPickedPhaseTimeDiffT = list(pickedPhaseTimeDiffT)
        tmpPickedPhaseTimeT     = list( pickedPhaseTimeT )
        loopFlag = 1
        while( loopFlag == 1 ):    
            for i in range(len(tmpArrivalsTimeDiffS)):
                for x in range(len(tmpPickedPhaseTimeDiffT)):
                    if( tmpArrivalsTimeDiffS[i] > 0.0 and tmpPickedPhaseTimeDiffT[x] >0.0):# match phases except direct s
                        if math.fabs( tmpArrivalsTimeDiffS[i] - tmpPickedPhaseTimeDiffT[x] ) <= arrTimeDiffTole:                
                            matchedPickedTimeDiffT.append(tmpPickedPhaseTimeDiffT[x])
                            matchedTaupTimeDiffT.append(tmpArrivalsTimeDiffS[i])
                            matchedTaupPhaseDigNameT.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameT.append(taupOriginalName[i])
                            matchedPickedTimeT.append(tmpPickedPhaseTimeT[x])
                            tmpPickedPhaseTimeDiffT.remove( tmpPickedPhaseTimeDiffT[x] )
                            tmpPickedPhaseTimeT.remove( tmpPickedPhaseTimeT[x] )
                            break
            loopFlag = 0
            

            
            
                        

        #%%-- Calculate the total number of matched phases having a unique name
        # (at least one depth-phase matches)
        if( len(matchedTaupTimeDiffZ) >= 1 and
            len(matchedTaupTimeDiffR) >= 1 and
            len(matchedTaupTimeDiffT) >= 1 ):
            totNumCalPhaGlobal[ist][indexCounter] = len( arrivals ) * 3 # 3 components
            totNumMatPhaGlobal[ist][indexCounter] = len( set(matchedTaupPhaseDigNameZ) ) +\
                                                    len( set(matchedTaupPhaseDigNameR) ) +\
                                                    len( set(matchedTaupPhaseDigNameT) )    
            percNumMatPhaGlobal[ist][indexCounter] = totNumMatPhaGlobal[ist][indexCounter] / totNumCalPhaGlobal[ist][indexCounter]
       
        #-- Calculate the RMS of arrival time differences of preliminary solution 
        # (at least one depth-phase matches)
        if( len(matchedTaupTimeDiffZ) >= 1 ):
            avgArrTimeDiffResEachStationZ[ist][indexCounter] = np.sum( np.fabs( np.array(matchedPickedTimeDiffZ) - np.array(matchedTaupTimeDiffZ) ) )/len(matchedTaupTimeDiffZ)
        if( len(matchedTaupTimeDiffR) >= 1 ):
            avgArrTimeDiffResEachStationR[ist][indexCounter] = np.sum( np.fabs( np.array(matchedPickedTimeDiffR) - np.array(matchedTaupTimeDiffR) ) )/len(matchedTaupTimeDiffR)
        if( len(matchedTaupTimeDiffT) >= 1 ):
            avgArrTimeDiffResEachStationT[ist][indexCounter] = np.sum( np.fabs( np.array(matchedPickedTimeDiffT) - np.array(matchedTaupTimeDiffT) ) )/len(matchedTaupTimeDiffT)
        
        if( len(matchedTaupTimeDiffZ) >= 1 and
            len(matchedTaupTimeDiffR) >= 1 and
            len(matchedTaupTimeDiffT) >= 1 ):
            avgArrTimeDiffResEachStationSum[ist][indexCounter] = ( avgArrTimeDiffResEachStationZ[ist][indexCounter] +\
                                                                   avgArrTimeDiffResEachStationR[ist][indexCounter] +\
                                                                   avgArrTimeDiffResEachStationT[ist][indexCounter] )/3.
                
                
                
        # save info of the matched phases of R, T, and Z components
        waveformGlobalZ[ist].append( stZ0[0].data )
        waveformGlobalR[ist].append( stR0[0].data )
        waveformGlobalT[ist].append( stT0[0].data )
        normWaveformGlobalZ[ist].append( stNorZ )
        normWaveformGlobalR[ist].append( stNorR )
        normWaveformGlobalT[ist].append( stNorT )
        template0GlobalZ[ist].append( templateZ[0].data )
        template0GlobalR[ist].append( templateR[0].data )
        template0GlobalT[ist].append( templateT[0].data )
        template60GlobalZ[ist].append( templateZ60 )
        template60GlobalR[ist].append( templateR60 )
        template60GlobalT[ist].append( templateT60 )
        template120GlobalZ[ist].append( templateZ120 )
        template120GlobalR[ist].append( templateR120 )
        template120GlobalT[ist].append( templateT120 )
        template170GlobalZ[ist].append( templateZ170 )
        template170GlobalR[ist].append( templateR170 )
        template170GlobalT[ist].append( templateT170 )
        histogramGlobalZ[ist].append( histZ )
        histogramGlobalR[ist].append( histR )
        histogramGlobalT[ist].append( histT )
        bigAmpGlobalZ[ist].append( bigAmpZ0 )
        bigAmpGlobalR[ist].append( bigAmpR0 )
        bigAmpGlobalT[ist].append( bigAmpT0 )
        bigAmpTimeGlobalZ[ist].append( pickedPhaseTimeZ )
        bigAmpTimeGlobalR[ist].append( pickedPhaseTimeR )
        bigAmpTimeGlobalT[ist].append( pickedPhaseTimeT )
        peaksCurveGlobalZ[ist].append( peaksCurveZ )
        peaksCurveGlobalR[ist].append( peaksCurveR )
        peaksCurveGlobalT[ist].append( peaksCurveT )
        finalCcGlobalZ[ist].append( PickCorrZ_span )
        finalCcGlobalR[ist].append( PickCorrR_span )
        finalCcGlobalT[ist].append( PickCorrT_span )
        finalpeaksPtsGlobalZ[ist].append( peaksZ )
        finalpeaksPtsGlobalR[ist].append( peaksR )
        finalpeaksPtsGlobalT[ist].append( peaksT )
        phaShiftAngGlobalZ[ist].append( phaShiftAngZ )
        phaShiftAngGlobalR[ist].append( phaShiftAngR )
        phaShiftAngGlobalT[ist].append( phaShiftAngT )
        phaShiftAngTimeGlobalZ[ist].append( phaShiftAngTimeZ )
        phaShiftAngTimeGlobalR[ist].append( phaShiftAngTimeR )
        phaShiftAngTimeGlobalT[ist].append( phaShiftAngTimeT )
        corrLengGlobalZ[ist].append( corrLengZ )
        corrLengGlobalR[ist].append( corrLengR )
        corrLengGlobalT[ist].append( corrLengT )
        DTGlobalZ[ist].append( DT )
        DTGlobalR[ist].append( DT )
        DTGlobalT[ist].append( DT )
        temBegNorGlobalZ[ist].append( temZtBegNor )
        temBegNorGlobalR[ist].append( temRtBegNor ) 
        temBegNorGlobalT[ist].append( temTtBegNor ) 
        leftBoundry1GlobalZ[ist].append( leftBoundry1Z )
        leftBoundry1GlobalR[ist].append( leftBoundry1R )
        leftBoundry1GlobalT[ist].append( leftBoundry1T )
        rightBoundry1GlobalZ[ist].append( rightBoundry1Z )
        rightBoundry1GlobalR[ist].append( rightBoundry1R )
        rightBoundry1GlobalT[ist].append( rightBoundry1T )
        st_begin_timeNorGlobalZ[ist].append( st_begin_timeNor )
        st_begin_timeNorGlobalR[ist].append( st_begin_timeNor )
        st_begin_timeNorGlobalT[ist].append( st_begin_timeNor )
        wantedTimeLengGlobalZ[ist].append( wantedTimeLengZ )
        wantedTimeLengGlobalR[ist].append( wantedTimeLengZ )
        wantedTimeLengGlobalT[ist].append( wantedTimeLengZ )
        
        depthCandidateArrGlobalR[ist][indexCounter].append( matchedPickedTimeR )
        depthCandidateArrGlobalT[ist][indexCounter].append( matchedPickedTimeT ) 
        depthCandidateArrGlobalZ[ist][indexCounter].append( matchedPickedTimeZ )
        depthCandidateArrDiffGlobalR[ist][indexCounter].append( matchedPickedTimeDiffR )
        depthCandidateArrDiffGlobalT[ist][indexCounter].append( matchedPickedTimeDiffT )
        depthCandidateArrDiffGlobalZ[ist][indexCounter].append( matchedPickedTimeDiffZ )
        depthCandidatePhaDigNameGlobalR[ist][indexCounter].append( matchedTaupPhaseDigNameR )                                                
        depthCandidatePhaDigNameGlobalT[ist][indexCounter].append( matchedTaupPhaseDigNameT )
        depthCandidatePhaDigNameGlobalZ[ist][indexCounter].append( matchedTaupPhaseDigNameZ )         
        depthCandidatePhaOrgNameGlobalR[ist][indexCounter].append( matchedTaupPhaseOrgNameR )                                                
        depthCandidatePhaOrgNameGlobalT[ist][indexCounter].append( matchedTaupPhaseOrgNameT )
        depthCandidatePhaOrgNameGlobalZ[ist][indexCounter].append( matchedTaupPhaseOrgNameZ )
        
        indexCounter += 1

    #%%-- find out the depth with maximum  number of matched phases of ist-th station
    idxMaxNumPha = np.argmax( totNumMatPhaGlobal[ist] )
    depthMaxNumPhaEachStation = idxMaxNumPha*srcDepthScanInc+srcDepthScanBeg
    idxMaxNumPhaEachStation[ist] = idxMaxNumPha
        
    
#%%######################################################
# Step 3: Preliminary solution                          #
#########################################################
#-- Calculate the total number of matched phases
sumGlobal = totNumMatPhaGlobal.sum(axis=0)

#-- sum the valid arrival time difference
for idepth in range( numScanDepth ):
    tmpCount = 0
    tmpSum   = 0.
    for ist in range( numSt ):        
        if( avgArrTimeDiffResEachStationSum[ist][idepth] < 9999 ):
            tmpSum   += avgArrTimeDiffResEachStationSum[ist][idepth]
            tmpCount += 1
    if tmpCount > 0:
        sumAvgArrTimeDiffResGlobal[idepth] = tmpSum / tmpCount
   
   
#-- the depth exceeds the set threshold
thresholdMaxNumb = np.max(sumGlobal)*0.9

prelimCandidatesGlobal = []
for i in range( len(scanDepthMembers) ):
    if sumGlobal[i] >= thresholdMaxNumb:
        candidates = sumAvgArrTimeDiffResGlobal[i], scanDepthMembers[i]
        prelimCandidatesGlobal.append( candidates )

tmpDepthRange1 = []
tmpAvgArrTimeDiffResEachStationSum1 = []
for i in range( len(scanDepthMembers) ):
    if sumGlobal[i] >= thresholdMaxNumb:
        tmpAvgArrTimeDiffResEachStationSum1.append( sumAvgArrTimeDiffResGlobal[i] )
        tmpDepthRange1.append(i)
    else:
        tmpAvgArrTimeDiffResEachStationSum1.append( 9999 )
prelimSolution = np.argmin( tmpAvgArrTimeDiffResEachStationSum1 ) * srcDepthScanInc + srcDepthScanBeg
print('prelimSolution=', prelimSolution, 'km' )





#%% -- plotting for debugging
# set figure layout
fig = plt.figure( constrained_layout=True, figsize=(5,2.5))
fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0.18)
gs0 = fig.add_gridspec(1, 1 )
gs00 = gs0[0].subgridspec(1,1)
ax1 = fig.add_subplot(gs00[0, 0])
#-- plot data
t = scanDepthMembers
ax1.plot(t, sumGlobal, color="black", linewidth=2., alpha=1)
ax11 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax11.scatter(t, sumAvgArrTimeDiffResGlobal, s=25, marker='s',
             facecolors='none', edgecolor='blue', zorder=100)
ax11.scatter(prelimSolution, sumAvgArrTimeDiffResGlobal[np.int(prelimSolution)-1],
             s=30, marker='s', facecolors='blue', edgecolor='blue', zorder=100)
ax11.set_ylim( 0, 1 )

#-- plot the number of matched of each station
for ist in range(numSt):
    ax1.plot(t, totNumMatPhaGlobal[ist],  linewidth=1, color="grey")

ax1.set_title('Step 3', fontsize=12)
# set labels
ax1.set_ylabel('Number of matches', fontsize=12)
ax1.set_xlabel('Depth (km)', fontsize=12)
ax11.set_ylabel('Sum of differential\narrival time residuals (s)', color='blue', fontsize=12)
#set grid and threshold lines
ax1.grid(True, linestyle='--', linewidth=0.25)
ax1.axvline( prelimSolution, linewidth=1, color='black', linestyle='--')
ax1.axhline( thresholdMaxNumb,linewidth=1, color='black', linestyle='--')   
#-- set axis
ax1.margins(x=0)
ax1.set_xticks( np.arange(scanDepthFrom,scanDepthTo, step=2) )
ax11.tick_params(axis='y', colors='blue')
   
#save
fig.tight_layout()
fig.savefig( "{0}/step3_locSrc{1}km.png".format( 
                    outfilePath,
                    format( prelimSolution, ".1f"),
                    dpi=360 ) )
plt.show()

#-- the end of debugging

    
    
#%%######################################################
# Step 4: Final solution based on travel time residuals #
#########################################################
wellBehavedStationId = []
wellBehavedStationIdx = []
wellBehavedStationDepth = []

for ist in range( numSt ):
    # avoid out of scanning depth range
    if (prelimSolution <=  scanDepthFrom +1 or
        prelimSolution >=  scanDepthTo -1 ): 
        break

  
    tmpData = avgArrTimeDiffResEachStationSum[ist][ np.min(tmpDepthRange1):np.max(tmpDepthRange1)+1 ]
    tmpMinValIdx = np.argmin( tmpData ) + np.min(tmpDepthRange1)
    prelimSolutionSingleStation = tmpMinValIdx * srcDepthScanInc + srcDepthScanBeg
    
    if ( np.abs( prelimSolution - prelimSolutionSingleStation ) <= 1 ):
        if ( len(depthCandidatePhaDigNameGlobalZ[ist][tmpMinValIdx]) > 0 and
             len(depthCandidatePhaDigNameGlobalR[ist][tmpMinValIdx]) > 0 and 
             len(depthCandidatePhaDigNameGlobalT[ist][tmpMinValIdx]) > 0 ):
            wellBehavedStationId.append( ist )
            wellBehavedStationIdx.append( tmpMinValIdx )
            wellBehavedStationDepth.append( prelimSolutionSingleStation )
            
 
print("wellBehavedStationId    =", wellBehavedStationId)
print("wellBehavedStationIdx   =", wellBehavedStationIdx)
print("wellBehavedStationDepth =", wellBehavedStationDepth)


#%%
if len( wellBehavedStationId ) < 1:
    print("\n\n\n")
    print("No good station for Step 4, the solution of Step 3 is the final focal depth!")
    print("\n\n\n")
    #-- write info of good stations
    with open( '{0}'.format( outPath ),  mode='a', newline='' ) as resultsFile:
        writer = csv.writer( resultsFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( format( prelimSolution, ".2f" ) ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ) ])
    
else:   
    numWellBehavedStation = len(wellBehavedStationId)
    rmsWellBehavedStation = []
    depthWellBehavedStation = []
    depthRangeWellBehavedStation= []

    for isw in range( numWellBehavedStation ):
        print( "Well-behaved station = ", isw+1, '/', numWellBehavedStation)
        ist = wellBehavedStationId[isw]
        idx = wellBehavedStationIdx[isw]
        depthCandidate = wellBehavedStationDepth[isw]
        rmsR = []
        rmsT = []
        rmsZ = []
        rmsRTZ = [] 
        exactDepthR = []
        exactDepthT = []
        exactDepthZ = []
        exactDepthRTZ = []
        exactDepthRange = []
        
        for srcDepth in np.arange( depthCandidate-1, depthCandidate+1, 0.1):
            taupOriginalName = []
            arrivalsTimeDiffP = []
            arrivalsTimeDiffS = []
            exactDepthRange.append( srcDepth )
            print( "Scanning depth candidate in Step 4 = {0}".format( format( srcDepth, '.1f') ) )
            
            #-- direc s phase
            phaList = [ "s", "Sg" ]
            arrivals, rays = subArrivalTimeForward( velModel, srcDepth, epiDisEachStation[ist], phaList, recDepth )
            arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
            while (len(arrivals) < 1): #avoid showing that there are no direct s phase
                arrivals, rays = subArrivalTimeForward( velModel, srcDepth, epiDisEachStation[ist], phaList, recDepth )
                arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
                srcDepth += 0.1
                #print("No direct s phase, trying srcDepth=", srcDepth)
            onsetCalS = arrivals[0].time
        
            #-- all depth-phase candidates
            phaList = [  "p", "Pg", "pPg", "sPg", "PvmP", "pPvmP", "sPvmP",  "Sg", "sSg", "SvmS", "sSvmS" ]
            arrivals, rays = subArrivalTimeForward( velModel, srcDepth, epiDisEachStation[ist], phaList, recDepth )
            arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )           

            #-- calculate time differences of P-wave and S-wave
            for i in range(len(arrivals)):
                taupOriginalName.append( arrivals[i].name )
                arrivalsTimeDiffP.append( arrivals[i].time - arrivals[0].time) # P-wave
                arrivalsTimeDiffS.append( arrivals[i].time - onsetCalS) # S-wave
                #because Taup will give several same phase names, here use number to identify each phase
                arrivals[i].name = i
            
               
            #-- calculate rms
            #-- Z
            tmp = 0.0
            count = 0
            for i in range( len( depthCandidatePhaDigNameGlobalZ[ist][idx][0] ) ):
                for j in range(len(arrivals)):
                    if ( depthCandidatePhaOrgNameGlobalZ[ist][idx][0][i] == taupOriginalName[j] ):
                            tmp += ( depthCandidateArrDiffGlobalZ[ist][idx][0][i] - arrivalsTimeDiffP[j])**2
                            count += 1
            if count > 0:
                tmp = math.sqrt( tmp/count )
            else:
                tmp =1.0
            rmsZ.append( tmp )
            exactDepthZ.append( srcDepth )
            #print( " rmsZ = {0}".format( format(  rmsZ[-1], '.3f') ) )
            
            #-- R
            tmp = 0.0
            count = 0
            for i in range( len( depthCandidatePhaDigNameGlobalR[ist][idx][0] ) ):
                for j in range(len(arrivals)):
                    if (depthCandidatePhaOrgNameGlobalR[ist][idx][0][i] == taupOriginalName[j] ):
                        tmp += ( depthCandidateArrDiffGlobalR[ist][idx][0][i] - arrivalsTimeDiffP[j])**2
                        count += 1
            if count > 0:
                tmp = math.sqrt( tmp/count )
            else:
                tmp =1.0
            rmsR.append( tmp )
            exactDepthR.append( srcDepth )
            #print( " rmsR = {0}".format( format(  rmsR[-1], '.3f') ) )
    
            #-- T
            tmp = 0.0
            count = 0
            for i in range( len( depthCandidatePhaDigNameGlobalT[ist][idx][0] ) ):
                for j in range(len(arrivals)):
                    if ( depthCandidatePhaOrgNameGlobalT[ist][idx][0][i] == taupOriginalName[j] ):
                        tmp += ( depthCandidateArrDiffGlobalT[ist][idx][0][i] - arrivalsTimeDiffS[j])**2
                        count += 1
            if count > 0:
                tmp = math.sqrt( tmp/count )
            else:
                tmp =1.0
            rmsT.append( tmp )
            exactDepthT.append( srcDepth )
            #print( " rmsT = {0}".format( format(  rmsT[-1], '.3f') ) )
            
            
            sumRmsRTZ = (rmsR[-1] + rmsT[-1] + rmsZ[-1]) / 3.0
            #sumRmsRTZ = ( rmsZ[-1] )
            rmsRTZ.append(sumRmsRTZ)
            exactDepthRTZ.append( srcDepth )
            
        
        #-- find out the best depth solution
        depthSolutionRTZ = exactDepthRTZ[ np.argmin(rmsRTZ) ]

        #--save rms and focal solution of each winner
        rmsWellBehavedStation.append( rmsRTZ )
        depthWellBehavedStation.append( depthSolutionRTZ )
        depthRangeWellBehavedStation.append( exactDepthRange )

        #-- write info of good stations
        with open( '{0}'.format( outPath ),  mode='a', newline='' ) as resultsFile:
            writer = csv.writer( resultsFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['{0}'.format( nameEachStation[ist][0] ),
                             '{0}'.format( azimuthEachStation[ist] ),
                             '{0}'.format( format( epiDisEachStation[ist], '.2f' ) ),
                             '{0}'.format( list( totNumMatPhaGlobal[ist] )),
                             '{0}'.format( format( depthSolutionRTZ, ".2f" ) ),
                             '{0}'.format( exactDepthRange ),
                             '{0}'.format( rmsRTZ ),
                             '{0}'.format( format( np.min(rmsRTZ), ".2f")) ])

    
    
        #%% -- plot for debugging
        if plotSteps1n2Flag == 1:
            #%%####################################################################
            # plot wavefrom, templates, cc, depth-phase matches, phase-shifted angles 
            #######################################################################
            ##### Vertical component
            #-- prepare data for plotting
            DT       = DTGlobalZ[ist][0]
            tempZ0   = template0GlobalZ[ist][0] / max( np.fabs( template0GlobalZ[ist][0] )) # phase-shift = 0 deg
            tempZ60  = template60GlobalZ[ist][0] / max( np.fabs(template60GlobalZ[ist][0]))# phase-shift = 60 deg
            tempZ120 = template120GlobalZ[ist][0] / max( np.fabs(template120GlobalZ[ist][0]))# phase-shift = 120 deg
            tempZ170 = template170GlobalZ[ist][0] / max( np.fabs(template170GlobalZ[ist][0]))# phase-shift = 170 deg
            dataZ    = waveformGlobalZ[ist][0] / max( np.fabs( waveformGlobalZ[ist][0] ))
            tempZ00  = template0GlobalZ[ist][0] / max( np.fabs( waveformGlobalZ[ist][0] ))
            bigAmpZ  = bigAmpGlobalZ[ist][0] * normWaveformGlobalZ[ist][0]
            stNorZ   = normWaveformGlobalZ[ist][0]
            orgCcZ   = peaksCurveGlobalZ[ist][0]
            finalCcZ = finalCcGlobalZ[ist][0]
            peaksPts = finalpeaksPtsGlobalZ[ist][0]
            corrLengZ= corrLengGlobalZ[ist][0]
            bigAmpTime       = bigAmpTimeGlobalZ[ist][0]
            phaShiftAngZ     = phaShiftAngGlobalZ[ist][0]
            phaShiftAngTimeZ = phaShiftAngTimeGlobalZ[ist][0] 
            temZtBegNor      = temBegNorGlobalZ[ist][0]
            leftBoundry1Z    = leftBoundry1GlobalZ[ist][0]
            rightBoundry1Z   = rightBoundry1GlobalZ[ist][0]
            wantedTimeLengZ  = wantedTimeLengGlobalZ[ist][0]
            finalArrZ        = depthCandidateArrGlobalZ[ist][idx][0]
            finalPhaOrgNameZ = depthCandidatePhaOrgNameGlobalZ[ist][idx][0]
            
            # set figure layout
            fig = plt.figure( constrained_layout=True, figsize=(8,4))
            fig.subplots_adjust(hspace=0.5)
            fig.subplots_adjust(wspace=0.05)
            gs0 = fig.add_gridspec(1, 2, width_ratios=[8,1] )
            gs00 = gs0[1].subgridspec(6,1)
            gs01 = gs0[0].subgridspec(10,1)
            
            
            ax0 = fig.add_subplot(gs00[0:2, 0])
            ax1 = fig.add_subplot(gs00[2, 0])
            ax2 = fig.add_subplot(gs00[3, 0])
            ax3 = fig.add_subplot(gs00[4, 0])
            ax4 = fig.add_subplot(gs00[5, 0])
            ax5 = fig.add_subplot(gs01[0:3, 0:])
            ax6 = fig.add_subplot(gs01[3:6, 0:])
            ax7 = fig.add_subplot(gs01[6, 0:])
            ax8 = fig.add_subplot(gs01[7:10, 0:])
            
            # plot data
            t1Z = np.arange( 0, len(tempZ0), 1)*DT
            t2Z = np.arange( 0, len(dataZ),  1)*DT
            tZcc= np.arange( 0, corrLengZ,   1)*DT
            
            #%%
            ax0.hist( histZ, normed=False, bins=11, orientation='horizontal')
            ax1.plot( t1Z, tempZ0, color='orange' )
            ax2.plot( t1Z, tempZ60 )
            ax3.plot( t1Z, tempZ120 )
            ax4.plot( t1Z, tempZ170 )
            ax5.plot( t2Z, dataZ )
            ax6.plot( t2Z, stNorZ, color='lightgray' )
            ax6.plot( t2Z, bigAmpZ )
            ax7.plot( tZcc, orgCcZ, color='lightgray')
            ax7.plot( tZcc, finalCcZ )
            
            # add templates
            tTempZ = np.arange( 0, len(tempZ0), 1)* DT+temZtBegNor
            ax5.plot( tTempZ, tempZ00, color='orange' )
            # add texts
            ax0.text( 2,   -0.4, r'$\mu-\sigma$', fontsize=10, rotation=0 )
            ax0.text( 2,    0.3, r'$\mu+\sigma$', fontsize=10, rotation=0 )
            ax1.text( 0.01, 0.45, '0 °', fontsize=10, color='black')
            ax2.text( 0.01, 0.45, '60 °', fontsize=10, color='black')
            ax3.text( 0.01, 0.45, '120 °', fontsize=10, color='black')
            ax4.text( 0.01, 0.45, '170 °', fontsize=10, color='black')
            
            # add matched phase and arrival time
            # find the phases sharing same arrival
            uniFinalArrZ = list(set(finalArrZ))
            for i in range( len( uniFinalArrZ ) ):
                phaNameForPlot = []
                count = 1
                for j in range( len( finalPhaOrgNameZ ) ):
                    if finalArrZ[j] == uniFinalArrZ[i]:
                        if (len(phaNameForPlot)) > 0:
                            phaNameForPlot.append( "{0}{1}".format( finalPhaOrgNameZ[j], count ) )
                            if finalPhaOrgNameZ[j] == "s":
                                phaNameForPlot[-1] = "s"
                            count += 1
                        else:
                            phaNameForPlot.append( "{0}".format( finalPhaOrgNameZ[j] ) )
                    
                nameAmpOffset = -0.7
                ax6.axvline( uniFinalArrZ[i], ymin=0, ymax=0.5, linewidth=1, color='black', linestyle='--')
                for k in range( len( phaNameForPlot ) ):
                    if k>=0 and k < len( phaNameForPlot ) -1:
                        ax6.text( uniFinalArrZ[i]-0.3, nameAmpOffset+0.5*k, "{0} + ".format( phaNameForPlot[k] ),
                                     fontsize=11, color='black', rotation=90)
                    else:
                        ax6.text( uniFinalArrZ[i]-0.3, nameAmpOffset+0.5*k, "{0}".format( phaNameForPlot[k] ),
                                     fontsize=11, color='black', rotation=90)
                    
            ax7.plot( bigAmpTime, finalCcZ[ peaksPts ], "o",
                     color='black', markersize=4, zorder=101)
            
            # plot phase-shifting angles
            ax8.plot( tZcc, finalCcZ, alpha=0 ) # just use its time axis
            ax8.scatter( phaShiftAngTimeZ, phaShiftAngZ,
                        s=20, color='black', zorder=101 )
            
            #plot CC and phase-shifting angle of matched phases 
            for i in range( len( uniFinalArrZ ) ):
                for j in range( len( phaShiftAngZ ) ):
                    if uniFinalArrZ[i] == phaShiftAngTimeZ[j]:
                            ax8.scatter( uniFinalArrZ[i], phaShiftAngZ[j],
                                        s=20, color='red', zorder=101 )
                for j in range( len( peaksPts ) ):
                    if uniFinalArrZ[i] == bigAmpTime[j]:
                            ax7.plot( uniFinalArrZ[i], finalCcZ[ peaksPts[j] ], "o",
                                     color='red',markersize=4, zorder=101)
            #set grid
            ax8.grid(True, linestyle='--', linewidth=0.5)
            
            # set title
            ax1.tick_params(axis='both', which='major', labelsize=10)
            #
            ax0.set_xscale('log')
            
            #set lim
            ax0.set_xlim(1e0, 1e3)
            ax0.set_ylim(-1.1, 1.1)
            ax1.set_ylim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            ax3.set_ylim(-1.2, 1.2)
            ax4.set_ylim(-1.2, 1.2)
            ax5.set_ylim(-1.2, 1.2)
            ax6.set_ylim(-1.3, 1.1)
            ax7.set_ylim(0, 1.2)
            ax8.set_ylim(-200, 200)
            ax8.set_yticks(np.arange(-180, 190, step=60))
            
            # set xticks
            ax0.xaxis.set_ticks_position('top')
            ax5.xaxis.set_ticks_position('top')
            ax6.xaxis.set_ticks_position('top')
            ax5.yaxis.set_ticks_position('left')
            ax6.yaxis.set_ticks_position('left')
            ax7.yaxis.set_ticks_position('left')
            ax8.yaxis.set_ticks_position('left')
            
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xticks([])
            ax4.set_xticks([])
            ax6.set_xticklabels([])
            ax7.set_xticks([])
            ax8.set_xticks([])
            
            ax0.set_yticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax3.set_yticks([])
            ax4.set_yticks([])        
            
            #remove axis margins
            ax1.margins(x=0)
            ax2.margins(x=0)
            ax3.margins(x=0)
            ax4.margins(x=0)
            ax5.margins(x=0)
            ax6.margins(x=0)
            ax7.margins(x=0)
            ax8.margins(x=0)
            
            # remove some spines
            ax6.spines['bottom'].set_visible(False)
            ax7.spines['top'].set_visible(False)
            
            # set labels
            ax0.xaxis.set_label_position('top')
            ax1.xaxis.set_label_position('top')
            ax5.xaxis.set_label_position('top')
            ax5.yaxis.set_label_position('left')
            ax6.yaxis.set_label_position('left')
            ax7.yaxis.set_label_position('left')
            ax8.yaxis.set_label_position('left')
            
            ax0.set_xlabel('Number', fontsize=12, labelpad=6)
            ax5.set_xlabel('Time (s)', fontsize=12, labelpad=8)
            ax5.set_ylabel('Amp.', fontsize=12)
            ax6.set_ylabel('Amp.', fontsize=12)
            ax7.set_ylabel('CC', fontsize=12, labelpad=15)
            ax8.set_ylabel('Shifted (°)', fontsize=12)
            
            # set zero lines
            ax7.axhline(ccThreshold, linewidth=0.8, linestyle='--', color='gray')
            
            # set span
            ax0.axhspan(leftBoundry1Z, rightBoundry1Z, facecolor='0.5', alpha=0.2, color='black')
            ax5.axhspan(leftBoundry1Z, rightBoundry1Z, facecolor='0.5', alpha=0.1, color='black')
                 
            # plot figure number
            ax5.set_title( "a)", x=-0.1, fontsize=16, color='black', loc='left' )
            ax5.text( 0.45, 0.7, "Z", fontsize=12, color='black' )
            
            #show
            plt.tight_layout()
            plt.savefig( "{0}/{1}_First2Steps_Z.png".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
            #plt.savefig( "{0}/{1}_First2Steps_Z.svg".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
            plt.show    
    
            #%%####################################################################
            # plot wavefrom, templates, cc, depth-phase matches, phase-shifted angles 
            #######################################################################
            ##### Radial component
            #-- prepare data for plotting
            DT       = DTGlobalR[ist][0]
            tempR0   = template0GlobalR[ist][0] / max( np.fabs( template0GlobalR[ist][0] )) # phase-shift = 0 deg
            tempR60  = template60GlobalR[ist][0] / max( np.fabs(template60GlobalR[ist][0]))# phase-shift = 60 deg
            tempR120 = template120GlobalR[ist][0] / max( np.fabs(template120GlobalR[ist][0]))# phase-shift = 120 deg
            tempR170 = template170GlobalR[ist][0] / max( np.fabs(template170GlobalR[ist][0]))# phase-shift = 180 deg
            dataR    = waveformGlobalR[ist][0] / max( np.fabs( waveformGlobalR[ist][0] ))
            tempR00  = template0GlobalR[ist][0] / max( np.fabs( waveformGlobalR[ist][0] ))
            bigAmpR  = bigAmpGlobalR[ist][0] * normWaveformGlobalR[ist][0]
            stNorR   = normWaveformGlobalR[ist][0]
            orgCcR   = peaksCurveGlobalR[ist][0]
            finalCcR = finalCcGlobalR[ist][0]
            peaksPts = finalpeaksPtsGlobalR[ist][0]
            corrLengR= corrLengGlobalR[ist][0]
            bigAmpTime       = bigAmpTimeGlobalR[ist][0]
            phaShiftAngR     = phaShiftAngGlobalR[ist][0]
            phaShiftAngTimeR = phaShiftAngTimeGlobalR[ist][0] 
            temRtBegNor      = temBegNorGlobalR[ist][0]
            leftBoundry1R    = leftBoundry1GlobalR[ist][0]
            rightBoundry1R   = rightBoundry1GlobalR[ist][0]
            wantedTimeLengR  = wantedTimeLengGlobalR[ist][0]
            finalArrR        = depthCandidateArrGlobalR[ist][idx][0]
            finalPhaOrgNameR = depthCandidatePhaOrgNameGlobalR[ist][idx][0]
            
            # set figure layout
            fig = plt.figure( constrained_layout=True, figsize=(8,4))
            fig.subplots_adjust(hspace=0.5)
            fig.subplots_adjust(wspace=0.05)
            gs0 = fig.add_gridspec(1, 2, width_ratios=[8,1] )
            gs00 = gs0[1].subgridspec(6,1)
            gs01 = gs0[0].subgridspec(10,1)
            
            
            ax0 = fig.add_subplot(gs00[0:2, 0])
            ax1 = fig.add_subplot(gs00[2, 0])
            ax2 = fig.add_subplot(gs00[3, 0])
            ax3 = fig.add_subplot(gs00[4, 0])
            ax4 = fig.add_subplot(gs00[5, 0])
            ax5 = fig.add_subplot(gs01[0:3, 0:])
            ax6 = fig.add_subplot(gs01[3:6, 0:])
            ax7 = fig.add_subplot(gs01[6, 0:])
            ax8 = fig.add_subplot(gs01[7:10, 0:])
            
            # plot data
            t1R = np.arange( 0, len(tempR0), 1)*DT
            t2R = np.arange( 0, len(dataR),  1)*DT
            tRcc= np.arange( 0, corrLengR,   1)*DT
            
            ax0.hist( histR, normed=False, bins=11, orientation='horizontal')
            ax1.plot( t1R, tempR0, color='orange' )
            ax2.plot( t1R, tempR60 )
            ax3.plot( t1R, tempR120 )
            ax4.plot( t1R, tempR170 )
            ax5.plot( t2R, dataR )
            ax6.plot( t2R, stNorR, color='lightgray' )
            ax6.plot( t2R, bigAmpR )
            ax7.plot( tRcc, orgCcR, color='lightgray')
            ax7.plot( tRcc, finalCcR )
            
            # add templates
            tTempR = np.arange( 0, len(tempR0), 1)* DT+temRtBegNor
            ax5.plot( tTempR, tempR00, color='orange' )
            # add texts
            ax0.text( 2,   -0.4, r'$\mu-\sigma$', fontsize=10, rotation=0 )
            ax0.text( 2,    0.3, r'$\mu+\sigma$', fontsize=10, rotation=0 )
            ax1.text( 0.01, 0.45, '0 °', fontsize=10, color='black')
            ax2.text( 0.01, 0.45, '60 °', fontsize=10, color='black')
            ax3.text( 0.01, 0.45, '120 °', fontsize=10, color='black')
            ax4.text( 0.01, 0.45, '170 °', fontsize=10, color='black')
            
            # add matched phase and arrival time
            # find the phases sharing same arrival
            uniFinalArrR = list(set(finalArrR))
            for i in range( len( uniFinalArrR ) ):
                phaNameForPlot = []
                count = 1
                for j in range( len( finalPhaOrgNameR ) ):
                    if finalArrR[j] == uniFinalArrR[i]:
                        if (len(phaNameForPlot)) > 0:
                            phaNameForPlot.append( "{0}{1}".format( finalPhaOrgNameR[j], count ) )
                            if finalPhaOrgNameR[j] == "s":
                                phaNameForPlot[-1] = "s"
                            count += 1
                        else:
                            phaNameForPlot.append( "{0}".format( finalPhaOrgNameR[j] ) )
                    
                nameAmpOffset = -0.7
                ax6.axvline( uniFinalArrR[i], ymin=0, ymax=0.5, linewidth=1, color='black', linestyle='--')
                for k in range( len( phaNameForPlot ) ):
                    if k>=0 and k < len( phaNameForPlot ) -1:
                        ax6.text( uniFinalArrR[i]-0.3, nameAmpOffset+0.5*k, "{0} + ".format( phaNameForPlot[k] ),
                                     fontsize=11, color='black', rotation=90)
                    else:
                        ax6.text( uniFinalArrR[i]-0.3, nameAmpOffset+0.5*k, "{0}".format( phaNameForPlot[k] ),
                                     fontsize=11, color='black', rotation=90)
                    
            ax7.plot( bigAmpTime, finalCcR[ peaksPts ], "o",
                     color='black', markersize=4, zorder=101)
            
            # plot phase-shifting angles
            ax8.plot( tRcc, finalCcR, alpha=0 ) # just use its time axis
            ax8.scatter( phaShiftAngTimeR, phaShiftAngR,
                        s=20, color='black', zorder=101 )
            
            #plot CC and phase-shifting angle of matched phases 
            for i in range( len( uniFinalArrR ) ):
                for j in range( len( phaShiftAngR ) ):
                    if uniFinalArrR[i] == phaShiftAngTimeR[j]:
                            ax8.scatter( uniFinalArrR[i], phaShiftAngR[j],
                                        s=20, color='red', zorder=101 )
                for j in range( len( peaksPts ) ):
                    if uniFinalArrR[i] == bigAmpTime[j]:
                            ax7.plot( uniFinalArrR[i], finalCcR[ peaksPts[j] ], "o",
                                     color='red',markersize=4, zorder=101)
            #set grid
            ax8.grid(True, linestyle='--', linewidth=0.5)
            
            # set title
            ax1.tick_params(axis='both', which='major', labelsize=10)
            #
            ax0.set_xscale('log')
            
            #set lim
            ax0.set_xlim(1e0, 1e3)
            ax0.set_ylim(-1.1, 1.1)
            ax1.set_ylim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            ax3.set_ylim(-1.2, 1.2)
            ax4.set_ylim(-1.2, 1.2)
            ax5.set_ylim(-1.2, 1.2)
            ax6.set_ylim(-1.3, 1.1)
            ax7.set_ylim(0, 1.2)
            ax8.set_ylim(-200, 200)
            ax8.set_yticks(np.arange(-180, 190, step=60))
            
            # set xticks
            ax0.xaxis.set_ticks_position('top')
            ax5.xaxis.set_ticks_position('top')
            ax6.xaxis.set_ticks_position('top')
            ax5.yaxis.set_ticks_position('left')
            ax6.yaxis.set_ticks_position('left')
            ax7.yaxis.set_ticks_position('left')
            ax8.yaxis.set_ticks_position('left')
            
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xticks([])
            ax4.set_xticks([])
            ax6.set_xticklabels([])
            ax7.set_xticks([])
            ax8.set_xticks([])
            
            ax0.set_yticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax3.set_yticks([])
            ax4.set_yticks([])        
            
            #remove axis margins
            ax1.margins(x=0)
            ax2.margins(x=0)
            ax3.margins(x=0)
            ax4.margins(x=0)
            ax5.margins(x=0)
            ax6.margins(x=0)
            ax7.margins(x=0)
            ax8.margins(x=0)
            
            # remove some spines
            ax6.spines['bottom'].set_visible(False)
            ax7.spines['top'].set_visible(False)
            
            # set labels
            ax0.xaxis.set_label_position('top')
            ax1.xaxis.set_label_position('top')
            ax5.xaxis.set_label_position('top')
            ax5.yaxis.set_label_position('left')
            ax6.yaxis.set_label_position('left')
            ax7.yaxis.set_label_position('left')
            ax8.yaxis.set_label_position('left')
            
            ax0.set_xlabel('Number', fontsize=12, labelpad=6)
            ax5.set_xlabel('Time (s)', fontsize=12, labelpad=8)
            ax5.set_ylabel('Amp.', fontsize=12)
            ax6.set_ylabel('Amp.', fontsize=12)
            ax7.set_ylabel('CC', fontsize=12, labelpad=15)
            ax8.set_ylabel('Shifted (°)', fontsize=12)
            
            # set zero lines
            ax7.axhline(ccThreshold, linewidth=0.8, linestyle='--', color='gray')
            
            # set span
            ax0.axhspan(leftBoundry1R, rightBoundry1R, facecolor='0.5', alpha=0.2, color='black')
            ax5.axhspan(leftBoundry1R, rightBoundry1R, facecolor='0.5', alpha=0.1, color='black')
                 
            # plot figure number
            ax5.set_title( "b)", x=-0.1, fontsize=16, color='black', loc='left' )
            ax5.text( 0.45, 0.7, "R", fontsize=12, color='black' )
            
            #show
            plt.tight_layout()
            plt.savefig( "{0}/{1}_First2Steps_R.png".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
            #plt.savefig( "{0}/{1}_First2Steps_R.svg".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
            plt.show  
    
            #%%####################################################################
            # plot wavefrom, templates, cc, depth-phase matches, phase-shifted angles 
            #######################################################################
            ##### Transverse component
            #-- prepare data for plotting
            DT       = DTGlobalT[ist][0]
            tempT0   = template0GlobalT[ist][0] / max( np.fabs( template0GlobalT[ist][0] )) # phase-shift = 0 deg
            tempT60  = template60GlobalT[ist][0] / max( np.fabs(template60GlobalT[ist][0]))# phase-shift = 60 deg
            tempT120 = template120GlobalT[ist][0] / max( np.fabs(template120GlobalT[ist][0]))# phase-shift = 120 deg
            tempT170 = template170GlobalT[ist][0] / max( np.fabs(template170GlobalT[ist][0]))# phase-shift = 180 deg
            dataT    = waveformGlobalT[ist][0] / max( np.fabs( waveformGlobalT[ist][0] ))
            tempT00  = template0GlobalT[ist][0] / max( np.fabs( waveformGlobalT[ist][0] ))
            bigAmpT  = bigAmpGlobalT[ist][0] * normWaveformGlobalT[ist][0]
            stNorT   = normWaveformGlobalT[ist][0]
            orgCcT   = peaksCurveGlobalT[ist][0]
            finalCcT = finalCcGlobalT[ist][0]
            peaksPts = finalpeaksPtsGlobalT[ist][0]
            corrLengT= corrLengGlobalT[ist][0]
            bigAmpTime       = bigAmpTimeGlobalT[ist][0]
            phaShiftAngT     = phaShiftAngGlobalT[ist][0]
            phaShiftAngTimeT = phaShiftAngTimeGlobalT[ist][0] 
            temTtBegNor      = temBegNorGlobalT[ist][0]
            leftBoundry1T    = leftBoundry1GlobalT[ist][0]
            rightBoundry1T   = rightBoundry1GlobalT[ist][0]
            wantedTimeLengT  = wantedTimeLengGlobalT[ist][0]
            finalArrT        = depthCandidateArrGlobalT[ist][idx][0]
            finalPhaOrgNameT = depthCandidatePhaOrgNameGlobalT[ist][idx][0]
            
            # set figure layout
            fig = plt.figure( constrained_layout=True, figsize=(8,4))
            fig.subplots_adjust(hspace=0.5)
            fig.subplots_adjust(wspace=0.05)
            gs0 = fig.add_gridspec(1, 2, width_ratios=[8,1] )
            gs00 = gs0[1].subgridspec(6,1)
            gs01 = gs0[0].subgridspec(10,1)
            
            
            ax0 = fig.add_subplot(gs00[0:2, 0])
            ax1 = fig.add_subplot(gs00[2, 0])
            ax2 = fig.add_subplot(gs00[3, 0])
            ax3 = fig.add_subplot(gs00[4, 0])
            ax4 = fig.add_subplot(gs00[5, 0])
            ax5 = fig.add_subplot(gs01[0:3, 0:])
            ax6 = fig.add_subplot(gs01[3:6, 0:])
            ax7 = fig.add_subplot(gs01[6, 0:])
            ax8 = fig.add_subplot(gs01[7:10, 0:])
            
            # plot data
            t1T = np.arange( 0, len(tempT0), 1)*DT
            t2T = np.arange( 0, len(dataT),  1)*DT
            tTcc= np.arange( 0, corrLengT,   1)*DT
            
            ax0.hist( histT, normed=False, bins=11, orientation='horizontal')
            ax1.plot( t1T, tempT0, color='orange' )
            ax2.plot( t1T, tempT60 )
            ax3.plot( t1T, tempT120 )
            ax4.plot( t1T, tempT170 )
            ax5.plot( t2T, dataT )
            ax6.plot( t2T, stNorT, color='lightgray' )
            ax6.plot( t2T, bigAmpT )
            ax7.plot( tTcc, orgCcT, color='lightgray')
            ax7.plot( tTcc, finalCcT )
            
            # add templates
            tTempT = np.arange( 0, len(tempT0), 1)*DT+temTtBegNor
            ax5.plot( tTempT, tempT00, color='orange' )
            # add texts
            ax0.text( 2,   -0.4, r'$\mu-\sigma$', fontsize=10, rotation=0 )
            ax0.text( 2,    0.3, r'$\mu+\sigma$', fontsize=10, rotation=0 )
            ax1.text( 0.01, 0.45, '0 °', fontsize=10, color='black')
            ax2.text( 0.01, 0.45, '60 °', fontsize=10, color='black')
            ax3.text( 0.01, 0.45, '120 °', fontsize=10, color='black')
            ax4.text( 0.01, 0.45, '170 °', fontsize=10, color='black')
            
            # add matched phase and arrival time
            # find the phases sharing same arrival
            uniFinalArrT = list(set(finalArrT))
            for i in range( len( uniFinalArrT ) ):
                phaNameForPlot = []
                count = 1
                for j in range( len( finalPhaOrgNameT ) ):
                    if finalArrT[j] == uniFinalArrT[i]:
                        if (len(phaNameForPlot)) > 0:
                            phaNameForPlot.append( "{0}{1}".format( finalPhaOrgNameT[j], count ) )
                            if finalPhaOrgNameT[j] == "s":
                                phaNameForPlot[-1] = "s"
                            count += 1
                        else:
                            phaNameForPlot.append( "{0}".format( finalPhaOrgNameT[j] ) )
                    
                nameAmpOffset = -0.7
                ax6.axvline( uniFinalArrT[i], ymin=0, ymax=0.5, linewidth=1, color='black', linestyle='--')
                for k in range( len( phaNameForPlot ) ):
                    if k>=0 and k < len( phaNameForPlot ) -1:
                        ax6.text( uniFinalArrT[i]-0.3, nameAmpOffset+0.5*k, "{0} + ".format( phaNameForPlot[k] ),
                                     fontsize=11, color='black', rotation=90)
                    else:
                        ax6.text( uniFinalArrT[i]-0.3, nameAmpOffset+0.5*k, "{0}".format( phaNameForPlot[k] ),
                                     fontsize=11, color='black', rotation=90)
                    
            ax7.plot( bigAmpTime, finalCcT[ peaksPts ],
                     "o", color='black', markersize=4, zorder=101)
            
            # plot phase-shifting angles
            ax8.plot( tTcc, finalCcT, alpha=0 ) # just use its time axis
            ax8.scatter( phaShiftAngTimeT, phaShiftAngT,
                        s=20, color='black', zorder=101 )
            
            #plot CC and phase-shifting angle of matched phases 
            for i in range( len( uniFinalArrT ) ):
                for j in range( len( phaShiftAngT ) ):
                    if uniFinalArrT[i] == phaShiftAngTimeT[j]:
                            ax8.scatter( uniFinalArrT[i], phaShiftAngT[j],
                                        s=20, color='red', zorder=101 )
                for j in range( len( peaksPts ) ):
                    if uniFinalArrT[i] == bigAmpTime[j]:
                            ax7.plot( uniFinalArrT[i], finalCcT[ peaksPts[j] ], "o",
                                     color='red',markersize=4, zorder=101)
            #set grid
            ax8.grid(True, linestyle='--', linewidth=0.5)
            
            # set title
            ax1.tick_params(axis='both', which='major', labelsize=10)
            #
            ax0.set_xscale('log')
            
            #set lim
            ax0.set_xlim(1e0, 1e3)
            ax0.set_ylim(-1.1, 1.1)
            ax1.set_ylim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            ax3.set_ylim(-1.2, 1.2)
            ax4.set_ylim(-1.2, 1.2)
            ax5.set_ylim(-1.2, 1.2)
            ax6.set_ylim(-1.3, 1.1)
            ax7.set_ylim(0, 1.2)
            ax8.set_ylim(-200, 200)
            ax8.set_yticks(np.arange(-180, 190, step=60))
            
            # set xticks
            ax0.xaxis.set_ticks_position('top')
            ax5.xaxis.set_ticks_position('top')
            ax6.xaxis.set_ticks_position('top')
            ax5.yaxis.set_ticks_position('left')
            ax6.yaxis.set_ticks_position('left')
            ax7.yaxis.set_ticks_position('left')
            ax8.yaxis.set_ticks_position('left')
            
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xticks([])
            ax4.set_xticks([])
            ax6.set_xticklabels([])
            ax7.set_xticks([])
            ax8.set_xticks([])
            
            ax0.set_yticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax3.set_yticks([])
            ax4.set_yticks([])        
            
            #remove axis margins
            ax1.margins(x=0)
            ax2.margins(x=0)
            ax3.margins(x=0)
            ax4.margins(x=0)
            ax5.margins(x=0)
            ax6.margins(x=0)
            ax7.margins(x=0)
            ax8.margins(x=0)
            
            # remove some spines
            ax6.spines['bottom'].set_visible(False)
            ax7.spines['top'].set_visible(False)
            
            # set labels
            ax0.xaxis.set_label_position('top')
            ax1.xaxis.set_label_position('top')
            ax5.xaxis.set_label_position('top')
            ax5.yaxis.set_label_position('left')
            ax6.yaxis.set_label_position('left')
            ax7.yaxis.set_label_position('left')
            ax8.yaxis.set_label_position('left')
            
            ax0.set_xlabel('Number', fontsize=12, labelpad=6)
            ax5.set_xlabel('Time (s)', fontsize=12, labelpad=8)
            ax5.set_ylabel('Amp.', fontsize=12)
            ax6.set_ylabel('Amp.', fontsize=12)
            ax7.set_ylabel('CC', fontsize=12, labelpad=15)
            ax8.set_ylabel('Shifted (°)', fontsize=12)
            
            # set zero lines
            ax7.axhline(ccThreshold, linewidth=0.8, linestyle='--', color='gray')
            
            # set span
            ax0.axhspan(leftBoundry1T, rightBoundry1T, facecolor='0.5', alpha=0.2, color='black')
            ax5.axhspan(leftBoundry1T, rightBoundry1T, facecolor='0.5', alpha=0.1, color='black')
                 
            # plot figure number
            ax5.set_title( "c)", x=-0.1, fontsize=16, color='black', loc='left' )
            ax5.text( 0.45, 0.7, "T", fontsize=12, color='black' )
            
            #show
            plt.tight_layout()
            plt.savefig( "{0}/{1}_First2Steps_T.png".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
            #plt.savefig( "{0}/{1}_First2Steps_T.svg".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
            plt.show      
    
       
    
        
    #-- calculte the final solution (median)
    numWinnerStep2 = len( depthWellBehavedStation )
    if( numWinnerStep2 > 1):
        finalDepthSolution = np.median(depthWellBehavedStation)
    else:
        finalDepthSolution = depthWellBehavedStation[0]
    #%%
    print( "finalDepthSolution = {0}".format( format( finalDepthSolution, ".2f" ) ), 'km' )    
         
    #%%############# PLOT ############################################
    if verboseFlag == 1:
        # -- plot the number of depth-phase matched of each assumed focal depth
        # and rms
        # set figure layout
        fig = plt.figure( constrained_layout=True, figsize=(5,5))
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.18)
        gs0 = fig.add_gridspec(1, 1 )
        gs00 = gs0[0].subgridspec(2,1)
        ax0 = fig.add_subplot(gs00[0, 0])
        ax1 = fig.add_subplot(gs00[1, 0])
        #-- ax0: the number of depth-phase matches
        #-- figure number
        t = scanDepthMembers
        ax0.plot(t, sumGlobal, linewidth=2., color="black")
        for ist in range(numSt):
            ax0.plot(t, totNumMatPhaGlobal[ist], linewidth=1, linestyle='-', color="grey")
        ax00 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax00.scatter(t, sumAvgArrTimeDiffResGlobal, s=15, marker='s',
                     facecolors='white', edgecolor='blue', zorder=100)
        ax00.scatter(prelimSolution, sumAvgArrTimeDiffResGlobal[prelimSolution-1],
                     s=25, marker='s', facecolors='blue', edgecolor='blue', zorder=100)
        ax00.axvline( np.array(prelimCandidatesGlobal).min(0)[1],
                     linewidth=1.5, color='blue', linestyle='--')
        ax00.axvline( np.array(prelimCandidatesGlobal).max(0)[1],
                     linewidth=1.5, color='blue', linestyle='--')
        ax00.set_ylim( 0, 1 )
        
        
        # set labels
        ax0.set_ylabel('Number of matches', fontsize=12)
        ax0.set_xlabel('Depth (km)', fontsize=12)
        ax00.set_ylabel('Sum of differential\narrival time residuals (s)',
                        color='blue', fontsize=12)
        ax00.tick_params(axis='y', colors='blue')
        #set grid
        ax0.grid(True, linestyle='--', linewidth=0.25)
        ax0.axhline( thresholdMaxNumb,linewidth=1.5, color='black', linestyle='--')
        ax0.margins(x=0)
        ax0.set_xticks( np.arange(scanDepthFrom,scanDepthTo, step=2))                
        
        #-- ax1: rms curve
        minRms = np.min( rmsWellBehavedStation )
        maxRms = np.max( rmsWellBehavedStation )
        minDepthRange = np.min(  depthRangeWellBehavedStation )
        maxDepthRange = np.max(  depthRangeWellBehavedStation )
        
        #-- This commend is only for the synthetic example in Section 3.2 of DSA paper
        if velModel == "ak135_Section3.2":
            for iwin in range( numWinnerStep2 ):
                line, = ax1.plot(depthRangeWellBehavedStation[iwin], rmsWellBehavedStation[iwin],
                                 label='RTZ', color='black', linewidth=0.5 )
        ax1.axvline( 13.5, linewidth=1.2, color='red', linestyle='-' )
        
        
        ax1.text( finalDepthSolution-0.3, minRms*1.2,
                  "(x={0}, y={1})".format(
                  format( finalDepthSolution, ".1f"),
                  format( minRms, ".2f") ),
                  fontsize=12, color='black',  rotation=0, zorder=110)
        ax1.scatter( finalDepthSolution, minRms, s=300,  marker='*',
                     facecolors='black', edgecolor='black', zorder=100)
    
        ax1.scatter( idepth, 0, s =300,  marker='*',
                    facecolors='white', edgecolor='black', zorder=100)
        
        #set xlim and ylim
        ax1.set_xlim( minDepthRange, maxDepthRange )
        ax1.set_xticks( np.arange( minDepthRange, maxDepthRange, step=0.2) )
        ax00.set_ylim( 0, 1 )
        
        # set labels
        ax1.set_ylabel('RMS (s)', fontsize=12)
        ax1.set_xlabel('Depth (km)', fontsize=12)
        #set grid
        ax1.grid(True, linestyle='--', linewidth=0.25)
        #-- figure number
        ax0.set_title( "a)", x=-0.2, fontsize=14, color='black', loc='left' )
        ax1.set_title( "b)", x=-0.2, fontsize=14, color='black', loc='left' )
        #save
        plt.tight_layout()
        plt.savefig( "{0}/Steps3-4_Prelim{1}km_Final{2}km.png".format(
                    outfilePath,
                    format( prelimSolution, ".1f"),
                    format( finalDepthSolution, '.1f') ),
                    dpi=360 )
        plt.savefig( "{0}/Steps3-4_Prelim{1}km_Final{2}km.svg".format(
                    outfilePath,
                    format( prelimSolution, ".1f"),
                    format( finalDepthSolution, '.1f') ),
                    dpi=360 )
        plt.show()
        
        
#%%       
resultsFile.close()

#%% calculate computing time
stop = timeit.default_timer()
elapsedTime = stop - start
print('Elapsed time: ', format( elapsedTime, '.1f'),
  'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )