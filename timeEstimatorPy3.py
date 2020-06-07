# This module will contain all the elements needed to compute the sensitivity
# and thus the time estimation for B-BOP
#
# Hard-coded information that need to be updated if this python script is distributed
#
# in filterTrans:
#   - The location of the band-pass profiles
#
# Development history
# 08/10/2019 Replace personal integration script by numpy.trapz that does exactly the 
#            same thing.
#            Still to be done for quality/robustness: turn all the common parts of the 
#            scripts into a single auxiliary script to be called by all (simpler to maintain)
# 06/05/2020 Changes for Py3 (Vincent Revéret)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:28:45 2019

@author: Marc Sauvage
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import math
#import var4te

#print( "This package contains the following functions")
#print( "   getForeground       a simple utility summing all foreground contributions")
#print( "   getNEP              a simple utility summing all NEP contributions")
#print( "   compTelForeground   computes the telescope contribution to the foreground")
#print( "   compZodiForegroud   computes the zodiacal light contribution to the foreground")
#print( "   compCIBForeground   computes the Cosmic Infrared Background contribution")
#print( "   compCMBForeground   computes the Cosmic Microwace Background contribution")
#print( "   comp4KForeground    computes the Fore Optics (4.8K) contribution to the foreground")
#print( "   comp2KForeground    computes the Ca;era Optics (1.8K) contribution to the foreground")
#print( "   filterTrans         returns the transmission of the filter of choice")
#print( "   reSample            is a utility function to resample a gridded function on another grid")
#print( "   getFwhm             is a utility that returns the FWHM of the beam at all central wavelengths")
#print( "   mkBackgroundFig     makes a plot of the backgrounds spectral densities (verification)")
#print( "   compTelNEP          computes the NEP associated to the telescope foreground")
#print( "   compZodiNEP         computes the NEP associated to the zodiacal light")
#print( "   compCMBNEP          computes the NEP associated to the CMB")
#print( "   compCIBNEP          computes the NEP associated to the CIB")
#print( "   comp4KNEP           computes the NEP associated to the Fore Optics (at 4.8K)")
#print( "   comp2KNEP           computes the NEP associated to the camera Optics (at 1.8K)")
#print( "   compSrcPixPow       computes the power falling on a pixel from an astronomical source")
#print( "   compPixPowSrc       computes the flux of a source from the power falling on the pixel")
#print( "   getObsTime          computes timing information (on-source, total) for an observation")
#print( "   getSensitivity      computes the sensitivity limit of an observation from S/N, area and obs mode")
    
# When using a distributed version of this script you need to update this path
#pathToFilter = '/Users/vreveret/Work/SPICA/B-BOP/InstrumentScientistStuff/TimeEstimator/Marc_Script/'
pathToFilter = os.getcwd()
# to the location where you have stored the file BandProfile.csv 


# Defines some constants and parameters for the whole code
# All units are SI
# physical constants (self-explanatory hopefully)
lightSpeed = 3e8
planck = 6.63e-34
bolzmann = 1.38e-23
hoverk = 4.80e-11

# 27/09/2019 - I have verified now that I get the same prediction of the spectral
#              density for the foregrounds as shown by AP in his slides, yet I have
#              exactly a factor of 2 in the background predictions per pixel per filter
#              It seems AP shows the background spectrum summing both polarisations but
#              computes the foreground prediction for a single polarisation...
# 07/10/2019 - We have clarified that: because the pixels is covered with devices that
#              have absorbing dipoles in only one orientation (each of the 4 thermometer
#              carries only one orientation of absorbing dipoles), the pixel is 
#              effectively sensitive to only 1/2 of the power for non-polarized light.
# 14/10/2019 - Prediction of total power sensitivities now complete and verified to be
#              compatible with previous estimations (when using same initial values).

# Astrophysical parameters
# Zodiacal light represented as a grey body, T=265K em=1e-7
zodiTemp = 265.0
#zodiTemp=var4te.zodiTemp_v
zodiEmissivity = 1e-7
# Cosmic Infrared Background represented by a grey body, T=30K, em=2.25e-6
cibTemp = 30.0
cibEmissivity = 2.25e-6
# Cosmic Microwave Background, actually a black-body at our level of accuracy! T=2.725K, em=1.0
cmbTemp = 2.725
cmbEmissivity = 1.0

# Photometric parameters
# As usual in the FIR we need a spectral convention to quote spectral densities
# for the power measured in our very wide bands. This is done assuming the spectrum 
# of the sources goes as some power (spectral index) of the frequency (usually -1)
fluxSpecIndex = -1.0

# Spacecraft parameters
# scanspeed in arcseconds/second, 20 "/s by default
scanSpeed = 20.0
# turn around time in between scan legs (to be used to compute the observing time)
turnAround = 0.0
# Initial overhead (before getting the first map data)
initOverhead = 0.0

# Telescope parameters - ?S 26-09-19 AP indicated that the telescope parameters
# he has used (T=6k, e=0.09) are meant to represent the equivalent black-body 
# that would contribute the same flux as the actual telescope (primary, secondary,
# baffles and all). In that respect if I make a computation with my code, the 
# "number of mirrors" in that effective telescope should be 1 (and not 2 or 3 as
# in the "real" telescope).
# Telescope diameter
telDiam = 2.5
#    Effective area (used in beam etendue) with 25% obscuration
telEffArea = np.pi * ((2.5/2.)**2 - (0.25*2.5/2)**2)
#    Mirror temperatures assumed the same for all mirrors
telTemp = 8.0
#    Mirror emissivity, assumed the same for all mirrors and constant
telEmissivity = 0.09
#    Number of mirrors in the telescope assembly
telNumMirrors = 1

# B-BOP parameters - Do not change those unless you really know what you are doing!
# band definitions - MS I believe (and hope) that is is the only installation dependent
# element.
bandProFile = pathToFilter+'BandProfile.csv'
#bandProFile = pathToFilter+'BandProfileAP.csv'
# Pixel sizes in arc-second, warning band 1 is in pixBand[0]
pixSizeBand = np.array([3.5,10,20])
# As a test this is AP's pixel sizes
#pixSizeBand = np.array([5,10,20])
# B-BOP's central wavelengths
bandWave = np.array([7e-5,2e-4,3.50e-4])
# As a test these are the AP central wavelengths
#bandWave = np.array([1.0e-4,2.0e-4,3.50e-4])
# Diameter of a circle that has the same surface as 2x2 pixels expressed 
# as function of the FWHM (of an unobscured 2.5m telescope)
diam4 = np.array([1.33,1.33,1.52])
# Fraction of a point source power that falls in this circle (assumption on beam shape)
eefD4 = 2./3.
# Instrumental transmission (assuming filterbands are normalised to max=1)
instTrans = 0.5
# Internal cold stop transmission
etaColdStop = 1.0
# Number of 4.8K (fore-optics) optical surfaces (this includes the pick-off mirror)
num4KSurf = 7
# Fore optic temperature
foreOptTemp = 4.8
# emissivity of these surfaces
foreOptEmissivity = 0.03
# transmission from the fore optics to the detectors 
foreOptTrans = instTrans * 1.0
# Number of 1.8K (camera optics) optical surface (this is an average number per channel)
# exact number is 7 for 70 mic, 3 for 200 mic and 4 for 350 mic
# Note that this component is never important...
num2KSurf = 5
# temperatuyre of the camera optics
camOptTemp = 1.8
# emissivity of these surfaces
camOptEmissivity = 0.03
# transmission from the camera optics to the detectors
camOptTrans = instTrans * 1.0
# This controls the fraction of the incident power that our polarisation-sensitive
# device are able to absorb. See getForeground for some details
pixAbsFrac = 0.5
# Detector NEP
detNEPReq = 3e-18
detNEPGoal = 1.5e-18
# B-BOP field of view in arcseconds
bbopFoV = 160.0


#######################################################
# Short utility function to get the complete backgroud contribution
#######################################################
def getForeground(band,incident=True):
    """"
    This code simply aggregates all the "foreground" source computations
    Important note: this computes the power that reaches the pixel, but not 
    necessarily the power that it is sensitive to: For these two quantities to
    be equal, the pixel surface has to be fully covered by absorbing
    devices.
    In B-BOP's case the pixel is indeed fully covered by absorbing devices however
        - a pixel is made of 4 absorbing thermometers, each occupying 1/4 of the
          pixel surface
        - each thermometer is covered by absorbing dipoles that are all oriented
          in the same direction. Therefore in the case of unpolarized radiation
          a thermometer is only sensitive to half of the power that falls within
          its "reach"
    
    if this script is called with incident = True, it will return the foreground
    power that is incident to the pixel, if incident = False it will return the
    fraction that the pixel is sensitive to (controlled by pixAbsFrac)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    result = compTelForeground(band)+compZodiForeground(band)+\
             compCMBForeground(band)+compCIBForeground(band)+\
             comp4KForeground(band)+comp2KForeground(band)
    
    if (incident):
        print ("Incident Foreground Power")
        return result
    else:
#        print("Effective Foreground Power (that which pixels can absorb)")
        return result*pixAbsFrac
   
    
#######################################################
# Short utility function to get the complete backgroud contribution
#######################################################
def getNEP(band,incident=True,withDet=False,goal=False):
    """"
    This code simply aggregates all the "NEP"  computations
    Important note: this computes the NEP associated to the power that reaches the pixel, but not 
    necessarily that which is relevant for sensitivity computation: For these two quantities to
    be equal, the pixel surface has to be fully covered by absorbing
    devices that are insensitive to the polarisation state of the incoming wave.
    In B-BOP's case the pixel is indeed fully covered by absorbing devices however
        - a pixel is made of 4 absorbing thermometers, each occupying 1/4 of the
          pixel surface
        - each thermometer is covered by absorbing dipoles that are all oriented
          in the same direction. Therefore in the case of unpolarized radiation
          a thermometer is only sensitive to approximately half of the power that falls within
          its "reach", and thus the NEP has to be corrected for this.
    
    if this script is called with incident = True, it will return the NEP associated to the 
    power that is incident to the pixel, if incident = False it will return the NEP that is 
    associated to the fraction that the pixel is sensitive to (multiplying by sqrt(pixAbsFrac))
    If withDet is True the detector NEP will be added, the default being the requirement value,
    otherwise if goal is True the gaol value (2 times better) is used
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    if (incident) and (withDet):
        print("ERROR - It is not correct to add the NEP of the incident radiation")
        print("ERROR - to that of the detector as the detector is not sensitive to")
        print("ERROR - the full incident radiation")
        return
    
    resultSq = compTelNEP(band,debug=False)**2+compZodiNEP(band,debug=False)**2+\
               compCMBNEP(band,debug=False)**2+compCIBNEP(band,debug=False)**2+\
               comp4KNEP(band,debug=False)**2+comp2KNEP(band,debug=False)**2
    
    if (incident):
#        print("Incident Foreground Power")
        return math.sqrt(resultSq)
    else:
        if (withDet):
            if (goal):
#                print("Including detector goal NEP")
                return math.sqrt(resultSq*pixAbsFrac + detNEPGoal**2)
            else:
#                 print("Including detector goal NEP")
                 return math.sqrt(resultSq*pixAbsFrac + detNEPReq**2)
        else:
#            print ("Effective Foreground NEP (that which pixels are sensitive to)")
            return math.sqrt(resultSq*pixAbsFrac)
   
    
#######################################################
# This function computes the power collected in the pixel
# coming from the telescope mirrors
#######################################################
def compTelForeground(band,debug=False):
    """
    This code returns the telescope thermal contribution in a given band.
    It is using SI units therefore this should be in W.
    It takes only one parameter that is the filter band, 1, 2 or 3
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=debug)
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / telTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
        
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,powSpecDens*telEmissivity)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1}$')
   
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    power *= telEmissivity
    power *= instTrans
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
    
    if (debug):
        print ("Beam Etendue is: ",telEffArea*pixSolidAngle)
    
    # Finally we need to account for the fact that the telescope is comprised of
    # a number of emitting surfaces so that
    power *= telNumMirrors
    
    return power

#######################################################
# This function computes the zodiacal light contribution
# to the foreground emission
#######################################################
def compZodiForeground(band,debug=False,status=True):
    """
    This function returns the contribution of the zodiacal light in the
    foreground collected by a band.
    It has a single parameter, the band that should be 1,2, or 3
    The zodiacal light is considered a black-body filling the beam and its
    parameters (temperature and emissivity) are defined in the parameter 
    section.
    """
    # MS - Because we treat the zodiacal light as a grey-body, this code is
    #      formally identical to that which computes the emission of the 
    #      telescope mirrors.
    #      This is also due to the assumption that we consider that the 
    #      transmission factor due to the telescope mirrors is negligible
    #      with respect to that due to the instrument, so we have applied
    #      the same value to a light source on the telescope or beyond.

    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        status = False
        return status
    
        # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)

    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=debug)
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / zodiTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    power *= zodiEmissivity
    power *= instTrans
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
    
    return power

   
#######################################################
# This function computes the Cosmic Infrared Background contribution
# to the foreground emission
#######################################################
def compCIBForeground(band,debug=False,status=True):
    """
    This function returns the contribution of the Cosmic Infrared Background in the
    foreground collected by a band.
    It has a single parameter, the band that should be 1,2, or 3
    The zodiacal light is considered a black-body filling the beam and its
    parameters (temperature and emissivity) are defined in the parameter 
    section.
    """
    # MS - Because we treat the CIB as a grey-body, this code is
    #      formally identical to that which computes the emission of the 
    #      telescope mirrors.
    #      This is also due to the assumption that we consider that the 
    #      transmission factor due to the telescope mirrors is negligible
    #      with respect to that due to the instrument, so we have applied
    #      the same value to a light source on the telescope or beyond.

    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        status = False
        return status
    
        # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)

    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=debug)
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / cibTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))

    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    power *= cibEmissivity
    power *= instTrans
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
    
    return power

#######################################################
# This function computes the Cosmic Infrared Background contribution
# to the foreground emission
#######################################################
def compCMBForeground(band,debug=False,status=True):
    """
    This function returns the contribution of the Cosmic Microwave Background in the
    foreground collected by a band.
    It has a single parameter, the band that should be 1,2, or 3
    The zodiacal light is considered a black-body filling the beam and its
    parameters (temperature and emissivity) are defined in the parameter 
    section.
    the outXXX keyword returns the frequency grid and the background level
    """
    # MS - Because we treat the CIB as a grey-body, this code is
    #      formally identical to that which computes the emission of the 
    #      telescope mirrors.
    #      This is also due to the assumption that we consider that the 
    #      transmission factor due to the telescope mirrors is negligible
    #      with respect to that due to the instrument, so we have applied
    #      the same value to a light source on the telescope or beyond.

    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
       status = False
       return status
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)

    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=debug)
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / cmbTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
                               
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
     
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by other instrumental terms in the formula that had no dependency on frequency
    power *= cmbEmissivity
    power *= instTrans
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
    
    return power

#######################################################
# This function computes the power collected in the pixel
# coming from the 4k mirrors, which constitute the fore-optics
#######################################################
def comp4KForeground(band):
    """
    This code returns the thermal contribution of the 4K fore opticsin a given band.
    It is using SI units therefore this should be in W.
    It takes only one parameter that is the filter band, 1, 2 or 3
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f')
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / foreOptTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
        
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
   
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    power *= foreOptEmissivity
    power *= foreOptTrans
    # it is debatable whether these components apply to the emission collected
    # from internal surfaces in the same way as they apply to the external
    # surfaces.
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
        
    # Finally we need to account for the fact that the fore Optics is comprised of
    # a number of emitting surfaces so that
    power *= num4KSurf
    
    return power

#######################################################
# This function computes the power collected in the pixel
# coming from the 4k mirrors, which constitute the fore-optics
#######################################################
def comp2KForeground(band):
    """
    This code returns the thermal contribution of the 2K camera optics in a given band.
    It is using SI units therefore this should be in W.
    It takes only one parameter that is the filter band, 1, 2 or 3
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f')
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / camOptTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
        
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
   
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    power *= camOptEmissivity
    power *= camOptTrans
    # it is debatable whether these components apply to the emission collected
    # from internal surfaces in the same way as they apply to the external
    # surfaces.
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
        
    # Finally we need to account for the fact that the fore Optics is comprised of
    # a number of emitting surfaces so that
    power *= num2KSurf
    
    return power

#######################################################
# This function returns the band profile, sampled along 
# a user-provided grid
#######################################################
def filterTrans(band,grid,worf='f',debug=False):
    """
    This code returns the filter transmission for a given band, sampled
    on the input grid.
    the band parameter can have value:
        1 - the 70micron band
        2 - the 200 micron band
        3 - the 350 micron band
    worf means wavelength or frequency, meaning that the input grid is 
    given either as a wavelength grid or as a frequency grid.
    """
    
    # I first test the input
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    if (worf != 'f') and (worf != 'w'):
        print ("ERROR - worf must either be w for wavelength or f for frequency")
        return
    
    # Here I read in the band transmission profiles. These must be stored
    # in a file with the first column being the wavelength, and the next three
    # columns being the transmission.
    # I assume the file is stored as a csv file. It's actual location and name
    # is defined in the parameter section of this script
    theProfile = open(bandProFile,'r')
    # create a reader for this file
    reader = csv.reader(theProfile,dialect='excel',delimiter=';')
    # this file contains one line of header that I do not want
    next(reader)
    # Define the wavelength, band 1,2,3 array
    wave = []
    band1 = []
    band2 = []
    band3 = []
    for row in reader:
        wave.append(float(row[0])*1e-6)
        band1.append(row[1])
        band2.append(row[2])
        band3.append(row[3])
    theProfile.close()
    
    # out of this I get lists. Things will be simpler with arrays
    wave = np.asarray(wave)
    band1 = np.asarray(band1)
    band2 = np.asarray(band2)
    band3 = np.asarray(band3)
    
    if (debug):
        # We plot the resulting profile to check that we are reading in things correcly
        plt.figure(figsize=(12,6))
        plt.plot(wave,band1,label='Band 1')
        plt.plot(wave,band2,label='Band 2')
        plt.plot(wave,band3,label='Band 3')
        plt.legend(loc=0,numpoints=1,shadow=True)
        

    # if the user has provided a grid in frequency we must convert the 
    # original grid in frequency
    if (worf == 'f'):
        if (debug): print("Converting original grid from wavelength to frequency")
        # and for everything to work I need to reverse the order of all the arrays
        # so that the sampling increase as a function of index.
        inputGrid = np.flipud(lightSpeed/wave)
        band1 = np.flipud(band1)
        band2 = np.flipud(band2)
        band3 = np.flipud(band3)      
    else:
        inputGrid = wave
    # now resamples, using a personal function
    if (band == 1):
        reBand = reSample(grid,inputGrid,band1)
    elif (band == 2):
        reBand = reSample(grid,inputGrid,band2)
    elif (band == 3):
        reBand = reSample(grid,inputGrid,band3)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.plot(grid,reBand,label='re-sampled band profile')
        plt.legend(loc=0,numpoints=1,shadow=True)
        
    return reBand

######################################################################
# check bands for error - this test is repeated over and over
######################################################################
def checkBandForError(band):
    if (band != 1) and (band != 2) and (band != 3):
        print ("ERROR - The only acceptable values for band are 1, 2, or 3")
        return False
    else:
        return True
    
###################################################################
# Computes the telescope's FWHM. This is straightforward but as
# it is done at different places in the code, it's better to put
# this in a function
###################################################################
def getFwhm(diameter,wavelengths):
    """
    This functions compute the FWHM for a pure airy profile
    All units must be SI
    """
    # Note: the FWHM is not the diameter of the first dark ring
    fwhm = 1.028*wavelengths/diameter
    # convert to arcseconds
    fwhm /= math.pi
    fwhm *= 180.
    fwhm *=3600
    
    return fwhm

###################################################################
# This is the function that resamples an Array
# It is very likely that there is a python function to do just that
# but this code has a pretty long heritage (IDL originally)
###################################################################
def reSample(outGrid,inGrid,inData,force=True,getHelp=False,verbose=False):
  """
  Correct use is
  outData = reSample(outGrid,inGrid,inData,force=T/F)

  where

  inGrid    is the input grid on which the data is currently sampled
  inData    is the data (can be 1,2, or 3D)
  outGrid   is the new grid on which you want to resample the data
  force     if True (default) will force extrapolation when the new grid is
            wider that the input grid
  verbose   T/F can be used to toggle the code from verbose to silent
  """
  # History
  # 06-Jan-01: interp1d works only when the x vector is increasing with index.
  #            so add a warning and check for this
  if (getHelp):
    print("Correct use is")
    print ("outData = reSample(outGrid,inGrid,inData,force=T/F)")
    print ("")
    print ("where")
    print ("")
    print ("inGrid    is the input grid on which the data is currently sampled")
    print ("inData    is the data (can be 1,2, or 3D)")
    print ("          Note that I assume that the first axis of inData is the sampling axis")
    print ("outGrid   is the new grid on which you want to resample the data")
    print ("force     if True (default) will force extrapolation when the new grid is")
    print ("          wider that the input grid")
    print ("")
    print ("WARNING   this code uses interpolate.interp1d which requires the grids to be")
    print ("          increasing with index.")
    print ("")
    print ("all inputs must be numpy arrays, I assume that the grid contains the sampling nodes")
    print ("arranged in ascending order.")
    return

  # First let's get the dimensions of the inputs
  nOrig = len(inGrid)
  nNew = len(outGrid)

  # check that the grid is indeed increasing.
  if (inGrid[-1] <= inGrid[0]):
    print ("ERROR -- the interpolation works only for grids that are increasing with index")
    return
  
  # now check that the data array has its first dimension corresponding to the same
  # dimension as the grid array
  if (nOrig != inData.shape[0]):
    print ("ERROR -- The input grid and the data are incompatible in size")
    print ("ERROR -- I expect that the first dimention of the data is the sampling axis")
    return

  # now merge it into a 3D array so that the resampling
  # code is generic.
  dimData = len(inData.shape)
  if (dimData == 1):
    # the data is monodimensional, i.e. is it a "spectrum"
    cubeIn = np.zeros((nOrig,1,1))
    cubeOut = np.zeros((nNew,1,1))
    # fill cube with the data
    cubeIn[:,0,0] = inData[:].copy()
  elif (dimData == 2):
    # the data is bidimensional, i.e. it is a "position velocity diagram"
    cubeIn = np.zeros((nOrig,inData.shape[1],1))
    cubeOut = np.zeros((nNew,inData.shape[1],1))
    # fill the cube with the data
    cubeIn[:,:,0] = inData[:,:].copy()
  elif (dimData == 3):
    # the data is tridimensional, i.e. it is a spectral cube
    cubeIn = np.zeros((nOrig,inData.shape[1],inData.shape[2]))
    cubeOut = np.zeros((nNew,inData.shape[1],inData.shape[2]))
    # fill the cube with the data
    cubeIn[:,:,:] = inData[:,:,:].copy()
  else:
    print ("ERROR -- The data has more than 3 dimensions, and I don't know how to deal with it")
    return

  # interpolation functions will have a hard time working if the new grid is larger than the
  # original one. I check whether this is the case and if so, and if the user has selected the
  # force option, I pad the original data with one more sample point beyond the new grid. The
  # value placed here is the same as the first or last value in the data cube.
  if (outGrid.min() < inGrid.min()) or (outGrid.max() > inGrid.max()):
    # check whether the user is aware of that
    if (force==False):
      print ("WARNING -- The output grid is wider than the input grid.")
      print ("WARNING -- IF you really want to extrapolate your data, use force=True")
      return
    else:
      if (verbose):
        print ("INFO -- The output grid is wider than the input grid.")
        print ("INFO -- Extrapolating the data to fix this")
      if (outGrid.min() < inGrid.min()):
        # I add one sampling point below the minimum of outGrid
        newInGrid = np.zeros(len(inGrid)+1)
        newInGrid[0] = outGrid.min()-0.1*np.abs(outGrid.mean())
        newInGrid[1:] = inGrid[:].copy()
        # add a new slice to the cube which is equal to the first slice
        newCubeIn = np.zeros((len(inGrid)+1,cubeIn.shape[1],cubeIn.shape[2]))
        newCubeIn[0,:,:] = cubeIn[0,:,:].copy()
        newCubeIn[1:,:,:] = cubeIn[:,:,:].copy()
        # replace the old variables with the new ones
        inGrid = newInGrid
        cubeIn = newCubeIn
      if (outGrid.max() > inGrid.max()):
        # I add one sampling point beyond the maximum of outGrid
        newInGrid = np.zeros(len(inGrid)+1)
        newInGrid[-1] = outGrid.max()+0.1*np.abs(outGrid.mean())
        newInGrid[0:len(inGrid)] = inGrid[:].copy()
        # add a new slice to the cube which is equal to the last slice
        # add a new slice to the cube which is equal to the first slice
        newCubeIn = np.zeros((len(inGrid)+1,cubeIn.shape[1],cubeIn.shape[2]))
        newCubeIn[-1,:,:] = cubeIn[-1,:,:].copy()
        newCubeIn[0:len(inGrid),:,:] = cubeIn[:,:,:].copy()
        # replace the old variables with the new ones
        inGrid = newInGrid
        cubeIn = newCubeIn

  # At this stage we are done and safe, the old grid fully encompasses the new grid.
  # to do the interpolation I am going to use the inter1d function of scipy
  for i in range(cubeIn.shape[1]):
    for j in range(cubeIn.shape[2]):
      # I am using linear interpolation. For cubic, simply use king='cubic' in the function's
      # argument
      interpFunc = interp1d(inGrid,cubeIn[:,i,j])
      cubeOut[:,i,j] = interpFunc(outGrid)

  # now that interpolation has occured, we simply need to reshape the output into
  # its original dimensions.
  if (dimData == 1):
    result = np.zeros(nNew)
    result[:] = cubeOut[:,0,0]
  elif (dimData == 2):
    result = np.zeros((nNew,cubeOut.shape[1]))
    result[:,:] = cubeOut[:,:,0]
  else:
    result = cubeOut

  return result
        
#########################################
# make the background plot
#########################################
def mkBackgroundFig(save=False):
    """
    This code uses the background makes a plot of the background
    before they are integrated in the pixel
    Background computations are cut/paste from the functions
    """
    
    
    # Get the CMN bacground value
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 10mic->1000mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 20000
    waveGrid = np.linspace(3e11,3e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(3.,300.,num=numEltsGrid)
    
    # CMB Contribution
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / cmbTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(cmbEmissivity * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(cmbEmissivity * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(cmbEmissivity * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    cmbSpecDens = np.asarray(powSpecDens)
    
    #Telescope contribution
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / telTemp)    
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(telEmissivity * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(telEmissivity * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(telEmissivity * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    telSpecDens = np.asarray(powSpecDens)
    # multiply by the number of mirrors
    telSpecDens *= telNumMirrors
    
    # Zodiacal light contribution
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / zodiTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(zodiEmissivity * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(zodiEmissivity * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(zodiEmissivity * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    zodiSpecDens = np.asarray(powSpecDens)
    
    # CIB contribution
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / cibTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(cibEmissivity * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(cibEmissivity * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(cibEmissivity * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    cibSpecDens = np.asarray(powSpecDens) 
    
    
    # Note that here, we effectively treat the contribution of the fore optics
    # and the camera optics as if they were external to the instrument (you do
    # not see the application of a transmission or a cold stop efficiency)
    # fore optics contribution
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / foreOptTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(foreOptEmissivity * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(foreOptEmissivity * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(foreOptEmissivity * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    foreOptSpecDens = np.asarray(powSpecDens)        
    # Finally we need to account for the fact that the fore Optics is comprised of
    # a number of emitting surfaces so that
    foreOptSpecDens *= num4KSurf

    # Camera optics contribution
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / camOptTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(camOptEmissivity * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(camOptEmissivity * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(camOptEmissivity * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    camOptSpecDens = np.asarray(powSpecDens)
    camOptSpecDens *= num2KSurf
    
    # To overplot the band edges, we determine them by locations 
    # where transmission is greater than half Max.
    # in pltaxvspan
    #plt.axvspan(80, 60, color='g', alpha=0.2, lw=0)
    bandCol=['b','g','r']
    bandMin = []
    bandMax = []
    for i in range(3):
        # Get the filter transmission for this grid
        filt = np.asarray(filterTrans(i+1,waveGrid,worf='f'))
        # Use the where function to locate the band
        idx = np.where(filt>filt.max()/2.)
        bandMin.append(waveGrid[idx[0].min()])
        bandMax.append(waveGrid[idx[0].max()])
        
#        plt.axvspan(1e6*lightSpeed/waveGrid[idx[0].min()],\
#                    1e6*lightSpeed/waveGrid[idx[0].max()],\
#                    color=bandCol[i],alpha=0.2)

        
    plt.figure(figsize=(12,6))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both')
    plt.plot(lightSpeed/waveGrid,cmbSpecDens,label='CMB')
    plt.plot(lightSpeed/waveGrid,telSpecDens,label='Telescope ('+str(telTemp)+'K)')
    plt.plot(lightSpeed/waveGrid,zodiSpecDens,label='Zodiacal light')
    plt.plot(lightSpeed/waveGrid,cibSpecDens,label='CIB')
    plt.plot(lightSpeed/waveGrid,foreOptSpecDens,label='Fore Optics')
    plt.plot(lightSpeed/waveGrid,camOptSpecDens,label='Camera Optics')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Background spectral density (W.m$^{-2}$.Hz$^{-1}$.sr$^{-1}$)')
    plt.xlim(1e-5,1e-3)
    plt.ylim(1e-22,5e-17)
    plt.legend(loc='upper left',numpoints=1,shadow=True)
    for i in range(3):
        plt.axvspan(lightSpeed/bandMin[i],lightSpeed/bandMax[i],\
                    color=bandCol[i],alpha=0.2)
    if (save):
        print ("File will be written in "+os.getcwd())
        fileName = os.getcwd()+'/EachBackgrounds'+str(telTemp)+'K.png'
        plt.savefig(fileName)


    totBkg = cmbSpecDens+telSpecDens+zodiSpecDens+cibSpecDens+\
             foreOptSpecDens+camOptSpecDens
    plt.figure(figsize=(12,6))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both')
    plt.plot(1e6*lightSpeed/waveGrid,1e20*totBkg,\
             label='Total Background (TelT='+str(telTemp)+'K)')
    plt.xlabel('Wavelength ($\mu$m)')
    plt.ylabel('Background spectral density (MJy.sr$^{-1}$)')
    plt.xlim(10,1000)
    plt.ylim(0.5,2000)
    plt.legend(loc='upper left',numpoints=1,shadow=True)
    for i in range(3):
        plt.axvspan(1e6*lightSpeed/bandMin[i],1e6*lightSpeed/bandMax[i],\
                    color=bandCol[i],alpha=0.2)
    if (save):
        print ("File will be written in "+os.getcwd())
        fileName = os.getcwd()+'/TotBackground'+str(telTemp)+'K.png'
        plt.savefig(fileName)
    
#######################################################
# This function computes NEP associated the telescope mirrors
# thermal emission
#######################################################
def compTelNEP(band,debug=True):
    """
    This code returns the telescope contribution to the NEP in a given band.
    It is using SI units therefore this should be in W.Hz^-1/2.
    It takes only one parameter that is the filter band, 1, 2 or 3
    Note that I compute here the NEP that is due to the thermal source assuming
    the detector can absorb whatever polarisation. As explained elsewhere this
    is not the case for the B-BOP detector so when computing the B-BOP background
    NEP one will have to divide this output by sqrt(2)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # now let's build the function under the integral
    twoH2Nu4C2 = 9.74e-40 * waveGridNorm**4
    hNuKT = hoverk * (waveGrid / telTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    nepSqInt = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            expTermInt = (1./hNuKT[i])
        elif (hNuKT[i]<100.):
            expTermInt = (1. / (math.exp(hNuKT[i])-1))
        else:
            expTermInt = (math.exp(-1*hNuKT[i]))
        # some terms have for the moment no frequency dependence so they can be
        # made into a single constant
        nepConst = telEmissivity * instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * expTermInt))

    # convert it as an array
    nepSqInt = np.asarray(nepSqInt)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,nepSqInt)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1/2}$')
   
    # now integrate that function
    nepSquare = np.trapz(nepSqInt,x=waveGrid)
    # multiply by the beam etendue
    nepSquare *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    nepSquare *= pixSolidAngle
    
    # account for the existence of two polarisation state per photon energy state
    nepSquare *=2
        
    # Finally we need to account for the fact that the telescope is comprised of
    # a number of emitting surfaces so that we have to add as many NEP^2 as there
    # are surfaces.
    nepSquare *= telNumMirrors
    
    return math.sqrt(nepSquare)

#######################################################
# This function computes NEP associated the B-BOP mirrors
# located in the fore optics
#######################################################
def comp4KNEP(band,debug=True):
    """
    This code returns the telescope contribution to the NEP in a given band.
    It is using SI units therefore this should be in W.Hz^-1/2.
    It takes only one parameter that is the filter band, 1, 2 or 3
    Note that I compute here the NEP that is due to the thermal source assuming
    the detector can absorb whatever polarisation. As explained elsewhere this
    is not the case for the B-BOP detector so when computing the B-BOP background
    NEP one will have to divide this output by sqrt(2)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # now let's build the function under the integral
    twoH2Nu4C2 = 9.74e-40 * waveGridNorm**4
    hNuKT = hoverk * (waveGrid / foreOptTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    nepSqInt = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            expTermInt = (1./hNuKT[i])
        elif (hNuKT[i]<100.):
            expTermInt = (1. / (math.exp(hNuKT[i])-1))
        else:
            expTermInt = (math.exp(-1*hNuKT[i]))
        # some terms have for the moment no frequency dependence so they can be
        # made into a single constant
        nepConst = foreOptEmissivity * instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * expTermInt))

    # convert it as an array
    nepSqInt = np.asarray(nepSqInt)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,nepSqInt)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1/2}$')
   
    # now integrate that function
    nepSquare = np.trapz(nepSqInt,x=waveGrid)
    # multiply by the beam etendue
    nepSquare *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    nepSquare *= pixSolidAngle
    
    # account for the existence of two polarisation state per photon energy state
    nepSquare *=2
        
    # Finally we need to account for the fact that the telescope is comprised of
    # a number of emitting surfaces so that we have to add as many NEP^2 as there
    # are surfaces.
    nepSquare *= num4KSurf
    
    return math.sqrt(nepSquare)

#######################################################
# This function computes NEP associated the B-BOP mirrors
# located in the camera optics
#######################################################
def comp2KNEP(band,debug=True):
    """
    This code returns the camera optics contribution to the NEP in a given band.
    It is using SI units therefore this should be in W.Hz^-1/2.
    It takes only one parameter that is the filter band, 1, 2 or 3
    Note that I compute here the NEP that is due to the thermal source assuming
    the detector can absorb whatever polarisation. As explained elsewhere this
    is not the case for the B-BOP detector so when computing the B-BOP background
    NEP one will have to divide this output by sqrt(2)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # now let's build the function under the integral
    twoH2Nu4C2 = 9.74e-40 * waveGridNorm**4
    hNuKT = hoverk * (waveGrid / camOptTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    nepSqInt = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            expTermInt = (1./hNuKT[i])
        elif (hNuKT[i]<100.):
            expTermInt = (1. / (math.exp(hNuKT[i])-1))
        else:
            expTermInt = (math.exp(-1*hNuKT[i]))
        # some terms have for the moment no frequency dependence so they can be
        # made into a single constant
        nepConst = camOptEmissivity * instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * expTermInt))

    # convert it as an array
    nepSqInt = np.asarray(nepSqInt)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,nepSqInt)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1/2}$')
   
    # now integrate that function
    nepSquare = np.trapz(nepSqInt,x=waveGrid)
    # multiply by the beam etendue
    nepSquare *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    nepSquare *= pixSolidAngle
    
    # account for the existence of two polarisation state per photon energy state
    nepSquare *=2
        
    # Finally we need to account for the fact that the telescope is comprised of
    # a number of emitting surfaces so that we have to add as many NEP^2 as there
    # are surfaces.
    nepSquare *= num2KSurf
    
    return math.sqrt(nepSquare)


#######################################################
# This function computes the NEP associated to the 
# Zodiacal light foreground
#######################################################
def compZodiNEP(band,debug=True):
    """
    This code returns the Zodiacal light contribution to the NEP in a given band.
    It is using SI units therefore this should be in W.Hz^-1/2.
    It takes only one parameter that is the filter band, 1, 2 or 3
    Note that I compute here the NEP that is due to the thermal source assuming
    the detector can absorb whatever polarisation. As explained elsewhere this
    is not the case for the B-BOP detector so when computing the B-BOP background
    NEP one will have to divide this output by sqrt(2)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # now let's build the function under the integral
    twoH2Nu4C2 = 9.74e-40 * waveGridNorm**4
    hNuKT = hoverk * (waveGrid / zodiTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    nepSqInt = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            expTermInt = (1./hNuKT[i])
        elif (hNuKT[i]<100.):
            expTermInt = (1. / (math.exp(hNuKT[i])-1))
        else:
            expTermInt = (math.exp(-1*hNuKT[i]))
        # some terms have for the moment no frequency dependence so they can be
        # made into a single constant
        nepConst = zodiEmissivity * instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * expTermInt))

    # convert it as an array
    nepSqInt = np.asarray(nepSqInt)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,nepSqInt)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1/2}$')
   
    # now integrate that function
    nepSquare = np.trapz(nepSqInt,x=waveGrid)
    # multiply by the beam etendue
    nepSquare *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    nepSquare *= pixSolidAngle
    
    # account for the existence of two polarisation state per photon energy state
    nepSquare *=2
            
    return math.sqrt(nepSquare)
   
#######################################################
# This function computes the NEP associated to the 
# Cosmological Microwave Background
#######################################################
def compCMBNEP(band,debug=True):
    """
    This code returns the CMB contribution to the NEP in a given band.
    It is using SI units therefore this should be in W.Hz^-1/2.
    It takes only one parameter that is the filter band, 1, 2 or 3
    Note that I compute here the NEP that is due to the thermal source assuming
    the detector can absorb whatever polarisation. As explained elsewhere this
    is not the case for the B-BOP detector so when computing the B-BOP background
    NEP one will have to divide this output by sqrt(2)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # now let's build the function under the integral
    twoH2Nu4C2 = 9.74e-40 * waveGridNorm**4
    hNuKT = hoverk * (waveGrid / cmbTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    nepSqInt = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            expTermInt = (1./hNuKT[i])
        elif (hNuKT[i]<100.):
            expTermInt = (1. / (math.exp(hNuKT[i])-1))
        else:
            expTermInt = (math.exp(-1*hNuKT[i]))
        # some terms have for the moment no frequency dependence so they can be
        # made into a single constant
        nepConst = cmbEmissivity * instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * expTermInt))

    # convert it as an array
    nepSqInt = np.asarray(nepSqInt)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,nepSqInt)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1/2}$')
   
    # now integrate that function
    nepSquare = np.trapz(nepSqInt,x=waveGrid)
    # multiply by the beam etendue
    nepSquare *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    nepSquare *= pixSolidAngle
    
    # account for the existence of two polarisation state per photon energy state
    nepSquare *=2
            
    return math.sqrt(nepSquare)

#######################################################
# This function computes the NEP associated to the 
# Cosmic Infrared Background
#######################################################
def compCIBNEP(band,debug=True):
    """
    This code returns the CIB contribution to the NEP in a given band.
    It is using SI units therefore this should be in W.Hz^-1/2.
    It takes only one parameter that is the filter band, 1, 2 or 3
    Note that I compute here the NEP that is due to the thermal source assuming
    the detector can absorb whatever polarisation. As explained elsewhere this
    is not the case for the B-BOP detector so when computing the B-BOP background
    NEP one will have to divide this output by sqrt(2)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # now let's build the function under the integral
    twoH2Nu4C2 = 9.74e-40 * waveGridNorm**4
    hNuKT = hoverk * (waveGrid / cibTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    nepSqInt = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            expTermInt = (1./hNuKT[i])
        elif (hNuKT[i]<100.):
            expTermInt = (1. / (math.exp(hNuKT[i])-1))
        else:
            expTermInt = (math.exp(-1*hNuKT[i]))
        # some terms have for the moment no frequency dependence so they can be
        # made into a single constant
        nepConst = cibEmissivity * instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * expTermInt))

    # convert it as an array
    nepSqInt = np.asarray(nepSqInt)
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,nepSqInt)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1/2}$')
   
    # now integrate that function
    nepSquare = np.trapz(nepSqInt,x=waveGrid)
    # multiply by the beam etendue
    nepSquare *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    nepSquare *= pixSolidAngle
    
    # account for the existence of two polarisation state per photon energy state
    nepSquare *=2
            
    return math.sqrt(nepSquare)

######################################################################
# This functions returns the incoming power from a source in the pixel
######################################################################
def compSrcPixPow(flux,band,extended=False):
    """
    This function returns the power from a source that falls on a pixel.
    it allows distinguishing between a point and an extended source
    As with the background prediction functions it computes the incident power,
    not the power that the stokes pixels are sensitive to (1/2 of it for 
    non-polarised signal).
    The flux is provided in mJy for point sources (extended=False) and 
    in MJy/sr for extended sources (extended=True)
    """
    
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Converts the input flux into SI
    if (extended):
        print ("Converting from MJy/sr into SI")
        inputFluxDens = flux * 1e-20 # now in W.m-2.Hz-1.sr-1
        # Need to be multiplied by the pixel solid angle.
        pixSolidAngle = (pixSizeBand[band-1]*(math.pi/(180*3600)))**2
        inputFluxDens *= pixSolidAngle
        # at this stage the flux in in W.m-2.Hz-1.pix-1
    else:
        print ("Converting from mJy into SI")
        inputFluxDens = flux * 1e-29 # now in W.m-2.Hz-1
        # to convert that into a surface brightness we make assumptions
        # the code is parameterized by eefD4 which is the fraction of energy that 
        # falls in a circle that has the same surface as 4 pixels (fully illuminated
        # by a point source as per optical design) and diam4 the diameter of this
        # circle expressed in units of the FWHM.
        # compute the FWHM (the function returns it at all wavelengths)
        fwhm = getFwhm(telDiam,bandWave)
        corrFrac = eefD4 * pixSizeBand[band-1]**2 * (4./(math.pi*fwhm[band-1]**2*diam4[band-1]**2))
        #print 1./corrFrac
        inputFluxDens *= corrFrac
        # at this stage, the flux is in W.m-2.Hz-1.pix-1
        
    # Now multiplies by the collecting surface
    inputFluxDens *= telEffArea
    
    # now we need to integrate inside the band.
    # create a spectrum.
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    centralFreq = lightSpeed/bandWave[band-1]
    spectrum = inputFluxDens*((waveGrid/centralFreq)**fluxSpecIndex)
    # now integrate not forgetting multiplying by the bandpass
    powerPerPix = np.trapz(spectrum*filterTrans(band,waveGrid,worf='f'),x=waveGrid)
    
    return powerPerPix
                              
########################################################################
# This function returns the source flux (mJy or MJy/sr) corresponding to
# and incident power on the pixel
########################################################################
def compPixPowSrc(power,band,extended=False):
    """
    This function returns the source flux (in mJy for point source or MJy/sr
    for extended sources) that corresponds to a given incident power on the
    pixel.
    Input power must be in W.pix-1
    """
    # First perform a input check
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    
    # Turn this power into a spectral density by dividing it by the bandwidth
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    centralFreq = lightSpeed/bandWave[band-1]
    bandWidth = np.trapz(filterTrans(band,waveGrid,worf='f')*\
                         (waveGrid/centralFreq)**fluxSpecIndex,x=waveGrid)
    
    power /= bandWidth
    # now power is in W.Hz-1.pix-1
    # turn that into a flux by dividing it by the collecting area
    power /=telEffArea
    # now power is in W.m-1.Hz-1.pix-1
    
    if (extended):
#        print ("converting to MJy/sr")
        # the source is extended so I just need to divide by the solid angle
        # subtended by a pixel (formall this factor is in sr.pix-1)
        pixSolidAngle = (pixSizeBand[band-1]*(math.pi/(180*3600)))**2
        flux = power/pixSolidAngle
        # normalize it to MJy/sr
        flux /= 1e-20
    else:
        print ("converting to mJy")
        # the source is a point source so I first determine how much of 
        # the source flux falls on one pixel. As explained in routine compSrcPixPow
        # we characterise this by eefD4 which is the fraction of the energy that 
        # falls on an aperture that occupies the same area as 2x2 pixels, and diam4,
        # the diameter of this circle expressed in units of the FWHM.
        # compute the FWHM (the function returns it at all wavelengths)
        fwhm = getFwhm(telDiam,bandWave)
        corrFrac = eefD4 * pixSizeBand[band-1]**2 * (4./(math.pi*fwhm[band-1]**2*diam4[band-1]**2))
        flux = power / corrFrac
        # Convert it in mJy
        flux /=1e-29
    
    return flux

######################################################################
# This routines simply makes the conversion of a power into a flux 
# density, assuming the spectral convention, the user provided filter
# and the collecting area
######################################################################
def convPow2Flux(power,band):
    """
    This function converts a power (in W, e.g. the power collected form a
    source through a filter) and converts it to a flux density (in mJy) using
    the current collecting area and the defined band passes and central 
    wavelength.
    It assumes the power is the integrated power from a source (e.g. a point source).
    """
    if (checkBandForError == False):
        # the band is not within the accepted band labels
        return
    else:
        # get the bandwidth
        numEltsGrid = 10000
        waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
        centralFreq = lightSpeed/bandWave[band-1]
        bandWidth = np.trapz(filterTrans(band,waveGrid,worf='f')*\
                             (waveGrid/centralFreq)**fluxSpecIndex,x=waveGrid)
        
        # now make the necessary divisions (I do not create a new variable but
        # this is no longer a power and gradually becomes a spectral density)
        power /= bandWidth
        power /=telEffArea
        # this is in W.m-2.Hz-1, convert it to mJy
        power /= 1e-29
        
        return power
        
######################################################################
# This function returns timing information for an observation defined
# by its scan speed, area and obsmode
######################################################################
def getObsTime(vscan,area,mode='optimal'):
    """
    This function returns timing information for an observation, e.g. the time
    "on-source" (i.e. on any point in the map) as well as the observation time
    (i.e. the time required to complete the observation).
    Its arguments are:
        vscan   the scan speed at which the spacecraft proceeds on a leg (in "/s)
        areea   the area to be mapped (in degrees) assumed to be square-shaped
        mode    is a parameter describing the observing mode (needs to be among allowed values)
    Note that in this code, vscan is an input parameter and not the paramterised value
    declared at the beginning of this package. This is because here we are providing
    an intermediate element in the sensitivity estimate.
    it returns [on source time, total observing time, efficiency]
    """
    
    # Verify the mode
    if (mode != 'optimal'):
        print ("Observing mode parameter is not in allowed values")
        print ("Allowed values are [optimal]")
        return
    
    if (mode == 'optimal'):
        # we're in the mode were we have found the optimal compromise between on-source
        # time and map execution time
        # Formulas used below come from document BBOP-DAp-RP-002
        # The formula for the obsTime may be slightly wrong because 
        # scanlegs do not come in fraction so somewhere there should be rounding
        # to the integer immediately above. Not considered major here.
        onSrcTime = bbopFoV / (0.84*vscan)
        obsTime = area * (3600./vscan) * (1.2 * (3600./bbopFoV+0.54))
        # determines the number of scan legs required for this observation
        # the +2 is because I cannot accept a fractional number of legs spacings
        # so the map is overdimensioned (+1 w.r.t to the integral part of the ratio
        # between map widths and leg spacing), and then for a given nunber of spacings
        # we have to make +1 number of legs
        nLegs = int(3600.*math.sqrt(area)/(bbopFoV*0.84))+2
        #print nLegs
        # add the turn around overhead
        obsTimeFull = obsTime + (nLegs-1)*turnAround
        # add the initial overhead
        obsTimeFull += initOverhead
        
    return [onSrcTime,obsTimeFull, obsTime/obsTimeFull]

########################################################################
# This function gets the sensitivity of a map.
# the user provided arguments are the source s/n that is targeted
# and the area to be mapped.
########################################################################
def getSensitivity(ston,area,mode='optimal',goal=False,silent=False):
    """
    This script returns the sensitivity (mJy for a point source, MJy/sr for an
    extended source) of a map on a user provided area and signal-to-noise ratio.
    It makes the computation for all three bands of B-BOP.
    its arguments are:
        - ston   signal to noise (dimensionless)
        - area   the area to map (degrees)
        - mode   this is the observing mode, chosen among accepted values
        - goal   (T/F) is a flag to use the goal detector NEP rather than the requirement
                 default is False
        - silent if true will not print the results
    Note that there is an underlying assumption that the signal to noise that
    matters to the used (the input here) is the one measured after object photometry
    has been performed, which means subtraction of a background in one way or another, 
    that leads to a factor sqrt(2) on the pixel rms (at least).
    Furthermore for point sources, the S/N from the user point of view is the S/N
    measured in a aperture around the source. We take the convention that the aperture
    is the same as the one used to characterise the Encircled Energy Fraction (in other
    words the aperture is fixed to the profile of the PSF). Here we have characterised it
    by eefD4 and diam4 which are the diameter of an aperture of surface equal to that
    of 4 pixels, and the fraction of energy inside that circle. So effectively the aperture
    photometry is performed on 4 pixels, which relaxes the level of the noise by a 
    factor 2 with respect to the noise per pixel.
    """
    
    # Check the input parameters
    if (mode != 'optimal'):
        print ("Observing mode parameter is not in allowed values")
        print ("Allowed values are [optimal]")
        return
    
    if (mode == 'optimal'):
        # collect the elements of the computation
        # Though we do not need it for the sensitivity, let's get the background
        forePower = [getForeground(1,incident=False),\
                     getForeground(2,incident=False),\
                     getForeground(3,incident=False)]
        forePower = np.asarray(forePower)
        # get the total NEP, with the detector contribution 
        pixNEP = [getNEP(1,incident=False,withDet=True,goal=goal),\
                  getNEP(2,incident=False,withDet=True,goal=goal),\
                  getNEP(3,incident=False,withDet=True,goal=goal)]
        pixNEP = np.asarray(pixNEP)
        # get the timing information
        timings = getObsTime(scanSpeed,area,mode=mode)
        onSourceTime = timings[0]
        #print "Total observing time: ",timings[1]
        
        # Extended case
        extSrcPow = ston*pixNEP/math.sqrt(onSourceTime)
        # Since we have been using absorbed quantities, there is a factor of 2
        # with respect to the incident power
        extSrcPow *=2
        # convert to MJy/sr
        extSrcFlux = [compPixPowSrc(extSrcPow[0],1,extended=True),\
                     compPixPowSrc(extSrcPow[1],2,extended=True),\
                     compPixPowSrc(extSrcPow[2],3,extended=True)]
        
        # Point source case
        # Here we have to be careful
        # Note to self: I need to replace this numerical factor of 4 by 
        # its expreession as a function of the diam4 aperture and pixel size
        # compute the number of "equivalent" pixels in the aperture that contains
        # eefD4 fraction of the energy.
        fwhm = getFwhm(telDiam,bandWave)
        nPixAper = (math.pi/4) * (diam4*fwhm)**2 / pixSizeBand**2
        # computes the point source power in the aperture
        pointSrcPow = ston * (np.sqrt(2*nPixAper)) * \
                      pixNEP / math.sqrt(2*onSourceTime)
        # Now correct for the fraction of the point source that is not in the aperture
        pointSrcPow /= eefD4
        # remember that these are absorbed quantities so the pixel receives twide that
        pointSrcPow *=2
        # convert to mJy - Warning: if I use compPixPowSrc I am assuming that
        # pointSrcPow is a power falling on a single pixel (and the routine will make
        # the correction). But this has already been taken into accound in the formula
        # above so now I just need to convert something that is in W into mJy using
        # the collecting area and the filter width.
        pointSrcFlux = [convPow2Flux(pointSrcPow[0],1),\
                        convPow2Flux(pointSrcPow[1],2),\
                        convPow2Flux(pointSrcPow[2],3)]
        
        if (silent == False):
            # prints the result
            print ("Observing parameters: S/N = {0:.1f}, area = {1:.2f}".format(ston,area))
            print ("Observing time = {0:.2f} (s) for an efficiency of {1:.2f}".format(timings[1],timings[2]))
            print ("Wavelengths:   {0:8.1f}   {1:8.1f}   {2:8.1f}".format(1e6*bandWave[0],1e6*bandWave[1],1e6*bandWave[2]))
            print ("Sensitivity to extended sources (MJy/sr)")
            print ("               {0:.2e}   {1:.2e}   {2:.2e}".format(extSrcFlux[0],extSrcFlux[1],extSrcFlux[2]))
            print ("Sensitivity to point sources (mJy)")
            print ("               {0:.2e}   {1:.2e}   {2:.2e}".format(pointSrcFlux[0],pointSrcFlux[1],pointSrcFlux[2]))
        
        return [extSrcFlux,pointSrcFlux]
        
        
        
        
        
        
        
