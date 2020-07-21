# This module contains all the elements needed to compute the sensitivity
# and thus the time estimation for B-BOP
#
# Hard-coded information that need to be updated if this python script is distributed
#   - The path to the location of the band-pass profiles (pathToFilter).
#
# This version is consistent with BBOP-DAp-RP-0002 v0.4
#
# Development history
# 27/09/2019 - I have verified now that I get the same prediction of the spectral
#              density for the foregrounds as shown by AP in his slides, yet I have
#              exactly a factor of 2 in the background predictions per pixel per filter
#              It seems AP shows the background spectrum summing both polarisations but
#              computes the foreground prediction for a single polarisation...
# 07/10/2019 - We have clarified that: because the pixels is covered with devices that
#              have absorbing dipoles in only one orientation (each of the 4 thermometer
#              carries only one orientation of absorbing dipoles), the pixel is 
#              effectively sensitive to only 1/2 of the power for non-polarized light.
# 08/10/2019 - Replace personal integration script by numpy.trapz that does exactly the 
#              same thing.
#              Still to be done for quality/robustness: turn all the common parts of the 
#              scripts into a single auxiliary script to be called by all, simpler to 
#              maintain, not completed yet.
# 14/10/2019 - Prediction of total power sensitivities now complete and verified to be
#              compatible with previous estimations (when using same initial values).
# 16/10/2019 - Corrected a printout in getNEP that always wrote that the goal NEP
#              was incorporated (when withDet=True). Computation was correct.
# 23/10/2019 - Implement the fact that the EEF at the area corresponding to the 4 central
#              pixels (a convention we use here) can be a function of the band.
# 07/11/2019 - Start the implementation of the polarisation part. For this I have
#              added a polar option to the getNEP function. Given that the meaning
#              of s/n is different in the total power and polarisation case, the
#              sensitivity computations are performed by two different functions.
#              Completed 15/11/2019
# 10/01/2020 - I start the conversion to python3 -> completed
# 13/02/2020 - telescope diameter was parameterised but not used.
# 02/04/2020 - Implementing the function that makes this into an actual time
#              estimator: input(target flux, S/N, map) output (obs Time)
# 06/04/2020 - I incorrectly assumed that one spiral branch in the detector (there are 4)
#              covered 1/4th of the pixel, whereas it covers 1/2 of the pixel.
#              This leads to many corrections.
# 12/05/2020 - As I was revising the equations I realized that the sensitivity
#              equations did not contain the optical tramsission applied to the
#              source.
# 14/05/2020 - Add a parameter to grossly mimic the effect of stray-light on the
#              background NEP. To use only with caution
# 02/06/2020 - Implement slew times consisting of:
#              initial acceleration to beging the map
#              the brake-step-accelerate times in between each scan leg
#              final decceleration at the end of the map
#              medium slew at the end of the map in case of repetition
#              leg to compute the total mapping time,
# 03/06/2020 - I start implementing modifications to tie the pixel size to the 
#              beam size so that the code automatically adjusts for future 
#              evolutions of the telescope aperture. Many elements of the code
#              are impacted by this change. I also implement the fact that the 
#              collecting area is now fixed by a requirement and thus the obscuration
#              is derived from the telescope diameter and the collecting area.
#              Given that the beam size is now properly computed, and that pixel
#              sizes may be adjusted to it, the field of view (bbopFoV), the 
#              diameter of an aperture that has the same surface as 4 pixels (diam4)
#              and the encircled energy fraction in that aperture (eefD4Band) are
#              no longer parameters of the code, but computed by the code.
#              08/06/2020 This is completed 
# 11/06/2020 - We now have an emissivity law for the telescope. I implement this
#              in the foreground computation and in the NEP computation.
#              this impact compTelForeground, mkForegroundFig, and compTelNEP
# 22/06/2020 - I start implementing modifications to allow for selection of the
#              background configuration between elicptic, galactic high,
#              Galactic low and extragalactic.
#              First modification affects foreground functions to output foreground
#              level at band center.
#              Then create a variable to select the background configurations and
#              modify the compXXForeground and compXXNep to work with it.
#              Add the functions to implement the ISM contribution to the 
#              foreground(compISMForeground and compISMNEP).
#              Make sure this component is now in getForeground and getNEP
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:49:21 2020

@author: sauvage
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import j1
from scipy.special import j0

import math

# print("This package contains the following functions")
# print("   getForeground       a simple utility summing all foreground contributions")
# print("   getNEP              a simple utility summing all NEP contributions")
# print("   compTelForeground   computes the telescope contribution to the foreground")
# print("   compISMForeground   computes the Galactic ISM contribution to the foreground")
# print("   compZodiForegroud   computes the zodiacal light contribution to the foreground")
# print("   compCIBForeground   computes the Cosmic Infrared Background contribution")
# print("   compCMBForeground   computes the Cosmic Microwace Background contribution")
# print("   comp4KForeground    computes the Fore Optics (4.8K) contribution to the foreground")
# print("   comp2KForeground    computes the Camera Optics (1.8K) contribution to the foreground")
# print("   filterTrans         returns the transmission of the filter of choice")
# print("   checkBandForError   common verification test")
# print("   reSample            is a utility function to resample a gridded function on another grid")
# print("   getBeamProfile      computes the beam profile and some of its properties")
# print("   getPixSizeBand      computes the optimal pixel size from the beam size")
# print("   getGeeomQuant       computes intermediate geometrical properties from pixel and beam sizes")
# print("   mkForegroundFig     makes a plot of the backgrounds spectral densities (verification)")
# print("   compTelNEP          computes the NEP associated to the telescope foreground")
# print("   compISMNEP          computes the NEP associated to the Galactic ISM foreground")
# print("   compZodiNEP         computes the NEP associated to the zodiacal light")
# print("   compCMBNEP          computes the NEP associated to the CMB")
# print("   compCIBNEP          computes the NEP associated to the CIB")
# print("   comp4KNEP           computes the NEP associated to the Fore Optics (at 4.8K)")
# print("   comp2KNEP           computes the NEP associated to the camera Optics (at 1.8K)")
# print("   compSrcPixPow       computes the power falling on a pixel from an astronomical source")
# print("   compPixPowSrc       computes the flux of a source from the power falling on the pixel")
# print("   convPow2Flux        converts a power (in W) to a flux density (mJy)")
# print("   getObsTime          computes timing information (on-source, total) for an observation")
# print("   getSensitivity      computes the sensitivity limit of an observation from S/N, area and obs mode")
# print("   getSensPolar        computes the sensitivity limit for polarised light")
# print("   getMyObsTime        computes the user obsering time to reach an input flux limit")



# Functions come and go, here is the list of functions that are still present but not used
#print("   getFwhm             is a utility that returns the FWHM of the beam at all central wavelengths")
#print("   getNEPOld           a simple utility summing all NEP contributions")

### Paramaters below can (or should) be adjusted by the user
# When using a distributed version of this script you need to update this path
#pathToFilter = '/Users/Sauvage/SPICA/Instrument B-BOP/Sensitivity/'
pathToFilter = os.getcwd()+'/'
# to the location where you have stored the file BandProfile.csv 

# Set this index to define which of the foureground cases to use 
# (0:Reference, 1:Ecliptic, 2:eXtragalactic, 3:Galactic Low, 4:Galactic High)
# The reference case is the original B-BOP case with no ISM component.
useFGCase = 0

# toggle this keyword to use default or adjusted pixel sizes. Adjusted meaning that
# we require a fixed number of pixels per FWHM and thus may need to afdjust their
# angular size when the telescope pqrqmeters evolve. We recomment to set it to 
# true
useDef = True
# Note to pro users: the code is faster if the pixel size is not computed in real
# time. You can use getPixSizeBand to find the pixel size adjusted to a given
# telescope diameter, put this values in defPixSizeBand and then run with
# useDef=True. Do this only if you understand how this code works!


### Changing parameters in this section is at your own risk and could lead to
### results inconsistent with the actual instrument/spacecraft
# Defines some constants and parameters for the whole code
# All units are SI
# physical constants (self-explanatory hopefully)
lightSpeed = 3e8
planck = 6.63e-34
bolzmann = 1.38e-23
hoverk = 4.80e-11

# Parameter definition section - To explore the effect of a parameter change there
# are two possibilities: either edit it below, and re-load the script (runfile), or
# change their value on the command line as they exist in the namespace. Some parameters
# however only impact other parameter definitions (e.g. obscFrac below) and thus the 
# second method will have no effect (dependent parameter is not recomputed).
#
# Foreground parameters
# Here we define an internal reference case for continuities with past studies.
# Since June 2020 we have standardized the forground cases throughout the
# instruments and so there are correcting factors on the emissivities
# There are five possible cases:
# 'R' is the reference case, 'E' is the ecliptic case, 'X' is the extragalactic case
# 'GL' is the Galactic Low Case and 'GH' is the Galactic High case
foreCases = ['R','E','X','GL','GH']
# Zodiacal light represented as a black body, T=265K em=1e-7
zodiTemp = 265.0
zodiEmissivity = 1e-7
zodiCorrEm = [1,1.80,1/2.36,1/1.12,1./1.63]
# Galactic ISM represented as a grey body, which parameters have been derived
# from a fit to the ISM component in the Euclid Background model. The emissivity
# has a wavelength dependency in -1.5
# The reference emissivity corresponds to the Extragal case (X)
# in our reference case there is no ISM contribution as it is the science target
ismTemp = 20.7
ismEmissivity = 1.4e-5
ismCorrEm = [0,6.8,1.,49.3,536.]
# Cosmic Infrared Background represented by a black body, T=30K, em=2.25e-6
# The CIB is isotropic so the correction factor to the emissivity does not
# depend on the foreground case. It turns out that our reference is significantly
# higher (but the CIB is not dominant).
cibTemp = 30.0
cibEmissivity = 2.25e-6
cibCorrEm = [1,0.69,0.69,0.69,0.69]
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
# This is decomposed into:
# brake = 10s
# step to next leg = 20s 
# accelerate = 10s
#turnAround = 40.0
timeBrakeAccel = 10.0
timeStep = 20.0
# In case of repetition of the map, we need to slew back to the beginning
# if the motion is larger than 5' this is a medium slow, otherwise a step
# a medium slew is 60s
timeMedSlew=60.0
# Initial overhead (before getting the first map data)
# As of this date this initial overhead should not be added to the observing time
# it is accounted for elsewhere
initOverhead = 0.0
# 02/06/2020 ambiguities remains: do we consider that the spacecraft starts and 
# finishes a map in a staring position? This adds another acceleration and 
# deceleration. Does the spacecraft returns at the beginning position at the end
# of the map (to be ready in case of repetitions)?

# Telescope parameters - ?S 26-09-19 AP indicated that the telescope parameters
# he has used (T=6k, e=0.09) are meant to represent the equivalent black-body 
# that would contribute the same flux as the actual telescope (primary, secondary,
# baffles and all). In that respect if I make a computation with my code, the 
# "number of mirrors" in that effective telescope should be 1 (and not 2 or 3 as
# in the "real" telescope).
# Telescope diameter
telDiam = 2.7
# Obscuration fraction - the fraction of the aperture that is blocked by structures
#obscFrac = 0.25
#    Effective area (used in beam etendue) with 25% obscuration
# This was wrong!! but with obsFrac=0.25 it does give the correct surface of 4.6 m2
#telEffArea = np.pi * ((telDiam/2.)**2 - (obscFrac*telDiam/2)**2)
# the telescope effective area is controlled by requirement MR0480 > 4.6m2
telEffArea = 4.6
# The obscuration fraction is the fraction of the geometrical aperture that is
# occupied by structures (e.g. secondary and its support). The obscuration fraction 
# is defined by this relation:
# Effective area = (1-obscFrac)*Geometrical area
obscFrac = 1. - (4*telEffArea)/(np.pi*telDiam**2)
# Note that defined this way obsFrac is the square of the obscuration fraction
# that enters the Airy formula (here it is a ratio of surfaces, for Airy it is
# a ratio of radii or diameters)
# Mirror temperatures assumed the same for all mirrors
telTemp = 8.0
# Mirror emissivity, assumed the same for all mirrors. We now have a law where the 
# emissivity is characterized as a function of wavelength. It is of the form
# em = Acoeff/sqrt(lambda) + Bcoeff/lambda + Ccoeff. We tabulate these coeffs
# the Ccoeff does not exist in the law but it is a trick to use a constant
# emissivity.
telACoeff = 0.01308
telBCoeff = 0.3276
telCCoeff = 0.0
# This is a case of a constant emissivity at 4.5% per mirror surface
#telACoeff = 0.0
#telBCoeff = 0.0
#telCCoeff = 0.045
# Number of mirrors in the telescope assembly 
telNumMirrors = 2

# B-BOP parameters - Do not change those unless you really know what you are doing!
# band definitions - MS I believe (and hope) that is is the only installation dependent
# element.
bandProFile = pathToFilter+'BandProfile.csv'
#bandProFile = pathToFilter+'BandProfileAP.csv'
# Pixel sizes in arc-second, warning band 1 is in pixBand[0]
# June 2020: now that we adjust pixel sizes to the beam there is an option to
# override this adjustment and use default values for the pixel sizes
defPixSizeBand = np.array([3.5,10,20])
# As a test this is AP's pixel sizes
#pixSizeBand = np.array([5,10,20])
# B-BOP's central wavelengths
bandWave = np.array([7e-5,2e-4,3.50e-4])
# As a test these are the AP central wavelengths
#bandWave = np.array([1.0e-4,2.0e-4,3.50e-4])
# Diameter of a circle that has the same surface as 2x2 pixels expressed 
# as function of the FWHM (of an unobscured 2.5m telescope)
# this is now computed from the pixel size and the actual beam profile
# diam4 = np.array([1.33,1.33,1.52])
# Fraction of a point source power that falls in this circle (assumption on beam shape)
# Given that pixels in band 3 are slightly larger than in the other two bands, w.r.t.
# the FWHM, the EEF in the same area is larger
# This is now computed from the pixel size and the actual beam profile
#eefD4Band = np.array([2./3.,2./3.,8./11.])
# B-BOP field of view in arcseconds - Now computed from the pixel size at band 2
# bbopFoV = 160.0
# Instrumental transmission (assuming filterbands are normalised to max=1)
instTrans = 0.5
# Internal cold stop transmission
etaColdStop = 1.0
# Number of 4.8K (fore-optics) optical surfaces (this includes the pick-off mirror)
num4KSurf = 7
# Fore optic temperature
foreOptTemp = 4.8
# emissivity of these surfaces (should be 0.03)
foreOptEmissivity = 0.03
# transmission from the fore optics to the detectors 
foreOptTrans = instTrans * 1.0
# Number of 1.8K (camera optics) optical surface (this is an average number per channel)
# This has changed after the re-design following the vertical telescope
# exact number is 8 for 70 mic, 5 for 200 mic and 6 for 350 mic
# Note that this component is never important...
#num2KSurf = 5 - This was the reprensentative number prior to telescope KDP
num2KSurf = 6
# temperature of the camera optics
camOptTemp = 1.8
# emissivity of these surfaces (should be 0.03)
camOptEmissivity = 0.03
# transmission from the camera optics to the detectors
camOptTrans = instTrans * 1.0
# This represents the fraction of the pixel geometric solid angle that is 
# covered by one branch of the bolometer (there are 4 branches overlapping)
pixFrac = 0.5
# Detector NEP
detNEPReq = 3e-18
detNEPGoal = 1.5e-18
# Stray light NEP multiplicative factor - it is applied to the thermal NEP
# components in getNEP. Set is to 1.0 to null this effect
strayLightFac = 1.0

#######################################################
# Short utility function to get the complete backgroud contribution
#######################################################
def getForeground(band,verb=False):
    """"
    This code simply aggregates all the "foreground" source computations
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    result = compTelForeground(band)+compZodiForeground(band)+\
             compCMBForeground(band)+compCIBForeground(band)+\
             compISMForeground(band)+\
             comp4KForeground(band)+comp2KForeground(band)
    
    if (verb):
        print("Incident Foreground Power: {0:.2e} W".format(result))     
    return result
   
#######################################################
# Short utility function to get the complete background NEP contribution
#######################################################
def getNEP(band,withDet=False,goal=False,silent=False):
    """"
    This code simply aggregates all the "NEP" computations.
    Important note: to understand why we no longer make a difference in the NEP
    to consider whether one is working in total power or polarisation, as well as
    why we no longer make a difference between incident and absorbed NEP it is
    better to read BBOP-DAp-RP-0002.
    We recall that for B-BOP the pixel is  fully covered by absorbing devices:
        - a pixel is made of 4 absorbing thermometers, each occupying 1/2 of the
          pixel surface (there is a clever overlapping structure).
        - each thermometer is covered by absorbing dipoles that are all oriented
          in the same direction. Therefore the thermometer "filters" the incoming
          power according to its polarisation state. Because polarisation fraction 
          are small in practice, a thermometer is only sensitive to approximately 
          half of the power that falls within its "reach", and the comXxxNEP 
          functions take that into account.
    
    Finally, a measurement with B-BOP always take into account the signal from 
    the 4 thermometers, either in addition or in subtraction. The NEP from each
    thermometer is thus added quadratically to provide the total thermal NEP in
    each case.
    
    withDet     if True the detector NEP will be added quadratically to the 
                thermal compoment, the default being the requirement value.
    goal        if True the goal value of the detector (2 times better) is used.
    silent      if True the code does not print anything
    22/04/2020 - MS - This is the new NEP compilation code
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # develop this to introduce a zero-order way of treating stray light, i.e.
    # that it acts as a multiplicative factor on the NEP
    resultSq = compTelNEP(band,debug=False)**2+compZodiNEP(band,debug=False)**2+\
               compCMBNEP(band,debug=False)**2+compCIBNEP(band,debug=False)**2+\
               compISMNEP(band,debug=False)**2+\
               comp4KNEP(band,debug=False)**2+comp2KNEP(band,debug=False)**2
    # introduce a zero-order way of treating stray light, i.e.
    # that it acts as a multiplicative factor on the NEP
    resultSq *= strayLightFac**2
    # since these are the NEP per thermometer, and that 4 of them contribute 
    # to the measurement
    resultSq *= 4
    
    if (withDet):
        if (goal):
            if not(silent):
                print("Total NEP (Including detector goal NEP - W.Hz^-1/2)")
            return math.sqrt(resultSq + detNEPGoal**2)
        else:
            if not(silent):
                print("Total NEP (Including detector requirement NEP - W.Hz^-1/2)")
            return math.sqrt(resultSq + detNEPReq**2)
    else:
        if not(silent):
            print("Thermal NEP (only due to the background components - W.Hz^-1/2)")
        return math.sqrt(resultSq)
    

#######################################################
# Short utility function to get the complete background NEP contribution - This version is deprecated
#######################################################
def getNEPOld(band,incident=True,polar=False,withDet=False,goal=False):
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
    otherwise if goal is True the gaol value (2 times better) is used.
    When polar is set to true, we consider that what is requested is the NEP associated to
    one of the dipole grid. This introduces a further correction because not only these grid
    are sensitive to only 1/2 of the radiation, they also only cover 1/2 of the pixel. So
    this is another sqrt(2) applied to the background components.
    22/04/2020 - MS - This is the old NEP compilation code, that was consistent with 
    the erroneous representation that the "polariser" pixels had only half the solid
    angle of the geometrical pixel.
    Since in correcting that I have changed pixAbsFrac into pixFrac I make this
    change in the code so that it can still be run (for what it's worth)
    """
    # Print a warning
    print("Running the deprecated version of getNEP")
    
    # First perform a input check
    if (checkBandForError(band) == False):
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
        if (polar):
            print("NEP associated to incident foreground power in one polarisation component")
            return math.sqrt(resultSq/2.)
        else:
            print("NEP associateed with Incident Foreground Power (total power mode)")       
            return math.sqrt(resultSq)
    else:
        if (withDet):
            if (goal):
                if (polar):
                    print("Polarised componenet NEP (Including detector goal NEP)")
                    return math.sqrt(resultSq*pixFrac/2. + detNEPGoal**2)
                else:
                    print("Total power NEP (Including detector goal NEP)")
                    return math.sqrt(resultSq*pixFrac + detNEPGoal**2)
            else:
                if(polar):
                    print("Polarised compoenent NEP (Including detector requirement NEP)")
                    return math.sqrt(resultSq*pixFrac/2. + detNEPReq**2)
                else:
                    print("Total power NEP (Including detector requirement NEP)")
                    return math.sqrt(resultSq*pixFrac + detNEPReq**2)
        else:
            if (polar):
                print("Effective foreground polarized component NEP (that which pixels are sensitive to)")
                return math.sqrt(resultSq*pixFrac/2.)
            else:
                print("Effective Foreground total power NEP (that which pixels are sensitive to)")
                return math.sqrt(resultSq*pixFrac)

#######################################################
# This function computes the power collected in the pixel
# coming from the telescope mirrors
#######################################################
def compTelForeground(band,debug=False,verb=False):
    """
    This code returns the telescope thermal contribution in a given band.
    It is using SI units therefore this should be in W.
    It takes only one parameter that is the filter band, 1, 2 or 3
    It computes the power that falls within the full pixel solid angle
    as coming from the telescope, irrespective of whether or not it is 
    polarised.
    if Debug is true, will plot the foreground spectrum
    if Verb is true, will print the foreground level at band center
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=debug)
    
    # build the emissivity function
    # remember em = telACoeff/sqrt(lambda) + telBCoeff/lambda +telCCoeff
    # with lambda in microns
    telEmissivity = telACoeff/np.sqrt(1e6*lightSpeed/waveGrid)+\
                    telBCoeff/(1e6*lightSpeed/waveGrid)+\
                    telCCoeff
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / telTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * telEmissivity[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * telEmissivity[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * telEmissivity[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
        
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
    if (verb):
        idx = np.abs(waveGrid - 3e8/bandWave[band-1]).argmin()
        fg = powSpecDens[idx] / 1e-20
        print('Foreground at {0:6.2f} micron is {1:5.2g} MJy/sr'.format(bandWave[band-1]*1e6,fg))
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,powSpecDens)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1}$')
   
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    # Emissivity is now a function of frequency
    # power *= telEmissivity
    power *= instTrans
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
    
    if (debug):
        print("Beam Etendue is: {:5.2g}".format(telEffArea*pixSolidAngle))
    
    # Finally we need to account for the fact that the telescope is comprised of
    # a number of emitting surfaces so that
    power *= telNumMirrors
    
    return power

#######################################################
# This function computes the power collected in the pixel
# coming from the ISM
#######################################################
def compISMForeground(band,debug=False,verb=False):
    """
    This code returns the Galactic ISM  contribution in a given band.
    It is using SI units therefore this should be in W.
    It takes only one parameter that is the filter band, 1, 2 or 3
    It computes the power that falls within the full pixel solid angle
    as coming from the ISM, irrespective of whether or not it is 
    polarised.
    if Debug is true, will plot the foreground spectrum
    if Verb is true, will print the foreground level at band center
    The shape of this emission has been obtained from a fit to the ISM
    Component in the Euclid Background Model.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.
    
    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=debug)
    
    # build the emissivity function
    emissivity = ismEmissivity * ismCorrEm[useFGCase] * (waveGrid/1.5e12)**1.5
    
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / ismTemp)
    
    # as math.exp only takes a scalar as an argument I need a loop here
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(filtBandpass[i] * emissivity[i]\
                               * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * emissivity[i]\
                               * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * emissivity[i]\
                               * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
        
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
    if (verb):
        print('Using Foreground prescription: ',foreCases[useFGCase])
        idx = np.abs(waveGrid - 3e8/bandWave[band-1]).argmin()
        fg = powSpecDens[idx] / 1e-20
        print('Foreground at {0:6.2f} micron is {1:5.2g} MJy/sr'.format(bandWave[band-1]*1e6,fg))
    
    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,powSpecDens)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1}$')
   
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    power *= instTrans
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
    
    if (debug):
        print("Beam Etendue is: {:5.2g}".format(telEffArea*pixSolidAngle))
        
    return power

#######################################################
# This function computes the zodiacal light contribution
# to the foreground emission
#######################################################
def compZodiForeground(band,debug=False,verb=False):
    """
    This function returns the contribution of the zodiacal light in the
    foreground collected by a band.
    It has a single parameter, the band that should be 1,2, or 3
    The zodiacal light is considered a black-body filling the beam and its
    parameters (temperature and emissivity) are defined in the parameter 
    section.
    It computes the power that falls within the full pixel solid angle
    as coming from the zodiacal light, irrespective of whether or not it is 
    polarised.

    """
    # MS - Because we treat the zodiacal light as a grey-body, this code is
    #      formally identical to that which computes the emission of the 
    #      telescope mirrors.
    #      This is also due to the assumption that we consider that the 
    #      transmission factor due to the telescope mirrors is negligible
    #      with respect to that due to the instrument, so we have applied
    #      the same value to a light source on the telescope or beyond.
    # 02/06/2020 Danger of copy/paste. in the plot the emissivity was that
    #      of the telescope..

    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return

    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.
    
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
            powSpecDens.append(filtBandpass[i] * zodiEmissivity*zodiCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * zodiEmissivity*zodiCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * zodiEmissivity*zodiCorrEm[useFGCase]\
                               * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
    if (verb):
        print('Using Foreground prescription: ',foreCases[useFGCase])
        idx = np.abs(waveGrid - 3e8/bandWave[band-1]).argmin()
        fg = powSpecDens[idx] / 1e-20
        print('Foreground at {0:6.2f} micron is {1:8.4g} MJy/sr'.format(bandWave[band-1]*1e6,fg))

    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,powSpecDens)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1}$')

    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
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
def compCIBForeground(band,debug=False,verb=False):
    """
    This function returns the contribution of the Cosmic Infrared Background in the
    foreground collected by a band.
    It has a single parameter, the band that should be 1,2, or 3
    The zodiacal light is considered a black-body filling the beam and its
    parameters (temperature and emissivity) are defined in the parameter 
    section.
    It computes the power that falls within the full pixel solid angle
    as coming from the CIB, irrespective of whether or not it is 
    polarised.    
    """
    # MS - Because we treat the CIB as a grey-body, this code is
    #      formally identical to that which computes the emission of the 
    #      telescope mirrors.
    #      This is also due to the assumption that we consider that the 
    #      transmission factor due to the telescope mirrors is negligible
    #      with respect to that due to the instrument, so we have applied
    #      the same value to a light source on the telescope or beyond.
    # 02/06/2020 Danger of copy/paste. in the plot the emissivity was that
    #      of the telescope..

    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
        
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
            powSpecDens.append(filtBandpass[i] * cibEmissivity*cibCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * cibEmissivity*cibCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * cibEmissivity*cibCorrEm[useFGCase]\
                               * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))

    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    
    if (verb):
        print('Using Foreground prescription: ',foreCases[useFGCase])
        idx = np.abs(waveGrid - 3e8/bandWave[band-1]).argmin()
        fg = powSpecDens[idx] / 1e-20
        print('Foreground at {0:6.2f} micron is {1:5.2g} MJy/sr'.format(bandWave[band-1]*1e6,fg))

    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,powSpecDens)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1}$')
    
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by the terms in the formula that had no dependency on frequency
    power *= instTrans
    power *= etaColdStop
    power *= telEffArea
    pixSolidAngle = ((np.pi*pixSizeBand[band-1])/(3600.*180.))**2
    power *= pixSolidAngle
    
    return power

#######################################################
# This function computes the Cosmic Microwave Background contribution
# to the foreground emission
#######################################################
def compCMBForeground(band,debug=False,verb=False):
    """
    This function returns the contribution of the Cosmic Microwave Background in the
    foreground collected by a band.
    It has a single parameter, the band that should be 1,2, or 3
    The zodiacal light is considered a black-body filling the beam and its
    parameters (temperature and emissivity) are defined in the parameter 
    section.
    It computes the power that falls within the full pixel solid angle
    as coming from the CMB, irrespective of whether or not it is 
    polarised.
    """
    # MS - Because we treat the CIB as a grey-body, this code is
    #      formally identical to that which computes the emission of the 
    #      telescope mirrors.
    #      This is also due to the assumption that we consider that the 
    #      transmission factor due to the telescope mirrors is negligible
    #      with respect to that due to the instrument, so we have applied
    #      the same value to a light source on the telescope or beyond.
    # 02/06/2020 Danger of copy/paste. in the plot the emissivity was that
    #      of the telescope..

    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
            powSpecDens.append(filtBandpass[i] * cmbEmissivity * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(filtBandpass[i] * cmbEmissivity * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(filtBandpass[i] * cmbEmissivity * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
                               
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
     
    if (verb):
        idx = np.abs(waveGrid - 3e8/bandWave[band-1]).argmin()
        fg = powSpecDens[idx] / 1e-20
        print('Foreground at {0:6.2f} micron is {1:5.2g} MJy/sr'.format(bandWave[band-1]*1e6,fg))

    if (debug):
        plt.figure(figsize=(12,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.plot(1e6*lightSpeed/waveGrid,powSpecDens)
        plt.xlabel('Wavelength in $\mu$m')
        plt.ylabel('Brightness in W.m$^{-2}$.sr$^{-1}$.Hz$^{-1}$')
    
    # now integrate that function
    power = np.trapz(powSpecDens,x=waveGrid)
    # multiply by other instrumental terms in the formula that had no dependency on frequency
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
    This code returns the thermal contribution of the 4K fore optics in a given band.
    It is using SI units therefore this should be in W.
    It takes only one parameter that is the filter band, 1, 2 or 3
    It computes the power that falls within the full pixel solid angle
    as coming from the fore optics, irrespective of whether or not it is 
    polarised.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
# coming from the 2k mirrors, which constitute the camera-optics
#######################################################
def comp2KForeground(band):
    """
    This code returns the thermal contribution of the 2K camera optics in a given band.
    It is using SI units therefore this should be in W.
    It takes only one parameter that is the filter band, 1, 2 or 3
    It computes the power that falls within the full pixel solid angle
    as coming from the camera optics, irrespective of whether or not it is 
    polarised.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    if (worf != 'f') and (worf != 'w'):
        print("ERROR - worf must either be w for wavelength or f for frequency")
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
    reader.__next__()
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
        print("ERROR - The only acceptable values for band are 1, 2, or 3")
        return False
    else:
        return True
    

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
    print("outData = reSample(outGrid,inGrid,inData,force=T/F)")
    print("")
    print("where")
    print("")
    print("inGrid    is the input grid on which the data is currently sampled")
    print("inData    is the data (can be 1,2, or 3D)")
    print("          Note that I assume that the first axis of inData is the sampling axis")
    print("outGrid   is the new grid on which you want to resample the data")
    print("force     if True (default) will force extrapolation when the new grid is")
    print("          wider that the input grid")
    print("")
    print("WARNING   this code uses interpolate.interp1d which requires the grids to be")
    print("          increasing with index.")
    print("")
    print("all inputs must be numpy arrays, I assume that the grid contains the sampling nodes")
    print("arranged in ascending order.")
    return

  # First let's get the dimensions of the inputs
  nOrig = len(inGrid)
  nNew = len(outGrid)

  # check that the grid is indeed increasing.
  if (inGrid[-1] <= inGrid[0]):
    print("ERROR -- the interpolation works only for grids that are increasing with index")
    return
  
  # now check that the data array has its first dimension corresponding to the same
  # dimension as the grid array
  if (nOrig != inData.shape[0]):
    print("ERROR -- The input grid and the data are incompatible in size")
    print("ERROR -- I expect that the first dimention of the data is the sampling axis")
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
    print("ERROR -- The data has more than 3 dimensions, and I don't know how to deal with it")
    return

  # interpolation functions will have a hard time working if the new grid is larger than the
  # original one. I check whether this is the case and if so, and if the user has selected the
  # force option, I pad the original data with one more sample point beyond the new grid. The
  # value placed here is the same as the first or last value in the data cube.
  if (outGrid.min() < inGrid.min()) or (outGrid.max() > inGrid.max()):
    # check whether the user is aware of that
    if (force==False):
      print("WARNING -- The output grid is wider than the input grid.")
      print("WARNING -- IF you really want to extrapolate your data, use force=True")
      return
    else:
      if (verbose):
        print("INFO -- The output grid is wider than the input grid.")
        print("INFO -- Extrapolating the data to fix this")
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

###################################################################
# Computes the telescope's FWHM. This is straightforward but as
# it is done at different places in the code, it's better to put
# this in a function
# This will be replaced by a more accurate computation in the function
# getBeamProfile
###################################################################
def getFwhm(diameter,wavelengths):
    """
    This functions compute the FWHM for a pure airy profile
    All units must be SI
    """
    print("You are aware that this function is deprecated?")
    # Note: the FWHM is not the diameter of the first dark ring
    fwhm = 1.028*wavelengths/diameter
    # convert to arcseconds
    fwhm /= math.pi
    fwhm *= 180.
    fwhm *=3600
    
    return fwhm

#####################################################################
# Computes the telescope beam we will need this beam profile in order
# to compute a number of elements in the code
#####################################################################
def getBeamProfile(full=False,fwhm=False,rayleigh=False,plot=False):
    """
    This function computes the beam profile based on the telescope diameter
    and the central obscuration.
    One can control the output with the following keywords (there is no default):
        full      returns all the computed elements (angle,beam,eef,fwhm,rayleigh)
        fwhm     returns only the FWHM
        rayleigh returns only the Rayleigh criterion
    """
    # 08/06/202 - I make a change: I was computing the profiles (beam and airy)
    # for a single central wavelength. But the Bessel functions have inherent
    # scaling so a profile from one wavelength can be deduced from another
    # using the proper relation on angle.
    # To avoid messing things up, I'm writing a temporary new function.
    
    refWave = bandWave[1]
    # The function will be computed for this reference band. Then all the 
    # other elements will be derived from the scaling.
    
    # define the x variable. This is (2pi/lambda).sin0
    # no need to explore it too far e.g. pi/1000 should be OK
    # (this is close to 10' off axis)
    # since the Airy function uses 1/angle, I avoid 0
    numElts = 5000
    angle = np.linspace(1e-6,np.pi/1000.,numElts)
    
    # define the obscuration fraction as it is expected for the Airy functions
    # a ratio of lengths, not of surfaces
    obs = math.sqrt(obscFrac)
    
    #computes the function variable - Warning: the variable integrates the 
    # radius of the aperture not its diameter
    var = 2*np.pi*(telDiam/2.)/refWave * np.sin(angle)
    
    # These formulae correspond to the ideal unobscured Airy case
    #IdAiry = (2*j1(angle)/angle)**2
    #IdEef = 1 - j0(angle)**2 - j1(angle)**2
    
    # compute the obscured Airy function.
    norm1 = 1./(1-obs**2)**2
    obAiry = norm1 * (2*j1(var)/var - obs*2*(j1(obs*var)/var))**2
    norm2 = 1./(1-obs**2)
    obEef = norm2 * (1 - j0(var)**2 - j1(var)**2 +\
                     obs**2 * (1 - j0(obs*var)**2 - j1(obs*var)**2))
    # There is an additional component to the EEF in the case of obscuration
    # that requires integration of the product of Bessel Functions!
    corrFac = np.zeros(numElts)
    for i in range(numElts):
        if (i == 0):
            # we need to treat the first element differently
            corrFac[i] = j1(var[i])*j1(obs*var[i])/2
        else:
            corrFac[i] = corrFac[i-1]+\
            (var[i]-var[i-1])*j1(var[i])*j1(obs*var[i])/var[i]
               
    obEef = obEef - norm2*4*obs*corrFac
    
    # At this stage we have angle, obAiry, and obEef that are consistent with
    # one another. Both the profile and the encircled energy fraction as also
    # valid for the two other wavelength provided that I rescale the angle variable
    # Thus I only need to create an array of 3xangle to represent the proper 
    # angular variable per wavelength.
    theAngles = np.zeros([3,numElts])
    theAngles[0,:] = np.arcsin(bandWave[0]/bandWave[1]*np.sin(angle))
    theAngles[1,:] = angle
    theAngles[2,:] = np.arcsin(bandWave[2]/bandWave[1]*np.sin(angle))
    
    
    # compute some parameters of this profile (all converted to arcseconds)
    # full width at half maximum
    halfMax = max(obAiry)/2.
    belowHM = [i for i in range(len(obAiry)) if obAiry[i]<=halfMax]
    fullWHM = np.array([2*theAngles[0,belowHM[0]]*(3600*180/np.pi),\
                     2*theAngles[1,belowHM[0]]*(3600*180/np.pi),\
                     2*theAngles[2,belowHM[0]]*(3600*180/np.pi)])
    # radius of the first dark ring - Also known as Rayleigh criterion
    # I work on the derivative of the Airy profile. The location of the first
    # dark ring is the first time this derivative ceases to be negative
    # Carefull: there is one less element in the derivative of an array than 
    # in the array itself
    dAiry = np.diff(obAiry)
    beyondDark = [i for i in range(len(obAiry)-1) if dAiry[i]>=0]
    radiusDark = np.array([theAngles[0,beyondDark[0]]*(3600*180/np.pi),\
                           theAngles[1,beyondDark[0]]*(3600*180/np.pi),\
                           theAngles[2,beyondDark[0]]*(3600*180/np.pi)])
    
    
    if (plot):
        # This plots everything in log log
        plt.figure(figsize=(12,6))
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True,which='both')
        plt.ylim(1e-7,1)
        plt.xlim(180*1e-6/math.pi,1.5)
        for i in range(len(bandWave)):
            plt.plot(180*theAngles[i,:]/math.pi,obAiry,\
                     label="{:.1f} $\mu$m obscured (".format(1e6*bandWave[i])+\
                     "{:.2f}".format(obs)+") Airy Profile")
            plt.plot(180*theAngles[i,:]/math.pi,obEef,\
                     label="{:.1f} $\mu$m obscured (".format(1e6*bandWave[i])+\
                     "{:.2f}".format(obs)+") Encircled Energy Fraction")
        plt.xlabel("radius ($^{\circ}$)")
        plt.ylabel("Normalized intensity")
        plt.legend(loc=0,numpoints=1,shadow=True)
        # draw vertical lines at key locations (0.5FWHM, 1.0 FWHM, 3.0FWHM)
        # The FWHM is that of the unobscured aperture, 1.028 Lambda/D
        # as angle/pi = lambda/D, in the plots the lines are located at
        # abscissa 0.514, 1.028 and 3.084.
        #plt.axvline(x=0.514)
        #plt.axvline(x=1.028)
        #plt.axvline(x=3.084)
        
        # this plots the central 5' in lin-lin
        # This plots everything in log log
        plt.figure(figsize=(12,6))
        plt.grid(True,which='both')
        plt.ylim(0,1)
        plt.xlim(0,50)
        for i in range(len(bandWave)):
            plt.plot(180*theAngles[i,:]/math.pi*3600,obAiry,\
                     label="{:.1f} $\mu$m obscured (".format(1e6*bandWave[i])+\
                     "{:.2f}".format(obs)+") Airy Profile")
            plt.plot(180*theAngles[i,:]/math.pi*3600,obEef,\
                     label="{:.1f} $\mu$m obscured (".format(1e6*bandWave[i])+\
                     "{:.2f}".format(obs)+") Encircled Energy Fraction")
        plt.xlabel('radius (")')
        plt.ylabel("Normalized intensity")
        plt.legend(loc=0,numpoints=1,shadow=True)
        
        print('For bands centered at:')
        print("For bands centered at: "\
              "{0:8.1f}   {1:8.1f}   {2:8.1f}".format(1e6*bandWave[0],1e6*bandWave[1],1e6*bandWave[2]))
        print('full width at half maximum is ("): '\
              '{0:.2f}   {1:.2f}   {2:.2f}'.format(fullWHM[0],fullWHM[1],fullWHM[2]))
        print('Radius of the first dark ring, or Rayleigh criterion, is ("):'+\
              '{0:.2f}  {1:.2f}  {2:.2f}'.format(radiusDark[0],radiusDark[1],radiusDark[2]))
    

    if (full):
        return(theAngles,obAiry,obEef,fullWHM,radiusDark)
    elif (fwhm):
        return fullWHM
    elif (rayleigh):
        return radiusDark
    else:
        return

###########################################################
# This code computes the pixel size per band from the beam  
###########################################################
def getPixSizeBand(over=False,verb=False):
    """
    We compute here the pixel size. This is done because the telescope diameter
    is still not fixed and we will adjust our optical design so that we sample
    the beam appropriately.
    On top of a sampling criterium we also try to accomodate for a similar fov in
    all bands, considering that there is always a ratio of 4 in the number of 
    pixels per band
    if keyword over is set to true, the pixel size computation is overidden
    and replaced by the default size set in the preamble
    """
    
    if (over):
        # the user requests overriding the pixel computation to use the default
        optPix = defPixSizeBand
        newPix = defPixSizeBand
    else:
        # first I obtain the Rayleigh criterion for the whole bands
        rayleigh = getBeamProfile(rayleigh=1)
        # the Rayleigh criterion is given in arcsec
        # We consider that we want to have at least 2 pixels within the rayleigh
        # criterion, but we also need to have some reasonable scaling between 
        # the bands. In particular, the pixel size of band 3 should be twice that
        # of band 2.
        # the optimal pixel size has a width half the Rayleigh criterion.
        optPix = rayleigh/2.
        # We don't want pixels with weird surfaces and we'd rather sacrifice some
        # sampling to increased sensitivity
        newPix = np.round(rayleigh)/2.
        for i in range(len(optPix)):
            if (newPix[i]<optPix[i]):
                newPix[i] +=0.5
        # and we want the pixel size of band three to be exactly twice that of
        # band 2
        newPix[2]=2*newPix[1]
        
    if (verb):
        print("Optimal pixel size:",optPix)
        print("New pixel size:    ",newPix)
    
    return newPix
    
#############################################################################
# This function computes geometrical properties necessary for the photometry
############################################################################
def getGeomQuant(pixSizeBand):
    """
    Here we compute geometrical quantities that depend on the beam and on the 
    pixel size and are used in photometry.
    These are:
        bbopFov      the field of view 16 times the pixel size of band.
        diam4        the diameter of a circle that has the same surface as 4 pixels
        eefD4Band    the encircled energy fraction of the beam in that circle
    While we are going to call the beam profile function, we assume that the
    pixel sizes have been set by the calling function.
    """
    
    # the bbop field of view is made of 16 pixels in band 2
    bbopFoV = 16*pixSizeBand[1]
    # diam4 is the diameter of a circle that would have the same surface as 4
    # pixels, expressed in units of the FWHM
    # first compute it in arcsec
    diam4 = 4*pixSizeBand/np.sqrt(np.pi)
    # Now I need the FWHM, and since the same code returns the EEF I make a single
    # call
    data = getBeamProfile(full=1)
    theAngles = data[0]
    eef = data[2]
    fwhm = data[3]

    eefD4 = []
    for i in range(len(pixSizeBand)):        
        # find the encircled energy fraction here.
        idx = np.abs(theAngles[i,:]-np.pi*(diam4[i]/2)/(3600*180)).argmin()
        eefD4.append(eef[idx])
                    
    # Convert the eef list into an array
    eefD4 = np.array(eefD4)
    # express the diameter as a function of the FWHM
    diam4 /= fwhm
    
    return bbopFoV,diam4,eefD4
    
    
#########################################
# make the foreground plot
#########################################
def mkForegroundFig(totOnSep=False,save=False):
    """
    This code makes a plot of the foreground
    before they are integrated in the pixel
    Foreground computations are cut/paste from the functions
        save = True        will save the figures in a file
        totOnSep = True    will superimpose the total background of the figure 
                           that separates them
    """
    
    
    # Get the CMB bacground value
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
    # build the emissivity function
    # remember em = telACoeff/sqrt(lambda) + telBCoeff/lambda +telCCoeff
    # with lambda in microns
    telEmissivity = telACoeff/np.sqrt(1e6*lightSpeed/waveGrid)+\
                    telBCoeff/(1e6*lightSpeed/waveGrid)+\
                    telCCoeff
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / telTemp)    
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(telEmissivity[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(telEmissivity[i] * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(telEmissivity[i] * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    telSpecDens = np.asarray(powSpecDens)
    # multiply by the number of mirrors
    telSpecDens *= telNumMirrors
    
    # build the emissivity function
    ismEm = ismEmissivity * ismCorrEm[useFGCase] * (waveGrid/1.5e12)**1.5
    # now let's build the function under the integral
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / ismTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(ismEm[i] * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(ismEm[i]* twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(ismEm[i]* twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    ismSpecDens = np.asarray(powSpecDens)
            
        
    # convert it as an array
    powSpecDens = np.asarray(powSpecDens)
    # Zodiacal light contribution
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / zodiTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(zodiEmissivity*zodiCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(zodiEmissivity*zodiCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(zodiEmissivity*zodiCorrEm[useFGCase]\
                               * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
    # convert it as an array
    zodiSpecDens = np.asarray(powSpecDens)
    
    # CIB contribution
    twoHNu3C2 = 1.47e-17 * waveGridNorm**3
    hNuKT = hoverk * (waveGrid / cibTemp)
    powSpecDens = []
    for i in range(numEltsGrid):
        # avoid divergence in computation
        if (hNuKT[i]<1e-3):
            powSpecDens.append(cibEmissivity*cibCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1./hNuKT[i]))
        elif (hNuKT[i]<100.):
            powSpecDens.append(cibEmissivity*cibCorrEm[useFGCase]\
                               * twoHNu3C2[i] * (1. / (math.exp(hNuKT[i])-1)))
        else:
            powSpecDens.append(cibEmissivity*cibCorrEm[useFGCase]\
                               * twoHNu3C2[i] * math.exp(-1*hNuKT[i]))
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
    plt.plot(lightSpeed/waveGrid,ismSpecDens,label='Galactic ISM')
    plt.plot(lightSpeed/waveGrid,cibSpecDens,label='CIB')
    plt.plot(lightSpeed/waveGrid,foreOptSpecDens,label='Fore Optics')
    plt.plot(lightSpeed/waveGrid,camOptSpecDens,label='Camera Optics')
    if (totOnSep):
        totBkg = cmbSpecDens+telSpecDens+zodiSpecDens+cibSpecDens+\
                     ismSpecDens+foreOptSpecDens+camOptSpecDens
        plt.plot(lightSpeed/waveGrid,totBkg,\
             label='Total Background (TelT='+str(telTemp)+'K)')

    plt.xlabel('Wavelength (m)')
    plt.ylabel('Background spectral density (W.m$^{-2}$.Hz$^{-1}$.sr$^{-1}$)')
    plt.title('Foregrounds in the B-BOP band using case '+foreCases[useFGCase])
    plt.xlim(1e-5,1e-3)
    plt.ylim(1e-22,5e-17)
    plt.legend(loc='upper left',numpoints=1,shadow=True)
    for i in range(3):
        plt.axvspan(lightSpeed/bandMin[i],lightSpeed/bandMax[i],\
                    color=bandCol[i],alpha=0.2)
    if (save):
        print("File will be written in "+os.getcwd())
        fileName = os.getcwd()+'/EachBackgrounds'+str(telTemp)+'K.png'
        plt.savefig(fileName)


    totBkg = cmbSpecDens+telSpecDens+zodiSpecDens+cibSpecDens+\
             ismSpecDens+foreOptSpecDens+camOptSpecDens
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
        print("File will be written in "+os.getcwd())
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
    10/04/2020 - I modify the computation, I am now computing the NEP that is
    relevant for one bolometer branch (out of 4) of the detector. The main change
    is that a branch is sensitive to only one polarisation, and it is covering
    1/2 of the pixel area. The proper summation of NEPs will have to be done 
    in function getNEP.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # build the emissivity function
    # remember em = telACoeff/sqrt(lambda) + telBCoeff/lambda +telCCoeff
    # with lambda in microns
    telEmissivity = telACoeff/np.sqrt(1e6*lightSpeed/waveGrid)+\
                    telBCoeff/(1e6*lightSpeed/waveGrid)+\
                    telCCoeff

    
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
        nepConst = instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * telEmissivity[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * telEmissivity[i] * expTermInt))

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
    # Take into account that one bolometer branch fills only pixFrac of the full
    # pixel angle
    nepSquare *= (pixSolidAngle*pixFrac)
    
    # account for the existence of two polarisation state per photon energy state
    # 10/04/2020 I no longer return the NEP corresponding to the two polarisation
    # I return the NEP per branch of the bolometers.
    # nepSquare *=2
        
    # Finally we need to account for the fact that the telescope is comprised of
    # a number of emitting surfaces so that we have to add as many NEP^2 as there
    # are surfaces.
    nepSquare *= telNumMirrors
    
    return math.sqrt(nepSquare)

#######################################################
# This function computes NEP associated the telescope mirrors
# thermal emission
#######################################################
def compISMNEP(band,debug=True):
    """
    This code returns the Galactic ISM contribution to the NEP in a given band.
    It is using SI units therefore this should be in W.Hz^-1/2.
    It takes only one parameter that is the filter band, 1, 2 or 3
    10/04/2020 - I modify the computation, I am now computing the NEP that is
    relevant for one bolometer branch (out of 4) of the detector. The main change
    is that a branch is sensitive to only one polarisation, and it is covering
    1/2 of the pixel area. The proper summation of NEPs is done in function getNEP.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

    # Now define the function that we will need to integrate
    # first define the grid. We will use a single frequency grid (overkill)
    # we cover the range 30mic->500mic, corresponding to 600 GHz to 10 THz
    numEltsGrid = 10000
    waveGrid = np.linspace(6e11,1e13,num=numEltsGrid)
    # same but normalized by 10^11
    waveGridNorm = np.linspace(6.,100.,num=numEltsGrid)
    
    # Get the filter transmission for this grid
    filtBandpass = filterTrans(band,waveGrid,worf='f',debug=False)
    
    # build the emissivity function
    emissivity = ismEmissivity * ismCorrEm[useFGCase] * (waveGrid/1.5e12)**1.5
    
    # now let's build the function under the integral
    twoH2Nu4C2 = 9.74e-40 * waveGridNorm**4
    hNuKT = hoverk * (waveGrid / ismTemp)
    
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
        nepConst = instTrans * etaColdStop
        nepSqInt.append(twoH2Nu4C2[i] * nepConst * filtBandpass[i] * emissivity[i] * expTermInt *\
                         (1 + nepConst * filtBandpass[i] * emissivity[i] * expTermInt))

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
    # Take into account that one bolometer branch fills only pixFrac of the full
    # pixel angle
    nepSquare *= (pixSolidAngle*pixFrac)
        
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
    10/04/2020 - I modify the computation, I am now computing the NEP that is
    relevant for one bolometer branch (out of 4) of the detector. The main change
    is that a branch is sensitive to only one polarisation, and it is covering
    1/2 of the pixel area. The proper summation of NEPs will have to be done 
    in function getNEP.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
        # The zodi emissivity can have different values to represent the different
        # bacground cases.
        nepConst = zodiEmissivity*zodiCorrEm[useFGCase] * instTrans * etaColdStop
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
    # Take into account that one bolometer branch fills only pixFrac of the full
    # pixel angle
    nepSquare *= (pixSolidAngle*pixFrac)
    
    # account for the existence of two polarisation state per photon energy state
    # 10/04/2020 I no longer return the NEP corresponding to the two polarisation
    # I return the NEP per branch of the bolometers.
    #nepSquare *=2
            
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
    10/04/2020 - I modify the computation, I am now computing the NEP that is
    relevant for one bolometer branch (out of 4) of the detector. The main change
    is that a branch is sensitive to only one polarisation, and it is covering
    1/2 of the pixel area. The proper summation of NEPs will have to be done 
    in function getNEP.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
    # Take into account that one bolometer branch fills only pixFrac of the full
    # pixel angle
    nepSquare *= (pixSolidAngle*pixFrac)
    
    # account for the existence of two polarisation state per photon energy state
    # 10/04/2020 I no longer return the NEP corresponding to the two polarisations
    # I return the NEP per branch of the bolometers.
    #nepSquare *=2
            
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
    10/04/2020 - I modify the computation, I am now computing the NEP that is
    relevant for one bolometer branch (out of 4) of the detector. The main change
    is that a branch is sensitive to only one polarisation, and it is covering
    1/2 of the pixel area. The proper summation of NEPs will have to be done 
    in function getNEP.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
        nepConst = cibEmissivity*cibCorrEm[useFGCase] * instTrans * etaColdStop
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
    # Take into account that one bolometer branch fills only pixFrac of the full
    # pixel angle
    nepSquare *= (pixSolidAngle*pixFrac)
    
    # account for the existence of two polarisation state per photon energy state
    # 10/04/2020 I no longer return the NEP corresponding to the two polarisations
    # I return the NEP per branch of the bolometers.
    #nepSquare *=2
            
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
    10/04/2020 - I modify the computation, I am now computing the NEP that is
    relevant for one bolometer branch (out of 4) of the detector. The main change
    is that a branch is sensitive to only one polarisation, and it is covering
    1/2 of the pixel area. The proper summation of NEPs will have to be done 
    in function getNEP.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
    # Take into account that one bolometer branch fills only pixFrac of the full
    # pixel angle
    nepSquare *= (pixSolidAngle*pixFrac)
    
    # account for the existence of two polarisation state per photon energy state
    # 10/04/2020 I no longer return the NEP corresponding to the two polarisations
    # I return the NEP per branch of the bolometers.
    #nepSquare *=2
        
    # Finally we need to account for the fact that the fore optics is comprised of
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
    10/04/2020 - I modify the computation, I am now computing the NEP that is
    relevant for one bolometer branch (out of 4) of the detector. The main change
    is that a branch is sensitive to only one polarisation, and it is covering
    1/2 of the pixel area. The proper summation of NEPs will have to be done 
    in function getNEP.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.

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
    # Take into account that one bolometer branch fills only pixFrac of the full
    # pixel angle
    nepSquare *= (pixSolidAngle*pixFrac)
    
    # account for the existence of two polarisation state per photon energy state
    # 10/04/2020 I no longer return the NEP corresponding to the two polarisations
    # I return the NEP per branch of the bolometers.
    #nepSquare *=2
        
    # Finally we need to account for the fact that the camera optics is comprised of
    # a number of emitting surfaces so that we have to add as many NEP^2 as there
    # are surfaces.
    nepSquare *= num2KSurf
    
    return math.sqrt(nepSquare)

######################################################################
# This functions returns the incoming power from a source in the pixel
######################################################################
def compSrcPixPow(flux,band,extended=False,silent=False):
    """
    This function returns the power from a source that falls on a pixel.
    it allows distinguishing between a point and an extended source
    As with the background prediction functions it computes the incident power,
    not the power that the stokes pixels are sensitive to (1/2 of it for 
    non-polarised signal).
    The flux must be provided in mJy for point sources (extended=False) and 
    in MJy/sr for extended sources (extended=True)
    Note that this function is provided as an utility. It is not used by
    the time estimator code.
    """
    
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.
    # We will need as well diam4 and eefD4Band
    data = getGeomQuant(pixSizeBand)
    diam4 = data[1]
    eefD4Band = data[2]

    # Converts the input flux into SI
    if (extended):
        if (not(silent)):
            print("Converting from MJy/sr into SI")
        inputFluxDens = flux * 1e-20 # now in W.m-2.Hz-1.sr-1
        # Need to be multiplied by the pixel solid angle.
        pixSolidAngle = (pixSizeBand[band-1]*(math.pi/(180*3600)))**2
        inputFluxDens *= pixSolidAngle
        # at this stage the flux in in W.m-2.Hz-1.pix-1
    else:
        if (not(silent)):
            print("Converting from mJy into SI")
        inputFluxDens = flux * 1e-29 # now in W.m-2.Hz-1
        # to convert that into a surface brightness we make assumptions
        # the code is parameterized by eefD4Band which is the fraction of energy that 
        # falls in a circle that has the same surface as 4 pixels (fully illuminated
        # by a point source as per optical design) and diam4 the diameter of this
        # circle expressed in units of the FWHM.
        # compute the FWHM (warning, the function returns the value for all the bands)
        fwhm = getBeamProfile(fwhm=1)
        #eefD4Band is now a function of the band
        corrFrac = eefD4Band[band-1] * pixSizeBand[band-1]**2 * (4./(math.pi*fwhm[band-1]**2*diam4[band-1]**2))
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
def compPixPowSrc(power,band,extended=False,silent=False):
    """
    This function returns the source flux in the sky (in mJy for point source or MJy/sr
    for extended sources) that corresponds to a given incident power on the
    pixel.
    Input power must be in W.pix-1
    """
    # First perform a input check
    if (checkBandForError(band) == False):
        # the band is not within the accepted band labels
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.
    # We will need as well diam4 and eefD4Band
    data = getGeomQuant(pixSizeBand)
    diam4 = data[1]
    eefD4Band = data[2]

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
    # now power is in W.m-2.Hz-1.pix-1
    # and take into account the optical transmission of the system
    power /= instTrans
    
    if (extended):
        if (not(silent)):
            print("converting to MJy/sr")
        # the source is extended so I just need to divide by the solid angle
        # subtended by a pixel (formally this factor is in sr.pix-1)
        pixSolidAngle = (pixSizeBand[band-1]*(math.pi/(180*3600)))**2
        flux = power/pixSolidAngle
        # normalize it to MJy/sr
        flux /= 1e-20
    else:
        if (not(silent)):
            print("converting to mJy")
        # the source is a point source so I first determine how much of 
        # the source flux falls on one pixel. As explained in routine compSrcPixPow
        # we characterise this by eefD4Band which is the fraction of the energy that 
        # falls on an aperture that occupies the same area as 2x2 pixels, and diam4,
        # the diameter of this circle expressed in units of the FWHM.
        # compute the FWHM. Warning: the function returns the value for all bands
        fwhm = getBeamProfile(fwhm=1)
        # eefD4Band is now a function of the band
        corrFrac = eefD4Band[band-1] * pixSizeBand[band-1]**2 * (4./(math.pi*fwhm[band-1]**2*diam4[band-1]**2))
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
    12/05/2020 It needs to take into account the optical transmission of the system
    """
    if (checkBandForError(band) == False):
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
        power /= telEffArea
        power /= instTrans
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
        area   the area to be mapped (in degrees) assumed to be square-shaped
        mode    is a parameter describing the observing mode (needs to be among allowed values)
    Note that in this code, vscan is an input parameter and not the paramterised value
    declared at the beginning of this package. This is because here we are providing
    an intermediate element in the sensitivity estimate.
    it returns [on source time, total observing time, efficiency, nLegs]
    02/06/2020 Returns as well the number of scan legs for information and change
    how the internal slew times are added to allow flexibility for future implementation
    of the observing mode
    08/06/2020 Now the field of view is a function of the pixel size and no longer
    a fixed parameter
    """
    
    # Verify the mode
    if (mode != 'optimal'):
        print("Observing mode parameter is not in allowed values")
        print("Allowed values are [optimal]")
        return
    
    if (mode == 'optimal'):
        pixSizeBand = getPixSizeBand(over=useDef)
        bbopFoV = getGeomQuant(pixSizeBand)[0]
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
        # for each leg of the scan we need to accelerate to it and brake at the end
        # this assumes that spacecraft is delivered standing still to begin the 
        # observation, and is positioned standing still at the end of the obs.
        # We need then one step motion betwen the legs, so one less than the 
        # number of legs. 
        # Since here we are computing the time for 1 map, we assume the observing 
        # time clock stops at the end of the map.
        obsTimeFull = obsTime + (nLegs-1)*timeStep + nLegs*2*timeBrakeAccel
        # add the initial overhead
        obsTimeFull += initOverhead
        
    return [onSrcTime,obsTimeFull, obsTime/obsTimeFull,nLegs]

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
    matters to the user (the input here) is the one measured after object photometry
    has been performed, which means subtraction of a background in one way or another, 
    that leads to a factor sqrt(2) on the pixel rms (at least).
    Furthermore for point sources, the S/N from the user point of view is the S/N
    measured in a aperture around the source. We take the convention that the aperture
    is the same as the one used to characterise the Encircled Energy Fraction (in other
    words the aperture is fixed to the profile of the PSF). Here we have characterised it
    by eefD4Band and diam4 which are the diameter of an aperture of surface equal to that
    of 4 pixels, and the fraction of energy inside that circle. So effectively the aperture
    photometry is performed on 4 pixels, which relaxes the level of the noise by a 
    factor 2 with respect to the noise per pixel.
    The function returns as a list of lists:
        the point source sensitivities
        the extended source sensitivities
        the timing information
    12/04/2020 Includes now corrected formulae that acknowledge the fact that
    the detector pixels are sensitive to 100% of the incoming power, and include
    the fact that optical transmission also applies to the source...
    """
    
    # Check the input parameters
    if (mode != 'optimal'):
        print("Observing mode parameter is not in allowed values")
        print("Allowed values are [optimal]")
        return
    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.
    # We will need as well diam4 and eefD4Band
    data = getGeomQuant(pixSizeBand)
    diam4 = data[1]
    eefD4Band = data[2]

    if (mode == 'optimal'):
        # collect the elements of the computation
        # Though we do not need it for the sensitivity, let's get the background
        forePower = [getForeground(1),getForeground(2),getForeground(3)]
        forePower = np.asarray(forePower)
        if not(silent):
            bandStr = str(bandWave[0]*1e6)+' '+str(bandWave[1]*1e6)+' '+str(bandWave[2]*1e6)
            print('Foreground power at '+bandStr+' are: {0:8.3g}, {1:8.3g} and {2:8.3g} W.'\
                  .format(forePower[0],forePower[1],forePower[2]))
        # get the total NEP, with the detector contribution 
        # 22/04/2020 - MS: since I have corrected the issue of polarizer solid
        # angle the distinction between incident and absorbed has disappeared.
        if not(silent):
            if (goal):
                print('NEP includes GOAL detector NEP')
            else:
                print('NEP uses REQuirement detector NEP')
        pixNEP = [getNEP(1,withDet=True,goal=goal,silent=True),\
                  getNEP(2,withDet=True,goal=goal,silent=True),\
                  getNEP(3,withDet=True,goal=goal,silent=True)]
        pixNEP = np.asarray(pixNEP)
        # get the timing information
        timings = getObsTime(scanSpeed,area,mode=mode)
        onSourceTime = timings[0]
        #print "Total observing time: ",timings[1]
        
        # 22/04/2020 MS: I need to check the formulae for the sensitivity as
        # they could be wrong.
        # Extended case
        extSrcPow = ston*pixNEP/math.sqrt(onSourceTime)
        # Since we have been using absorbed quantities, there is a factor of 2
        # with respect to the incident power
        # 12/05/2020 - This was a wrong understanding of how the detectors are
        # built
        #extSrcPow *=2
        # convert to MJy/sr
        extSrcFlux = [compPixPowSrc(extSrcPow[0],1,extended=True,silent=True),\
                     compPixPowSrc(extSrcPow[1],2,extended=True,silent=True),\
                     compPixPowSrc(extSrcPow[2],3,extended=True,silent=True)]
        
        # Point source case
        # Here we have to be careful
        # Compute the number of pixels in the aperture for which we know the
        # encircled energy fraction
        # Here I need the array of fwhm. This is what getBeamProfile provides
        fwhm = getBeamProfile(fwhm=1)
        nPixAper = (math.pi/4) * (diam4*fwhm)**2 / pixSizeBand**2
        # computes the point source power in the aperture
        pointSrcPow = ston * (np.sqrt(nPixAper)) * \
                      pixNEP / math.sqrt(onSourceTime)
        # Now correct for the fraction of the point source that is not in the aperture
        pointSrcPow /= eefD4Band
        # remember that these are absorbed quantities so the pixel receives twice that
        # 12/05/2020 - this was a wrong understanding of how the detectors are
        # built
        #pointSrcPow *=2
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
            print("Sensitivity figures derived under foreground case "+foreCases[useFGCase])
            print("Observing parameters: S/N = {0:.1f}, area = {1:.2f}".format(ston,area))
            print("Observing time = {0:.2f} (s) for an efficiency of {1:.2f}".format(timings[1],timings[2]))
            print("Number of scan legs = ",timings[3])
            print("pixel size used = ",pixSizeBand)
            print("Wavelengths:   {0:8.1f}   {1:8.1f}   {2:8.1f}".format(1e6*bandWave[0],1e6*bandWave[1],1e6*bandWave[2]))
            print("Sensitivity to extended sources (MJy/sr)")
            print("               {0:.2e}   {1:.2e}   {2:.2e}".format(extSrcFlux[0],extSrcFlux[1],extSrcFlux[2]))
            print("Sensitivity to point sources (mJy)")
            print("               {0:.2e}   {1:.2e}   {2:.2e}".format(pointSrcFlux[0],pointSrcFlux[1],pointSrcFlux[2]))
        
        return [extSrcFlux,pointSrcFlux,timings]

###################################################################################
# This function gets the sensitivity of a map when polarisation parameters are the
# primary objective of the measurement. In that case the user provides an area to
# cover, the expected/reprentative polarisation fraction, and  the desired S/N on
# that polarisation fraction.
###################################################################################
def getSensPolar(pfrac,ston,area,mode='optimal',goal=False,silent=False):
    """
    This script returns the sensitivity (mJy for a point source, MJy/sr for an
    extended source) of a map when polarisation parameters are the primary goal, 
    i.e. on a user provided representative polarisation fraction, signal-to-noise
    on the polarisation fraction, and area to cover.
    its arguments are:
        - pfrac  the polarisation fraction (%)
        - ston   signal to noise on the polarisation fraction (dimensionless)
        - area   the area to map (degrees)
        - mode   this is the observing mode, chosen among accepted values
                    only 'optimal' is allowed for now.
        - goal   (T/F) is a flag to use the goal detector NEP rather than the requirement
                 default is False
        - silent if true will not print the results
    Note that there is an underlying assumption that the signal that matters to 
    the user, i.e. the sensitivity limit is the one measured after object photometry
    has been performed, which means subtraction of a background in one way or another, 
    and for point sources, measurement of the properties in an aperture around the source.
    We take the convention that the aperture.
    is the same as the one used to characterise the Encircled Energy Fraction (in other
    words the aperture is fixed to the profile of the PSF). Here we have characterised it
    by eefD4Band and diam4 which are the diameter of an aperture of surface equal to that
    of 4 pixels, and the fraction of energy inside that circle. So effectively the aperture
    photometry is performed on 4 pixels.
    This introduces correction factors to sensisitivies derived for a single pixel
    that will be explicited in code below.
    The function returns as a list of lists:
        the point source sensitivities
        the extended source sensitivities
        the timing information
    12/05/2020 - I correct for the mistake resulting from my wrong understanding
    of how the detectors are built and the filling factors of the grids, and as
    well for having forgotten that optical transmission also applies to the source.
    I have also changed the way NEPs are computed which affects this code.
    """
    
    # Check the input parameters
    if (mode != 'optimal'):
        print("Observing mode parameter is not in allowed values")
        print("Allowed values are [optimal]")
        return
    
    # There is ambiguity in how the user will understand the polarisation fraction 
    # so I make a test
    if (pfrac >= 1.0):
        print("Assuming polarisation fraction has been provided in %")
        pfrac = pfrac/100.

    
    # We will need the pixel size
    pixSizeBand = getPixSizeBand(over=useDef)
    # if over=True it will return the default pixel size, useDef is set in the
    # preamble.
    # We will need as well diam4 and eefD4Band
    data = getGeomQuant(pixSizeBand)
    diam4 = data[1]
    eefD4Band = data[2]

    if (mode == 'optimal'):
        # get the total NEP, with the detector contribution 
        # Though we do not need it for the sensitivity, let's get the background
        forePower = [getForeground(1),getForeground(2),getForeground(3)]
        forePower = np.asarray(forePower)
        if not(silent):
            bandStr = str(bandWave[0]*1e6)+' '+str(bandWave[1]*1e6)+' '+str(bandWave[2]*1e6)
            print('Foreground power at '+bandStr+' are: {0:8.3g}, {1:8.3g} and {2:8.3g} W.'\
                  .format(forePower[0],forePower[1],forePower[2]))
        # 12/05/2020 the comments below do not apply anymore. Same NEP in polar
        # or total power mode as all grids always contribute to the measurement.
        # the polar option divides the background contribution by sqrt(2)
        # because only 1/2 of the pixel is sensitive to a given polarisation.
        if not(silent):
            if (goal):
                print('NEP includes GOAL detector NEP')
            else:
                print('NEP uses REQuirement detector NEP')
        pixNEP = [getNEP(1,withDet=True,goal=goal,silent=True),\
                  getNEP(2,withDet=True,goal=goal,silent=True),\
                  getNEP(3,withDet=True,goal=goal,silent=True)]
        pixNEP = np.asarray(pixNEP)
        # get the timing information
        timings = getObsTime(scanSpeed,area,mode=mode)
        # Note that in the case of a polarisation observation, we must consider
        # that we have 2 kinds of pixels and that we need both information to
        # derive the polarisation fraction and angle so the on-source time per
        # kind of pixels is divided by two
        onSourceTime = timings[0]/2.
        # In the case of a "polarisation" measurement we have a number of elements
        # to consider:
        # As we express the s/n on the polarisation fraction, this is not the 
        # S/N that enters the NEP equation, and as the source is only one 
        # component of the incident power on the pixel, there is a complex
        # relation between the two leading to the presence in the limiting 
        # flux equation of s/n.(1/sqrt(2)).sqrt(1/p^2 + 2)
        # These terms alreay take into account the fact that a background subtraction
        # must occur to get to the source component.
        
        # Extended source case
        # Contrary to the total power case I cannot simply break down this 
        # expression into components. See the sensitivity note for
        # explanations
        extSrcPow = ston*math.sqrt(2)*math.sqrt(1/pfrac**2+2)*pixNEP/math.sqrt(2*onSourceTime)
        # convert to MJy/sr
        extSrcFlux = [compPixPowSrc(extSrcPow[0],1,extended=True,silent=True),\
                     compPixPowSrc(extSrcPow[1],2,extended=True,silent=True),\
                     compPixPowSrc(extSrcPow[2],3,extended=True,silent=True)]
        
        # Point source case.
        # Further considerations apply.
        # First a point source measurement is always the result of the integration
        # in an aperture. We assume that the s/n specified by the user on the polarisation
        # fraction is that in the integrated flux and thus that the s/n on the 
        # polarisation fraction per pixel in the aperture can be sqrt(Naper) lower
        # where Naper is the number of pixels in the aperture.
        # Compute the number of pixels in the aperture for which we know the
        # encircled energy fraction
        # Here I need the array of fwhm. This is  what getBeamProfile provides
        fwhm = getBeamProfile(fwhm=1)
        nPixAper = (math.pi/4) * (diam4*fwhm)**2 / pixSizeBand**2
        # As with the extended case it is too complex to break down the limiting flux 
        # equation into its component
        pointSrcPow = math.sqrt(2)*np.sqrt(nPixAper)*ston*math.sqrt(2+1/pfrac**2)*pixNEP/math.sqrt(2*onSourceTime)
        # recall that we only have a certain EEF in that aperture
        pointSrcPow /= eefD4Band
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
            print("Sensitivity figures derived under foreground case "+foreCases[useFGCase])
            print("Observing parameters: p = {0:.2f}, S/N = {1:.1f}, area = {2:.2f}".format(pfrac,ston,area))
            print("Observing time = {0:.2f} (s) for an efficiency of {1:.2f}".format(timings[1],timings[2]))
            print("Number of scan legs = ",timings[3])
            print("pixel size used = ",pixSizeBand)
            print("Wavelengths:   {0:8.1f}   {1:8.1f}   {2:8.1f}".format(1e6*bandWave[0],1e6*bandWave[1],1e6*bandWave[2]))
            print("Sensitivity to extended sources (MJy/sr)")
            print("               {0:.2e}   {1:.2e}   {2:.2e}".format(extSrcFlux[0],extSrcFlux[1],extSrcFlux[2]))
            print("Sensitivity to point sources (mJy)")
            print("               {0:.2e}   {1:.2e}   {2:.2e}".format(pointSrcFlux[0],pointSrcFlux[1],pointSrcFlux[2]))
        
        return [extSrcFlux,pointSrcFlux,timings]
    
###################################################################################
# This function gets the observing time needed to reach a user-provided flux with
# a user-provided S/N. This is simply using previous functions in a reversed order
###################################################################################
def getMyObsTime(flux,band,ston,src='PS',area=1.0,pfrac=0.0,mode='optimal'):
    """
    This function computes the user observing time as a function of:
        flux     the input source flux
        band     the band this applies to (1,2,or 3, default to 2)
        ston     the signal to noise ratio to reach
        src      PS or ES, the type of sources for which the flux is indicated.
        area     is the area to map, defaulted to 1 square degree
        pfrac    is the polarisation fraction. if not 0, then it is assumed that
                 the S/N provided applies to the polarisation fraction and not to
                 the total flux.
        mode     is the observing mode, only "optimal" implemented as of now
    Note that this function is a trivial one, since it assumes that sensitivity is 
    scaling as the square root of the observing time, but it's implemented as a
    convenience to the user.
    """
    
    # Systematically prints this warning to the user
    if (src=='PS'):
        print("Point Source flux is assumed to be in mJy")
    elif (src=='ES'):
        print("Extended Source flux is assumed to be in MJy/sr")
    else:
        print("Value of source type (keyword src) should be 'PS' or 'ES'")
        return
    
    # first thing we do is to derive the "instantaneous" sensitivity, that of performing
    # a single scan
    if (pfrac == 0):
        sens = getSensitivity(ston,area,mode=mode,silent=True)
        extSrcFlux = sens[0]
        pointSrcFlux = sens[1]
        timings = sens[2]
    else:
        sens = getSensPolar(pfrac,ston,area,mode=mode,silent=True)
        extSrcFlux = sens[0]
        pointSrcFlux = sens[1]
        timings = sens[2]
    
    # now I need to compare the sensitivity achieved to the one requested and
    # scale the observing time accordingly.
    if (src == 'PS'):
        fluxRatio = flux / pointSrcFlux[band-1]
    elif (src == 'ES'):
        fluxRatio = flux / extSrcFlux[band-1]
    else:
        print('The type keyword can only be ES or PS')
        return np.nan
    
    # Now from this flux ratio we use the simple assumption that post-processing
    # sensitivity scales as the square root of time.
    # Now if the ratio is greater than 1, unfortunately we cannot yet go faster
    # to the time is the minimal time given by the observing mode.
    print("Observing time figures derived under foreground case "+foreCases[useFGCase])

    if (fluxRatio > 1):
        print("The minimum sensitivity with a single pass is already better than your limit flux")
        if (type == 'PS'):
            print("The minimum map will give you a sensitivity of: {0:.2e}".format(pointSrcFlux[band-1]))
        else:
             print("The minimum map will give you a sensitivity of: {0:.2e}".format(extSrcFlux[band-1]))           
        print("Observing time (in seconds) is: {0:8.1f}".format(timings[1]))
        print("with an efficiency of: {0:.2f}".format(timings[2]))
        print("Number of scan legs required: {0:.1f}".format(timings[3]))
        return [timings[1],timings[2]]
    else:
        # We can only repeat the number of maps an integer number of times
        nMap = int(1./fluxRatio**2)+1
        # By doing this we overestimate the number of maps slightly but we 
        # Garantee the sensitivity.
        # in the timing we need to add the time to go back at the beginning of the map
        # if number of scan legs is an even number the length is the diagonal of the map
        # if number of scan legs is an odd number the length is the square of the map
        if (timings[3] % 2 == 0):
            # the number of scan legs is an even number
            slewBackDist = np.sqrt(2*area)*60
        else:
            slewBackDist = np.sqrt(area)*60
        # if the slew length is less than 5' is should be a short slew
        if (slewBackDist < 5.0):
            slewBackTime = (nMap-1)*timeStep
        else:
            slewBackTime = (nMap-1)*timeMedSlew
        # Recompute the efficiency
        obsTimeMap = timings[1]*timings[2]
        efficiency = (obsTimeMap*nMap)/(timings[1]*nMap+slewBackTime)
        print("Number of repetitions needed to achieve sensitivity is: {0:.1f}".format(nMap))
        print("Observing time (in seconds) is: {0:8.1f}".format(timings[1]*nMap+slewBackTime))
        print("with an efficiency of: {0:.2f}".format(efficiency))
        print("Number of scan legs required: {0:.1f}".format(timings[3]))
        print("Time spent slew back to map start (in second) is: {0:.1f}".format(slewBackTime))
        return [timings[1]*nMap+slewBackTime,efficiency]

print("--- Time Estimator is ready for use ---")
    