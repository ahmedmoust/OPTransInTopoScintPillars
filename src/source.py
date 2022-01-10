'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


import sys

import numpy as np
import pandas as pd

import opticalPhoton


def resetIPython():
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
    #get_ipython().run_line_magic('clear', '')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    return



def generateRandomUnitVector(numberOfVectors):
    '''
    method to generate unit vectors with uniformly distributed directions over 4pi

    Parameters
    ----------
    numberOfVectors : int
                      Number of unit vectors to generate.

    Returns
    -------
    vector : (nx3) array of floats
            (numberOfVectors X 3) array of unit vectors.

    '''
    
    phi          = 2*np.pi*np.random.rand(numberOfVectors, 1)
    dir_gamma    = 2*np.random.rand(numberOfVectors, 1) - 1
    dir_alpha    = np.sqrt(1-(dir_gamma**2)) * np.cos(phi)
    dir_beta     = np.sqrt(1-(dir_gamma**2)) * np.sin(phi)
    vector       = np.concatenate((dir_alpha, dir_beta, dir_gamma), axis=1)
    vector      /= np.array([np.linalg.norm(vector, axis = 1)]).transpose()
    
    return vector



def generateOpticalPhotonsInfo(position, energy, scintMaterial):
    '''
    a method to sample optical photons emission information using the scintillator material and deposited energy

    Parameters
    ----------
    position : (1x3) array of floats
               Emission position in mm of the optical photons to be generated.
    energy : float
             Deposited energy in keV to be used to generate optical photons.
    scintMaterial : dict
                    Scintillator material information to be used to generate the optical photons.

    Returns
    -------
    OPlist : list
             OpticalPhoton objects with the sampled emission information. 

    '''
    
    #print(' - generating optical photons info for the given energy deposition...')
    
    # == retrieve relevant scintillator material information
    scintLY                   = scintMaterial['lightYield'] * 1E-3                # photons/keV
    scintEmissionWavelengths  = scintMaterial['emissionSpectrum']['wavelength']   # nm
    scintEmissionSpectrum     = scintMaterial['emissionSpectrum']['amplitude']    # a.u.
    scintEmissionTimes        = scintMaterial['timeResponse']['time']             # ns
    scintEmissionTimeResponse = scintMaterial['timeResponse']['amplitude']        # a.u.
    
    # == random-sample the optical photon emission information
    opCount           = int(np.random.normal(energy*scintLY, np.sqrt(energy*scintLY)))
    #
    opWavelegnths     = np.random.choice(scintEmissionWavelengths, size=opCount,
                                         p=scintEmissionSpectrum)                          # nm
    #
    position         *= 1E3                                                                # um
    opPositions       = [(position[0], position[1], position[2]),]*opCount
    opPositions       = np.asarray(opPositions)
    #
    opMomentumDir     = generateRandomUnitVector(opCount)
    #
    opPolarizationDir = generateRandomUnitVector(opCount)
    #
    opTimes           = np.random.choice(scintEmissionTimes, size=opCount,
                                         p=scintEmissionTimeResponse)                      # ns
    
    # == store the sampled info into a list of opticalPhoton objects
    OPlist = []
    for opNumber in range(opCount):
        OPlist.append(opticalPhoton.opticalPhoton(np.array([opPositions[opNumber]]),
                                                  opWavelegnths[opNumber],
                                                  np.array([opMomentumDir[opNumber]]),
                                                  np.array([opPolarizationDir[opNumber]]),
                                                  opTimes[opNumber]))
    
    return OPlist



def energyDepositions(energyDepositionsFileName, scintMaterial, generatePhotonsInfo = False):
    '''
    a method to read-in energy deposition information and optionally generating optical photon emission information
    Note: generating optical photon information at once for large numbers of energy depositions upfront consumes a lot of memory;
          it is otherwise better to use the generateOpticalPhotonsInfo() method to generate them for each individual energy depositions
          while performing tracking. That way, only the information of the photon being tracked is kept in memory.
          Information of only 'detected' photons could then be saved to desk to save memory.

    Parameters
    ----------
    energyDepositionsFileName : string
                                Name of the energy depositions hdf file
    scintMaterial : dict
                    Scintillator material information to be used to generate the optical photons.
    generatePhotonsInfo : bool, optional
                          Whether to also generate optical photons emission info after loading the energy depositions.
                          The default is False.

    Raises
    ------
    AttributeError: if the loaded energy depositions dataframe doesn't have both 'energy' & 'position' columns'

    Returns
    -------
    if generatePhotonsInfo == True:
        sourceData : dict
                     opticalPhoton objects for each primary event.
    if generatePhotonsInfo == False:
        energyDepositions : dataframe
                            Energy depositions.

    '''
    
    print('\n')
    print('Loading an energy depositions dataframe...')
    
    # == load the energy depositons dataframe and checking if it has all the necessary information
    energyDepositions = pd.read_hdf(energyDepositionsFileName+'.hdf')
    if 'energy' and 'position' not in energyDepositions.columns:
        raise AttributeError("The loaded energy depositions dataframe does not contain all necessary information...\n"
                            + "Please make sure the dataframe has 'energy' & 'position' columns.")
    
    # == generate optical photon emission information for each primary event of energy deposition
    if generatePhotonsInfo:
        sourceData = {}
        for event, energy in enumerate(energyDepositions['energy']):
            event_OPlist = generateOpticalPhotonsInfo(energyDepositions['position'].iloc[event], energy, scintMaterial)
            sourceData[event] = event_OPlist
        return sourceData
    else:
        return energyDepositions



def energyDepositionDistribution(energyDepositionDistributionFileName, scintMaterial, generatePhotonsInfo = False, **kwargs):
    '''
    a method to read-in energy deposition distribution and optionally generating optical photon emission information
    Note: generating optical photon information at once for large numbers of primary histories upfront consumes a lot of memory;
          it is otherwise better to use the generateOpticalPhotonsInfo() method to generate them for each individual primary history
          while performing tracking. That way, only the information of the photon being tracked is kept in memory.
          Information of only 'detected' photons could then be saved to desk to save memory.

    Parameters
    ----------
    energyDepositionDistributionFileName : string
                                           Name of the energy deposition distribution hdf file
    scintMaterial : dict
                    Scintillator material information to be used to generate the optical photons.
    generatePhotonsInfo : bool, optional
                          Whether to also generate optical photons emission info after loading the energy deposition distribution.
                          The default is False.
    **kwargs : two necessary arguments if generatePhotonsInfo == True
               primaryHistories : int
                                 Number of primary energy depostions to sample and generate optical photon emission info for.
               primaryEmissionPosition : array of floats
                                         Primary energy depostion location coordinates in mm.

    Raises
    ------
    AttributeError: if the loaded energy deposition distribution dataframe doesn't have both 'energy' & 'amplitudes' columns'.

    Returns
    -------
    if generatePhotonsInfo == True:
        sourceData : dict
                     opticalPhoton objects for each primary event.
    if generatePhotonsInfo == False:
        energyDepositions : dataframe
                            Energy deposition distribution.

    '''
    
    print('\n')
    print('Loading an energy deposition distribution...')
    
    # == load the energy depositon distribution dataframe and checking if it has all the necessary information
    energyDepositionDistribution = pd.read_hdf(energyDepositionDistributionFileName+'.hdf')
    if 'energy' and 'amplitude' not in energyDepositionDistribution.columns:
        raise AttributeError("The loaded energy deposition distribution dataframe does not contain all necessary information...\n"
                            + "Please make sure the dataframe has 'energy' & 'amplitude' columns.")
    #= normalize to sum = 1.0 so that it could be used for sampling later
    energyDepositionDistribution['amplitude'] /= np.sum(energyDepositionDistribution['amplitude'])
    
    # == generate optical photon emission information for each primary event of energy deposition
    if generatePhotonsInfo:
        primaryHistories        = kwargs['primaryHistories']
        primaryEmissionPosition = kwargs['primaryEmissionPosition']            # mm
        #
        energyDepositions = np.random.choice(energyDepositionDistribution['energy'], size=primaryHistories, p=energyDepositionDistribution['amplitude'])  # keV
        #
        sourceData = {}
        for event, energy in enumerate(energyDepositions):
            event_OPlist = generateOpticalPhotonsInfo(primaryEmissionPosition*1E3, energy, scintMaterial)
            sourceData[event] = event_OPlist
        return sourceData
    else:
        return energyDepositionDistribution



def opticalPhotonsInfo(opticalPhotonsInformationFileName, generatePhotonsInfo = False):
    '''
    a method to read-in optical photon emission information dataframe

    Parameters
    ----------
    opticalPhotonsInformationFileName : string
                                        Name of the optical photon emission information hdf file
    generatePhotonsInfo : bool, optional
                          Whether to also store the optical photons emission info into opticalPhoton objects
                             after loading the optical photon emission information.
                          The default is False.

    Raises
    ------
    AttributeError: if the loaded optical photons emission info dataframe doesn't have all of the
                        'position', 'wavelength', 'momentumDirection', 'polarizationDirection' and 'time' columns.

    Returns
    -------
    if generatePhotonsInfo == True:
        sourceData : list
                     opticalPhoton objects for each primary event.
    if generatePhotonsInfo == False:
        energyDepositions : dataframe
                            Optical photon emission information.

    '''
    
    print('\n')
    print('Loading an optical photon information file...')
    
    # == load the optical photon emission information dataframe and checking if it has all the necessary information
    opticalPhotonInformation = pd.read_hdf(opticalPhotonsInformationFileName+'.hdf')
    if 'position' and 'wavelength' and 'momentumDirection' and 'polarizationDirection' and 'time' not in opticalPhotonInformation.columns:
        raise AttributeError("The loaded optical photon information dataframe does not contain all necessary information...\n"
                            + "Please make sure the dataframe has 'position', 'wavelength', 'momentumDirection', 'polarizationDirection' and 'time' columns.")
    
    # == store the optical photon emission information into opticalPhoton objects
    if generatePhotonsInfo:
        sourceData = []
        for opNumber in range(len(opticalPhotonInformation)):
            sourceData.append(opticalPhoton.opticalPhoton(np.array([opticalPhotonInformation['position'].iloc[opNumber]]*1E3),
                                                          opticalPhotonInformation['wavelength'].iloc[opNumber],
                                                          np.array([opticalPhotonInformation['momentumDirection'].iloc[opNumber]]),
                                                          np.array([opticalPhotonInformation['polarizationDirection'].iloc[opNumber]]),
                                                          time = opticalPhotonInformation['time'].iloc[opNumber],
                                                          )
                              )
        return sourceData
    else:
        return opticalPhotonInformation



def isotropicOpticalPhotonSourceInfo(scintMaterial, numberOfOpticalPhotons, opEmissionPosition):
    '''
    a method to generate optical photon emission information for an isotropic point source

    Parameters
    ----------
    scintMaterial : dict
                    Scintillator material information to be used to generate the optical photons.
    numberOfOpticalPhotons : int
                             Number of photons to generate information for.
    opEmissionPosition : array of floats
                         Isotropic point source location coordinates in mm.

    Returns
    -------
    sourceData : list
                 opticalPhoton objects.

    '''
    
    print('Generating optical photon info for an isotropic point source...')
    
    numberOfOpticalPhotons = int(numberOfOpticalPhotons)
    
    # == retrieve relevant scintillator material information
    scintEmissionWavelengths  = scintMaterial['emissionSpectrum']['wavelength']   # nm
    scintEmissionSpectrum     = scintMaterial['emissionSpectrum']['amplitude']    # a.u.
    scintEmissionTimes        = scintMaterial['timeResponse']['time']             # ns
    scintEmissionTimeResponse = scintMaterial['timeResponse']['amplitude']        # a.u.
    
    # == random-sample the optical photon emission information
    opWavelegnths       = np.random.choice(scintEmissionWavelengths, size=numberOfOpticalPhotons,
                                         p=scintEmissionSpectrum)                 # nm
    #
    opEmissionPosition *= 1E3                                                     # um
    opPositions         = [(opEmissionPosition[0], opEmissionPosition[1], opEmissionPosition[2]),]*numberOfOpticalPhotons
    opPositions         = np.asarray(opPositions)
    #
    opMomentumDir       = generateRandomUnitVector(numberOfOpticalPhotons)
    #
    opPolarizationDir   = generateRandomUnitVector(numberOfOpticalPhotons)
    #
    opTimes             = np.random.choice(scintEmissionTimes, size=numberOfOpticalPhotons,
                                         p=scintEmissionTimeResponse)             # ns
    
    # == store the sampled info into a list of opticalPhoton objects
    sourceData = []
    for opNumber in range(numberOfOpticalPhotons):
        sourceData.append(opticalPhoton.opticalPhoton(np.array([opPositions[opNumber]]),
                                                      opWavelegnths[opNumber],
                                                      np.array([opMomentumDir[opNumber]]),
                                                      np.array([opPolarizationDir[opNumber]]),
                                                      opTimes[opNumber]))
    
    return sourceData
    
                                                  
#%% main


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 'energyDepositions' or 'energyDepositionDistribution' or 'opticalPhotonInfo' or 'isotropicOpticalPhotonSource'
    '''
    
    resetIPython()
    
    command  = sys.argv[1]
    
    
    if command == 'energyDepositions':
        
        energyDepositionsFileName = sys.argv[2]
        scintMaterial             = sys.argv[3]
        
        import materials
        materialsLibrary = materials.materials().materialsLibrary
        sourceData = energyDepositions(energyDepositionsFileName, materialsLibrary[scintMaterial], generatePhotonsInfo = True)
        
        
    if command == 'energyDepositionDistribution':
        
        energyDepositionDistributionFileName = sys.argv[2]
        scintMaterial                        = sys.argv[3]
        primaryHistories                     = int(sys.argv[4])
        tmp = sys.argv[5][1:-1]; tmp = tmp.split(', ')
        primaryEmissionPosition              = [float(i) for i in tmp]
        
        import materials
        materialsLibrary = materials.materials().materialsLibrary
        sourceData = energyDepositionDistribution(energyDepositionDistributionFileName, materialsLibrary[scintMaterial],
                                                  generatePhotonsInfo = True,
                                                  primaryHistories=primaryHistories, primaryEmissionPosition=primaryEmissionPosition,
                                                  )
    
        
    if command == 'opticalPhotonInfo':
        
        opticalPhotonsInformationFileName = sys.argv[2]
        
        sourceData = opticalPhotonsInfo(opticalPhotonsInformationFileName, generatePhotonsInfo = True)
        
        
    if command == 'isotropicOpticalPhotonSource':
        
        scintMaterial                  = sys.argv[2]
        numberOfOpticalPhotons         = int(sys.argv[3])
        tmp = sys.argv[4][1:-1]; tmp = tmp.split(', ')
        opticalPhotonEmissionPosition                = [float(i) for i in tmp]
        
        import materials
        materialsLibrary = materials.materials().materialsLibrary
        sourceData = isotropicOpticalPhotonSourceInfo(materialsLibrary[scintMaterial], numberOfOpticalPhotons, opticalPhotonEmissionPosition)
        