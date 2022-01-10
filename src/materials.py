'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


import sys

import numpy as np
import pandas as pd
import pickle
from scipy import interpolate


def resetIPython():
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
    #get_ipython().run_line_magic('clear', '')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    return


class materials:
    
    def __init__(self, firstTime = True, materialsLibraryFileName = 'materialsLibrary'):
        '''
        materials library instantiation

        Parameters
        ----------
        firstTime : bool, optional
                    Flag marking whether the materials library is being built for the first time.
                    The default is True.
        materialsLibraryFileName : string, optional
                                   File name of the materials library to be built and saved, or loaded.
                                   The default is 'materialsLibrary'.

        Returns
        -------
        None.

        '''
        
        if firstTime:
            print('\n')
            #print('Building and saving the materials library to: {}.pkl...'.format(materialsLibraryFileName))
            print('Building the materials library...')
            self.materialsLibrary = self.buildMaterials()
            #self.saveMaterialsLibrary(materialsLibraryFileName)
        else:
            print('\n')
            print('Loading the previously created materials library: {}.pkl...'.format(materialsLibraryFileName))
            self.loadMaterialsLibrary(materialsLibraryFileName)
        
        
        
    def buildMaterials(self):
        '''
        building default materials with pre-coded properties

        Returns
        -------
        materialsLibrary : dict
                           Keys are the available materials with pre-coded properties

        '''
        
        materialsLibrary = {}
        
        materialsLibrary['EJ-204']     = self.buildEJ204()
        materialsLibrary['EJ-550']     = self.buildEJ550()
        materialsLibrary['Air']        = self.buildAir()
        materialsLibrary['SensLGlass'] = self.buildSensLGlass()
        materialsLibrary['Teflon']     = self.buildTeflon()
        
        print(' - list of built materials:'+' '.join(list(materialsLibrary.keys())))
        
        return materialsLibrary
    
    
    
    def saveMaterialsLibrary(self, materialsLibraryFileName):
        '''
        saving the materials library for later quick loading

        Parameters
        ----------
        materialsLibraryFileName : string
                                   File name of the materials library to be saved.

        Raises
        ------
        AttributeError:  if the materialsLibrary wasn't previously built and stored as an attribute.

        Returns
        -------
        None.

        '''
        
        # == check if the materialsLibrary was bulit and stored as an attribute
        if not hasattr(self, 'materialsLibrary'):
            raise AttributeError('The materialsLibrary attribute was not found...\n'
                                 +'Did you forget to first build the materialsLibrary?')
        
        # = save the materialsLibrary for later quick loading
        pickle.dump(self.materialsLibrary, open(materialsLibraryFileName+'.pkl','wb'), protocol=2)
        
        return
    
    
    
    def loadMaterialsLibrary(self, materialsLibraryFileName):
        '''
        loading a previously created materials library

        Parameters
        ----------
        materialsLibraryFileName : string
                                   File name of the materials library to be loaded.

        Returns
        -------
        None.

        '''
        
        self.materialsLibrary = pickle.load(open(materialsLibraryFileName+'.pkl', 'rb'))
        
        return
    
    
    
    def addMeterial(self, materialsLibraryFileName, name, lightYield, refractiveIndex, attenuationLength, riseTime, fallTime, emissionSpectrumAmplitude, energies=None, wavelengths=None):
        '''
        adding a new material to a previously created materials library

        Parameters
        ----------
        materialsLibraryFileName : string
                                   File name of the modified materials library to be saved.
        name : string
               Name of material to be added and used as a key in the materialsLibrary dictionary.
        lightYield : float
                     Number of photons generated per MeV of deposited energy.
        refractiveIndex : float
                          Index of refraction.
        attenuationLength : float
                            Attenuation length of optical photons in mm.
        riseTime : float
                   Rise time of the material response to radiaiton in ns.
        fallTime : float
                   Fall time of the material response to radiaiton in ns.
        emissionSpectrumAmplitude : numpy array of floats or None
                                    Emission probability at each entry of the energies and/or wavelengths arrays.
                                    By default, normalized to area = 1.0.
                           
        energies : numpy array of floats or None, optional
                   Energies in eV at which the emission probability is defined.
                   If not entered, estimated from 'wavelengths' if entered.
                   The default is None.
        wavelengths : numpy array of floats or None, optional
                      Wavelengths in nm at which the emission probability is defined.
                      If not entered, estimated from 'energies' if entered.
                      The default is None.

        Raises
        ------
        AttributeError: if somehow the materialsLibrary attribute was not found; shouldn't happen since a library is always created by default whenever the class is instantiated.
        ValueError: if each of the name, refractiveIndex, and attenuationLength of the material to be added is not specified.

        Returns
        -------
        None.

        '''
        
        
        # == check if the default materialsLibrary was previously stored as an attribute
        if not hasattr(self, 'materialsLibrary'):
            raise AttributeError('The materialsLibrary attribute was not found...\n'
                                 +'Did you forget to first load a previously created materialsLibrary?')
        
        if name == None:
            raise ValueError('A material name must be entered...')
        if refractiveIndex == None:
            raise ValueError('A material index of refraction must be entered...')
        if attenuationLength == None:
            raise ValueError('A material attenuation length must be entered; if it is unknown exactly but expected to be irrelevant, enter an arbitrary high number...')
        
        print(' - Adding the following material to the materials library: name={}, lightYield={:2.1f} photons/MeV, refractiveIndex={:1.2f}, \
               attenuationLength={:2.2f} mm, riseTime={} ns, and fallTime={} ns...'.format(name, lightYield, refractiveIndex, attenuationLength, riseTime, fallTime))
        
        data = {}
        timeResponse     = pd.DataFrame(columns=['time', 'amplitude'])
        emissionSpectrum = pd.DataFrame(columns=['energy', 'wavelength', 'amplitude'])
        
        data['name']               = name
        data['lightYield']         = lightYield         # photons/MeV
        data['refractiveIndex']    = refractiveIndex
        data['attenuationLength']  = attenuationLength  # mm
        
        data['riseTime']           = riseTime           # ns
        data['fallTime']           = fallTime           # ns
        timeResponseFunction       = lambda t, falltime, risetime: np.exp(-t/falltime) - np.exp(-t/risetime)
        timeResponse['time']       = np.arange(0.0, 32+0.05, 0.05)
        timeResponse['time']       = np.around(timeResponse['time'], decimals = 2)
        timeResponse['amplitude']  = timeResponseFunction(timeResponse['time'], data['fallTime'], data['riseTime'])
        timeResponse['amplitude'] /= np.sum(timeResponse['amplitude'])
        data['timeResponse']       = timeResponse
        
        if energies and wavelengths:
            emissionSpectrum['energy']     = energies             # eV
            emissionSpectrum['wavelength'] = wavelengths          # nm
        elif energies:
            emissionSpectrum['energy']     = energies             # eV
            emissionSpectrum['wavelength'] = 1240.0 / energies    # nm
        elif wavelengths:
            emissionSpectrum['energy']     = 1240.0 / wavelengths # eV
            emissionSpectrum['wavelength'] =  wavelengths         # nm
        else:
            emissionSpectrum['energy']     = None
            emissionSpectrum['wavelength'] = None
        
        if emissionSpectrumAmplitude:
            emissionSpectrum['amplitude'] = emissionSpectrumAmplitude
            # normalizing to area = 1.0
            emissionSpectrum['amplitude'] /= np.sum(emissionSpectrum['amplitude'])
        else:
            emissionSpectrum['amplitude'] = None
        
        data['emissionSpectrum'] = emissionSpectrum
        
        # = adding the new material to the materials library
        self.materialsLibrary[name] = data
        
        # = saving the modified materials library
        print(' - saving the modified materials library to: {}.pkl...'.format(materialsLibraryFileName))
        self.saveMaterialsLibrary(materialsLibraryFileName)
        
        return
    
    
    
    def deleteMeterial(self, materialsLibraryFileName, name):
        '''
        deleting a material from a previously created materials library

        Parameters
        ----------
        materialsLibraryFileName : string
                                   File name of the modified materials library to be saved.
        name : string
               Name of material to be deleted; must match the key in the materialsLibrary dictionary.

        Raises
        ------
        AttributeError: if somehow the materialsLibrary attribute was not found; shouldn't happen since a library is always created by default whenever the class is instantiated.

        Returns
        -------
        None.

        '''
        
        
        # == check if the default materialsLibrary was previously stored as an attribute
        if not hasattr(self, 'materialsLibrary'):
            raise AttributeError('The materialsLibrary attribute was not found...\n'
                                 +'Did you forget to first load a previously created materialsLibrary?')
        
        # = deleting the material
        print(' - deleting the material: {} from the materials library: {}.pkl...'.format(name, materialsLibraryFileName))
        del self.materialsLibrary[name]
        
        
        # = saving the midified materials library
        print(' - saving the modified materials library to: {}.pkl...'.format(materialsLibraryFileName))
        self.saveMaterialsLibrary(materialsLibraryFileName)
        
        return
    
    
    
    @staticmethod
    def buildEJ204():
        
        data             = {}
        timeResponse     = pd.DataFrame(columns=['time', 'amplitude'])
        emissionSpectrum = pd.DataFrame(columns=['energy', 'wavelength', 'amplitude'])
        
        
        data['name']               = 'EJ-204'
        data['lightYield']         = 10400 # photons/MeV; source: Eljen website, https://eljentechnology.com/products/plastic-scintillators/ej-200-ej-204-ej-208-ej-212
        data['refractiveIndex']    = 1.58  # source: Eljen website
        data['attenuationLength']  = 1600  # mm;  source: Eljen website, https://eljentechnology.com/products/plastic-scintillators/ej-200-ej-204-ej-208-ej-212
        
        data['riseTime']           = 1.3   # ns; source: Joshua Brown measurements at SNL
        data['fallTime']           = 2.0   # ns; source: Joshua Brown measurements at SNL
        timeResponseFunction       = lambda t, falltime, risetime: np.exp(-t/falltime) - np.exp(-t/risetime)
        timeResponse['time']       = np.arange(0.0, 32+0.05, 0.05)
        timeResponse['time']       = np.around(timeResponse['time'], decimals = 2)
        timeResponse['amplitude']  = timeResponseFunction(timeResponse['time'], data['fallTime'], data['riseTime'])
        timeResponse['amplitude'] /= np.sum(timeResponse['amplitude'])
        data['timeResponse']       = timeResponse

        # energies array in eV
        emissionSpectrum['energy'] = np.array([ 2.49995, 2.502, 2.50405, 2.5061, 2.50815, 2.51021, 2.51228, 2.51434, 2.51641, 2.51848, 2.52056, 2.52264, 2.52472, 2.52681, 2.52889, 2.53099,
                                                2.53308, 2.53518, 2.53729, 2.53939, 2.5415, 2.54362, 2.54574, 2.54786, 2.54998, 2.55211, 2.55424, 2.55638, 2.55851, 2.56066, 2.5628, 2.56495,
                                                2.5671, 2.56926, 2.57142, 2.57359, 2.57575, 2.57792, 2.5801, 2.58228, 2.58446, 2.58665, 2.58883, 2.59103, 2.59323, 2.59543, 2.59763, 2.59984,
                                                2.60205, 2.60427, 2.60649, 2.60871, 2.61094, 2.61317, 2.6154, 2.61764, 2.61988, 2.62213, 2.62438, 2.62663, 2.62889, 2.63115, 2.63342, 2.63569,
                                                2.63796, 2.64024, 2.64252, 2.64481, 2.64709, 2.64939, 2.65168, 2.65399, 2.65629, 2.6586, 2.66091, 2.66323, 2.66555, 2.66788, 2.67021, 2.67254,
                                                2.67488, 2.67722, 2.67956, 2.68191, 2.68427, 2.68663, 2.68899, 2.69136, 2.69373, 2.6961, 2.69848, 2.70086, 2.70325, 2.70564, 2.70804, 2.71044,
                                                2.71284, 2.71525, 2.71766, 2.72008, 2.7225, 2.72493, 2.72736, 2.72979, 2.73223, 2.73467, 2.73712, 2.73957, 2.74203, 2.74449, 2.74696, 2.74942,
                                                2.7519, 2.75438, 2.75686, 2.75935, 2.76184, 2.76434, 2.76684, 2.76934, 2.77185, 2.77437, 2.77689, 2.77941, 2.78194, 2.78447, 2.78701, 2.78955,
                                                2.7921, 2.79465, 2.7972, 2.79977, 2.80233, 2.8049, 2.80748, 2.81006, 2.81264, 2.81523, 2.81782, 2.82042, 2.82302, 2.82563, 2.82825, 2.83086,
                                                2.83349, 2.83611, 2.83875, 2.84138, 2.84403, 2.84667, 2.84933, 2.85198, 2.85465, 2.85731, 2.85998, 2.86266, 2.86534, 2.86803, 2.87072, 2.87342,
                                                2.87612, 2.87883, 2.88154, 2.88426, 2.88698, 2.88971, 2.89244, 2.89518, 2.89793, 2.90067, 2.90343, 2.90619, 2.90895, 2.91172, 2.9145, 2.91728,
                                                2.92006, 2.92285, 2.92565, 2.92845, 2.93126, 2.93407, 2.93689, 2.93971, 2.94254, 2.94537, 2.94821, 2.95106, 2.95391, 2.95676, 2.95962, 2.96249,
                                                2.96536, 2.96824, 2.97112, 2.97401, 2.97691, 2.97981, 2.98271, 2.98563, 2.98854, 2.99147, 2.9944, 2.99733, 3.00027, 3.00322, 3.00617, 3.00913,
                                                3.01209, 3.01506, 3.01804, 3.02102, 3.024, 3.027, 3.03, 3.033, 3.03601, 3.03903, 3.04205, 3.04508, 3.04812, 3.05116, 3.0542, 3.05726,
                                                3.06032, 3.06338, 3.06645, 3.06953, 3.07262, 3.07571, 3.0788, 3.0819, 3.08501, 3.08813, 3.09125, 3.09438, 3.09751, 3.10065, 3.1038, 3.10695,
                                                3.11011, 3.11328, 3.11645, 3.11963, 3.12282, 3.12601, 3.12921, 3.13241, 3.13562, 3.13884, 3.14207, 3.1453, 3.14853, 3.15178, 3.15503, 3.15829,
                                                3.16155, 3.16483, 3.1681, 3.17139, 3.17468, 3.17798, 3.18129, 3.1846, 3.18792, 3.19125, 3.19458, 3.19792, 3.20127, 3.20462, 3.20798, 3.21135,
                                                3.21473, 3.21811, 3.2215, 3.2249, 3.2283, 3.23171, 3.23513, 3.23856, 3.24199, 3.24543, 3.24888, 3.25233, 3.2558, 3.25926, 3.26274 ])

        # wavelengths array in nm
        emissionSpectrum['wavelength'] = np.array([ 496.0099202, 495.6035172, 495.1977796, 494.7927058, 494.3882942, 493.9825752, 493.5755569, 493.1711702, 492.7654873, 492.3604714, 491.9541689, 
                                                    491.5485365, 491.1435724, 490.7373328, 490.3337037, 489.9268666, 489.5226365, 489.1171436, 488.7103957, 488.3062468, 487.900846, 487.4942012,
                                                    487.0882337, 486.6829418, 486.2783238, 485.8724741, 485.4673014, 485.0609064, 484.6570856, 484.2501543, 483.8457937, 483.440223, 483.0353317,
                                                    482.6292395, 482.2238296, 481.8172281, 481.4131806, 481.0079444, 480.6015271, 480.195796, 479.7907493, 479.3845321, 478.9808524, 478.5741578,
                                                    478.1681532, 477.762837, 477.3582073, 476.9524278, 476.5473377, 476.1411067, 475.7355678, 475.330719, 474.9247397, 474.5194534, 474.1148581,
                                                    473.7091426, 473.3041208, 472.8979875, 472.4925506, 472.0878083, 471.6819646, 471.2768181, 470.8705789, 470.4650395, 470.060198, 469.6542739,
                                                    469.2490501, 468.8427524, 468.4389273, 468.032264, 467.6280698, 467.2210521, 466.8164997, 466.410893, 466.0059904, 465.6000421, 465.1948003,
                                                    464.7885212, 464.3829512, 463.9780883, 463.5721976, 463.1670165, 462.7625431, 462.3570515, 461.9505489, 461.5447605, 461.1396844, 460.7336068,
                                                    460.3282437, 459.9235933, 459.5179508, 459.1130233, 458.7071118, 458.3019175, 457.8957475, 457.4902968, 457.0855635, 456.6798637, 456.2748835,
                                                    455.868945, 455.4637282, 455.0575611, 454.6521178, 454.2473963, 453.8417337, 453.4367949, 453.030923, 452.625777, 452.2197058, 451.8143626,
                                                    451.408102, 451.0042118, 450.5977688, 450.1920577, 449.7870766, 449.3811948, 448.976045, 448.5700022, 448.1646933, 447.7601161, 447.3546548,
                                                    446.9483162, 446.5427151, 446.1378494, 445.732115, 445.3271179, 444.9212597, 444.5161406, 444.110168, 443.7049362, 443.3004433, 442.8935234,
                                                    442.4889289, 442.0834967, 441.6772337, 441.2717166, 440.8669435, 440.4613477, 440.0564976, 439.6508321, 439.245914, 438.8401879, 438.4336604,
                                                    438.0294328, 437.6228609, 437.2185846, 436.8119771, 436.4076611, 436.0010267, 435.5966796, 435.1900271, 434.7856577, 434.3789957, 433.9746125,
                                                    433.5694655, 433.1635612, 432.7584161, 432.3525207, 431.947386, 431.541508, 431.1363921, 430.7305398, 430.325451, 429.9196328, 429.5145792,
                                                    429.1088033, 428.7037933, 428.2980678, 427.891633, 427.4874426, 427.0810731, 426.6754755, 426.2706475, 425.8651244, 425.4589123, 425.0534745,
                                                    424.6488086, 424.243461, 423.8374378, 423.4321911, 423.0262754, 422.6211372, 422.2153366, 421.8103146, 421.4046368, 420.9997386, 420.5941911,
                                                    420.1880002, 419.7825932, 419.3779678, 418.972706, 418.5668137, 418.1617072, 417.7559766, 417.3510326, 416.9454709, 416.5392975, 416.1339146,
                                                    415.72932, 415.3227292, 414.9183213, 414.5119289, 414.1063318, 413.7015277, 413.2961367, 412.8901646, 412.4849892, 412.0792388, 411.674286,
                                                    411.2687641, 410.8626791, 410.4573952, 410.0529101, 409.6465147, 409.2409241, 408.8361358, 408.4308023, 408.024929, 407.6198616, 407.2142604,
                                                    406.8081309, 406.4028107, 405.9982974, 405.5919353, 405.1863857, 404.7816464, 404.3763962, 403.9706405, 403.5643848, 403.1589454, 402.7543199,
                                                    402.3492002, 401.9435918, 401.5375, 401.1322281, 400.7264783, 400.3215486, 399.9161466, 399.5102777, 399.1052318, 398.6997244, 398.2937609,
                                                    397.8886233, 397.4830348, 397.0770009, 396.6717957, 396.2661502, 395.8613336, 395.4560821, 395.0504008, 394.644295, 394.2390233, 393.8345831,
                                                    393.4284753, 393.0232042, 392.617524, 392.2126805, 391.806195, 391.4017866, 390.9957463, 390.5905477, 390.1849603, 389.778989, 389.3738617,
                                                    388.9683555, 388.5624755, 388.1574417, 387.7520388, 387.3462719, 386.9413534, 386.5360757, 386.1304436, 385.7244621, 385.3193334, 384.91386,
                                                    384.5080468, 384.1030883, 383.6977947, 383.2921706, 382.886221, 382.4811304, 382.0757188, 381.6699909, 381.2651238, 380.8587751, 380.454459,
                                                    380.0486707 ])

        # Produced by Klaus using an automated process to extract data points
        # from the plots on Eljen website,  https://eljentechnology.com/products/plastic-scintillators/ej-200-ej-204-ej-208-ej-212
        emissionSpectrum['amplitude'] = np.array([ 0.0292553, 0.0305851, 0.0305851, 0.0319149, 0.0319149, 0.0319149, 0.0332447, 0.0345745, 0.0359043, 0.0372341, 0.0372341, 0.0372341,
                                                    0.0372341, 0.0385639, 0.0398937, 0.0398937, 0.0412234, 0.0425532, 0.0425532, 0.0425532, 0.0425532, 0.0425532, 0.0425532, 0.043883,
                                                    0.0452128, 0.0465426, 0.0465426, 0.0478724, 0.0478724, 0.0478724, 0.0492022, 0.0505319, 0.0518617, 0.0518617, 0.0531915, 0.0531915,
                                                    0.0531915, 0.0545213, 0.0545213, 0.0558511, 0.0571809, 0.0585107, 0.0585107, 0.0598405, 0.0625, 0.0638298, 0.0638298, 0.0664894,
                                                    0.0664894, 0.0678192, 0.069149, 0.069149, 0.0704788, 0.0718085, 0.0731383, 0.0744681, 0.0744681, 0.0771277, 0.0784575, 0.0797873,
                                                    0.081117, 0.0824468, 0.0837766, 0.0851064, 0.0851064, 0.0890958, 0.0890958, 0.0930851, 0.0944149, 0.0970745, 0.0984043, 0.101064,
                                                    0.103723, 0.105053, 0.109043, 0.113032, 0.113032, 0.115692, 0.118351, 0.121011, 0.12367, 0.12766, 0.128989, 0.132979,
                                                    0.134308, 0.139628, 0.142287, 0.144947, 0.147606, 0.151596, 0.155585, 0.158245, 0.162234, 0.166223, 0.170213, 0.174202,
                                                    0.179521, 0.182181, 0.1875, 0.191489, 0.195479, 0.200798, 0.204787, 0.210106, 0.215426, 0.222075, 0.227394, 0.232713,
                                                    0.239362, 0.246011, 0.25133, 0.256649, 0.261968, 0.268617, 0.272606, 0.277926, 0.283245, 0.288564, 0.293883, 0.299202,
                                                    0.304521, 0.307181, 0.3125, 0.317819, 0.323138, 0.325798, 0.331117, 0.336436, 0.341755, 0.347075, 0.351064, 0.355053,
                                                    0.361702, 0.367021, 0.371011, 0.37633, 0.381649, 0.384308, 0.390958, 0.394947, 0.398936, 0.402926, 0.405585, 0.412234,
                                                    0.416223, 0.418883, 0.421543, 0.426862, 0.430851, 0.43484, 0.4375, 0.441489, 0.445479, 0.449468, 0.452128, 0.454787,
                                                    0.458777, 0.461436, 0.465426, 0.468085, 0.472075, 0.474734, 0.478723, 0.481383, 0.485372, 0.488032, 0.492021, 0.496011,
                                                    0.49867, 0.50266, 0.506649, 0.511968, 0.514628, 0.519947, 0.525266, 0.530585, 0.535904, 0.542553, 0.547872, 0.555851,
                                                    0.56117, 0.569149, 0.577128, 0.583777, 0.591755, 0.599734, 0.606383, 0.615691, 0.62633, 0.636968, 0.647606, 0.664894,
                                                    0.683511, 0.700798, 0.720745, 0.740691, 0.759309, 0.783245, 0.803191, 0.821809, 0.845745, 0.871011, 0.892287, 0.910904,
                                                    0.922872, 0.932181, 0.942819, 0.953457, 0.962766, 0.968085, 0.976064, 0.981383, 0.985372, 0.993351, 0.994681, 0.99734,
                                                    0.99867, 0.99867, 0.99867, 0.99734, 0.994681, 0.993351, 0.986702, 0.982713, 0.977394, 0.972074, 0.968085, 0.961436,
                                                    0.956117, 0.948138, 0.941489, 0.933511, 0.924202, 0.913564, 0.904255, 0.892287, 0.881649, 0.868351, 0.855053, 0.840426,
                                                    0.827128, 0.81117, 0.795213, 0.779255, 0.760638, 0.744681, 0.727394, 0.707447, 0.680851, 0.652926, 0.630319, 0.602394,
                                                    0.574468, 0.549202, 0.522606, 0.494681, 0.470745, 0.446808, 0.417553, 0.393617, 0.369681, 0.341755, 0.319149, 0.297872,
                                                    0.276596, 0.256649, 0.239362, 0.222075, 0.203458, 0.19016, 0.174202, 0.158245, 0.143617, 0.131649, 0.118351, 0.107713,
                                                    0.0997341, 0.0904256, 0.0851064, 0.0771277, 0.0718085, 0.0651596, 0.0598405, 0.0545213, 0.0492022, 0.0465426, 0.043883 ])
        # normalizing to area = 1.0
        emissionSpectrum['amplitude'] /= np.sum(emissionSpectrum['amplitude'])
        
        data['emissionSpectrum'] = emissionSpectrum
        
        return data
    
    
    
    @staticmethod
    def buildEJ550():
        
        data = {}
        timeResponse     = pd.DataFrame(columns=['time', 'amplitude'])
        emissionSpectrum = pd.DataFrame(columns=['energy', 'wavelength', 'amplitude'])
        
        data['name']                   = 'EJ-550'
        data['lightYield']             = None 
        data['refractiveIndex']        = 1.46  # source: Eljen website, https://eljentechnology.com/products/accessories/ej-550-ej-552
        data['attenuationLength']      = 1600  # mm;  high and arbitray value to prevent abosrption; only refractiveIndex is important in the optical coupling layer
        data['riseTime']               = None  
        data['fallTime']               = None
        timeResponse['time']           = None
        timeResponse['amplitude']      = None
        data['timeResponse']           = timeResponse
        emissionSpectrum['energy']     = None
        emissionSpectrum['wavelength'] = None
        emissionSpectrum['amplitude']  = None
        data['emissionSpectrum']       = emissionSpectrum
        
        return data
    
    
    
    @staticmethod
    def buildAir():
        
        data = {}
        timeResponse     = pd.DataFrame(columns=['time', 'amplitude'])
        emissionSpectrum = pd.DataFrame(columns=['energy', 'wavelength', 'amplitude'])
        
        data['name']                   = 'Air'
        data['lightYield']             = None 
        data['refractiveIndex']        = 1.0  
        data['attenuationLength']      = 3000.0  # mm
        data['riseTime']               = None  
        data['fallTime']               = None
        timeResponse['time']           = None
        timeResponse['amplitude']      = None
        data['timeResponse']           = timeResponse  
        emissionSpectrum['energy']     = None
        emissionSpectrum['wavelength'] = None
        emissionSpectrum['amplitude']  = None
        data['emissionSpectrum']       = emissionSpectrum
        
        return data
    
    
    
    @staticmethod
    def buildSensLGlass():
        
        data = {}
        timeResponse     = pd.DataFrame(columns=['time', 'amplitude'])
        emissionSpectrum = pd.DataFrame(columns=['energy', 'wavelength', 'amplitude'])
        
        data['name']                   = 'SensLGlass'
        data['lightYield']             = None 
        data['refractiveIndex']        = 1.53  # source: SensL datasheet, https://sensl.com/downloads/ds/DS-MicroJseries.pdf
        data['attenuationLength']      = 1600  # mm;  high and arbitray value to prevent abosrption; only refractiveIndex is important in the optical coupling layer
        data['riseTime']               = None  
        data['fallTime']               = None 
        timeResponse['time']           = None
        timeResponse['amplitude']      = None
        data['timeResponse']           = timeResponse  
        emissionSpectrum['energy']     = None
        emissionSpectrum['wavelength'] = None
        emissionSpectrum['amplitude']  = None
        data['emissionSpectrum']       = emissionSpectrum
        
        return data
    
    
    
    @staticmethod
    def buildTeflon():
        
        data = {}
        timeResponse     = pd.DataFrame(columns=['time', 'amplitude'])
        emissionSpectrum = pd.DataFrame(columns=['energy', 'wavelength', 'amplitude'])
        lambertianFraction = {}
        
        data['name']                   = 'Teflon'
        data['lightYield']             = None
        # use these values if the reflector is modeled as a hollow volume with finite thickness
        # data['refractiveIndex']        = 1.33  # sources: Min K. Yang, Roger H. French, Edward W. Tokarsky, "Optical properties of Teflon AF amorphous fluoropolymers", J. Micro/Nanolith. MEMS MOEMS 7(3), 033010 (Jul–Sep 2008).
        #                                        #          Masato Yamawaki et al 2008 Jpn. J. Appl. Phys. 47 1104.
        # data['attenuationLength']      = 1600  # mm;  high and arbitray value to prevent abosrption; only refractiveIndex is important in the optical coupling layer
        # use these values if the reflector is modeled as a zero-thickness layer to represent the air gap in between
        data['refractiveIndex']        = 1.0    
        data['attenuationLength']      = 3000.0  # mm
        data['riseTime']               = None  
        data['fallTime']               = None 
        timeResponse['time']           = None
        timeResponse['amplitude']      = None
        data['timeResponse']           = timeResponse 
        emissionSpectrum['energy']     = None
        emissionSpectrum['wavelength'] = None
        emissionSpectrum['amplitude']  = None
        data['emissionSpectrum']       = emissionSpectrum
        data['reflectivity']           = 0.945 # sources: M. Janecek, "Reflectivity Spectra for Commonly Used Reflectors",
                                               #          IEEE Trans. on Nucl. Sci., vol. 59, no. 3, pp. 490-497, 2012.
                                               #          B. Pichler, E. Lorenz, R. Mirzoyan, L. Weiss and S. Ziegler, "Production of a diffuse very high reflectivity material for light collection in nuclear detectors",
                                               #          Nucl. Inst. and Meth. in Phys. Res. A, vol. 442, no. 1-3, pp. 333-336, 2000.
       
        #lambertianFractionFunction       = lambda theta: -9.182E-5*np.exp(0.09479*theta) + 0.9799*np.exp(-9.27E-5*theta) # exponential fit with R2=0.9967 and RMSE=0.0037 to data from
                                                                                                                         # M. Janecek and W. Moses, "Optical Reflectance Measurements for Commonly Used Reflectors",
                                                                                                                         # IEEE Transactions on Nuclear Science, vol. 55, no. 4, pp. 2432-2437, 2008.
        #lambertianFraction['angle']      = np.arange(0.0, 90+1.0, 1.0)
        #lambertianFraction['angle']      = np.around(lambertianFraction['angle'], decimals = 2)
        #lambertianFraction['amplitude']  = lambertianFractionFunction(lambertianFraction['angle'])
        #lambertianFraction['amplitude'] /= np.sum(lambertianFraction['amplitude'])
        
        lambertianFraction['doubleExpFitParams']      = {'a1': -9.182E-5, 'b1': 0.09479,
                                                         'a2': 0.9799,    'b2': -9.27E-5}           # exponential fit with R2=0.9967 and RMSE=0.0037 to data in Figure 12 from
                                                                                                    # M. Janecek and W. Moses, "Optical Reflectance Measurements for Commonly Used Reflectors",
                                                                                                    # IEEE Transactions on Nuclear Science, vol. 55, no. 4, pp. 2432-2437, 2008.
        data['lambertianFraction']      = lambertianFraction
        
        specularLobeSigma  = np.array( [ [   0.0,  10.0,  30.0, 50.0, 62.0, 74.0, 90.0],
                                         [ 32.07, 25.97, 13.77, 13.4, 11.4, 7.60, 2.52] ])          # composite fit to diffusion lobe (cosine) and specular lobe (Gaussian) to to data in Figure 7 from
                                                                                                    # M. Janecek and W. Moses, "Optical Reflectance Measurements for Commonly Used Reflectors",
                                                                                                    # IEEE Transactions on Nuclear Science, vol. 55, no. 4, pp. 2432-2437, 2008.
        interpolation_func         = interpolate.interp1d(specularLobeSigma[0, :], specularLobeSigma[1, :])
        inAngles                   = np.arange(0.0, 90.0+1.0, 1.0)
        specularLobeSigma          = np.zeros([2, len(inAngles)])
        specularLobeSigma[0, :]    = inAngles
        specularLobeSigma[1, :]    = interpolation_func(inAngles)
        data['specularLobeSigma']  = specularLobeSigma
        
        return data
        
    
    
#%% main


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 'new' or 'load' or 'add' or 'delete'
    '''
    
    resetIPython()
    
    command  = sys.argv[1]
    
    if command == 'new':
        
        if len(sys.argv) > 2:
            materialsLibraryFileName        = sys.argv[2]
            materialsLibrary = materials(materialsLibraryFileName = materialsLibraryFileName)
        else:
            materialsLibrary = materials()
            
            
    if command == 'load':
        
        if len(sys.argv) > 2:
            materialsLibraryFileName        = sys.argv[2]
            materialsLibrary = materials(False, materialsLibraryFileName = materialsLibraryFileName)
        else:
            materialsLibrary = materials(False)
            
            
    if command == 'add':
        
        materialsLibraryFileName        = sys.argv[2]
        name                            = sys.argv[3]
        lightYield                      = float(sys.argv[4])
        refractiveIndex                 = float(sys.argv[5])
        attenuationLength               = float(sys.argv[6])
        
        materialsLibrary = materials(materialsLibraryFileName = materialsLibraryFileName)
        materialsLibrary.addMeterial(materialsLibraryFileName, name, lightYield, refractiveIndex, attenuationLength, riseTime=None, fallTime=None, emissionSpectrum=None, energies=None, wavelengths=None)
        
        
    if command == 'delete':
        
        materialsLibraryFileName        = sys.argv[2]
        name                            = sys.argv[3]
        
        materialsLibrary = materials(materialsLibraryFileName = materialsLibraryFileName)
        materialsLibrary.deleteMeterial(materialsLibraryFileName, name)
        
        
        