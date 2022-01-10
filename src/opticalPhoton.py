'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


import sys


def resetIPython():
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
    #get_ipython().run_line_magic('clear', '')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    return



class opticalPhoton:
    
    def __init__(self, position, wavelength, momentumDirection, polarizationDirection,
                 time = 0.0, traveledDistance = 0.0, weight = 1.0,
                 volume = 'pillar', alive = True,
                 trackingHistory = None):
        '''

        optical photon obeject instantiation

        Parameters
        ----------
        position : (1x3) array of floats
                   Position relative to the pillar center in units of um.
        wavelength : float
                     In units of nm.
        momentumDirection : (1x3) array of floats
                            Directional unit vector.
        polarizationDirection : list of 3 float numbers
                                Directional unit vector.
        time : float, optional
               In units of ns.
               The default is 0.0.
        traveledDistance : float, optional
                           Cumulative traveled distance up to the moment of current photon information.
                           The default is 0.0.
        weight : float, optional
                 Weight representing the importance of the photon.
                 The default is 1.0.
        volume : string, optional
                 In which the photon exists.
                 The default is 'pillar'.
        alive :  bool, optional
                 Tracking status of the optical photon.
                 The default is True.

        Returns
        -------
        None.

        '''
        
        self.position         = position
        self.wavelength       = wavelength
        self.momentumDir      = momentumDirection
        self.polarizationDir  = polarizationDirection
        self.time             = time
        self.traveledDistance = traveledDistance
        self.weight           = weight
        self.volume           = volume
        self.alive            = alive
    
    
#%% main


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 
    '''
    
    resetIPython()  
    
    command  = sys.argv[1] 