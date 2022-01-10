'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


import sys

import numpy as np
import pickle

import trimesh

import materials
import surface



def resetIPython():
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
    #get_ipython().run_line_magic('clear', '')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    return


class volume:
    
    def __init__(self, name, material, width, length, center, surfaceTrimeshes = None, touchingVolumes = None):
        '''
        volume instantiation

        Parameters
        ----------
        name : string
               Preferably descriptive of the modeled component.
        material : materials object
                   Of the volume.
        width : float
                Dimension in mm of the cuboid's regular cross section.
        length : float
                 Dimension in mm of the cuboid's third dimesnion.
        center : array of floats
                 (x, y, z) coordinates in mm of the cuboid center.
        surfaceTrimeshes : dict, optional
                           Trimeshes for each side surface of the volume.
                           The default is None.
        touchingVolumes : dict, optional
                          Touching volume name.
                          The default is None.

        Returns
        -------
        None.

        '''
        
        print(' - Instantiating the volume: {} with material:{}, width={:2.1f} mm, length={:2.1f} mm, '
                  'and positioned at {}...'.format(name, material['name'], width, length, center))
        
        self.name     = name
        self.material = material
        self.width    = width*1E3    #um
        self.length   = length*1E3   #um
        
        # == create a trimesh volume of the cuboid to intersect light rays with in order to first identify 
        #    the intersecting plane before switching to intersecting with this plane's respective surface mesh
        translationMatrix = np.array([[1, 0, 0, center[0]*1E3],
                                      [0, 1, 0, center[1]*1E3],
                                      [0, 0, 1, center[2]*1E3],
                                      [0, 0, 0,             1]
                                      ])
        self.volumeTrimesh    = trimesh.creation.box(extents=np.array([width, width, length])*1E3, transform=translationMatrix)
        self.surfaceTrimeshes = surfaceTrimeshes
        self.touchingVolumes  = touchingVolumes
        
        
        
    def addSurfaceTrimeshes(self, surfaceTrimeshes, alreadyCreated = False):
        '''
        adding triangulated meshes to each of the volume surfaces either by loading previously created ones or creating new

        Parameters
        ----------
        surfaceTrimeshes : dict
                           If alreadyCreated == True: {'normalOrientation': surfaceTrimeshes.pkl}.
                           If alreadyCreated == False: {'normalOrientation': {'finish': 'pointCloudFileName'}}.
        alreadyCreated : bool, optional
                         Whether the surface trimeshes were already created (triangulated, oriented, modified, etc.) and stored.
                         The default is False.

        Raises
        ------
        ValueError: if the entered dict doesn't contain information for all 6 side surfaces of the cuboid.

        Returns
        -------
        None.

        '''
        
        # == check if information for all surfaces are entered
        surfacesNormaldict = {'-x': (-1.0, 0.0, 0.0),
                              '+x':  (1.0, 0.0, 0.0),
                              '-y': (0.0, -1.0, 0.0),
                              '+y':  (0.0, 1.0, 0.0),
                              '-z': (0.0, 0.0, -1.0),
                              '+z':  (0.0, 0.0, 1.0)
                              }
        for surf in surfacesNormaldict.keys():
            if surf not in list(surfaceTrimeshes.keys()):
                raise ValueError('A surface is missing from the entered surfaces list...\n'
                                 +'Please specifiy information for all surfaces:'+' '.join(surfacesNormaldict.keys()))
        
        if alreadyCreated:
            # = load surfaces directly if they were previously created and stored
            for surf, normal in surfacesNormaldict.items():
                surfaceTrimeshes[normal] = pickle.load(open(surfaceTrimeshes[surf]+'.pkl', 'rb'))
            self.surfaceTrimeshes = surfaceTrimeshes
        else:
            # = create new surfaces
            for surf, normal in surfacesNormaldict.items():
                finish, pointCloudFileName = list(surfaceTrimeshes[surf].keys())[0], list(surfaceTrimeshes[surf].values())[0]
                aSurface = surface.surface(surfaceFinish=finish, normalOrientation=surf)
                aSurface.loadPointCloud(pointCloudFileName)
                aSurface.createTriangularMesh()
                aSurface.applyDefaultOrientation()
                surfaceTrimeshes[normal] = aSurface
            self.surfaceTrimeshes = surfaceTrimeshes
        
        return
    
    
    
    def addTouchingVolumes(self, touchingVolumes):
        '''
        adding the names of the touching volumes of all side planes

        Parameters
        ----------
        touchingVolumes : dict
                          Touchong volume of each of the volume side planes

        Raises
        ------
        ValueError: if the entered dict doesn't contain information for all 6 side surfaces of the cuboid.

        Returns
        -------
        None.

        '''
        
        # == check if information for all surfaces are entered
        surfacesNormaldict = {'-x': (-1.0, 0.0, 0.0),
                              '+x':  (1.0, 0.0, 0.0),
                              '-y': (0.0, -1.0, 0.0),
                              '+y':  (0.0, 1.0, 0.0),
                              '-z': (0.0, 0.0, -1.0),
                              '+z':  (0.0, 0.0, 1.0)
                              }
        for surf in surfacesNormaldict.keys():
            if surf not in list(touchingVolumes.keys()):
                raise ValueError('A touching surface is missing from the list...\n'
                                 +'Please specifiy information for all surfaces:'+' '.join(surfacesNormaldict.keys()))
        
        for surf, normal in surfacesNormaldict.items():
                touchingVolumes[normal] = pickle.load(open(touchingVolumes[surf]+'.pkl', 'rb'))
        self.touchingVolumes = touchingVolumes
        
        return
            
#%% main


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 'new' or 'addCreatedSurfaces' or 'createAndAddSurfaces'
    '''
    
    resetIPython()
    
    command  = sys.argv[1]
    
    if command == 'new':
        
        name      = sys.argv[2]
        material  = sys.argv[3]
        width     = float(sys.argv[4])
        length    = float(sys.argv[5])
        center    = np.array([float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])])
        
        materialsLibrary = materials.materials().materialsLibrary
        
        aVolume = volume(name, materialsLibrary[material], width, length, center)
        
    
    if command == 'addCreatedSurfaceTrimeshes':
        
        name                     = sys.argv[2]
        material                 = sys.argv[3]
        width                    = float(sys.argv[4])
        length                   = float(sys.argv[5])
        center                   = np.array([float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])])
        surfaceNormalOrientation = sys.argv[9]
        surfaceTrimeshesFileName = sys.argv[10]
        
        materialsLibrary = materials.materials().materialsLibrary
        
        aVolume = volume(name, materialsLibrary[material], width, length, center)
        aVolume.addSurfaceTrimeshes({surfaceNormalOrientation: surfaceTrimeshesFileName}, alreadyCreated = True)