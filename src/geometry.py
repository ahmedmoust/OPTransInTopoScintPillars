'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


import sys

import numpy as np
import pickle
import copy

import volume


def resetIPython():
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
    #get_ipython().run_line_magic('clear', '')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    return


class geometry:
    
    def __init__(self, firstTime = True, geometryFileName = None):
        '''
        geometry instantiation

        Parameters
        ----------
        firstTime : bool, optional
                    Flag marking whether the geometry is being built for the first time.
                    The default is True.
        geometryFileName : string, optional
                           File name of the geometry to be built, or loaded.
                           The default is None.

        Returns
        -------
        None.

        '''
        
        if firstTime:
            print('Instantiating a geometry object...')
            self.volumes = {}
            return
        else:
            print('Loading the previously created geometry: {}.pkl...'.format(geometryFileName))
            self.loadGeometry(geometryFileName)
            return
            
      
    
    def loadGeometry(self, geometryFileName):
        '''
        loading a previously created geometry

        Parameters
        ----------
        geometryFileName : string
                           File name of the geometry to be loaded.

        Returns
        -------
        None.

        '''
        
        self.volumes = pickle.load(open(geometryFileName+'.pkl', 'rb'))
        
        return
    
    
    def saveGeometry(self, geometryFileName):
        '''
        saving the current geometry

        Parameters
        ----------
        geometryFileName : string
                           File name of the geometry to be saved.

        Returns
        -------
        None.

        '''
        
        pickle.dump(self.volumes, open(geometryFileName+'.pkl','wb'), protocol=2)

        return
    
    
    
    def createGeometry(self, volumesConfiguration):
        '''
        creating full geometry given the configuration of each volume

        Parameters
        ----------
        volumesConfiguration : dict
                               Dimensions and properties of the volume to be created.

        Raises
        ------
        ValueError: if the dictionary for each volume doesn't have all necessary information.

        Returns
        -------
        None.

        '''
        
        # == check if all the necessary information to create a volume is included in the dictionary, then create the volume
        for vol in volumesConfiguration:
            if 'name'and 'material' and 'width' and 'length' and 'center' not in list(vol.keys()):
                raise ValueError(f"A configuration setting was not sepcified for volume:{vol}...\n"
                                 +"Please specify the name, material, width, length, and center for the given volume.")
                
            self.volumes[vol['name']] = volume.volume(vol['name'],
                                                      vol['material'],
                                                      vol['width'],
                                                      vol['length'],
                                                      vol['center'], 
                                                      vol['surfaceTrimeshes']
                                                      )
        # == check for volume overlaps
        self.checkOverlaps()
        # == identify and store the touching volumes of each created volume
        self.identifyTouchingVolumes()
        
        return
        
    
    
    def addVolume(self, name, material, width, length, center, surfaceTrimeshes=None):
        '''
        adding a single volume knowing its properties to create it

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

        Returns
        -------
        None.

        '''
        
        self.volumes[name] = volume.volume(name, material, width, length, center, surfaceTrimeshes=surfaceTrimeshes)
        
        return
    
    
    
    def addSurfaceTrimeshes(self, volumeName, surfaceTrimeshes, alreadyCreated):
        '''
        adding triangulated meshes to each of the volume surfaces either by loading previously created ones or creating new

        Parameters
        ----------
        volumeName : string
                     Volume to which the surface trimeshes are to be added.
        surfaceTrimeshes : dict
                           If alreadyCreated == True: {'normalOrientation': surfaceTrimeshes.pkl}.
                           If alreadyCreated == False: {'normalOrientation': {'finish': 'pointCloudFileName'}}.
        alreadyCreated : bool, optional
                         Whether the surface trimeshes were already created (triangulated, oriented, modified, etc.) and stored.
                         The default is False.

        Raises
        ------
        ValueError: if the entered volume name to add its surfaces trimeshes was not previously created

        Returns
        -------
        None.

        '''
        
        # == cheack if the entered volume name was previously created
        if volumeName not in list(self.volumes.keys()):
            raise ValueError(f"The {volumeName} volume hasn't been previously created...\n"
                                 +"Please first create a volume before adding its surface trimeshes.")
        
        # == add the surface trmeshes
        self.volumes[volumeName].addSurfaceTrimeshes(surfaceTrimeshes, alreadyCreated)
        
        return
    
    
    
    def checkGeometry(self):
        '''
        checking geometry's volumes for correct placements, i.e. touching volumes without gaps or overlaps

        Raises
        ------
        ValueError: if placement issues were detected.

        Returns
        -------
        None.

        '''
        
        print('checking volume overlaps...')
        
        #= set ray directions pointing towards each of the 6 sides of a volume
        ray_directions = np.array([[ 1.0,  0.0,  0.0],
                                   [-1.0,  0.0,  0.0],
                                   [ 0.0,  1.0,  0.0],
                                   [ 0.0, -1.0,  0.0],
                                   [ 0.0,  0.0,  1.0],
                                   [ 0.0,  0.0, -1.0]
                                  ])
        
        #= empty dictionary to record the placement issues counter of each volume
        volumesIssuesRecord = {}
        
        #= loop over all the created volumes
        for currentVol in self.volumes.values():
            
            #= skip the enclosing environment volume and reflector, if any
            if currentVol.name == 'environment' or currentVol.name == 'reflector': continue
        
            issuesCounter = 0
            
            #= set each ray origin as the center of the current volume
            ray_origin = np.round(np.array([currentVol.volumeTrimesh.centroid]), 9)
            
            #= loop over all six ray directions and intersect each with all the other volumes
            for direction in ray_directions:
                # emoty array to record intersections with the other volumes
                volumeIntersects = np.empty((0,), dtype=[('volume', object), ('intersect_loc', np.ndarray), ('intersect_normal', np.ndarray)])
                
                for otherVol in self.volumes.values():
                    #= skip intersecting with current volume
                    if otherVol.name == currentVol.name or otherVol.name == 'environment' or otherVol.name == 'reflector': continue
                    
                    #= copying the trimesh before intersecting prevents an issue that arises when trying to
                    #  save the final geometry to a pickle file + another issue while later intersecting with
                    #  that volume during tracking when mistakenly no intersection is detected (exact cause
                    #  of both issues is not understood) 
                    volTrimesh = copy.deepcopy(otherVol.volumeTrimesh)
                    intersect_locs, _, intersect_tris = volTrimesh.ray.intersects_location(ray_origins    = ray_origin,
                                                                                           ray_directions = np.array([direction]))
                    intersect_faceNormals = volTrimesh.face_normals[intersect_tris]
                    #= record intersections
                    for i in range(len(intersect_locs)):
                        volumeIntersects = np.append(volumeIntersects, np.array((otherVol.name,
                                                                                 np.round(np.array([intersect_locs[i]]), 9),
                                                                                 intersect_faceNormals[i]),
                                                                                 dtype = volumeIntersects.dtype))
                if len(volumeIntersects['intersect_loc']) != 0:
                    #= estimate the distance between the photon origin and all intersections
                    distances = np.round(np.linalg.norm(ray_origin-np.vstack(volumeIntersects['intersect_loc']), axis = 1), 9)
                    #= find the shortest distance
                    closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
                    #= get the axis along which the ray is directed
                    rayIncidenceAxisIndex = np.where(np.abs(direction) == 1.0)[0]
                    #= ovelap if the shortest distance is within the current volume bounds, and nontouching if beyond them
                    if distances[closestIntersectIndex] < np.round(currentVol.volumeTrimesh.extents[rayIncidenceAxisIndex], 9)/2.0:
                        issuesCounter += 1
                        print(f" - Warning! {volumeIntersects['volume'][closestIntersectIndex][0]} intersects with {currentVol.name}...")
                    elif distances[closestIntersectIndex] > np.round(currentVol.volumeTrimesh.extents[rayIncidenceAxisIndex], 9)/2.0:
                        issuesCounter += 1
                        print(f" - Warning! {volumeIntersects['volume'][closestIntersectIndex][0]} is not in perfect touch with {currentVol.name}...")
                            
            #= check and record if any issues were detected for the current volume
            if issuesCounter == 0:
                print(f" - {currentVol.name}... OK!")
            else:
                volumesIssuesRecord[currentVol.name] = issuesCounter
        
        #= stop if any issues were detected for all the geometry volumes
        if len(volumesIssuesRecord) != 0:
            raise ValueError('Volumes placement issues detected...\n'
                                 'Please correct any placement issues before continuing.')
            
        return
 
    
 
    def identifyTouchingVolumes(self):
        '''
        idntifying the touching volumes of each of the geometry's volumes on all of their 6 sides

        Returns
        -------
        None.

        '''
        
        print('Identifying touching volumes...')
        
        #= set ray directions pointing towards each of the 6 sides of a volume
        ray_directions = np.array([[ 1.0,  0.0,  0.0],
                                   [-1.0,  0.0,  0.0],
                                   [ 0.0,  1.0,  0.0],
                                   [ 0.0, -1.0,  0.0],
                                   [ 0.0,  0.0,  1.0],
                                   [ 0.0,  0.0, -1.0]
                                  ])
        
        #= loop over all the created volumes
        for currentVol in self.volumes.values():
            
            #= skip the enclosing environment volume and reflector, if any
            if currentVol.name == 'environment' or currentVol.name == 'reflector' : continue
            
            #= empty dictionary to record the touching volumes of all 6 sides of the current volume
            touchingVolumes = {}
            
            #= set each ray origin as the center of the current volume
            ray_origin = np.round(np.array([currentVol.volumeTrimesh.centroid]), 9)
            
            #= loop over all six ray directions and intersect each with all the other volumes
            for direction in ray_directions:
                # emoty array to record intersections with the other volumes
                volumeIntersects = np.empty((0,), dtype=[('volume', object), ('intersect_loc', np.ndarray), ('intersect_normal', np.ndarray)])
                
                for otherVol in self.volumes.values():
                    #= skip intersecting with current volume
                    if otherVol.name == currentVol.name or otherVol.name == 'environment': continue
                    
                    #= copying the trimesh before intersecting prevents an issue that arises when trying to
                    #  save the final geometry to a pickle file + another issue while later intersecting with
                    #  that volume during tracking when mistakenly no intersection is detected (exact cause
                    #  of both issues is not understood) 
                    volTrimesh = copy.deepcopy(otherVol.volumeTrimesh)
                    intersect_locs, _, intersect_tris = volTrimesh.ray.intersects_location(ray_origins    = ray_origin,
                                                                                           ray_directions = np.array([direction]))
                    intersect_faceNormals = volTrimesh.face_normals[intersect_tris]
                    #= record intersections
                    for i in range(len(intersect_locs)):
                        volumeIntersects = np.append(volumeIntersects, np.array((otherVol.name,
                                                                                 np.round(np.array([intersect_locs[i]]), 9),
                                                                                 np.array([intersect_faceNormals[i]])),
                                                                                 dtype = volumeIntersects.dtype))
                if len(volumeIntersects['intersect_loc']) != 0:
                    #= estimate the distance between the photon origin and all intersections
                    distances = np.round(np.linalg.norm(ray_origin-np.vstack(volumeIntersects['intersect_loc']), axis = 1), 9)
                    #= find the shortest distance
                    closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
                    #= if more than one, it must be due to a volume enclsoing either the currentVol or the otherVol
                    #  since it wasn't detected as an error in the prior executed checkGeometry(), it must be of a reflector
                    if len(closestIntersectIndex) > 1:
                        reflectorIntersectIndices = np.where(volumeIntersects['volume'] == 'reflector')[0]
                        volumeIntersects          = np.delete(volumeIntersects, reflectorIntersectIndices)
                        distances = np.round(np.linalg.norm(ray_origin-np.vstack(volumeIntersects['intersect_loc']), axis = 1), 9)
                        closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
                        if len(closestIntersectIndex) > 1:
                            raise ValueError(f"Eliminating the reflector volume intersection has failed during identifying the next touhcing volume of {currentVol.name}...\n")
                    touchingVolumes[tuple(direction)] = volumeIntersects['volume'][closestIntersectIndex][0]
                else:
                    touchingVolumes[tuple(direction)] = 'environment'
            
            #= record all the touching volumes of the current volume
            currentVol.touchingVolumes = touchingVolumes
        
        return

#%% main


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 
    '''
    
    resetIPython()   
    
    command  = sys.argv[1]