'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


import sys

import numpy as np
import copy


def resetIPython():
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
    #get_ipython().run_line_magic('clear', '')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    return



def getVolumeTrimeshIntersection(volumeList, opPosition, opMomentumDir, opVolume):
    '''
    intersecting geometry volumes to find the first intersection point, plane, and next volume

    Parameters
    ----------
    volumeList    : dict of volume objects
                    All Geometry volume objects.
    opPosition    : (1x3) array of floats
                    Photon's origin position in um.
    opMomentumDir : (1x3) array of floats
                    Photon's momentum direction unit vector.
    opVolume      : string
                    Photon's current associated volume.

    Returns
    -------
    volumeIntersectPoint : (1x3) array of floats
                           Initial intersection point on a volume in um.
    intersectPlaneNormal : (1x3) array of floats
                           Normal unit vector of the intersected plane.
    nextVolume           : string
                           Next volume to the intersected plane.

    '''
    
    #### == the photon is inside any of the geometry volumes
    if opVolume != 'environment' and opVolume != 'reflector':
        #= intersect with the current volume
        intersect_locs, _, intersect_tris = volumeList[opVolume].volumeTrimesh.ray.intersects_location(ray_origins    = opPosition,
                                                                                                       ray_directions = opMomentumDir)
        # this check is unnecessary but put here as a fail safe
        if len(intersect_locs) == 0:
            nextVolume               = None
            volumeIntersectPoint     = None
            intersectPlaneNormal     = None
        else:
            distances              = np.round(np.linalg.norm(opPosition[0]-np.round(np.vstack(intersect_locs), 9), axis = 1), 9)
            farthestIntersectIndex = np.argmax(distances)
            volumeIntersectPoint   = np.round(np.array([intersect_locs[farthestIntersectIndex]]), 9)
            intersectPlaneNormal   = np.array([volumeList[opVolume].volumeTrimesh.face_normals[intersect_tris[farthestIntersectIndex]]])
            nextVolume             = volumeList[opVolume].touchingVolumes[tuple(intersectPlaneNormal[0])]
            #= reverse the normal to point towards the photon's incidence volume
            intersectPlaneNormal   *= -1
        
    #### == the photon is at the outside 'environment' or between the reflector and a geometry volume
    else:
        #= create a record array to record the intersections info with all volumes
        volumeIntersectsInfo = np.empty((0,), dtype=[('volume', object), ('intersect_loc', np.ndarray), ('intersect_normal', np.ndarray)])
        
        #= loop over all the created volumes in the geometry
        for key, vol in volumeList.items():
            #= intersect with the current volume
            intersect_locs, _, intersect_tris = vol.volumeTrimesh.ray.intersects_location(ray_origins = opPosition,
                                                                                          ray_directions = opMomentumDir)
            intersect_triFaceNormals = np.array(vol.volumeTrimesh.face_normals[intersect_tris])
            
            # # this next elimination check is not necessary when using the pyembree accelerator
            # #= eliminating the photon origin point that is considered an intersection point if it is exactly
            # #  on a volume trimesh boundary.
            # #  Note: this is NOT redundant to the previous check. A transmitted photon to the outside environment
            # #        with an origin exactly on the boundary of a volume won't be detected by the previous check
            # #        since the photon's associated volume is not that volume and rather the outside environment
            # trueIntersectIndices = []
            # for l, loc in enumerate(intersect_locs):
            #     if not np.array_equal(loc, opPosition[0]): trueIntersectIndices.append(l)
            # intersect_locs = intersect_locs[trueIntersectIndices]
            # intersect_tris = intersect_tris[trueIntersectIndices]
            # intersect_triFaceNormals = np.array(vol.volumeTrimesh.face_normals[intersect_tris])
            
            #= eliminating the replica second intersection that trimesh sometimes produce when intersecting
            #  the 'environment' or 'reflector' volumes. It's a bug of unknown reason!
            if (vol.name == 'environment' or vol.name == 'reflector') and len(intersect_locs) > 1:
                if np.isclose(intersect_locs, intersect_locs[0]).all():
                    intersect_locs = np.array([intersect_locs[0]])
                    intersect_triFaceNormals = np.array([intersect_triFaceNormals[0]])

            #= do NOT record if no volume intersections
            if len(intersect_locs) == 0: continue
            #= record all intersection point locations and local surface normals
            for i in range(len(intersect_locs)):
                volumeIntersectsInfo = np.append(volumeIntersectsInfo, np.array((vol.name,
                                                                                 np.round(np.array([intersect_locs[i]]), 9),
                                                                                 np.array([intersect_triFaceNormals[i]])),
                                                                                 dtype = volumeIntersectsInfo.dtype))
       
        # == if only one intersection, it must be with the outside 'environment' volume
        if len(volumeIntersectsInfo) == 1:
            nextVolume               = None
            volumeIntersectPoint     = None
            intersectPlaneNormal     = None
        else:
            # this replaces the next two checks on line 119 & 131
            #= eliminating multiple closestIntersectIndex in case of touching volumes, eg. pillar-OG or reflector-PD;
            # otherwise, two closest intersection will be detected in the next elimination check.
            distances = np.round(np.linalg.norm(opPosition-np.vstack(volumeIntersectsInfo['intersect_loc']), axis = 1), 9)
            closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
            # several closest intersections means the photon is spacially inside a volume that is touching another despite it is supposed to be outside!
            if len(closestIntersectIndex) > 1:
                touchingIndices = np.where( (volumeIntersectsInfo['volume'] == 'pillar') | (volumeIntersectsInfo['volume'] == 'reflector'))[0]
                volumeIntersectsInfo = np.delete(volumeIntersectsInfo, touchingIndices)
                
            # #= eliminating the intersection with the pillar volume at its boundary with a touching volume
            # #  if the photon had transmitted to the outside 'environment' but under the pillar average
            # #  surface and headed towards that boundary; otherwise, two closest intersection will be
            # #  detected in the next elimination check.
            # pillarIntersectIndex = np.where(volumeIntersectsInfo['volume'] == 'pillar')[0]
            # if len(pillarIntersectIndex) != 0:
            #     distances = np.round(np.linalg.norm(opPosition-np.vstack(volumeIntersectsInfo['intersect_loc']), axis = 1), 9)
            #     closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
            #     # several closest intersections means the photon is spacially inside a volume that is touching another despite it is supposed to be outside!
            #     if len(closestIntersectIndex) > 1:
            #         volumeIntersectsInfo = np.delete(volumeIntersectsInfo, pillarIntersectIndex)
                    
            # #= eliminating the intersection with the reflector volume at its boundary with a touching volume
            # #  if the photon is in the air gap and headed towards that boundary; otherwise, two closest 
            # #  intersection will be detected in the next elimination check.
            # reflectorIntersectIndex = np.where(volumeIntersectsInfo['volume'] == 'reflector')[0]
            # if len(reflectorIntersectIndex) != 0:
            #     distances = np.round(np.linalg.norm(opPosition-np.vstack(volumeIntersectsInfo['intersect_loc']), axis = 1), 9)
            #     closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
            #     # several closest intersections means the photon is headed towards a touching volume with the reflector
            #     if len(closestIntersectIndex) > 1:
            #         volumeIntersectsInfo = np.delete(volumeIntersectsInfo, reflectorIntersectIndex)
                    
            #= eliminating the intersection with a volume other than the photon's current associated volume
            #  if it is slightly inside it.
            #  This happens when the origin of a photon that is associated with the outside 'environment' or 'reflector'
            #  is slightly above the irregular surface trimesh of another volume (which means it is spacially in that 
            #  other volume) and that trimesh.ray.intersects_location() would undesirably detect and intersect the
            #  neighbouring volume.
            distances = np.round(np.linalg.norm(opPosition-np.vstack(volumeIntersectsInfo['intersect_loc']), axis = 1), 9)
            closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
            # check if that one intersection is not with the outside 'environment' or 'refelctor' volumes
            if volumeIntersectsInfo['volume'][closestIntersectIndex] != opVolume:
                # check if there is only one intersection with that volume and only then eliminate it; otherwise, two intersections means
                # that the photon is headed towards another volume from its outside and therfore both intersections should be kept.
                if len(np.where(volumeIntersectsInfo['volume'] == volumeIntersectsInfo['volume'][closestIntersectIndex])[0]) == 1:
                    # eliminate it
                    volumeIntersectsInfo = np.delete(volumeIntersectsInfo, closestIntersectIndex)
            
            # == if only one intersection, it must be with the outside 'environment' volume        
            if len(volumeIntersectsInfo) == 1: 
                nextVolume               = None
                volumeIntersectPoint     = None
                intersectPlaneNormal     = None
            else:
                #= estimate the distance between the photon origin and all intersections
                distances = np.round(np.linalg.norm(opPosition-np.vstack(volumeIntersectsInfo['intersect_loc']), axis = 1), 9)
                #= find the shortest distances
                closestIntersectIndex = np.where(distances == distances[np.argmin(distances)])[0]
                #= eliminate the others
                volumeIntersectsInfo = volumeIntersectsInfo[closestIntersectIndex]
                #
                nextVolume               = volumeIntersectsInfo['volume'][0]
                volumeIntersectPoint     = volumeIntersectsInfo['intersect_loc'][0]
                intersectPlaneNormal     = volumeIntersectsInfo['intersect_normal'][0]
                if nextVolume == opVolume:
                    intersectPlaneNormal    *= -1
    
    return volumeIntersectPoint, intersectPlaneNormal, nextVolume



def sampleSurfaceTrimeshPoint(samplingBounds, volumeIntersectPlaneNormal):
    '''
    sampling an initial point on a surface trimesh in its frame of reference

    Parameters
    ----------
    samplingBounds             : (2x3) array of floats
                                 Bounds on the allowed area to sample within in um.
    volumeIntersectPlaneNormal : (1x3) array of floats
                                 Normal unit vector of the intersected plane of the volume.
                                 For cuboid-shaped volumes, the two lateral coordinates will have 0.0 values.

    Returns
    -------
    trimeshSampledPoint        : (1x3) array of floats
                                 Initial sampled point on the surface mesh in its frame of reference in um.

    '''
    
    # == limit the mesh area from within a point is sampled.
    #    This is done to leave areas on the mesh periphery to allow photon interactions without running out of
    #    mesh area in the potential case of multiple local interactions.
    #    Note: This limit on sampling is fine ONLY IF the mesh already contain multiples of the expected surface
    #    features, not only one set, which is recommended anyways for used measured point clouds.
    #    Leave out an arbitrary 5%
    
    #= get the lateral coordinates indices
    lateralCoordIndices = np.where(volumeIntersectPlaneNormal[0, :] == 0.0)[0]
    
    #= sample the two lateral coordinates within the limitied area
    ax1_point  = np.random.uniform(samplingBounds[0, lateralCoordIndices[0]], samplingBounds[1, lateralCoordIndices[0]])
    ax2_point  = np.random.uniform(samplingBounds[0, lateralCoordIndices[1]], samplingBounds[1, lateralCoordIndices[1]])
    #
    trimeshSampledPoint = np.zeros([1, 3])
    trimeshSampledPoint[0, lateralCoordIndices[0]] = ax1_point
    trimeshSampledPoint[0, lateralCoordIndices[1]] = ax2_point
    
    return trimeshSampledPoint



def shiftSurfaceTrimesh(surfaceTrimesh, volumeIntersectPlaneNormal, volumeIntersectPoint, trimeshSampledPoint):
    '''
    shifting the surface trimesh such that the two lateral coordinates of the sampled mesh point match
    the two of the volume intersection point

    Parameters
    ----------
    surfaceTrimesh             : trimesh object
                                 Surface trimesh to be shifted.
    volumeIntersectPlaneNormal : (1x3) array of floats
                                 Normal unit vector of the intersected plane of the volume.
                                 For cuboid-shaped volumes, the two lateral coordinates will have 0.0 values.
    volumeIntersectPoint       : (1x3) array of floats
                                 Initial intersection point in the volume frame of reference in um.
    trimeshSampledPoint        : (1x3) array of floats
                                 Initial sampled point on the surface mesh in its frame of reference in um.

    Returns
    -------
    surfaceTrimesh             : trimesh object
                                 Shifted surface trimesh that is ready to perform the intersecting.

    '''
    
    #= get the lateral and vertical coordinates indices
    lateralCoordIndices = np.where(volumeIntersectPlaneNormal[0, :] == 0.0)[0]
    verticalCoordIndex  = np.where(np.abs(volumeIntersectPlaneNormal[0, :]) == 1.0)[0]
    
    #= calculate the shift values for each coordinate
    ax1_shift  = volumeIntersectPoint[0, lateralCoordIndices[0]] - trimeshSampledPoint[0, lateralCoordIndices[0]]
    ax2_shift  = volumeIntersectPoint[0, lateralCoordIndices[1]] - trimeshSampledPoint[0, lateralCoordIndices[1]]
    #
    shiftVector = np.zeros(3)
    shiftVector[lateralCoordIndices[0]] = ax1_shift
    shiftVector[lateralCoordIndices[1]] = ax2_shift
    shiftVector[verticalCoordIndex]     = volumeIntersectPoint[0, verticalCoordIndex]
   
    #= shift the surface mesh
    surfaceTrimesh = surfaceTrimesh.apply_translation(shiftVector)
    
    return surfaceTrimesh



def getSurfaceTrimeshIntersection(surfaceTrimesh, opPosition, opMomentumDir):
    '''
    finding the intersection with the trimesh

    Parameters
    ----------
    surfaceTrimesh            : trimesh object
                                Surface trimesh to be intersected.
    opPosition                : (1x3) array of floats
                                Photon's origin position in um.
    opMomentumDir             : (1x3) array of floats
                                Photon's momentum direction unit vector.

    Returns
    -------
    trimeshIntersecPoint      : (1x3) array of floats
                                Intersection point with the trimesh in um.
    trimeshIntersecFaceNormal : (1x3) array of floats
                                Normal unit vector of the intersected face.

    '''
    
    #= intersect with the trimesh
    trimeshIntersecPoints, _, trimeshIntersecTriangles = surfaceTrimesh.ray.intersects_location(ray_origins = opPosition,
                                                                                                ray_directions = opMomentumDir)
    
    if len(trimeshIntersecPoints) == 0: return np.array([]), np.array([])
    
    #= eliminating the photon origin point that is considered as an intersection point if it is exactly on a surface trimesh
    distances = np.round(np.linalg.norm(opPosition-trimeshIntersecPoints, axis = 1), 9)
    # true intersections will have distances > 0.0; use a slightly higher number instead becasue of the machine precision
    trueIntersecPointsIndices = np.where(distances > 1E-9)[0]
    trimeshIntersecPoints     = trimeshIntersecPoints[trueIntersecPointsIndices]
    trimeshIntersecTriangles  = trimeshIntersecTriangles[trueIntersecPointsIndices]
    distances                 = distances[trueIntersecPointsIndices]
    if len(distances) == 0: return np.array([]), np.array([])
    #= get the first intersection
    trimeshIntersecPoint      = np.array([trimeshIntersecPoints[np.argmin(distances)]])
    trimeshIntersecTriangle   = np.array([trimeshIntersecTriangles[np.argmin(distances)]])
        
    #= find the face normal
    trimeshIntersecFaceNormal = np.array(surfaceTrimesh.face_normals[trimeshIntersecTriangle])
    
    return trimeshIntersecPoint, trimeshIntersecFaceNormal



def prepareSurfaceTrimesh(opPosition, opMomentumDir, opVolume, volumeIntersectPoint, volumeIntersectPlaneNormal):
    '''
    preparing the surface tirmesh and finding first intersection

    Parameters
    ----------
    opPosition                 : (1x3) array of floats
                                 Photon's position in um.
    opMomentumDir              : (1x3) array of floats
                                 Photon's momentum direction unit vector.
    opVolume                   : trimesh volume object
                                 The photon current associated volume.
    volumeIntersectPoint       : (1x3) array of floats
                                 Initial intersection point on a volume in um.
    volumeIntersectPlaneNormal : (1x3) array of floats
                                 Normal unit vector of the volume's intersected plane.

    Returns
    -------
    surfaceTrimesh             : trimesh object
                                 Surface trimesh to be shifted.
    trimeshIntersectPoint      : (1x3) array of floats
                                 Intersection point with the trimesh in um.
    trimeshIntersectFaceNormal : (1x3) array of floats
                                 Local normal unit vector at surface trimesh intersection point.

    '''

    # counter of times the photon position happen to fall under the surface trimesh it's supposed to hit
    # This would happen if the photon is at a corner and reflecting off a surface to the adjacent one within
    # the scale of the surface trimesh features.
    originUnderSurfaceMeshCounter = 0
    #= retrieve the bounds on the surface trimesh marking the allowed area to sample a potential intersection point
    #  from within
    #  Note: The bounds are set to leave 5% on the area periphery to ensure that the photon doesn't run out of surface
    #        trimesh area while transporting over its features.
    #        The 5% is arbitray. Limiting the area allowed to sample the intersection point within is fine ONLY if the
    #        modeled area has several repetitions of features, if any; otherwise, the parts of the features would be
    #        cut out and therefore the modeled surface trimesh won't be a good representation of the actual surface.
    samplingBounds      = opVolume.surfaceTrimeshes[tuple(map(tuple,volumeIntersectPlaneNormal))[0]].samplingBounds # um
    #= keep sampling a surface trimesh point until a viable configuration is found where the photon position
    #  happens to be above the surface trimesh it is supposed to hit, or up to 20 times (arbitrary) if no viable
    #  point is sampled
    #  Note: if no viable point is found, this loosly represents photon trapping at a corner and is therefore killed.
    while True:
        #= get a sampled point on the surface trimesh that marks a potential intersection point as the photon is set to
        #  head towards it
        trimeshSampledPoint = sampleSurfaceTrimeshPoint(samplingBounds, volumeIntersectPlaneNormal)                                                  # um
        #= retrieve the surface trimesh to intersect from the volume that has it and acquired volume intersection plane
        surfaceTrimesh      = copy.deepcopy(opVolume.surfaceTrimeshes[tuple(map(tuple,volumeIntersectPlaneNormal))[0]].trimesh)
        #= shift the entire surface trimesh such that the sampled point coincides with the previously acquired tentative
        #  volume intersection point
        #  Note: This simplifies tracking as it unifies the volume and surface trimeshes frame of references
        surfaceTrimesh      = shiftSurfaceTrimesh(surfaceTrimesh, volumeIntersectPlaneNormal, volumeIntersectPoint, trimeshSampledPoint)
        #= get the true intersection point with the surface trimesh and local normal
        trimeshIntersectPoint, trimeshIntersectFaceNormal = getSurfaceTrimeshIntersection(surfaceTrimesh, opPosition, opMomentumDir)     # um
        #= resample for up to 20 times (arbitrary) then kill if no true intersection
        if len(trimeshIntersectPoint) == 0:
            if originUnderSurfaceMeshCounter < 20:
                originUnderSurfaceMeshCounter += 1
                continue
            else:
                break
        else:
            break
        
    return surfaceTrimesh, trimeshIntersectPoint, trimeshIntersectFaceNormal



def calculateCosineIncidenceAngle(opMomentumDir, intersectNormal):
    '''
    calculating a photon's incidence angle cosine

    Parameters
    ----------
    opMomentumDir     : (1x3) array of floats
                        Photon's momentum direction unit vector.
    intersectNormal   : (1x3) array of floats
                        Normal unit vector of the intersected face.

    Raises
    ------
    ValueError: if the incidence angle cosine is +ve, i.e. incidence angle is higher than 90 deg.

    Returns
    -------
    cosIncidenceAngle : float
                        Cosine of the incidence angle.

    '''
    
    # the dot product yields cosine of the inner angle between two unit vectors
    cosIncidenceAngle = np.dot(opMomentumDir, intersectNormal.transpose())
    
    if cosIncidenceAngle > 0.0:
        raise ValueError('The calculated incidence angle is higher than 90 deg...\n'
                         +' - photon momentum direction: {}, intersection normal:{}'.format(opMomentumDir, intersectNormal))
    
    return -1*cosIncidenceAngle



def calculateSineTransmissionAngle(cosIncidenceAngle, currentVolumeRIndex, nextVolumeRIndex):
    '''
    calculating a photon's transmission angle cosine

    Parameters
    ----------
    cosIncidenceAngle   : float
                          Cosine of the incidence angle.
    currentVolumeRIndex : float
                          Index of refraction of the incidence volume.
    nextVolumeRIndex    : float
                          Index of refraction of the transmission volume.

    Raises
    ------
    ValueError: if the transmission angle sine is -ve, i.e. transmission angle is higher than 90 deg.

    Returns
    -------
    None if TIR else the transmission angle sine

    '''
    
    sinIncidenceAngle = np.sqrt(1 - cosIncidenceAngle**2)
    
    sinTransmissionAngle = (currentVolumeRIndex/nextVolumeRIndex)*sinIncidenceAngle
    
    #= check if tranmission is feasible
    if sinTransmissionAngle > 1.0:
        return None
    else:
        if sinTransmissionAngle < 0.0:
            incidenceAngle = np.pi - np.arccos(np.clip(cosIncidenceAngle, -1.0, 1.0))
            raise ValueError('The calculated transmission angle is higher than 90 deg...\n'
                             +' - photon incidence angle: {}deg, incidence n:{}, transmission n:{}'.format(np.degrees(incidenceAngle),
                                                                                                           currentVolumeRIndex,
                                                                                                           nextVolumeRIndex))
        return sinTransmissionAngle



def calculateReflectionProbability(cosIncidenceAngle, sinTransmissionAngle, currentVolumeRIndex, nextVolumeRIndex):
    '''
    calculating the photon reflection probability

    Parameters
    ----------
    cosIncidenceAngle    : float
                           Cosine of the incidence angle.
    sinTransmissionAngle : float
                           Sine of the transmission angle.
    currentVolumeRIndex  : float
                           Index of refraction of the incidence volume.
    nextVolumeRIndex     : float
                           Index of refraction of the transmission volume.

    Raises
    ------
    ValueError: if the calculated reflection probability is higher than 1.0.

    Returns
    -------
    R : float
        The reflection probability of the incident photon.

    '''
    
    cosTransmissionAngle = np.sqrt(1 - sinTransmissionAngle**2)
    
    #= calculate the reflection coefficient in the perpendicular polarization direction
    rPerpendicular = (currentVolumeRIndex*cosIncidenceAngle - nextVolumeRIndex*cosTransmissionAngle)\
                   / (currentVolumeRIndex*cosIncidenceAngle + nextVolumeRIndex*cosTransmissionAngle)
    #= calculate the reflection coefficient in the parallel polarization direction
    rParallel = (currentVolumeRIndex*cosTransmissionAngle - nextVolumeRIndex*cosIncidenceAngle)\
              / (currentVolumeRIndex*cosTransmissionAngle + nextVolumeRIndex*cosIncidenceAngle)
    
    #= average both coefficients to estimate the reflection probabilty ???????????????????  (to be uppdated - AM 5/7/2021)
    R = 0.5*(rPerpendicular**2 + rParallel**2)
    
    #= stop if the calculated probability is higher that 1.0
    if R > 1.0:
        incidenceAngle    = np.pi - np.arccos(np.clip(cosIncidenceAngle, -1.0, 1.0))
        transmissionAngle = np.arcsin(sinTransmissionAngle)
        raise ValueError('The calculated reflection probability is higher that 1.0...\n'
                         +' - estimated incidence angle:{}deg, transmission angle:{}deg'.format(np.degrees(incidenceAngle),
                                                                                                np.degrees(transmissionAngle)))
    
    return R



def doReflection(opMomentumDir, opPolarizationDir, intersectNormal):
    '''
    calculating the reflected ray new momentum and polarization directions
                                   P_r = I - 2(I.n)n
    For the derivation, refer to:
        https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel

    Parameters
    ----------
    opMomentumDir               : (1x3) array of floats
                                  Photon's momentum direction unit vector.
    opPolarizationDir           : (1x3) array of floats
                                  Photon's polarization direction unit vector.
    intersectNormal             : (1x3) array of floats
                                  Normal unit vector of the intersected face.

    Returns
    -------
    opReflectionMomentumDir     : (1x3) array of floats
                                  Reflected photon's momentum direction unit vector.
    opReflectionPolarizationDir : (1x3) array of floats
                                  Reflected photon's polarization direction unit vector.

    '''
    
    #= calculate the momentum direction of the reflected ray
    opReflectionMomentumDir = opMomentumDir\
              - 2*np.dot(opMomentumDir, intersectNormal.transpose())*intersectNormal
    #= due to limited numerical precision, renormalize to yield an exact unit vector
    opReflectionMomentumDir /= np.linalg.norm(opReflectionMomentumDir)
    
    # =====????????????????????????????????????????????? (to be uppdated - AM 5/7/2021)
    opReflectionPolarizationDir = opPolarizationDir
    
    return opReflectionMomentumDir, opReflectionPolarizationDir



def doTransmission(opMomentumDir, opPolarizationDir, intersectNormal,
                   cosIncidenceAngle, sinTransmissionAngle, currentVolumeRIndex, nextVolumeRIndex):
    '''
    calculating the transmitted ray new momentum and polarization directions
                            P_t = (n_i/n_t)*(I + cos(theta_i)n) - cos(theta_t)n
    For a derivation, refer to:
        https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel

    Parameters
    ----------
    opMomentumDir                 : (1x3) array of floats
                                    Photon's momentum direction unit vector.
    opPolarizationDir             : (1x3) array of floats
                                    Photon's polarization direction unit vector.
    intersectNormal               : (1x3) array of floats
                                    Normal unit vector of the intersected face.
    cosIncidenceAngle             : float
                                    Cosine of the incidence angle.
    sinTransmissionAngle          : float
                                    Sine of the transmission angle.
    currentVolumeRIndex           : float
                                    Index of refraction of the incidence volume.
    nextVolumeReIndex             : float
                                    Index of refraction of the transmission volume.

    Yields
    ------
    opTransmissionMomentumDir     : (1x3) array of floats
                                    Transmitted photon's momentum direction unit vector.
    opTransmissionPolarizationDir : (1x3) array of floats
                                    Transmitted photon's polarization direction unit vector.

    '''
    
    cosTransmissionAngle = np.sqrt(1 - sinTransmissionAngle**2)
    
    #= calculate the momentum direction of the transmitted ray
    opTransmissionMomentumDir = (currentVolumeRIndex/nextVolumeRIndex)*(opMomentumDir+cosIncidenceAngle*intersectNormal)\
                                    - intersectNormal*cosTransmissionAngle
    #= due to limited numerical precision, renormalize to yield an exact unit vector
    opTransmissionMomentumDir /= np.linalg.norm(opTransmissionMomentumDir)
    
    # =====????????????????????????????????????????????? (to be uppdated - AM 5/7/2021)
    opTransmissionPolarizationDir = opPolarizationDir
        
    return opTransmissionMomentumDir, opTransmissionPolarizationDir



def doLambertian(opMomentumDir, opPolarizationDir, intersectNormal):
    '''
    calculating the reflected ray new momentum and polarization directions
    
    There are two methods:
        a. around the z-axis:
            theta = arcsin(sqrt(rand)), phi = 2*pi*rand
        b. around a given normal N:
            used in GEANT4 and implemented below
            
    For more information about the correct way of sampling from lambertian distribution (important and contrary to intuition), refer to:
        https://www.particleincell.com/2015/cosine-distribution/
        https://www.sciencedirect.com/science/article/pii/S0042207X02001732
        

    Parameters
    ----------
    opMomentumDir               : (1x3) array of floats
                                  Photon's momentum direction unit vector.
    opPolarizationDir           : (1x3) array of floats
                                  Photon's polarization direction unit vector.
    intersectNormal             : (1x3) array of floats
                                  Normal unit vector of the intersected face.

    Returns
    -------
    opReflectionMomentumDir     : (1x3) array of floats
                                  Reflected photon's momentum direction unit vector.
    opReflectionPolarizationDir : (1x3) array of floats
                                  Reflected photon's polarization direction unit vector.

    '''
    
    #= sample a momentum direction of the reflected ray
    while True:
        phi                           = 2*np.pi*np.random.rand()
        dir_gamma                     = 2*np.random.rand() - 1
        dir_alpha                     = np.sqrt(1-(dir_gamma**2)) * np.cos(phi)
        dir_beta                      = np.sqrt(1-(dir_gamma**2)) * np.sin(phi)
        opReflectionMomentumDir       = np.array([[dir_alpha, dir_beta, dir_gamma]])
        opReflectionMomentumDir      /= np.linalg.norm(opReflectionMomentumDir)
        
        cosTheta = np.dot(opReflectionMomentumDir, intersectNormal.transpose())
        
        if cosTheta < 0.0:
            opReflectionMomentumDir  *= -1
            cosTheta                 *= -1
        
        if np.random.rand() < cosTheta:
            break
    
    # =====????????????????????????????????????????????? (to be uppdated - AM 9/27/2021)
    opReflectionPolarizationDir = opPolarizationDir
    
    return opReflectionMomentumDir, opReflectionPolarizationDir



def traveledDistanceAndTime(opPosition, intersectPoint, currentVolumeRIndex):
    '''
    estimating the traveled distance and time between two interactions

    Parameters
    ----------
    opPosition          : (1x3) array of floats
                          Photon's origin position in um.
    intersectPoint      : (1x3) array of floats
                          Intersection point with a surface in um.
    currentVolumeRIndex : float
                          Index of refraction of the volume within which the photon traveled.

    Returns
    -------
    traveledDistance    : float
                          Distance between the photon origin and the interaction in um.
    traveledTime        : float
                          Traveling time between the photon origin and the interaction in ns.

    '''
    
    traveledDistance    = np.round(np.linalg.norm(intersectPoint - opPosition), 9) # um
    
    photonSpeedInMedium = 299792458*1E-3 / currentVolumeRIndex                     # um/ns
    
    traveledTime        = traveledDistance/photonSpeedInMedium                     # ns
    
    return traveledDistance, traveledTime
    


def updatePhotonInfo(oPhoton, opTrackingHistory,
                     intersectPoint, interactType, intersectNormal,
                     newMomentumDir, newPolarizationDir,
                     cosIncidenceAngle, sinTransmissionAngle,
                     traveledDistance, traveledTime, attenuationLength):
    '''
    updating the optical photon attribute values & tracking history after an interaction

    Parameters
    ----------
    oPhoton              : opticalPhoton object
    opTrackingHistory    : structured numpy array
                           Complete previous tracking history of the photon.
    intersectPoint       : (1x3) array of floats
                           Intersection point with a surface at current step in um.
    interactType         : string
                           Type of the photon interaction at current step.
    intersectNormal      : (1x3) array of floats
                           Normal unit vector of the intersected face.
    newMomentumDir       : (1x3) array of floats
                           Photon's new momentum direction unit vector.
    newPolarizationDir   : (1x3) array of floats
                           Photon's new polarization direction unit vector.
    cosIncidenceAngle    : float
                           Cosine of the incidence angle.
    sinTransmissionAngle : float
                           Sine of the transmission angle.
    traveledDistance     : float
                           Distance the photon traveled before the last interaction in um.
    traveledTime         : float
                           Photon's traveling time to the last interaction in ns.
    attenuationLength    : float
                           Optical photon's attenuation length in the medium through which it traveled in mm.

    Returns
    -------
    oPhoton              : opticalPhoton object
                           Updated.
    opTrackingHistory    : structured numpy array
                           Updated with current step.

    '''
    
    incidenceAngle    = np.arccos(np.clip(cosIncidenceAngle, -1.0, 1.0))
    if sinTransmissionAngle is not None:
        outAngle = np.arcsin(sinTransmissionAngle)
    else:
        if opTrackingHistory[-1]['volume'] == b'reflector' and oPhoton.volume == 'reflector':
            cosOutAngle = np.dot(newMomentumDir, -1*intersectNormal.transpose())
            outAngle    = np.arccos(np.clip(cosOutAngle, -1.0, 1.0))
        else:
            outAngle = incidenceAngle
    
    #= retrieve info from last step to update current one
    p                 = opTrackingHistory[-1]['id']
    step              = opTrackingHistory[-1]['step']+1
    inMoemntumDir     = opTrackingHistory[-1]['outMomentumDir']
    inPolarizationDir = opTrackingHistory[-1]['outPolarizationDir']
    relTime           = opTrackingHistory[-1]['relTime']
    
    intersectPlaneOrientation = {(-1.0, 0.0, 0.0): '-x',
                                  (1.0, 0.0, 0.0): '+x',
                                 (0.0, -1.0, 0.0): '-y',
                                  (0.0, 1.0, 0.0): '+y',
                                 (0.0, 0.0, -1.0): '-z',
                                  (0.0, 0.0, 1.0): '+z'
                                 }
    
    #= update the the photon attributes
    #  Note: the volume is updated outside following the interaction type
    oPhoton.position          = intersectPoint
    oPhoton.momentumDir       = newMomentumDir
    oPhoton.polarizationDir   = newPolarizationDir
    oPhoton.time             += traveledTime
    oPhoton.traveledDistance += traveledDistance
    oPhoton.weight           *= np.exp(-(traveledDistance*1E-3)/attenuationLength)
    
    #= update the photon tracking history with the current step
    opTrackingHistory = np.append( opTrackingHistory,
                                   np.array( [ (p,
                                                step,
                                                oPhoton.position*1E-3,
                                                interactType,
                                                intersectPlaneOrientation[tuple(map(tuple,intersectNormal))[0]],
                                                oPhoton.volume,
                                                inMoemntumDir,
                                                oPhoton.momentumDir,
                                                np.degrees(incidenceAngle),
                                                np.degrees(outAngle),
                                                inPolarizationDir,
                                                oPhoton.polarizationDir,
                                                oPhoton.time,
                                                relTime+traveledTime,
                                                oPhoton.traveledDistance*1E-3,
                                                oPhoton.weight
                                                )
                                              ], dtype = opTrackingHistory.dtype)
                                   )
    
    return oPhoton, opTrackingHistory



def pointIsWithinVolumeBounds(opVolume, trimeshIntersectPoint, volumeIntersectPlaneNormal):
    '''
    checking whether the trimesh intersection point is outside the volume boundaries in the lateral directions

    Parameters
    ----------
    opVolume                   : trimesh volume object
                                 The photon current associated volume.
    trimeshIntersectPoint      : (1x3) array of floats
                                 Intersection point with the trimesh in um.
    volumeIntersectPlaneNormal : (1x3) array of floats
                                 Normal unit vector of the volume's intersected plane.

    Returns
    -------
    bool

    '''
    
    #= get the lateral coordinates indices
    lateralCoordIndices = np.where(volumeIntersectPlaneNormal[0, :] == 0.0)[0]
    
    #= get the volume bounds
    #  Note: rounding the volume bounds is because of the floating point errors in some
    #        of the constructed volume trimeshes.
    ax1_bounds = np.round([opVolume.bounds[0, lateralCoordIndices[0]], opVolume.bounds[1, lateralCoordIndices[0]]], 9)
    ax2_bounds = np.round([opVolume.bounds[0, lateralCoordIndices[1]], opVolume.bounds[1, lateralCoordIndices[1]]], 9)
    
    #= get the intersection point lateral coordinates
    ax1_coordinate = trimeshIntersectPoint[0, lateralCoordIndices[0]]
    ax2_coordinate = trimeshIntersectPoint[0, lateralCoordIndices[1]]
    
    if ax1_coordinate > ax1_bounds[0] and ax1_coordinate < ax1_bounds[1]\
        and ax2_coordinate > ax2_bounds[0] and ax2_coordinate < ax2_bounds[1]:
        return True
    else:
        return False
                


def interactWithLocalSurface(opMomentumDir, opPolarizationDir, localSurfaceNormal, currentVolumeRIndex, nextVolumeRIndex):
    '''
    performing interaction with the a local surface given the local normal

    Parameters
    ----------
    opMomentumDir        : (1x3) array of floats
                           Photon's current momentum direction unit vector.
    opPolarizationDir    : (1x3) array of floats
                           Photon's current polarization direction unit vector.
    localSurfaceNormal   : (1x3) array of floats
                           Local normal unit vector of the intersected surface.
    currentVolumeRIndex  : float
                           Index of refraction of the photon's current volume
    nextVolumeRIndex     : float
                           Index of refraction of the photon's next volume

    Returns
    -------
    interactType         : string
                           Type of the photon interaction at current step.
    cosIncidenceAngle    : float
                           Cosine of the incidence angle.
    sinTransmissionAngle : float
                           Sine of the transmission angle.
    newMomentumDir       : (1x3) array of floats
                           Photon's new momentum direction unit vector.
    newPolarizationDir   : (1x3) array of floats
                           Photon's new polarization direction unit vector.

    '''
    
    
    #= reverse local normal if incident on mesh from the outside
    if np.matmul(opMomentumDir, localSurfaceNormal.transpose()) > 0:
        localSurfaceNormal *= -1

    #= calculate the incidence angle cosine
    cosIncidenceAngle    = calculateCosineIncidenceAngle(opMomentumDir, localSurfaceNormal)
    #= calculate the transmission angle sine in case of a potential refraction interaction at the boundary
    #  Returns None if refraction is physically prohibited.
    sinTransmissionAngle = calculateSineTransmissionAngle(cosIncidenceAngle,
                                                          currentVolumeRIndex,
                                                          nextVolumeRIndex)
    ### == refraction is NOT physically prohibited
    if sinTransmissionAngle is not None:
        #= calculate the reflection probability
        reflectionProb = calculateReflectionProbability(cosIncidenceAngle, sinTransmissionAngle,
                                                        currentVolumeRIndex,
                                                        nextVolumeRIndex)
        ## sample reflection
        if np.random.rand() < reflectionProb:
            interactType = 'reflection'
            # calculate photon's new momentum and polarization direction
            newMomentumDir, newPolarizationDir = doReflection(opMomentumDir, opPolarizationDir,
                                                              localSurfaceNormal)
        ## sample transmission
        else:
            interactType = 'transmission'
            # calculate photon's new momentum and polarization direction
            newMomentumDir, newPolarizationDir = doTransmission(opMomentumDir, opPolarizationDir,
                                                                localSurfaceNormal,
                                                                cosIncidenceAngle, sinTransmissionAngle,
                                                                currentVolumeRIndex,
                                                                nextVolumeRIndex)
    ### == refraction is physically prohibited
    else:
        interactType = 'TIR'
        # calculate photon's new momentum and polarization direction
        newMomentumDir, newPolarizationDir = doReflection(opMomentumDir, opPolarizationDir,
                                                          localSurfaceNormal)
        
    return interactType, cosIncidenceAngle, sinTransmissionAngle, newMomentumDir, newPolarizationDir
                


def interactWithReflector(reflectorMaterial, reflectorIntersectPlaneNormal, opMomentumDir, opPolarizationDir):
    '''
    Perform interaction with the reflector volume

    Parameters
    ----------
    reflectorMaterial             : material object
    reflectorIntersectPlaneNormal : (1x3) array of floats
                                    Normal unit vector of the reflector intersected plane.
    opMomentumDir                 : (1x3) array of floats
                                    Photon's current momentum direction unit vector.
    opPolarizationDir             : (1x3) array of floats
                                    Photon's current polarization direction unit vector.

    Returns
    -------
    interactType                  : string
                                    Type of the photon interaction at current step.
    cosIncidenceAngle             : float
                                    Cosine of the incidence angle.
    sinTransmissionAngle          : float
                                    Sine of the transmission angle.
    newMomentumDir                : (1x3) array of floats
                                    Photon's new momentum direction unit vector.
    newPolarizationDir            : (1x3) array of floats
                                    Photon's new polarization direction unit vector.

    '''
    
    #= calculate the incidence angle cosine
    cosIncidenceAngle   = calculateCosineIncidenceAngle(opMomentumDir, reflectorIntersectPlaneNormal)
    ## sample reflection
    if np.random.rand() < reflectorMaterial['reflectivity']:
        interactType = 'reflection'
        sinTransmissionAngle = None
        # calculate the incidence angle
        incidenceAngle      = np.degrees(np.arccos(np.clip(cosIncidenceAngle, -1.0, 1.0)))
        # calculate the lambertian reflection probability
        fitParams = reflectorMaterial['lambertianFraction']['doubleExpFitParams']
        lambertianFraction = fitParams['a1']*np.exp(fitParams['b1']*incidenceAngle) + fitParams['a2']*np.exp(fitParams['b2']*incidenceAngle)
        # sample lambertian reflection
        if np.random.rand() < lambertianFraction:
            # calculate photon's new momentum and polarization direction
            newMomentumDir, newPolarizationDir =  doLambertian(opMomentumDir, opPolarizationDir,
                                                               reflectorIntersectPlaneNormal)
        # sample specular lobe reflection
        else:
            # sample the reflected ray polar angle (relative to surface normal) & azimuth angle (relative to the plane of incidence)
            gaussianSigma = reflectorMaterial['specularLobeSigma'][1, np.where(reflectorMaterial['specularLobeSigma'][0, :] == int(np.round(incidenceAngle, 0)))[0]]
            while True:
                outPolarAngle     = np.random.normal(loc=incidenceAngle, scale=gaussianSigma)
                if outPolarAngle >= 0.0 and outPolarAngle <= 90.0:
                    break
            outAzimuthAngle     = np.random.normal(loc=0.0, scale=gaussianSigma)
            
            # calculate photon's new momentum and polarization direction
            incidencePlanePerp  = np.cross(reflectorIntersectPlaneNormal, opMomentumDir)
            incidencePlanePerp /= np.linalg.norm(incidencePlanePerp)
            incidencePlanePara  = np.cross(incidencePlanePerp, reflectorIntersectPlaneNormal)
            incidencePlanePara /= np.linalg.norm(incidencePlanePara)
            
            paraComponent = incidencePlanePara * np.sin(np.radians(outPolarAngle)) * np.cos(np.radians(outAzimuthAngle))
            perpComponent = incidencePlanePerp * np.sin(np.radians(outPolarAngle)) * np.sin(np.radians(outAzimuthAngle))
            normComponent = reflectorIntersectPlaneNormal * np.cos(np.radians(outPolarAngle))
            
            newMomentumDir     = paraComponent + perpComponent + normComponent
            newMomentumDir    /= np.linalg.norm(newMomentumDir)
            newPolarizationDir = opPolarizationDir
            
    ## sample transmission
    else:
        interactType = 'transmission'
        sinTransmissionAngle = np.sqrt(1 - np.dot(opMomentumDir, -1*reflectorIntersectPlaneNormal.transpose())**2)
        newMomentumDir, newPolarizationDir = opMomentumDir, opPolarizationDir
        
    return interactType, cosIncidenceAngle, sinTransmissionAngle, newMomentumDir, newPolarizationDir



def trackOverSurfaceTrimesh(volumeList, opTrackingHistory, oPhoton, volumeIntersectPoint, volumeIntersectPlaneNormal, nextVolume):
    '''
    Perfrom tracking over a surface trimesh

    Parameters
    ----------
    volumeList                 : dict of volume objects
                                 All Geometry volume objects.
    opTrackingHistory          : structured numpy array
                                 Complete previous tracking history of the photon.
    oPhoton                    : opticalPhoton object
    volumeIntersectPoint       : (1x3) array of floats
                                 Initial intersection point in the volume frame of reference in um.
    volumeIntersectPlaneNormal : (1x3) array of floats
                                 Normal unit vector of the intersected global face.
    nextVolume                 : string
                                 Detected next volume to interact with.

    Returns
    -------
    oPhoton                    : opticalPhoton object
                                 Updated.
    opTrackingHistory          : structured numpy array
                                 Updated.

    '''
    
    #= determine which of the optical photon volume and nextVolume has roughness
    surfaceTrimeshesVolume = oPhoton.volume if volumeList[oPhoton.volume].surfaceTrimeshes else nextVolume
            
    #= prepare surface trimesh and get first intersection point
    surfaceTrimesh, trimeshIntersectPoint, trimeshIntersectFaceNormal = prepareSurfaceTrimesh(oPhoton.position,
                                                                                              oPhoton.momentumDir,
                                                                                              volumeList[surfaceTrimeshesVolume],
                                                                                              volumeIntersectPoint,
                                                                                              volumeIntersectPlaneNormal)
    if len(trimeshIntersectPoint) == 0:
        oPhoton.alive = False
        print(' - a photon was killed due to being trapped at a corner ...')
        print('   photon volume: {}, position: {}, momentum direction: {}, intersected plane orientation: {} ...'.format(surfaceTrimeshesVolume,
                                                                                                                         oPhoton.position*1E-3,
                                                                                                                         oPhoton.momentumDir,
                                                                                                                         volumeIntersectPlaneNormal))
    
    # counter of local intersections at the current surface trimesh
    # Up to 20 intersections (arbitrary) then the photon is killed as this situation represents feature-trapping
    localSurfaceMeshIntersectCounter = 0
    #= keep tracking while the photon keep having local intersections with the current surface trimesh
    while len(trimeshIntersectPoint) != 0:
        #= get the extents of the volume the photon is currently transporting over its surface trimesh and check
        #  if it's still within it
        #  If not, the photon escaped the volume while transporting the surface trimesh features and so is killed.
        #  Note: this is not a robust method, but is assumed to be sufficient should this be rare to happen.
        boundingVolume = volumeList[surfaceTrimeshesVolume].volumeTrimesh
        if not pointIsWithinVolumeBounds(boundingVolume, trimeshIntersectPoint, volumeIntersectPlaneNormal):
            oPhoton.alive = False
            print(' - a photon was killed due to escaping its volume bounds while transporting trimesh features ...')
            print('   intersected trimesh orientation: {} ...'.format(volumeIntersectPlaneNormal))
            break
        localSurfaceMeshIntersectCounter += 1
        
        
        #= retrieve photon's current volume properties before any updating of the volume
        refractiveIndex                = volumeList[oPhoton.volume].material['refractiveIndex']                         
        attenuationLength              = volumeList[oPhoton.volume].material['attenuationLength']                         # mm
        
        #= interact with surface
        interactType, cosIncidenceAngle, sinTransmissionAngle, newMomentumDir, newPolarizationDir = interactWithLocalSurface(oPhoton.momentumDir,
                                                                                                                             oPhoton.polarizationDir,
                                                                                                                             trimeshIntersectFaceNormal,
                                                                                                                             volumeList[oPhoton.volume].material['refractiveIndex'],
                                                                                                                             volumeList[nextVolume].material['refractiveIndex'])
        # exchange the photon current and next volumes in case of transmission
        if interactType == 'transmission':
            oPhoton.volume, nextVolume = nextVolume, oPhoton.volume
        
        #= estimate traveled distance and spent time
        traveledDistance, traveledTime = traveledDistanceAndTime(oPhoton.position, trimeshIntersectPoint,
                                                                 refractiveIndex)
        #= update the photon attributes and tracking history
        oPhoton, opTrackingHistory = updatePhotonInfo(oPhoton, opTrackingHistory,
                                                      trimeshIntersectPoint, interactType, volumeIntersectPlaneNormal,
                                                      newMomentumDir, newPolarizationDir,
                                                      cosIncidenceAngle, sinTransmissionAngle,
                                                      traveledDistance, traveledTime, attenuationLength)
        
        #= kill if up to 20 (arbitrary) local surface trimesh occur
        if localSurfaceMeshIntersectCounter > 20:
            oPhoton.alive = False
            print(' - a photon was killed due to being trapped inside trimesh features ...')
            print('   intersected trimesh orientation: {} ...'.format(volumeIntersectPlaneNormal))
            break
        
        #= get the next local intersection point and local normal
        trimeshIntersectPoint, trimeshIntersectFaceNormal = getSurfaceTrimeshIntersection(surfaceTrimesh,
                                                                                          oPhoton.position,
                                                                                          oPhoton.momentumDir)
        
        
        
        
        ### == interact with the 'reflector' volume, if any
        if len(trimeshIntersectPoint) == 0 and oPhoton.volume == 'reflector':
            #= intersect with the reflector volume, if any
            reflectorIntersectPoint, _, reflectorIntersectTri = volumeList['reflector'].volumeTrimesh.ray.intersects_location(ray_origins    = oPhoton.position,
                                                                                                                              ray_directions = oPhoton.momentumDir)
            reflectorIntersectPoint         = np.round(np.array([reflectorIntersectPoint[0]]), 9)
            reflectorIntersectPlaneNormal   = np.round(np.array([volumeList['reflector'].volumeTrimesh.face_normals[reflectorIntersectTri[0]]]), 9)
            
            #= interact locally with the reflector only if both reflector volumeIntersectPlaneNormal are parallel;
            #  otherwise, the photon could have hit the reflector "non-true" boundary with the another volume and the
            #  getVolumeTrimeshIntersection() method is needed to detect that
            if (reflectorIntersectPlaneNormal == volumeIntersectPlaneNormal).all():
                # retrieve photon's current volume properties before any updating of the volume
                refractiveIndex                = volumeList[oPhoton.volume].material['refractiveIndex']                         
                attenuationLength              = volumeList[oPhoton.volume].material['attenuationLength']                         # mm
                #= perform interaction with reflector
                interactType, cosIncidenceAngle, sinTransmissionAngle, newMomentumDir, newPolarizationDir = interactWithReflector(volumeList[oPhoton.volume].material,
                                                                                                                                  -1*reflectorIntersectPlaneNormal,
                                                                                                                                  oPhoton.momentumDir,
                                                                                                                                  oPhoton.polarizationDir)
                #= update the photon's volume to the outside 'environment' only in case of transmission
                if interactType == 'transmission':
                    oPhoton.volume = 'environment'
                #= estimate traveled distance and spent time
                traveledDistance, traveledTime = traveledDistanceAndTime(oPhoton.position,
                                                                         reflectorIntersectPoint,
                                                                         refractiveIndex)  # um, ns
                #= update the photon attributes and tracking history
                oPhoton, opTrackingHistory = updatePhotonInfo(oPhoton, opTrackingHistory,
                                                              reflectorIntersectPoint, interactType, reflectorIntersectPlaneNormal,
                                                              newMomentumDir, newPolarizationDir,
                                                              cosIncidenceAngle, sinTransmissionAngle,
                                                              traveledDistance, traveledTime, attenuationLength)
                
                if interactType != 'transmission':
                    #= update remaining info before heading back towards the surfaceTrimeshesVolume
                    nextVolume = surfaceTrimeshesVolume
                    localSurfaceMeshIntersectCounter = 0
                    #= check if the photon returned to and iteracted with the surfaceTrimesh
                    trimeshIntersectPoint, trimeshIntersectFaceNormal = getSurfaceTrimeshIntersection(surfaceTrimesh,
                                                                                                      oPhoton.position,
                                                                                                      oPhoton.momentumDir)
    
    
    return oPhoton, opTrackingHistory



def trackPhoton(opTrackNumber, oPhoton, volumeList):
    '''
    tracking an optical photon until it's either detected or killed

    Parameters
    ----------
    opTrackNumber     : int
                        Track number of the photon to be tracked.
    oPhoton           : opticalPhoton object
    volumeList        : dict of volume objects
                        All Geometry volume objects.

    Returns
    -------
    opTrackingHistory : structured numpy array
                        The photon's tracking history

    '''
    #= specify the data types of the tracking history elements (columns)
    dtype = [('id', np.int32),
             ('step', np.int32),
             ('position', np.float64, (3, )),
             ('type', 'S15'),
             ('planeOrientation', 'S15'),
             ('volume', 'S15'),
             ('inMomentumDir', np.float64, (3, )),
             ('outMomentumDir', np.float64, (3, )),
             ('inAngle', np.float64),
             ('outAngle', np.float64),
             ('inPolarizationDir', np.float64, (3, )),
             ('outPolarizationDir', np.float64, (3, )),
             ('absTime', np.float64),
             ('relTime', np.float64),
             ('distance', np.float64),
             ('weight', np.float64)
            ]
    #= create a structured array with the first step of the current photon tracking history
    opTrackingHistory = np.array( [ (opTrackNumber,
                                     0.0,
                                     oPhoton.position*1E-3,
                                     'emission',
                                     None,
                                     oPhoton.volume,
                                     None,
                                     oPhoton.momentumDir,
                                     None,
                                     None,
                                     None,
                                     oPhoton.polarizationDir,
                                     oPhoton.time,
                                     0.0,
                                     0.0,
                                     oPhoton.weight
                                    )
                                  ], dtype = dtype)
    
    # == do as long as the photon status is 'alive'
    while oPhoton.alive:
        
        #= find the volumes with which the photon intersects
        volumeIntersectPoint, volumeIntersectPlaneNormal, nextVolume = getVolumeTrimeshIntersection(volumeList,
                                                                                                    oPhoton.position,
                                                                                                    oPhoton.momentumDir,
                                                                                                    oPhoton.volume)
        # break if no found intersections
        if nextVolume is None:
            oPhoton.alive = False
            continue
        
        #  reversing the orientation of the intersected plane if the photon's current associated volume has defined
        #  surfaceTrimeshes.
        #  This is becasue the surfaceTrimeshes are given designations that match the volume trimesh directions, i.e.
        #  pointing outwards the volume.
        if volumeList[oPhoton.volume].surfaceTrimeshes: volumeIntersectPlaneNormal *= -1
        
        
        
        
        
        
        #### == interaction with the reflector
        if oPhoton.volume == 'reflector' and nextVolume == 'reflector':
            # retrieve photon's current volume properties before any updating of the volume
            refractiveIndex                = volumeList[oPhoton.volume].material['refractiveIndex']                         
            attenuationLength              = volumeList[oPhoton.volume].material['attenuationLength']                         # mm
            #= perform interaction with reflector
            interactType, cosIncidenceAngle, sinTransmissionAngle, newMomentumDir, newPolarizationDir = interactWithReflector(volumeList[oPhoton.volume].material,
                                                                                                                              volumeIntersectPlaneNormal,
                                                                                                                              oPhoton.momentumDir,
                                                                                                                              oPhoton.polarizationDir)
            #= update the photon's volume to the outside 'environment' only in case of transmission
            if interactType == 'transmission':
                oPhoton.volume = 'environment'
            #= estimate traveled distance and spent time
            traveledDistance, traveledTime = traveledDistanceAndTime(oPhoton.position,
                                                                     volumeIntersectPoint,
                                                                     refractiveIndex)  # um, ns
            #= update the photon attributes and tracking history
            oPhoton, opTrackingHistory = updatePhotonInfo(oPhoton, opTrackingHistory,
                                                          volumeIntersectPoint, interactType, -1*volumeIntersectPlaneNormal,
                                                          newMomentumDir, newPolarizationDir,
                                                          cosIncidenceAngle, sinTransmissionAngle,
                                                          traveledDistance, traveledTime, attenuationLength)
            continue
        
        
        
        
        
        #### == the two volumes at the intersection plane are perfectly polished
        if volumeList[oPhoton.volume].surfaceTrimeshes == None and volumeList[nextVolume].surfaceTrimeshes == None:
            # retrieve photon's current volume properties before any updating of the volume
            refractiveIndex                = volumeList[oPhoton.volume].material['refractiveIndex']                         
            attenuationLength              = volumeList[oPhoton.volume].material['attenuationLength']                         # mm
            #= interact with surface
            interactType, cosIncidenceAngle, sinTransmissionAngle, newMomentumDir, newPolarizationDir = interactWithLocalSurface(oPhoton.momentumDir,
                                                                                                                                 oPhoton.polarizationDir,
                                                                                                                                 volumeIntersectPlaneNormal,
                                                                                                                                 volumeList[oPhoton.volume].material['refractiveIndex'],
                                                                                                                                 volumeList[nextVolume].material['refractiveIndex'])
            # update photon's current volume in case of transmission
            if interactType == 'transmission':
                oPhoton.volume = nextVolume
            
            #= estimate traveled distance and spent time
            traveledDistance, traveledTime = traveledDistanceAndTime(oPhoton.position,
                                                                     volumeIntersectPoint,
                                                                     refractiveIndex)       # um, ns
            #= update the photon attributes and tracking history
            oPhoton, opTrackingHistory = updatePhotonInfo(oPhoton, opTrackingHistory,
                                                          volumeIntersectPoint, interactType, -1*volumeIntersectPlaneNormal,
                                                          newMomentumDir, newPolarizationDir,
                                                          cosIncidenceAngle, sinTransmissionAngle,
                                                          traveledDistance, traveledTime, attenuationLength)
        
        
        
        #### == either of the two volumes at the intersection plane has roughness through a surface trimesh
        else:
            oPhoton, opTrackingHistory = trackOverSurfaceTrimesh(volumeList,
                                                                 opTrackingHistory,
                                                                 oPhoton,
                                                                 volumeIntersectPoint,
                                                                 volumeIntersectPlaneNormal,
                                                                 nextVolume)
            
            
        #= kill if the photon weight drops under a minimum of 1E-4 (arbitrary)
        if oPhoton.weight < 1E-4:
            oPhoton.alive = False
            print(' - a photon was killed because its weight dropped below the set minimum ...')
        
        #= score as a detected photon and kill if it reaches and refracts to any of the photodetector volumes
        if oPhoton.volume == 'PD_l' or oPhoton.volume == 'PD_r':
            oPhoton.alive = False
            
        
    return opTrackingHistory

#%% main


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 
    '''
    
    resetIPython()
    
    command  = sys.argv[1]