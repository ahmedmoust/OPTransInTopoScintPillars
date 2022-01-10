from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f')
#get_ipython().run_line_magic('clear', '')
#get_ipython().run_line_magic('matplotlib', 'auto')

import sys
sys.path.append('../../src')

import tracker

import numpy as np
import pickle
import copy
from deepdiff import DeepDiff


#= load geometry file
geometryFileName     = 'bare_dm_geometry_pillar_200mm'
volumeList           = pickle.load(open(f"../../data/geometries/{geometryFileName}.pkl", 'rb'))

#= test cases
testCases = { # photon is inside the pillar and headed towards the right OG volume
              'case_1': {'position'                  : np.array([[0.0, 0.0, 0.0]]),
                         'momentumDir'               : np.array([[0.0, 0.0, 1.0]]),
                         'volume'                    : 'pillar',
                         'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 2]]]),
                         'nextVolume'                : 'OG_r',
                         'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                         'results'                   : {'trimeshIntersectPoint' : np.array([[0.0, 0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 2]]]),
                                                        'incidenceAboveTrimesh' : False}
                        },
              
              # photon is inside the left OG volume and headed towards the pillar
              'case_2': {'position'                  : np.array([[0.0, 0.0, volumeList['OG_l'].volumeTrimesh.bounds[0, 2]+0.5*volumeList['OG_l'].volumeTrimesh.extents[2]]]),
                         'momentumDir'               : np.array([[0.0, 0.0, 1.0]]),
                         'volume'                    : 'OG_l',
                         'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['OG_l'].volumeTrimesh.bounds[1, 2]]]),
                         'nextVolume'                : 'pillar',
                         'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                         'results'                   : {'trimeshIntersectPoint' : np.array([[0.0, 0.0, volumeList['OG_l'].volumeTrimesh.bounds[1, 2]]]),
                                                        'incidenceAboveTrimesh' : True}
                        },
              
              # photon is inside the pillar and headed towards the outside
              'case_3': {'position'                  : np.array([[0.0, 0.0, 0.0]]),
                         'momentumDir'               : np.array([[0.0, -1.0, 0.0]]),
                         'volume'                    : 'pillar',
                         'volumeIntersectPoint'      : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[0, 1], 0.0]]),
                         'nextVolume'                : 'environment',
                         'volumeIntersectPlaneNormal': np.array([[0.0, 1.0, 0.0]]),
                         'results'                   : {'trimeshIntersectPoint' : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[0, 1], 0.0]]),
                                                        'incidenceAboveTrimesh' : False}
                        },
              
              # photon is outside and headed towards the pillar
              'case_4': {'position'                  : np.array([[volumeList['pillar'].volumeTrimesh.bounds[1, 0]+5.0, 0.0, 0.0]]),
                         'momentumDir'               : np.array([[-1.0, 0.0, 0.0]]),
                         'volume'                    : 'environment',
                         'volumeIntersectPoint'      : np.array([[volumeList['pillar'].volumeTrimesh.bounds[1, 0], 0.0, 0.0]]),
                         'nextVolume'                : 'pillar',
                         'volumeIntersectPlaneNormal': np.array([[1.0, 0.0, 0.0]]),
                         'results'                   : {'trimeshIntersectPoint' : np.array([[volumeList['pillar'].volumeTrimesh.bounds[1, 0], 0.0, 0.0]]),
                                                        'incidenceAboveTrimesh' : True}
                        },
              
            }


for key, testCase in testCases.items():
    
    #print('Testing:', key)
    results = {}
    
    #= reversing the orientation of the intersected plane if the photon's current associated volume has defined
        #  surfaceTrimeshes.
        #  This is becasue the surfaceTrimeshes are given designations that match the volume trimesh directions, i.e.
        #  pointing outwards the volume.
    volumeIntersectPlaneNormal     = testCase['volumeIntersectPlaneNormal']
    if volumeList[testCase['volume']].surfaceTrimeshes: volumeIntersectPlaneNormal *= -1
            
    #= retrieve the bounds on the surface trimesh marking the allowed area to sample a potential intersection point
    #  from within
    #  Note: The bounds are set to leave 5% on the area periphery to ensure that the photon doesn't run out of surface
    #        trimesh area while transporting over its features.
    #        The 5% is arbitray. Limiting the area allowed to sample the intersection point within is fine ONLY if the
    #        modeled area has several repetitions of features, if any; otherwise, the parts of the features would be
    #        cut out and therefore the modeled surface trimesh won't be a good representation of the actual surface.
    samplingBounds      = volumeList[testCase['volume']].surfaceTrimeshes[tuple(map(tuple,volumeIntersectPlaneNormal))[0]].samplingBounds\
                            if volumeList[testCase['volume']].surfaceTrimeshes\
                                else volumeList[testCase['nextVolume']].surfaceTrimeshes[tuple(map(tuple,volumeIntersectPlaneNormal))[0]].samplingBounds     # um
    #= get a sampled point on the surface trimesh that marks a potential intersection point as the photon is set to
    #  head towards it
    trimeshSampledPoint = tracker.sampleSurfaceTrimeshPoint(samplingBounds, volumeIntersectPlaneNormal)                              # um
    #= retrieve the surface trimesh to intersect from the volume that has it and acquired volume intersection plane
    surfaceTrimesh      = copy.deepcopy(volumeList[testCase['volume']].surfaceTrimeshes[tuple(map(tuple,volumeIntersectPlaneNormal))[0]].trimesh)\
                            if volumeList[testCase['volume']].surfaceTrimeshes\
                                else copy.deepcopy(volumeList[testCase['nextVolume']].surfaceTrimeshes[tuple(map(tuple,volumeIntersectPlaneNormal))[0]].trimesh)
    #= shift the entire surface trimesh such that the sampled point coincide with the previously acquired tentative
    #  volume intersection point
    #  Note: This simplifies tracking as it unifies the volume and surface trimeshes frame of references
    surfaceTrimesh      = tracker.shiftSurfaceTrimesh(surfaceTrimesh, volumeIntersectPlaneNormal, testCase['volumeIntersectPoint'], trimeshSampledPoint)
    #= get the true intersection point with the surface trimesh and local normal
    results['trimeshIntersectPoint'], trimeshIntersectFaceNormal = tracker.getSurfaceTrimeshIntersection(surfaceTrimesh, testCase['position'], testCase['momentumDir'])    # mm
    results['incidenceAboveTrimesh'] = True if np.matmul(testCase['momentumDir'], trimeshIntersectFaceNormal.transpose()) < 0 else False
    
    #= check if results match the expected corect ones
    diff = DeepDiff(testCase['results'], results, significant_digits=3)
    print('\n')
    if len(diff) == 0: print('{} is checked and passed!'.format(key))
    else: print('{} failed testing! ... \n Differences are: {}'.format(key, diff))
    