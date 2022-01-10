from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f')
#get_ipython().run_line_magic('clear', '')
#get_ipython().run_line_magic('matplotlib', 'auto')

import sys
sys.path.append('../../src')

import tracker

import numpy as np
import pickle
from deepdiff import DeepDiff


#= load geometry file
reflected = True
geometryFileName     = 'wrapped_4dm_geometry_pillar_200mm'
volumeList           = pickle.load(open(f"../../data/geometries/{geometryFileName}.pkl", 'rb'))


#### === bare scintillator test cases
if not reflected:
    testCases = { # photon is inside the pillar and headed towards the outisde environment
                  'case_1': {'position'   : np.array([[0.0, 0.0, 0.0]]),
                             'momentumDir': np.array([[0.0, 1.0, 0.0]]),
                             'volume'     : 'pillar',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 1], 0.0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, -1.0, 0.0]]),
                                             'nextVolume'                : 'environment'
                                            }
                            },
                  
                  # photon is inside the pillar and headed towards the right OG and PD volumes
                  'case_2': {'position'   : np.array([[0.0, 0.0, 0.0]]),
                             'momentumDir': np.array([[0.0, 0.0, 1.0]]),
                             'volume'     : 'pillar',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                                             'nextVolume'                : 'OG_r'
                                            }
                            },
                  
                  # photon is inside the left OG volume and headed towards the pillar
                  'case_3': {'position'   : np.array([[0.0, 0.0, volumeList['OG_l'].volumeTrimesh.bounds[0, 2]+0.5*volumeList['OG_l'].volumeTrimesh.extents[2]]]),
                             'momentumDir': np.array([[0.0, 0.0, 1.0]]),
                             'volume'     : 'OG_l',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['OG_l'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                                             'nextVolume'                : 'pillar'
                                            }
                            },
                  
                  # photon is at outside environment and headed towards the pillar
                  'case_4': {'position'   : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[0, 1]-5.0*1E3, 0.0]]),
                             'momentumDir': np.array([[0.0, 1.0, 0.0]]),
                             'volume'     : 'environment',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[0, 1], 0.0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, -1.0, 0.0]]),
                                             'nextVolume'                : 'pillar'
                                            }
                            },
                  
                  # photon is at outside environment and headed towards the PD volume
                  'case_5': {'position'   : np.array([[0.0,
                                                       volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.005*1E3,
                                                       volumeList['PD_l'].volumeTrimesh.bounds[1, 2]+5.0*1E3]]),
                             'momentumDir': np.array([[0.0, 0.0, -1.0]]),
                             'volume'     : 'environment',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.005*1E3,
                                                                                      volumeList['PD_l'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, 1.0]]),
                                             'nextVolume'                : 'PD_l'
                                            }
                            },
                  
                  # photon is slightly below the pillar surface trimesh and reflecting back
                  'case_6': {'position'   : np.array([[0.0,
                                                       volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.005*volumeList['pillar'].volumeTrimesh.extents[1],
                                                       0.0]]),
                             'momentumDir': np.array([[0.0, 1.0, 0.0]]),
                             'volume'     : 'pillar',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[1, 1],
                                                                                      0.0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, -1.0, 0.0]]),
                                             'nextVolume'                : 'environment'
                                            }
                            },
                  
                  # photon is slightly above the pillar surface trimesh and transmitting to outside environment
                  'case_7': {'position'   : np.array([[0.0,
                                                       volumeList['pillar'].volumeTrimesh.bounds[0, 1]+0.005*volumeList['pillar'].volumeTrimesh.extents[1],
                                                       0.0]]),
                             'momentumDir': np.array([[0.0, -1.0, 0.0]]),
                             'volume'     : 'environment',
                             'results'    : {'volumeIntersectPoint'      : None,
                                             'volumeIntersectPlaneNormal': None,
                                             'nextVolume'                : None
                                            }
                            },
                  
                  # photon is slightly above the pillar surface trimesh and transmitting to OG volume
                  'case_8': {'position'   : np.array([[0.0, 0.0, volumeList['OG_r'].volumeTrimesh.bounds[0, 2]-0.005*volumeList['pillar'].volumeTrimesh.extents[2] ]]),
                             'momentumDir': np.array([[0.0, 0.0, 1.0]]),
                             'volume'     : 'OG_r',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['OG_r'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                                             'nextVolume'                : 'PD_r'
                                            }
                            },
                  
                  # photon is slightly below the pillar surface trimesh and transmitting to OG volume
                  'case_9': {'position'   : np.array([[0.0, 0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 2]+0.005*volumeList['OG_r'].volumeTrimesh.extents[2] ]]),
                             'momentumDir': np.array([[0.0, 0.0, -1.0]]),
                             'volume'     : 'pillar',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['pillar'].volumeTrimesh.bounds[0, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, 1.0]]),
                                             'nextVolume'                : 'OG_l'
                                            }
                            },
                  
                  # photon is transmitted to outside but slightly below the pillar surface trimesh and headed towards OG volume
                  'case_10': {'position'  : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 1]-0.005*volumeList['pillar'].volumeTrimesh.extents[1], 0.0]]),
                             'momentumDir': np.array([[0.0, 0.0, 1.0]]),
                             'volume'     : 'environment',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[1, 1]-0.005*volumeList['pillar'].volumeTrimesh.extents[1],
                                                                                      volumeList['OG_r'].volumeTrimesh.bounds[0, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                                             'nextVolume'                : 'OG_r'
                                            }
                            },
                }

#### === wrapped scintillator test cases
else:
    testCases = { # photon is inside the pillar and headed towards the outisde
                  'case_1': {'position'   : np.array([[0.0, 0.0, 0.0]]),
                             'momentumDir': np.array([[0.0, 1.0, 0.0]]),
                             'volume'     : 'pillar',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 1], 0.0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, -1.0, 0.0]]),
                                             'nextVolume'                : 'reflector'
                                            }
                            },
                  
                  # photon is inside the pillar and headed towards the right OG and PD volumes
                  'case_2': {'position'   : np.array([[0.0, 0.0, 0.0]]),
                             'momentumDir': np.array([[0.0, 0.0, 1.0]]),
                             'volume'     : 'pillar',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                                             'nextVolume'                : 'OG_r'
                                            }
                            },
                  
                  # photon is inside the left OG volume and headed towards the pillar
                  'case_3': {'position'   : np.array([[0.0, 0.0, volumeList['OG_l'].volumeTrimesh.bounds[0, 2]+0.5*volumeList['OG_l'].volumeTrimesh.extents[2]]]),
                             'momentumDir': np.array([[0.0, 0.0, 1.0]]),
                             'volume'     : 'OG_l',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, 0.0, volumeList['OG_l'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                                             'nextVolume'                : 'pillar'
                                            }
                            },
                  
                  # photon is at the airgap on the right OG boundary and headed towards the reflector
                  'case_4': {'position'   : np.array([[0.0,
                                                       volumeList['OG_l'].volumeTrimesh.bounds[0, 1],
                                                       volumeList['OG_l'].volumeTrimesh.bounds[0, 2]+0.5*volumeList['OG_l'].volumeTrimesh.extents[2]]]),
                             'momentumDir': np.array([[0.0, -1.0, 0.0]]),
                             'volume'     : 'reflector',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['reflector'].volumeTrimesh.bounds[0, 1],
                                                                                      volumeList['OG_l'].volumeTrimesh.bounds[0, 2]+0.5*volumeList['OG_l'].volumeTrimesh.extents[2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 1.0, 0.0]]),
                                             'nextVolume'                : 'reflector'
                                            }
                            },
                  
                  # photon is slightly below the pillar surface trimesh and reflecting back
                  'case_5': {'position'   : np.array([[0.0,
                                                       volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.005*volumeList['pillar'].volumeTrimesh.extents[1],
                                                       0.0]]),
                             'momentumDir': np.array([[0.0, 1.0, 0.0]]),
                             'volume'     : 'pillar',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[1, 1],
                                                                                      0.0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, -1.0, 0.0]]),
                                             'nextVolume'                : 'reflector'
                                            }
                            },
                  
                  # photon is slightly above the pillar surface trimesh and transmitting to the air gap
                  'case_6': {'position'   : np.array([[0.0,
                                                       volumeList['pillar'].volumeTrimesh.bounds[0, 1]+0.005*volumeList['pillar'].volumeTrimesh.extents[1],
                                                       0.0]]),
                             'momentumDir': np.array([[0.0, -1.0, 0.0]]),
                             'volume'     : 'reflector',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['reflector'].volumeTrimesh.bounds[0, 1],
                                                                                      0.0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 1.0, 0.0]]),
                                             'nextVolume'                : 'reflector'
                                            }
                            },
                  
                  # photon is transmitted to airgap but slightly below the pillar surface trimesh and headed towards OG volume
                  'case_7': {'position'  : np.array([[0.0, volumeList['pillar'].volumeTrimesh.bounds[1, 1]-0.005*volumeList['pillar'].volumeTrimesh.extents[1], 0.0]]),
                             'momentumDir': np.array([[0.0, 0.0, 1.0]]),
                             'volume'     : 'reflector',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[1, 1]-0.005*volumeList['pillar'].volumeTrimesh.extents[1],
                                                                                      volumeList['OG_r'].volumeTrimesh.bounds[0, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, -1.0]]),
                                             'nextVolume'                : 'OG_r'
                                            }
                            },
                  
                  # photon is at the reflector and headed back to the pillar
                  'case_8': {'position'   : np.array([[0.0,
                                                       volumeList['reflector'].volumeTrimesh.bounds[0, 1],
                                                       0]]),
                             'momentumDir': np.array([[0.0, 1.0, 0.0]]),
                             'volume'     : 'reflector',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[0, 1],
                                                                                      0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, -1.0, 0.0]]),
                                             'nextVolume'                : 'pillar'
                                            }
                            },
                  
                  # photon is at outside environment and headed towards the reflector
                  'case_9': {'position'   : np.array([[0.0, volumeList['reflector'].volumeTrimesh.bounds[0, 1]-5.0*1E3, 0.0]]),
                             'momentumDir': np.array([[0.0, 1.0, 0.0]]),
                             'volume'     : 'environment',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0, volumeList['reflector'].volumeTrimesh.bounds[0, 1], 0.0]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, -1.0, 0.0]]),
                                             'nextVolume'                : 'reflector'
                                            }
                            },
                  
                  # photon is at the airgap and headed towards the PD volume
                  'case_10': {'position'   : np.array([[0.0,
                                                       volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.000005*1E3,
                                                       volumeList['PD_l'].volumeTrimesh.bounds[1, 2]+5.0*1E3]]),
                             'momentumDir': np.array([[0.0, 0.0, -1.0]]),
                             'volume'     : 'reflector',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.000005*1E3,
                                                                                      volumeList['PD_l'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, 1.0]]),
                                             'nextVolume'                : 'PD_l'
                                            }
                            },
                  
                  # photon is at outside environment and headed towards the PD volume
                  'case_11': {'position'   : np.array([[0.0,
                                                       volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.005*1E3,
                                                       volumeList['PD_l'].volumeTrimesh.bounds[1, 2]+5.0*1E3]]),
                             'momentumDir': np.array([[0.0, 0.0, -1.0]]),
                             'volume'     : 'environment',
                             'results'    : {'volumeIntersectPoint'      : np.array([[0.0,
                                                                                      volumeList['pillar'].volumeTrimesh.bounds[0, 1]-0.005*1E3,
                                                                                      volumeList['PD_l'].volumeTrimesh.bounds[1, 2]]]),
                                             'volumeIntersectPlaneNormal': np.array([[0.0, 0.0, 1.0]]),
                                             'nextVolume'                : 'PD_l'
                                            }
                            },
                  
                  # photon is at the airgap and headed towards the reflector and the PD is on the line of sight
                  'case_12': {'position'  : np.array([[-2.49999016,  -0.801861, -98.47099947]])*1E3,
                             'momentumDir': np.array([[-0.06872865, -0.0224301, -0.99738321]])*1E3,
                             'volume'     : 'reflector',
                             'results'    : {'volumeIntersectPoint'      : np.array([[ -2.500446  ,  -0.802009766734, -98.477614573926]])*1E3,
                                             'volumeIntersectPlaneNormal': np.array([[1.0, 0.0, 0.0]]),
                                             'nextVolume'                : 'reflector'
                                            }
                            },
                  
                }                   




             
for key, testCase in testCases.items():
    
    #print('Testing:', key)
    results = {}
    
    results['volumeIntersectPoint'], results['volumeIntersectPlaneNormal'], results['nextVolume'] =\
        tracker.getVolumeTrimeshIntersection(volumeList, testCase['position'], testCase['momentumDir'], testCase['volume'])
    
    #= check if results match the expected corect ones
    diff = DeepDiff(testCase['results'], results, significant_digits=3)
    print('\n')
    if len(diff) == 0: print('{} is checked and passed!'.format(key))
    else: print('{} failed testing! ... \n Differences are: {}'.format(key, diff))