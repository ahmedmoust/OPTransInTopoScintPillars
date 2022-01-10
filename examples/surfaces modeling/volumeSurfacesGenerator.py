'''
    This example models all side surfaces of the pillar.
    Set the surface settings in the surfaceTrimeshSettings dict. The best settings could be identified for each
    surface separately using the surfaceModeling example file.
    
    - The example requires for running
        - the `surface.py` class file in the /src/ directory, this directory is temporarily added to the sys.path
        - a point cloud data file from the /data/point_clouds/ directory
'''

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f')
#get_ipython().run_line_magic('clear', '')
#get_ipython().run_line_magic('matplotlib', 'auto')


#= temporarily add the /src/ directory to the system path to find surface.py
#  - alternatively, have surface.py in the same directory of this example file
#    and comment out the following two lines.
import sys
sys.path.append('../../src')

import surface as surf

import numpy as np
import pickle



#== choose the surface finish
surfaceFinish = 'dm'
depth = 8
scale = 1.5
featureOreientation = 'perpendicular' # 'perpendicular' or 'parallel'


if surfaceFinish == 'dm':
    #= set settings for all surfaces - diamond milled
    surfaceTrimeshSettings = {'+x': {'finish'            : 'dm',
                                     'pointCloudFileName': '../../data/point_clouds/DM_480x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : 40.0 if featureOreientation == 'perpendicular' else -50.0,
                                     'cropLimits'        : {'ax1_limits': (-170, 170), 'ax2_limits': (-170, 170)}
                                     },
                              
                              '-x': {'finish'            : 'dm',
                                     'pointCloudFileName': '../../data/point_clouds/DM_480x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : -40.0 if featureOreientation == 'perpendicular' else 50.0,
                                     'cropLimits'        : {'ax1_limits': (-170, 170), 'ax2_limits': (-170, 170)}
                                     },
                              
                              '+y': {'finish'            : 'dm',
                                     'pointCloudFileName': '../../data/point_clouds/DM_480x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : -40.0 if featureOreientation == 'perpendicular' else 35,
                                     'cropLimits'        : {'ax1_limits': (-170, 170), 'ax2_limits': (-170, 170)}
                                     },
                              
                              '-y': {'finish'            : 'dm',
                                     'pointCloudFileName': '../../data/point_clouds/DM_480x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : 40.0 if featureOreientation == 'perpendicular' else -35,
                                     'cropLimits'        : {'ax1_limits': (-170, 170), 'ax2_limits': (-170, 170)}
                                     },
                              
                              '+z': {'finish'            : 'dm',
                                     'pointCloudFileName': '../../data/point_clouds/DM_480x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : 35.0,
                                     'cropLimits'        : {'ax1_limits': (-170, 170), 'ax2_limits': (-170, 170)}
                                     },
                              
                              '-z': {'finish'            : 'dm',
                                     'pointCloudFileName': '../../data/point_clouds/DM_480x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : -35.0,
                                     'cropLimits'        : {'ax1_limits': (-170, 170), 'ax2_limits': (-170, 170)}
                                     },
                              }

elif surfaceFinish == 'ac':
    #= set settings for all surfaces - as cast
    surfaceTrimeshSettings = {'+x': {'finish'            : 'ac',
                                     'pointCloudFileName': '../../data/point_clouds/AsCast_1200x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : None,
                                     'cropLimits'        : {'ax1_limits': (-90, 90), 'ax2_limits': (-90, 90)}
                                     },
                              
                              '-x': {'finish'            : 'ac',
                                     'pointCloudFileName': '../../data/point_clouds/AsCast_1200x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : None,
                                     'cropLimits'        : {'ax1_limits': (-90, 90), 'ax2_limits': (-90, 90)}
                                     },
                              
                              '+y': {'finish'            : 'ac',
                                     'pointCloudFileName': '../../data/point_clouds/AsCast_1200x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : None,
                                     'cropLimits'        : {'ax1_limits': (-90, 90), 'ax2_limits': (-90, 90)}
                                     },
                              
                              '-y': {'finish'            : 'ac',
                                     'pointCloudFileName': '../../data/point_clouds/AsCast_1200x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : None,
                                     'cropLimits'        : {'ax1_limits': (-90, 90), 'ax2_limits': (-90, 90)}
                                     },
                              
                              '+z': {'finish'            : 'ac',
                                     'pointCloudFileName': '../../data/point_clouds/AsCast_1200x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : None,
                                     'cropLimits'        : {'ax1_limits': (-90, 90), 'ax2_limits': (-90, 90)}
                                     },
                              
                              '-z': {'finish'            : 'ac',
                                     'pointCloudFileName': '../../data/point_clouds/AsCast_1200x.xyz',
                                     'triangParams'      : {'depth': depth, 'scale': scale},
                                     'rotAngle'          : None,
                                     'cropLimits'        : {'ax1_limits': (-90, 90), 'ax2_limits': (-90, 90)}
                                     },
                              }

   
for surfaceOrientation, settings in surfaceTrimeshSettings.items():
    
    #= instantiate the surface using its finish and the final normal vector orientation to be
    #  - raises value error if the options of the two attibutes are not met
    aSurface = surf.surface(settings['finish'], surfaceOrientation)
    
    #= read in the point cloud data
    #  - loads the data and shifts the heights to a mean=0
    aSurface.loadPointCloud(settings['pointCloudFileName'])
    
    #= triangulate the surface
    #  - uses the Poisson surface reconstruction method that takes in two params: depth and scale
    #  - the params have the two default values: depth=6 and scale=1.0
    #  - the values could be re-set and the surface re-triangulated later using the
    #    cropPointCloudAndReTriangulate() method
    #  - For more information on the Poisson reconstruction method, refer to the article:
    #    https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba 
    aSurface.createTriangularMesh()
    
    #= rotate the surface to face its designated normal vector orientation
    #  - this also cuts 5% of the traingulated surface edges to eliminate the surface-closure triangles created by Trimesh
    aSurface.applyDefaultOrientation()
    
    if settings['triangParams']:
        #= set new values for the traingulation parameters
        kwargs = {}
        kwargs['depth']      = settings['triangParams']['depth']
        kwargs['scale']      = settings['triangParams']['scale']
        
        #= perform the retriangulation; no cropping on the original point cloud since limits are not passed in kwargs
        aSurface.cropPointCloudAndReTriangulate(**kwargs)
        
    if settings['rotAngle']:
        #= set the rotation angle around the surface normal vector
        #  - +ve for counter clockwise and -ve for clockwise
        rotationAngle = settings['rotAngle']
        aSurface.rotateSurfaceFeatures(rotationAngle)
        
        #= set the limits of the area to crop triangulated surface to
        #  - the limits are determined after the visual investigation of the previously feature-rotated surface
        #    and should be chosen to ensure a rectangular shape
        #  - the limits are set for the two lateral axes, they are automatically identified based on
        #    the normal vector orientation of the surface
        #  - note that this done to align the edges with the pillar axes
        ax1_limits  = np.array(settings['cropLimits']['ax1_limits'])
        ax2_limits  = np.array(settings['cropLimits']['ax2_limits'])
        aSurface.cropTriangulatedArea(ax1_limits, ax2_limits)
    elif settings['cropLimits']:
        ax1_limits  = np.array(settings['cropLimits']['ax1_limits'])
        ax2_limits  = np.array(settings['cropLimits']['ax2_limits'])
        aSurface.cropTriangulatedArea(ax1_limits, ax2_limits)

    #= plot for visual confirmation
    aSurface.plotTriangulatedArea('')
        
    #= save the cropped surface object for later quick loading into the code
    pickle.dump(aSurface, open(surfaceOrientation+'_'+settings['finish']+'.pkl','wb'), protocol=2)
    
