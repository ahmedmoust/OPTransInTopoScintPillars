'''
    This example demonstrates how to model a side surface of the pillar.
    - Since the main code is built to model optical photon transport in scintillation pillars with
      realistic surface topographies, modeling the surface into the code requires tweaking to adjust
      the orientation of the surface features (features due to, for example, milling). This is in
      order to have the features following the actual orientation on the pillar relative to its long axis.
    - It is therefore recommended that this example file be used for reconstructing, tweaking and saving all 
      final 6 sides of the pillar before running the main code. The main code can then read-in the saved
      surfaces directly.
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


#= set a name to the surface
#  - it better matches the normal vector orientation designation
surfaceName        = '-z'

#= set acronym describing the surface finish/roughness
#  - for example: 'dm' for diamond-milled, 'ac' for as-cast
surfaceFinish      = 'dm'

#= set the intended normal vector orientation of the surface to be loaded
#  - this is the final orientation after rotating the surface into its place
#  - options: '+x'  '-x'  '+y'  '-y'  '+z'  '-z'
normalOrientation  = '-z'

#= the name of the point cloud data file
#  - data files are stored in the separate directory /data/point_clouds/
#  - for now, it is recommended to start with the largest area possible (corresponds to the lowest zoom XXXx)
#  - this is to capture most of the surface features after they are rotated and the area is cropped
#    to have its edges aligned with the pillar axes
pointCloudFileName = '../../data/point_clouds/DM_480x.xyz'
#pointCloudFileName = '../../data/point_clouds/AsCast_1200x.xyz'



'''
Step 1:
    creating a surface with a specific orientation
==================================================
'''

#= instantiate the surface using its finish and the final normal vector orientation to be
#  - raises value error if the options of the second attibute are not met
aSurface = surf.surface(surfaceFinish, normalOrientation)

#= read in the point cloud data
#  - loads the data and shifts the heights to a mean=0
aSurface.loadPointCloud(pointCloudFileName)

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

#= plot the trainulated area and the distribution of the surface normals' components
#  - examining the plots is an essential part of constructing and tweaking the surface
#  - it is later used to choose the rotation angle to adjust the alignment of the surface
#    features as well as the margins of the area to crop the original into
#  - the argument represents the current status of the traingulated area being plotted
#    for example, 'original', 'rotated', or 'cropped', this allows saving the plots with
#    different names to follow and visualize the tweaking procress
aSurface.plotTriangulatedArea('original')

#= save the constructed surface object for later quick loading
pickle.dump(aSurface, open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'.pkl','wb'), protocol=2)

#%%
'''
Step 2 (optional):
    cropping the original point cloud, retraingulating using different param values, and reorienting
====================================================================================================
'''

#= load the previously created surface object file
# surfaceName        = '+z'
# surfaceFinish      = 'dm'
# normalOrientation  = '+z'
aSurface = pickle.load(open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'.pkl', 'rb'))

kwargs = {}
#= set the limits of the area to crop the original point cloud to
#  - the limits are set for the two lateral axes, they are automatically identified based on
#    the final normal vector orientation of the surface
#  - note that this is a crop of the original point cloud before triangulating it
#  - another method cropTriangulatedArea() is used to crop an already triangulated area and
#    is recommended to be used after a triangulated area undergoes surface-features rotation
#    to align its edges with the pillar axes
kwargs['ax1_limits'] = np.array([-100, 100])    # um
kwargs['ax2_limits'] = np.array([-100, 100])    # um

#= set new values for the traingulation parameters
kwargs['depth']      = 7
kwargs['scale']      = 1.2

#= perform the cropping (if any) and retriangulation
aSurface.cropPointCloudAndReTriangulate(**kwargs)

#= plot for visual confirmation
aSurface.plotTriangulatedArea('retriangulated')

#= save the rotated surface object for later quick loading for cropping
pickle.dump(aSurface, open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'_retriangulated'+'.pkl','wb'), protocol=2)

#%%
'''
Step 3 (optional):
    A. rotating the surface features of an already triangulated and oriented area
==============================================================================
'''

#= load the previously created surface object file
# surfaceName        = '+x'
# surfaceFinish      = 'dm'
# normalOrientation  = '+x'
aSurface = pickle.load(open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'.pkl', 'rb'))

#= set new values for the traingulation parameters
kwargs = {}
kwargs['depth']      = 7
kwargs['scale']      = 1.2

#= perform the cropping (if any) and retriangulation
aSurface.cropPointCloudAndReTriangulate(**kwargs)

#= set the rotation angle around the surface normal vector
#  - +ve for counter clockwise and -ve for clockwise
rotationAngle = -35.0

#= perform rotation
aSurface.rotateSurfaceFeatures(rotationAngle)

#= plot for visual confirmation
aSurface.plotTriangulatedArea('rotated')

#= save the rotated surface object for later quick loading for cropping
pickle.dump(aSurface, open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'_rotated'+'.pkl','wb'), protocol=2)

#%%
'''
Step 3 (not optional if step 3A was performed):
    B. cropping the feature-rotated surface for edges alignment
============================================================
'''

#= load the previously created surface object file
# surfaceName        = '+z'
# surfaceFinish      = 'dm'
# normalOrientation  = '+z'
aSurface = pickle.load(open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'_rotated'+'.pkl', 'rb'))

#= set the limits of the area to crop triangulated surface to
#  - the limits are determined after the visual investigation of the previously feature-rotated surface
#    and should be chosen to ensure a rectangular shape
#  - the limits are set for the two lateral axes, they are automatically identified based on
#    the normal vector orientation of the surface
#  - note that this done to align the edges with the pillar axes
ax1_limits  = np.array([-170, 170])   # um
ax2_limits  = np.array([-170, 170])   # um

#= perform the cropping  
aSurface.cropTriangulatedArea(ax1_limits, ax2_limits)

#= plot for visual confirmation
aSurface.plotTriangulatedArea('cropped')
    
#= save the cropped surface object for later quick loading into the code
pickle.dump(aSurface, open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'_cropped'+'.pkl','wb'), protocol=2)