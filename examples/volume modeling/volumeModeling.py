'''
    This example demonstrates how to model the pillar volume.
    - This acts as an example of modeling any other volume
    - The example requires for running
        - the `materials.py`, `surfaces` and `volume.py` class files in the /src/ directory,
          this directory is temporarily added to the sys.path
        - a point cloud data file from the /data/point_clouds/ directory
'''

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f')
#get_ipython().run_line_magic('clear', '')
#get_ipython().run_line_magic('matplotlib', 'inline')


#= temporarily add the /src/ directory to the system path to find volume.py and the other 
#  dependent class files. Alternatively, have all of them in the same directory of this example file
#  and comment out the following two lines.
import sys
sys.path.append('../../src')

import materials
import volume as vol


import numpy as np
import pickle

#= bulid materials
materialsLibrary = materials.materials().materialsLibrary

#= give the volume a name
name      = 'pillar'

#= choose the volume material
#  - must match one of the materials library defined in the materials class
material  = 'EJ-204'

#= set the volume width in mm
#  - dimension of the pillar cross section; it's assumed the pillar has a square corss-sectional area
width     = 5.0

#= set the volume length in mm
#  - dimension of the pillar long axis
length    = 200.0

#= set the volume center coordinates in mm
center    = np.array([0.0, 0.0, 0.0])


'''
Step 1:
    creating the volume using the set information
=================================================
'''
aVolume = vol.volume(name, materialsLibrary[material], width, length, center)



'''
Step 2:
    add surface trimeshes to each of the volume 6 side surfaces
===============================================================
'''
# - if all surface meshes have been previously created and saved, just pass them as a dict
#   on the format: {'normalOrientation': surfaceTrimeshes.pkl} with the flag alreadyCreated == True
# - if the surface meshes have not been previously created and saved, pass a dict with all their info
#   on the format: {'normalOrientation': {'finish': 'pointCloudFileName'}} with the flag alreadyCreated == False
# - the surface meshes of a certain volume are stored as an attribute and could be operated-on later after the
#   volume has been created

pointCloudFileName = '../../data/point_clouds/DM_480x.xyz'

surfaceTrimeshes = {'+x': {'dm': pointCloudFileName},
                    '-x': {'dm': pointCloudFileName},
                    '+y': {'dm': pointCloudFileName},
                    '-y': {'dm': pointCloudFileName},
                    '+z': {'dm': pointCloudFileName},
                    '-z': {'dm': pointCloudFileName}
                    }
            
aVolume.addSurfaceTrimeshes(surfaceTrimeshes=surfaceTrimeshes, alreadyCreated = False)

#= save the created volume
pickle.dump(aVolume, open(name+'_volume.pkl','wb'), protocol=2)
