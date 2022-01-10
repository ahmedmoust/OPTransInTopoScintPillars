'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f')
#get_ipython().run_line_magic('clear', '')
#get_ipython().run_line_magic('matplotlib', 'inline')


#= add the /src/ directory to the system path to find the necessary class and method files
import sys
sys.path.append('src')

import materials
import geometry
import source
import tracker

import numpy as np
import pickle
import h5py

'''
    Geometry definition
    -------------------
    Step 1: Building materials
            - Default materials with their pre-coded properties are built and must be chosen from;
              otherwise, a new material with its properties must be added first before use.
    Step 2: Constructing cuboid volumes
            - Currently, the code only supports cuboid-shaped volumes (AM 04/19/2021).
            - After a volume is constructed, a surface trimesh of the sample topography could be added to it.
              It's better to pre-create and save the trimeshes for all surfaces external to this `main.py` using
              the `surfaces modeling` or `volumeSurfacesGenerator` example file. This allows mesh configuration 
              to specific settings before directly improting them here to a volume.
            - After all volumes are constructed and added to a volume list, it's better to save it for later
              quick loading.
            
'''

surfaceFinish             = '4dm'
pillarLength              = 200.0   # mm
emissionZPosition         = 0.0     # mm
reflector                 = 'bare'
previouslyCreatedGeometry = False
geometryFileName          = reflector+'_'+surfaceFinish+'_geometry_pillar_'+str(int(pillarLength))+'mm'
recordAllHistory          = True


# == bulid materials
materialsLibrary = materials.materials().materialsLibrary


if previouslyCreatedGeometry:
    print('\n')
    print('Loading previously created geometry...')
    print('--------------------------------------')
    volumeList = pickle.load(open(f"data/geometries/{geometryFileName}.pkl", 'rb'))
else:
    print('\n')
    print('Creating geometry...')
    print('--------------------')
    
    Geo = geometry.geometry()
    
    # == scintillation pillar
    pillar_material      = materialsLibrary['EJ-204']
    pillar_width         = 5.0              # mm
    pillar_length        = pillarLength     # mm
    pillar_center        = np.array([0.0, 0.0, 0.0])
    #= adding the pillar surface trimeshes
    # =============================================================================
    # loading already created meshes
    surfaceTrimeshesDir = "data/surfaceTrimeshes/"
    surfaceTrimeshes    = {'+x': surfaceTrimeshesDir+'+x_'+'dm',
                           '-x': surfaceTrimeshesDir+'-x_'+'dm',
                           '+y': surfaceTrimeshesDir+'+y_'+'ac' if surfaceFinish == '2ac-2dm' else surfaceTrimeshesDir+'+y_'+'dm',
                           '-y': surfaceTrimeshesDir+'-y_'+'ac' if surfaceFinish == '2ac-2dm' else surfaceTrimeshesDir+'-y_'+'dm',
                           '+z': surfaceTrimeshesDir+'+z_'+'dm',
                           '-z': surfaceTrimeshesDir+'-z_'+'dm'
                           }
    Geo.addVolume('pillar', pillar_material, pillar_width, pillar_length, pillar_center)
    Geo.addSurfaceTrimeshes('pillar', surfaceTrimeshes, alreadyCreated = True)
    # =============================================================================
    # # or creating new meshes then loading them
    # pointCloudFileName = 'data/point_clouds/AsCast_1200x.xyz' if surfaceFinish == 'ac-dm' else 'data/point_clouds/DM_480x.xyz'
    # surfaceTrimeshes = {'+x': {'dm': pointCloudFileName},
    #                     '-x': {'dm': pointCloudFileName},
    #                     '+y': {'ac': pointCloudFileName} if surfaceFinish == 'ac-dm' else {'dm': pointCloudFileName},
    #                     '-y': {'ac': pointCloudFileName} if surfaceFinish == 'ac-dm' else {'dm': pointCloudFileName},
    #                     '+z': {'dm': pointCloudFileName},
    #                     '-z': {'dm': pointCloudFileName}
    #                     }
    # pillar_volume.addSurfaceTrimeshes(surfaceTrimeshes = surfaceTrimeshes, alreadyCreated = False)
    
    # == optical gel
    optGel_material            = materialsLibrary['EJ-550']
    optGel_width               = pillar_width
    optGel_length              = 1E-2          # mm
    optGel_center              = np.array([0.0, 0.0, (pillar_length+optGel_length)/2.0])
    Geo.addVolume('OG_l', optGel_material, optGel_width, optGel_length, -1*optGel_center)
    Geo.addVolume('OG_r', optGel_material, optGel_width, optGel_length, optGel_center)
    
    # == PD volumes
    PD_material        = materialsLibrary['SensLGlass']
    PD_width           = pillar_width+1.0      # mm
    PD_length          = 0.5       # mm
    PD_center          = np.array([0.0, 0.0, (pillar_length+2*optGel_length+PD_length)/2.0])
    Geo.addVolume('PD_l', PD_material, PD_width, PD_length, -1*PD_center)
    Geo.addVolume('PD_r', PD_material, PD_width, PD_length, PD_center)
    
    if reflector == 'wrapped':
        # == reflector
        reflector_material    = materialsLibrary['Teflon']
        reflector_width       = pillar_width + 2*np.ceil(np.max([Geo.volumes['pillar'].surfaceTrimeshes[(1, 0, 0)].trimesh.bounds[1, 0], 
                                                                 Geo.volumes['pillar'].surfaceTrimeshes[(0, 1, 0)].trimesh.bounds[1, 1]
                                                                 ])*1E3)*1E-6    # mm
        reflector_length      = pillarLength+2*optGel_length      # mm
        reflector_center      = np.array([0.0, 0.0, 0.0])
        Geo.addVolume('reflector', reflector_material, reflector_width, reflector_length, reflector_center)
    
    # == environment
    environment_material      = materialsLibrary['Air']
    environment_width         = 1000.0       # mm
    environment_length        = 1000.0       # mm
    environment_center        = np.array([0.0, 0.0, 0.0])
    Geo.addVolume('environment', environment_material, environment_width, environment_length, environment_center)
    
    Geo.checkGeometry()
    Geo.identifyTouchingVolumes()
    
    volumeList = Geo.volumes
    
    #= save the created geometry
    pickle.dump(volumeList, open(f"data/geometries/{geometryFileName}.pkl",'wb'), protocol=2) 



'''
    Source definition
    -----------------
    Three options for a source:
        a. List of gamma energy depositions from an external simulation
           each energy deposition is used to generate optical photons with properties randomly sampled from the
           respective scintillation material properties.
           
        b. Gamma energy deposition distribution
           energy depositions are first sampled from the distributions then used to generate optical photons
           with properties randomly sampled from the respective scintillation material properties.
           
        c. List of optical photons emission information from an external simulation
        
        d. Isotropic optical photon point source with specified number of emissions
           optical photons are generated with properties randomly sampled from the respective scintillation
           material properties.
'''
print('\n')
print('Generating source photons...')
print('----------------------------')
# == define an isotropic point source of optical photons
pillar_material               = materialsLibrary['EJ-204']
numberOfOpticalPhotons        = 50
opticalPhotonEmissionPosition = np.array([0.0, 0.0, emissionZPosition])        # mm
sourcePhotons = source.isotropicOpticalPhotonSourceInfo(pillar_material, numberOfOpticalPhotons, opticalPhotonEmissionPosition)
pickle.dump(sourcePhotons, open('output/'+reflector+'_'+surfaceFinish+'_sourcePhotons'+'_pos_'+str(int(emissionZPosition))+'mm.pkl','wb'), protocol=2)
# == or load source photons file
# sourcePhotons = pickle.load(open('output/'+reflector+'_'+surfaceFinish+'_sourcePhotons'+'_pos_'+str(int(emissionZPosition))+'mm.pkl', 'rb'))



'''
    tracking
    --------
    Step 1: Finding the volume intersection point, plane, and next volume
            a. the photon is at the outside 'environment':
                - Intersecitng the photon with all 'volume' cuboid trimeshes.
            b. the photon is inside any of the othe geometry volumes:
                - Intersecting with only the photon's associated volume.
            - The acquired intersection point is prelimenary until intrsection with the a volume's
              'surface' trimesh is performed, if any, to get a true intersection point.
              
    Step 2: Determining whether the intersection plane is between perfectly polished surfaces or either is rough
            and tracking accordingly
            a. both are perfectly polished:
                1. estimate the incidence angle and whether transmission is physically non-prohibited.
                2. if transmission is non-prohibited: estimate reflection probability and either sample reflection or refraction
                   else: TIR is enforced
                3. get the new momentum and polarization directions accordingly.
                4. update the photon status attributes
            b. either surfaces is rough and has a surface trimesh:
                1. sample a potential surface trimesh point to direct the photon towards it.
                   keep sampling for up to 20 times until it's ensured that the photon origin point is above the surface trimesh;
                   otherwise, consider it corner-trapping and kill.
                2. shift the entire surface trimesh such that the acquired prelimenray volume intersection point coincide with the
                   sampled surface trimesh point, i.e. unifying both frame of references.
                3. intersect to find the true intersection point and local normal.
                4. estimate the incidence angle and whether transmission is physically non-prohibited.
                5. if transmission is non-prohibited: estimate reflection probability and either sample reflection or refraction
                   else: TIR is enforced
                6. get the new momentum and polarization directions accordingly.
                7. update the photon status attributes
                8. keep transporting over the surface trimesh for up to 20 local intersections; otherwise, consider it surface
                   feature-trapping and kill photon.
                   
   Step 3: Scoring if photon reaches and refracts to either photodetector volumes
           - otherwise, keep tracking
           - kill photon if its weight falls below a set minimum of 1E-4
            
'''
print('\n')
print('Tracking...')
print('-----------')


#= open an h5 file and initilize and empty trackingHistory dataset   
with h5py.File('output/'+reflector+'_'+surfaceFinish+'_trackingHistory'+'_pos_'+str(int(emissionZPosition))+'mm.hdf5', 'w') as outputFile:

    # == loop for the isotropic point source optical photons
    for p, oPhoton in enumerate(sourcePhotons):
        if p%100 == 0.0: print(str.format('working event: {} ({:2.2f}% complete)', p, float(p)/len(sourcePhotons)*100))
        
        #= track the current photon and return its tracking history
        opTrackingHistory = tracker.trackPhoton(p, oPhoton, volumeList)
        
        if p == 0:
            #= create the dataset for the first time
            trackingHistory = outputFile.create_dataset('trackingHistory',
                                                        data = opTrackingHistory if recordAllHistory else np.array([opTrackingHistory[0]]),
                                                        chunks=True,
                                                        maxshape=(None, ),
                                                        compression='lzf')
        elif recordAllHistory:
            #= append the photon tracking history to the dataset
            trackingHistory.resize((trackingHistory.shape[0] + opTrackingHistory.shape[0]), axis = 0)
            trackingHistory[-opTrackingHistory.shape[0]:] = opTrackingHistory
        else:
            if opTrackingHistory[-1]['volume'] == b'PD_l' or opTrackingHistory[-1]['volume'] == b'PD_r':
                #= append the photon tracking history to the dataset
                trackingHistory.resize((trackingHistory.shape[0] + opTrackingHistory.shape[0]), axis = 0)
                trackingHistory[-opTrackingHistory.shape[0]:] = opTrackingHistory
            else:
                #= append only the emission step to the dataset
                trackingHistory.resize((trackingHistory.shape[0] + 1), axis = 0)
                trackingHistory[-1] =  np.array([opTrackingHistory[0]])
        
    #= add attributes to dataset and close the h5 file
    trackingHistory.attrs['surfaceFinish']       = surfaceFinish
    trackingHistory.attrs['pillarLength']        = pillarLength
    trackingHistory.attrs['reflector']           = reflector
    trackingHistory.attrs['emissionZPosition']   = emissionZPosition