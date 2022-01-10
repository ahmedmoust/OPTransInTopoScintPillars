'''
	Copyleft 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
'''


import sys

import numpy as np
import pandas as pd
import pickle

import trimesh
import open3d

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def seabornInitiation():
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    plt.figure()
    plt.plot(x, y)
    sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    return

def resetIPython():
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
    #get_ipython().run_line_magic('clear', '')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    seabornInitiation()
    return


class surface:
    
    def __init__(self, surfaceFinish, normalOrientation):
        '''
        surface object instantiation
        
        Parameters
        ----------
        normalOrientation : string,
                            an assigned designation to the surface following its normal 
                            vector orientation from the list [-x, +x, -y, +y, -z, +z].
        surfaceFinish: string
                       short acronym describing the surface finish/roughness,
                       for ex.: 'ac' for as-cast and 'dm' for diamond-milled
        
        Raises
        ------
        ValueError : if the entered surface finish and normal vector orientation is not from the lists.

        Returns
        -------
        None.
        '''
        
        print('\n')
        print('Instantiating a surface with {} finish and {} normal vector orientation...'.format(surfaceFinish, normalOrientation))
        
        # == store the surface finish
        self.surfaceFinish = surfaceFinish
        
        # == assign a designation to the surface following its normal vector orientation
        surfaceNormalsDict = {'+x':  (1.0, 0.0, 0.0),
                              '-x': (-1.0, 0.0, 0.0),
                              '+y':  (0.0, 1.0, 0.0),
                              '-y': (0.0, -1.0, 0.0),
                              '+z':  (0.0, 0.0, 1.0),
                              '-z': (0.0, 0.0, -1.0)
                              }
        if normalOrientation not in surfaceNormalsDict.keys():
            raise ValueError('The entered normal vector orientation is not valid...\n'
                             +'Please choose from:'+' '.join(surfaceNormalsDict.keys()))
        self.normalOrientation = surfaceNormalsDict[normalOrientation]
        
        
        
    def loadPointCloud(self, fileName):
        '''
        a method to load a point cloud data file with the surface topography
        
        Parameters
        ----------
        fileName : string,
                   file name of the surface point cloud data. File must contain 3 columns: x y z, with units of um.

        Returns
        -------
        None.
        '''
        
        print(' - loading a point cloud from file: {}...'.format(fileName))
              
        # == read in the point cloud data file into a dataframe
        self.pointCloud   = pd.DataFrame(data=np.loadtxt(fileName), columns=['x', 'y', 'z'])
        # == shift the area center to the origin (0, 0, 0)
        self.pointCloud.x -= (np.max(self.pointCloud.x) - np.min(self.pointCloud.x)) / 2.0
        self.pointCloud.y -= (np.max(self.pointCloud.y) - np.min(self.pointCloud.y)) / 2.0
        # == shift to the mean height
        self.pointCloud.z = (self.pointCloud.z - np.mean(self.pointCloud.z))
        
        return
    
    
    
    def createTriangularMesh(self, depth=6, scale=1.0):
        '''
        a method to create a triangulated mesh from point cloud data using the Poisson surface reconstruction method
        
        Parameters
        ----------
        depth : int, optional
                The depth parameter of the Poisson surface reconstruction method.
                Use steps of 1. Higher value yields higher details but also yields artifacts.
                The default is 6.
        scale : float, optional
                The scale parameter of the Poisson surface reconstruction method.
                Use steps of 0.1. Lower value yields higher details but also yields artifacts.
                The default is 1.0.
        For more information on the Poisson reconstruction method, refer to the article:
        https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
        
        Raises
        ------
        AttributeError: raised if the pointCloud was not found as an instance attribute

        Returns
        -------
        None.
        '''
        
        print(' - triangulating the area from the point cloud using param values: depth={:2.1f}, scale={:1.1f}...'.format(depth, scale))
        
        # == check if a point cloud was loaded and stored as an attribute
        if not hasattr(self, 'pointCloud'):
            raise AttributeError('The pointCloud attribute was not found...\n'
                                 +'Did you forget to first load a point cloud?')
        
        # == create an initial mesh with open 3D
        pointCloud_open3d        = open3d.geometry.PointCloud()
        pointCloud_open3d.points = open3d.utility.Vector3dVector(self.pointCloud.to_numpy())
        # == estimate normal vectors for later usage in trangulating the surface
        pointCloud_open3d.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normal vectors
        pointCloud_open3d.estimate_normals()
        # == triangulate using the Poisson surface method
        poisson_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointCloud_open3d,
                                                                                    depth=int(depth),
                                                                                    scale=scale,
                                                                                    linear_fit=False)[0]
        # == cut to the original area from the artificially extended one after triangulation (caveat with Poisson surfaces method)
        bbox                = pointCloud_open3d.get_axis_aligned_bounding_box()
        croppedPoisson_mesh = poisson_mesh.crop(bbox)
        # == extract mesh triangles and vertices.
        open3d_triangles     = np.asarray(croppedPoisson_mesh.triangles)
        open3d_vertices      = np.asarray(croppedPoisson_mesh.vertices)
        open3d_vertexNormals = np.asarray(croppedPoisson_mesh.vertex_normals)  # different from the triangle normal vectors
        # == create Trimesh triangular mesh from open3d vertices and faces
        self.trimesh = trimesh.Trimesh(open3d_vertices, open3d_triangles, vertex_normals=open3d_vertexNormals)
        #self.trimesh = self.trimesh.process(validate=True)
        self.triangulationParams = {'depth': depth, 'scale': scale}
        
        return
        
    
    
    def applyDefaultOrientation(self):
        '''
        a method to orient a triangulated area based on its designated normal vector orientation
        Note: it is recommended to run plotTriangulatedArea() afterwards to generate plots 
              of the oriented area and the distribution of its normal vectors' components for visual
              investigatoin.
        
        Raises
        ------
        AttributeError: raised if the originalTrimesh was not found as an instance attribute

        Returns
        -------
        None
        '''
        
        print(' - rotating the triangulated area to its default orientation...')
        
        # == check if a triangular mesh was created and stored as an attribute
        if not hasattr(self, 'trimesh'):
            raise AttributeError('The trimesh attribute was not found...\n'
                                 +'Did you forget to first run createTriangularMesh(depth, scale) to triangulate the area?')
        
        # == rotate the triangulated area according to the designated orientation
        if self.normalOrientation   ==  (1.0, 0.0, 0.0): self.rotateArea('y',  90)
        elif self.normalOrientation == (-1.0, 0.0, 0.0): self.rotateArea('y', -90)
        elif self.normalOrientation ==  (0.0, 1.0, 0.0): self.rotateArea('x', -90)
        elif self.normalOrientation == (0.0, -1.0, 0.0): self.rotateArea('x',  90)
        elif self.normalOrientation ==  (0.0, 0.0, 1.0): pass
        elif self.normalOrientation == (0.0, 0.0, -1.0): self.rotateArea('x', 180)
        
        # == crop the triangulated area by a small margin (5%) to eliminate the closure triangles created by Trimesh
        #    This will also add/update the samplingBounds attribute, which is limits on the area to sample within it
        #    the first intersection point while tracking
        #= get the lateral coordinates indices
        lateralCoordIndices = np.where(np.asarray(self.normalOrientation) == 0.0)[0]
        #= get the crop limits
        ax1_limits = [self.trimesh.bounds[0, lateralCoordIndices[0]]+0.025*self.trimesh.extents[lateralCoordIndices[0]],
                      self.trimesh.bounds[1, lateralCoordIndices[0]]-0.025*self.trimesh.extents[lateralCoordIndices[0]]]
        ax2_limits = [self.trimesh.bounds[0, lateralCoordIndices[1]]+0.025*self.trimesh.extents[lateralCoordIndices[1]],
                      self.trimesh.bounds[1, lateralCoordIndices[1]]-0.025*self.trimesh.extents[lateralCoordIndices[1]]]
    
        self.cropTriangulatedArea(ax1_limits, ax2_limits)
        
        # == plot oriented area for visual investigation
        # print(' - Generating plots...')
        # self.plotTriangulatedArea('original')
        
        return
        
    
    
    def cropPointCloudAndReTriangulate(self, **kwargs):
        '''
        a method to crop the point cloud, re-triangulate it and re-orient it following its designated normal vector orientation

        Parameters
        ----------
        **kwargs : if passed, optional 2 sets of two arguments:
            ax1_limits : list of float numbers,
                     the limits on the first axis of the desired cropped area.
            ax2_limits : list of float numbers,
                         the limits on the second axis of the desired cropped area.
                         
            depth : int, optional
                The depth parameter of the Poisson surface reconstruction method.
                Use steps of 1. Higher value yields higher details but also yields artifacts.
                The default is 6.
            scale : float, optional
                    The scale parameter of the Poisson surface reconstruction method.
                    Use steps of 0.1. Higher value yields higher details but also yields artifacts.
                    The default is 1.0.            

        Returns
        -------
        None.
        '''
        
        if 'ax1_limits' in kwargs and 'ax2_limits' in kwargs:
            # == retrieve the low and high limits of the area two lateral axis
            ax1_low, ax1_high = kwargs['ax1_limits']
            ax2_low, ax2_high = kwargs['ax2_limits']
            
            # == crop the original point cloud and restore
            print(' - cropping the point cloud to axis1={}um, axis2={}um...'.format(kwargs['ax1_limits'],kwargs['ax2_limits']))
            self.pointCloud = self.pointCloud.iloc[np.where( (self.pointCloud.x > ax1_low) &
                                                             (self.pointCloud.x < ax1_high) &
                                                             (self.pointCloud.y > ax2_low) &
                                                             (self.pointCloud.y < ax2_high)
                                                            )
                                                   ]
            self.pointCloud.reset_index(drop=True, inplace=True)
        
        # == retriangulate
        if 'depth' in kwargs and 'scale' in kwargs:
            self.createTriangularMesh(depth=kwargs['depth'], scale=kwargs['scale'])
        else:
            self.createTriangularMesh()
        
        # == reapply the default orientation
        self.applyDefaultOrientation()
        
        return
    
    
    
    def rotateSurfaceFeatures(self, rotAngle):
        '''
        a method to rotate the surface features of the area already oriented towards its default,
        i.e. rotation will be around the area's designated normal vector orientation.
        Plots are then again generated for visual confirmation.
        Note: - it is recommended to run cropTriangulatedArea() afterwards to yield
                a rectangular area with edges parallel to the coordinate axes.
              - it is also recommended to run plotTriangulatedArea() afterwards to generate plots 
                of the oriented area and the distribution of its normal vectors' components for visual
                confirmation.
        
        Parameters
        ----------
        rotAngle : float,
                   angle of rotation in degrees - +ve for counter clockwise and -ve for clockwise.

        Returns
        -------
        None.
        '''
        
        print(' - rotating the surface features by {:2.2f} deg...'.format(rotAngle))
        
        # == rotate the previously oriented area around the surface designated orientation
        rotAxis = np.extract(np.abs(np.asarray(self.normalOrientation)) == 1.0, ['x', 'y', 'z'])[0]
        self.rotateArea(rotAxis, rotAngle)
        
        # == plot the area after surface features rotation for visual confirmation
        # print(' - Generating plots...')
        # self.plotTriangulatedArea('rotated')
        
        return
    
    
    
    def cropTriangulatedArea(self, ax1_limits, ax2_limits):
        '''
        a method to crop an already triangulated and oriented area
        Note: - it is recommended to crop the triangulated area, even by a small margin,
                to eliminate the issue of area edge closure by extra triangles that Trimesh performs.
              - it is also recommended to run plotTriangulatedArea() afterwards to generate plots 
                of the oriented area and the distribution of its normal vectors' components for visual
                confirmation.

        Parameters
        ----------
        ax1_limits : list of float numbers,
                     the limits on the first axis of the desired cropped area.
        ax2_limits : list of float numbers,
                     the limits on the second axis of the desired cropped area.

        Returns
        -------
        None.
        '''
        
        print(' - cropping the triangulated area to axis1={}um, axis2={}um...'.format(ax1_limits, ax2_limits))
        
        # == retrieve the low and high limits of the area two lateral axis
        ax1_low, ax1_high = ax1_limits
        ax2_low, ax2_high = ax2_limits
        
        # == create the bounding 4 parallel planes to the designated orientation
        lateralCoordIndices = np.where(np.asarray(self.normalOrientation) == 0.0)[0]
        #
        planesOrigin = np.zeros([4, 3])
        planesOrigin[0, lateralCoordIndices[0]] = ax1_low
        planesOrigin[1, lateralCoordIndices[0]] = ax1_high
        planesOrigin[2, lateralCoordIndices[1]] = ax2_low
        planesOrigin[3, lateralCoordIndices[1]] = ax2_high
        #
        shiftingArray = np.zeros(3)
        shiftingArray[lateralCoordIndices[0]] = (ax1_high - ax1_low)/2.0 + ax1_low
        shiftingArray[lateralCoordIndices[1]] = (ax2_high - ax2_low)/2.0 + ax2_low
        #
        planesNormal = -1*np.sign(planesOrigin)
        #
        # == crop the orientedTrimesh using the created bounding 4 planes
        self.trimesh = self.trimesh.slice_plane(planesOrigin, planesNormal)
        # == shift the cropped orientedTrimesh center to the origin (0, 0, 0)
        self.trimesh = self.trimesh.apply_translation(shiftingArray)
        
        # == estimate the limits on the area to sample within it the first intersection point while tracking
        samplingBounds = np.zeros([2, 3])
        samplingBounds[0, lateralCoordIndices[0]] = self.trimesh.bounds[0, lateralCoordIndices[0]]+0.025*self.trimesh.extents[lateralCoordIndices[0]]
        samplingBounds[1, lateralCoordIndices[0]] = self.trimesh.bounds[1, lateralCoordIndices[0]]-0.025*self.trimesh.extents[lateralCoordIndices[0]]
        samplingBounds[0, lateralCoordIndices[1]] = self.trimesh.bounds[0, lateralCoordIndices[1]]+0.025*self.trimesh.extents[lateralCoordIndices[1]]
        samplingBounds[1, lateralCoordIndices[1]] = self.trimesh.bounds[1, lateralCoordIndices[1]]-0.025*self.trimesh.extents[lateralCoordIndices[1]]
        self.samplingBounds = samplingBounds
        
        # == plot the area after surface features rotation for visual confirmation
        # print(' - Generating plots...')
        # self.plotTriangulatedArea('cropped')
        
        return
        
    
    
    def rotateArea(self, rotAxis, rotAngle):
        '''
        a method to rotate a triangulated area around a coordinate axis
        
        Parameters
        ----------
        rotAxis  : string,
                   axis around which rotation is done - 'x', 'y', 'z'.
        rotAngle : float,
                   angle of rotation in degrees - +ve for counter clockwise and -ve for clockwise.

        Returns
        -------
        None.
        '''
        
        # == convert to radians
        theta = rotAngle*(np.pi/180) 
        
        # == construct the rotation matrix
        if rotAxis == 'x':
            M = np.array([[1,             0,              0, 0],
                          [0, np.cos(theta), -np.sin(theta), 0],
                          [0, np.sin(theta),  np.cos(theta), 0],
                          [0,             0,              0, 1]
                        ])
        elif rotAxis == 'y':
            M = np.array([[ np.cos(theta), 0,  np.sin(theta), 0],
                          [             0, 1,              0, 0],
                          [-np.sin(theta), 0,  np.cos(theta), 0],
                          [             0, 0,              0, 1]
                        ])
        elif rotAxis == 'z':
            M = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                          [np.sin(theta),  np.cos(theta), 0, 0],
                          [            0,              0, 1, 0],
                          [            0,              0, 0, 1]
                        ])
        
        # == rotate the triangulated area
        self.trimesh = self.trimesh.apply_transform(M)
        
        # == store the applied rotation angle to the original trimesh
        self.rotAngle = rotAngle
        
        return
    
    
    
    def plotTriangulatedArea(self, tag):
        '''
        a method to plot the triangulated surface and the components of the surface nomals

        Parameters
        ----------
        tag : string,
              a description of the status of the triangulated area being plotted.
              Possible tags: 'original', 'rotated', 'cropped'
                        
        Returns
        -------
        None.
        '''
        
        #= retrieving the normal on the foramt of a string
        normalAxis = np.extract(np.abs(np.asarray(self.normalOrientation)) == 1.0, ['x', 'y', 'z'])[0]
        orientation = '+' if np.all(np.asarray(self.normalOrientation) >= 0) else '-'
        
        # == plotting the distribution of the normal vectors componnets
        plt.figure(figsize = (22, 16))
        plt.title("distribution of {} area triangles' normal vectors' components for the {} surface".format(tag+' '+self.surfaceFinish, orientation+normalAxis), loc = 'left', fontsize=22)
        bins = np.arange(-1.0, 1.0+0.001, 0.001)
        plt.hist(self.trimesh.face_normals[:, 0], bins=bins, histtype='step', color = 'xkcd:lightish blue', linestyle='-', linewidth = 1.1, label = 'x')
        plt.hist(self.trimesh.face_normals[:, 1], bins=bins, histtype='step', color = 'xkcd:wine', linestyle='-', linewidth = 1.1, label = 'y')
        plt.hist(self.trimesh.face_normals[:, 2], bins=bins, histtype='step', color = 'xkcd:blue green', linestyle='-', linewidth = 1.1, label = 'z')
        plt.xlabel('component value', fontsize=22)
        plt.xlim(-1.1, 1.1)
        plt.ylabel('count', fontsize=22)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(loc='upper right', fontsize=22, frameon=True, edgecolor='k', framealpha=1.0)
        sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
        plt.savefig('distribution_of_{}_area_triangles_normal_vectors_components_for_the_{}_surface.png'.format(tag+' '+self.surfaceFinish, orientation+normalAxis), bbox_inches='tight', dpi = 200)
        
        # == plotting the triangulated surface
        if normalAxis == 'x':
            x, y, z = (self.trimesh.vertices[:, 1], self.trimesh.vertices[:, 2], self.trimesh.vertices[:, 0])
            xlabel, ylabel, zlabel = ('y', 'z', 'x')
        elif normalAxis == 'y':
            x, y, z = (self.trimesh.vertices[:, 0], self.trimesh.vertices[:, 2], self.trimesh.vertices[:, 1])
            xlabel, ylabel, zlabel = ('x', 'z', 'y')
        elif normalAxis == 'z':
            x, y, z = (self.trimesh.vertices[:, 0], self.trimesh.vertices[:, 1], self.trimesh.vertices[:, 2])
            xlabel, ylabel, zlabel = ('x', 'y', 'z')
        #    
        fig = plt.figure(figsize = (24, 16))
        ax = fig.gca(projection='3d')
        plt.title('{} area height map for the {} surface'.format(tag+' '+self.surfaceFinish, orientation+normalAxis), loc = 'left', fontsize=22)
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin = -0.5, vmax = 0.5)
        alpha  = 0.3
        surf = ax.plot_trisurf(x, y, z, alpha=alpha, edgecolor='none', cmap=cmap, norm = norm)
        ax.set_xlabel(xlabel+' ($\mu m$)', fontsize = 22, labelpad=22)
        ax.set_ylabel(ylabel+' ($\mu m$)', fontsize = 22, labelpad=22)
        ax.set_zlabel(zlabel+' ($\mu m$)', fontsize = 22, rotation=10, labelpad=22)
        cbar = plt.colorbar(surf)
        #cbar = plt.colorbar(surf, shrink=0.8, aspect=4)
        #cbar.set_label('', size=22)
        cbar.ax.tick_params(labelsize=22)
        ax.view_init(azim=45.0, elev=60.0)
        ax.tick_params(axis='both', which='major', labelsize=22)
        sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
        plt.savefig('{}_area_height_map_for_the_{}_surface.png'.format(tag+'_'+self.surfaceFinish, orientation+normalAxis),  bbox_inches='tight', dpi = 300)

        return


#%% main


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 'create' or 'cropPointCloudAndReTriangulate' or 'rotateFeatures' or 'cropTriangulatedSurface'
    '''
    
    resetIPython()
    
    command  = sys.argv[1]
    
    if command == 'create':
        
        surfaceName        = sys.argv[2]
        surfaceFinish      = sys.argv[3]
        normalOrientation  = sys.argv[4]
        pointCloudFileName = sys.argv[5]
    
        aSurface = surface(surfaceFinish, normalOrientation)
        aSurface.loadPointCloud(pointCloudFileName)
        aSurface.createTriangularMesh()
        aSurface.applyDefaultOrientation()
        aSurface.plotTriangulatedArea('original')
        pickle.dump(aSurface, open(surfaceName+'_'+surfaceFinish+'_'+normalOrientation+'.pkl','wb'), protocol=2)
        
    if command == 'cropPointCloudAndReTriangulate':
        
        surfaceName = sys.argv[2]
        kwargs = {}
        if len(sys.argv) > 3:
            kwargs['ax1_limits'] = ( float(sys.argv[3].split(',')[0][1:]), float(sys.argv[3].split(',')[1][:-1]) )
            kwargs['ax2_limits'] = ( float(sys.argv[4].split(',')[0][1:]), float(sys.argv[4].split(',')[1][:-1]) )
        if len(sys.argv) > 5:
            kwargs['depth']       = float(sys.argv[5])
            kwargs['scale']       = float(sys.argv[6])
            
        aSurface = pickle.load(open(surfaceName+'.pkl', 'rb'))
        aSurface.cropPointCloudAndReTriangulate(**kwargs)
        aSurface.plotTriangulatedArea('retriangulated')
    
    if command == 'rotateFeatures':
        
        surfaceName   = sys.argv[2]
        rotationAngle = float(sys.argv[3])
        
        aSurface = pickle.load(open(surfaceName+'.pkl', 'rb'))
        aSurface.rotateSurfaceFeatures(rotationAngle)
        aSurface.plotTriangulatedArea('rotated')
        
        pickle.dump(aSurface, open(surfaceName+'_rotated'+'.pkl','wb'), protocol=2)
        
    if command == 'cropTriangulatedSurface':
        
        surfaceName   = sys.argv[2]
        ax1_limits  = ( float(sys.argv[3].split(',')[0][1:]), float(sys.argv[3].split(',')[1][:-1]) )
        ax2_limits  = ( float(sys.argv[4].split(',')[0][1:]), float(sys.argv[4].split(',')[1][:-1]) )
        
        aSurface = pickle.load(open(surfaceName+'.pkl', 'rb'))
        aSurface.cropTriangulatedArea(ax1_limits, ax2_limits)
        aSurface.plotTriangulatedArea('cropped')
        
        pickle.dump(aSurface, open(surfaceName+'_cropped'+'.pkl','wb'), protocol=2)
        
        