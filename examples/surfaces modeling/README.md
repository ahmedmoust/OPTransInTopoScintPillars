# Surface modeling #

This example demonstrates how to model a side surface of a scintilaltion pillar.


Since the main code is built to model optical photon transport in scintillation pillars with
realistic surface topographies, modeling the surface into the code requires tweaking to adjust
the orientation of the surface features (features due to, for example, milling). This is in
order to have the features following the actual orientation on the pillar relative to its long axis.
  
  
It is so recommended that this example file be used for reconstructing, tweaking and saving all final
6 sides of the pillar before running the main code. The main code can then read in the saved surfaces
directly.
  
  
All the necessary information are included as comments over every step in the example file `surfacesModeling.py`.


The example requires for running
- the `surface.py` class file in the `/src/` directory. This directory is temporarily added to the sys.path.
- a point cloud data file from the `/data/point_clouds` directory.
- Open3D: http://www.open3d.org/docs/release/introduction.html
- Trimesh: https://trimsh.org/index.html


Examples of the expected output could be found under the subdirectories, each is designated by the orientation of the
surface of which the subdirectory contains its created files.


## Example output of each step ##
1. Creating a surface with a specific orientation
   
   This loads a point cloud data of a scanned surface area, triangulates it, and reorients the area to match the given
   surface normal vector orientation.
   
   The plot shows a surface with +z normal vector orientation.
   
   ![Image of original triangulated surface](output_+z/original_dm_area_height_map_for_the_+z_surface.png?raw=true)
   
2. (optional) cropping the original point cloud, retraingulating using different param values, and reorienting
   
   This crops the original point cloud if limits were passed, triangulates the area using different parameter values if specified,
   then re-orients the triangulated area to match the surface normal vector orientation.
   
3. (optional)
   A. rotating the surface features of an already triangulated and oriented area
      
      This surface features rotation is performed around the surface normal vector.
	  
	  The plot shows a counter-clockwise rotation with a 35-deg angle.
   
   ![Image of feature-rotated surface](output_+z/rotated_dm_area_height_map_for_the_+z_surface.png?raw=true)
 
   B. cropping the feature-rotated surface for edges alignment
      
      This is performed by specifing limits on the feature-rotated area of the last step. The limits are determined by visually
	  investigating the plots generated from the last step.
	  
	  The plot shows the feature-rotated surface cropped to x=[-170, 170] & y=[-170, 170].
   
   ![Image of cropped feature-rotated surface](output_+z/cropped_dm_area_height_map_for_the_+z_surface.png?raw=true)
   
   

## Important notes:
1. The measured 3D topography must be large enough to contain several repetitions of any surface features. This is important so that no feature is lost due to cropping while preparing the surface, and during the partial area sampling of the mesh at the first mesh interaction.
2. The code does _not_ handle touching surfaces of two volumes with each having a surface mesh.
3. A touching volume to a volume with a surface trimesh must have a thickness larger than its roughness scale. For example, don't set an optical gel thickness of 1 um when the pillar end roughness scale is on the order of 10 um! Otherwise, photon tracking will throw an error.
