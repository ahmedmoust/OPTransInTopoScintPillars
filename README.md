# Optical Photon Transport in Scintillation Pillars with Topographic Surfaces #

This is a proof-of-concept code for modeling optical photon transport in scintillation pillars with side surfaces modeled using their measured 3D topography. It tracks photons throughout the entire volume of a scintillation pillar to the points of collection on photodetectors. At boundary interactions, the code reflects/transmits photons around the local miscroscopic features of the side surfaces. These surfaces are reconstructed from their 3D topography measured using a surface profilometer, like an atomic force microscope. The code is presented in a publication currently under review.

The code methodology is inspired by the DAVIS LUT model[^1] [^2] [^3], which employes pre-computed look-up tables (LUTs) of the angular distributions of reflection and tranmission at a single modeled surface.

## Description
Comprehensive comments are added to the code files and they describe each step. Briefly, the code structure is as follows:
1. Material information are added to the [materials] class.
2. Each optical photon is defined as an instance of the class [opticalPhoton] with its step-wise information (position, momentum, time, etc.).
3. [source] class is used to create instances of source opticalPhoton. Methods for creating sources include: (a) generating optical photons following an external distribution of radiation energy depositions (e.g. from a GEANT4 simulation), (b) reading photons' information from a file, and (c) an isotropic point source of optical photons.
4. [surface] class is used to create an instance of a scintillaltor topographic side surface. The [surfaceExample] provides an example with detailed description of the process.
5. [volume] class is used to create an instance of a geometry volume.
6. [geometry] class is used to create an instance of full geometry. It has methods to check for volume overlaps as well.
7. [tracker] is the essential part of the code as it contains tracking methods. Crucial tests to the tracker are included in [trackerTest].
8. [main] controls the entire code. It consists of three sections:
a. Geometry definition
b. Source definition
c. Tracking control

###### code units
|  quantity                   |    unit  |
|  -------------------------- | -------- |
| length                      |     um   |
| time                        |     ns   |
| optical photon energy       |     eV   |
| gamma energy deposition     |    keV   |
| wavelength                  |     nm   |

## Required packages
The code requires the following packages:
- [Open3D], for surface reconstruction from a 3D point cloud.
- [Trimesh], for geometry's volumes setup and ray-tracing.
    - For faster ray tracing, Intel's [Embree] accelerator is required, with its python wrapper [pyEmbree].
    - Last pyEmbree `v0.1.6` is only compatible with Embree `v2.17.7`, which in turn is only compatible with python `v3.6`.
    - pyEmbree fails to install/work on Windows. The code therefore has only been run on UNIX-based systems (see tested OSs below).
    - The pyembree acceleration option is activated _during_ volume creation by Trimesh, i.e. volumes created on systems without a functioning pyembree won't utilize the accelerator when used in on systems that has a functioning installation.
- [h5py], for outputing the tracking history.


The code has been tested with the following:
- 64-bit Intel with Ubuntu 18.04 & 64-bit AMD with CentOS Linux 7
- gcc `7.5.0`
- python `3.6.13`
- ipython `v7.16.1`
- cython `v0.29.23`
- open3d `v0.11.2`
- embree `v2.17.7`
- pyembree `v0.1.6`
- trimesh `v3.9.18`
- h5py `3.1.0`


[^1]: E. Roncali, S. R. Cherry, Simulation of light transport in scintillators based on 3D characterization of crystal surfaces, Physics in Medicine and Biology 58 (7) (2013) 2185-2198. [doi:10.1088/0031-9155/58/7/2185].
[^2]: E. Roncali, M. Stockhoff, S. R. Cherry, An integrated model of scintillator-reector properties for advanced simulations of optical transport, Physics in Medicine and Biology 62 (12) (2017) 4811-4830. [doi:10.1088/1361-6560/aa6ca5].
[^3]: M. Stockhoff, S. Jan, A. Dubois, S. R. Cherry, E. Roncali, Advanced optical simulation of scintillation detectors in GATEV8.0: first implementation of a reectance model based on measured data, Physics in Medicine and Biology 62 (12) (2017) 645 L1-L8. [doi:10.1088/1361-6560/aa7007].


[Open3D]: <http://www.open3d.org/docs/release/introduction.html>
[Trimesh]: < https://trimsh.org/index.html>
[Embree]: <https://www.embree.org/>
[pyEmbree]: <https://github.com/scopatz/pyembree>
[h5py]: <https://www.h5py.org/>
[doi:10.1088/0031-9155/58/7/2185]: <https://dx.doi.org/10.1088/0031-9155/58/7/2185>
[doi:10.1088/1361-6560/aa6ca5]: <https://dx.doi.org/10.1088/1361-6560/aa6ca5>
[doi:10.1088/1361-6560/aa7007]: <https://dx.doi.org/10.1088/1361-6560/aa7007>

[materials]: <src/materials.py>
[opticalPhoton]: <src/opticalPhoton.py>
[source]: <src/source.py>
[surface]: <src/surface.py>
[surfaceExample]: <examples/surfaces modeling/README.md>
[volume]: <src/volume.py>
[geometry]: <src/geometry.py>
[tracker]: <src/tracker.py>
[trackerTest]: <tests/tracker/volumeIntersections.py>
[main]: <main.py>



## License

	Copyleft (ɔ) 2022 Ahmed Moustafa and John Mattingly

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

