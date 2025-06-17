# DENSS: DENsity from Solution Scattering

Author: Thomas Grant | email: tdgrant@buffalo.edu
#### [Nature Methods paper describing DENSS](https://www.nature.com/articles/nmeth.4581)
If you use DENSS in your work, please cite:

Grant, Thomas D. (2018). Ab initio electron density determination directly from solution scattering data. Nature Methods. http://dx.doi.org/10.1038/nmeth.4581.

#### Official DENSS Website [https://www.tdgrant.com/denss](https://www.tdgrant.com/denss)
tdgrant.com/denss is the official home of DENSS. Packed with detailed instructions for installing and running DENSS on your own computer. It also contains useful tips and best practices.

#### DENSSWeb server: [denss.ccr.buffalo.edu](https://denss.ccr.buffalo.edu)
Try out DENSS without installing the code using the DENSSWeb server. Run DENSSWeb in standard Slow and Membrane modes suitable for publication. Many file types are accepted, including raw data. You can also enforce symmetry. Only takes about 5 minutes for a fully averaged reconstruction. Thanks to Andrew Bruno and the CCR for DENSSWeb!

## NEWS

#### DENSS now on PyPI and installable with pip
DENSS v1.8.0 has been refactored to be more compatible with modern Python conventions, including convenient package installation with pip. DENSS now requires Python > 3.6+. As a result, all command line programs have slightly different names, using hyphens rather than dots, and losing the trailing `.py` extension. For example, the old `denss.align.py` is now `denss-align`.

#### New SWAXS calculator for atomic models
DENSS v1.7.0 introduces a new tool for calculating highly accurate SWAXS profiles from atomic models and fitting experimental data (like CRYSOL, FoXS and others). The denss.pdb2mrc.py script will accept atomic models in the PDB format and generate a high-resolution electron density map incorporating solvent terms such as excluded volume and the hydration shell. The article describing the method is available on bioRxiv (link to be added).

#### DENSS v1.7.0 released
DENSS v1.7.0 has now been released. Included in the updates are many bug fixes, speed and stability improvements. A new script called denss.mrc2sas.py is included to fit a calculated scattering profile from a density map to experimental SWAXS data, and calculate chi2 as a metric of the quality of fit. Similar fits are also now produced by denss.py, denss.refine.py, and denss.all.py.

#### Automatic Dmax estimation and fitting of raw data
When provided raw experimental data, DENSS will now automatically estimate the maximum dimension and fit the profile with a smooth curve. This uses the same algorithm used in denss.fit_data.py for fitting. Additionally, denss.fit_data.py now outputs a .fit file with header information including the Dmax. denss.py can now take this .fit as input and read the Dmax value directly from the header, thus removing the need to expressly set the -d option in denss.py. Users are still advised to manually run denss.fit_data.py to ensure accurate Dmax and fits are obtained. These features are available in version 1.6.1+.

#### DENSS version 1.6.0 released
Shrinkwrap in DENSS has been updated to be more robust, particularly for particles with negative contrast used in MEMBRANE mode. Several bug fixes. Thanks to intern Esther Gallmeier!

#### DENSS now updated for GPU acceleration with CUDA
A new --gpu option when running denss.py v1.5.0+ with NVIDIA CUDA capable graphics cards is available. Tests show this option speeds DENSS up by more than 20x compared to CPU-only. Requires CuPy to be installed. Will attempt to update this in the future for more GPUs. Thanks to intern Esther Gallmeier!

#### DENSS is now python3 compatible!
DENSS has been updated to support python 3 in addition to python 2. I will attempt to maintain python 2 support for as long as possible.

#### New MEMBRANE mode
DENSS now has a new mode for membrane proteins. Membrane proteins are often solubilized in detergents or lipid nanodiscs. The hydrophobic regions of these molecules often have lesser scattering density than the bulk solvent, resulting in a negative contrast relative to the solvent. The default setting for DENSS enforces a positivity restraint that will not allow any density to be negative. While this is appropriate for most standard biomolecules such as proteins and nucleic acids, is it not appropriate for molecules containing regions of negative contrast. To accommodate this scenario, there is a new MEMBRANE mode in addition to the previously available FAST and SLOW modes. This mode disables the positivity restraint and starts shrink-wrap immediately.

#### New symmetry averaging feature
A new feature in denss v1.4.6 allows for the use of symmetry if known. The options for imposing symmetry are --ncs, --ncs_axis, and --ncs_steps. Currently only symmetry along a single axis is supported, though multiple axes will be supported in the future. Symmetry is imposed by first aligning the principal axes of inertia with the XYZ axes (largest to smallest). Then symmetry averaging is performed along the selected axis at the given step(s). More frequent steps makes for stronger restraint, but consequently more bias. Can select a different axis (in case the largest principal axis is not the symmetry axis). Note that the averaging procedure is still symmetry agnostic. Also, you may need to manually filter sets of maps in case the wrong symmetry axis was chosen in some cases, then perform averaging separately with the denss.align_and_average.py script.

#### New script for performing simple operations on MRC files
A new script, `denss.mrcops.py`, is included in v1.4.5 that includes handy tools for resampling or reshaping an MRC formatted electron density map.

#### New refinement script
A new script called `denss.refine.py` is available for refining an averaged electron density map. Final averaged maps from denss.all.py or superdenss are unlikely to have scattering profiles matching the data (since they are an average from many different maps). To refine the averaged map so that its scattering profile matches the data, one can input the averaged map to denss.refine.py using the --rho_start option. `denss.refine.py` works similarly to denss.py, and will take the same input file used in the original run of denss.all.py.

#### New Averaging Procedure
A new procedure for aligning and averaging electron density maps with DENSS is now available with v1.4.3. The new procedure is written entirely in Python from the ground up and no longer requires EMAN2 to be installed. The new procedure can be accessed with a series of new scripts, named as denss.xxx.py where xxx is the name of the script. denss.all.py acts as the old superdenss, running twenty reconstructions, aligning and averaging all maps, including enantiomer generation and selection. Thanks to intern Nhan Nguyen for helping to write the new code.

#### New interactive GUI for fitting data
A new script (denss.fit_data.py) is provided with DENSS v1.3.0 for calculating smooth fits to experimental data using a convenient interactive GUI.

#### New script for calculating profiles from MRC files
A new script (denss.rho2dat.py) is provided with DENSS v1.3.0 for calculating scattering profiles from MRC files.


### About DENSS
DENSS is an algorithm used for calculating ab initio electron density maps directly from solution scattering data. DENSS implements a novel iterative structure factor retrieval algorithm to cycle between real space density and reciprocal space structure factors, applying appropriate restraints in each domain to obtain a set of structure factors whose intensities are consistent with experimental data and whose electron density is consistent with expected real space properties of particles.

DENSS utilizes the Fast Fourier Transform for moving between real and reciprocal space domains. Each domain is represented by a grid of points (Cartesian), N x N x N. N is determined by the size of the system and the desired resolution. The real space size of the box is determined by the maximum dimension of the particle, D, and the desired sampling ratio. Larger sampling ratio results in a larger real space box and therefore a higher sampling in reciprocal space (i.e. distance between data points in q). Smaller voxel size in real space corresponds to higher spatial resolution and therefore to larger q values in reciprocal space.

The core functions are stored in the `saxstats.py` module. The actual script to run DENSS is `denss.py`.

## Installation
DENSS v1.8.0+ can be installed at the command line simply by typing
```
pip install denss
```

DENSS can also be installed from source by typing at the command prompt in the directory where you downloaded DENSS:
```
pip install .
```

## Requirements
DENSS requires that Python (3.6+), NumPy (v1.10+) and SciPy are installed. These packages are often installed by default, or are available for your operating system using package managers such as PIP or [Anaconda](https://www.continuum.io/downloads). The current code was built using the Anaconda package management system on Mac OS X. 

## Input files
DENSS uses smooth fits to experimental scattering profiles (rather than the noisy experimental data). Multiple file formats are currently acceptable: .dat files (3-column ASCII text files: q, I, error), .fit files (4-columns: q, I, error, fit), or .out files (GNOM). DENSS will check if the data in .dat format are raw or smoothed data. If raw, DENSS will roughly estimate Dmax and fit the data with a smooth curve. However, it is best to use the denss.fit_data.py script described below. If a .fit is given (the output of denss.fit_data.py), then Dmax will be read directly from the header of that file. DENSS will extract Dmax from files if possible, otherwise will estimate it automatically. 

A script called `denss-fit-data` is provided which can be used to fit experimental data with a smooth curve based on an extended version of Peter Moore's approach (Moore 1979) using a trigonometric series. The denss.fit_data.py script includes a simple interactive GUI for selecting Dmax and the smoothing factor alpha and displays the experimental data, the smooth fit to the data, and the real space pair distribution function. denss.fit_data.py will save a .fit file containing the smooth fit to the data which can then be used as input to denss.py (see below). Additionally, useful parameters calculated from the fit, such as the radius of gyration and Porod volume, are displayed. The manuscript describing the mathematical derivation and the algorithm of this new approach is currently in preparation.

`denss-fit-data` can be run simply from the command line as:
```
denss-fit-data -f <experimental_data.dat>
```
where <experimental_data.dat> is the noisy scattering profile, given as a three-column ASCII text file with columns q, I, error. An interactive GUI will appear showing the experimental scattering profile on the left along with the fit to the data, and the associated pair distribution function (P(r)) on the right. Two interactive sliders on the bottom left can be adjusted for Dmax (the maximum particle dimension) and the alpha smoothing factor. See `denss.fit_data.py -h` for more options. denss.fit_data.py will output two files, one containing the fitted, smooth scattering profile (ends with .fit) and one containing the corresponding P(r) curve (ends with pr.dat). Parameters such as Dmax, alpha, Rg, volume, etc. are printed to the terminal screen and also are written in the header of the .fit file. The .fit file can be used directly as input to denss.py, which will then read the Dmax value from the header.

DENSS also accepts [GNOM](https://www.embl-hamburg.de/biosaxs/gnom.html) .out files created by [ATSAS](https://www.embl-hamburg.de/biosaxs/software.html) (credit for .out parsing - Jesse Hopkins).

`lysozyme.out` is a GNOM .out file from real lysozyme data. `6lyz.dat` is a simulated scattering profile from lysozyme PDB 6LYZ using FoXS. `6lyz.out` is a GNOM .out file created from the `6lyz.dat` data file. Any of these files can be used as input to DENSS for testing.

It is best to provide denss.py with a data file where q is defined as 4 pi sin(theta)/lambda in angstroms. However, since some beamlines provide data in 1/nm, the -u option of denss can be set to "nm" in that case.

## Usage
DENSS can be run with basic defaults as follows:
```
denss -f <saxs.out>
```
In this case, DENSS uses the maximum dimension from the .out file. You can override this maximum dimension by specifying the -d parameter. Similarly you can provide a .fit file from denss-fit-data:
```
denss -f <saxs.fit>
```
If you provide a raw experimental profile that has not been fitted, denss.py will attempt to estimate Dmax for you from the data and fit the data with the same algorithm denss-fit-data uses. However, it is advised that you manually run denss-fit-data or GNOM to ensure accurate fit and estimation of Dmax.

Examples: using the supplied 6lyz.out data, DENSS can be run with:
```
denss -f 6lyz.out
```

On Windows, how to run DENSS scripts depends heavily on your setup. We recommend selecting the option to "Add Python to PATH" when installing Python. If your Python Scripts directory is in your path, you may be able to simply type: 
```
denss -f 6lyz.out
```
Alternatively, Windows users can also use virtual environments such as `venv` or `conda`, and follow the instructions above.

However, if it is not in your path, you may need to give it the full path, such as:
```
C:\Python39\Scripts\denss.exe -f 6lyz.out
```

By default, DENSS assumes data are given in 1/Å (q = 4π sin(θ)/λ, where 2θ is the scattering angle and λ is the wavelength of the incident beam). The -u option can be used set the scale to inverse nanometers (-u nm).

Options you may want to set are:
```
  -h, --help            show this help message and exit
  -f FILE, --file FILE  SAXS data file for input (either .dat, .fit, or .out)
  -u UNITS, --units UNITS
                        Angular units ("a" [1/angstrom] or "nm" [1/nanometer]; default="a")
  -d DMAX, --dmax DMAX  Estimated maximum dimension (Default=100)
  -n NSAMPLES, --nsamples NSAMPLES
                        Number of samples, i.e. grid points, along a single
                        dimension. (Sets voxel size, overridden by --voxel.
                        Best optimization with n=power of 2. Default=64)
  -ncs NCS, --ncs NCS   Rotational symmetry
  -ncs_steps NCS_STEPS [NCS_STEPS ...], --ncs_steps NCS_STEPS [NCS_STEPS ...]
                        List of steps for applying NCS averaging (default=3000)
  -ncs_axis NCS_AXIS, --ncs_axis NCS_AXIS
                        Rotational symmetry axis (options: 1, 2, or 3 corresponding to xyz principal axes)
  -s STEPS, --steps STEPS
                        Maximum number of steps (iterations)
  -o OUTPUT, --output OUTPUT
                        Output map filename
  -m MODE, --mode MODE  Mode. F(AST) sets default options to run quickly for
                        simple particle shapes. S(LOW) useful for more complex
                        molecules. M(EMBRANE) mode allows for negative contrast. (default SLOW)
```
By default DENSS runs in SLOW mode, which is generally suitable for the vast majority of particles, including those with complex shapes. You can override all the default parameters set by the SLOW mode by explicitly setting any of the options.

Additional advanced options can be seen by typing `denss -h`.

## Results
As the program runs, the current status will be printed to the screen like so:
```
Step  Chi2      Rg      Support Volume
----- --------- ------- --------------
 2259  1.31e+00  14.34        42135
```
Where `Step` represents the number of iterations so far, `Chi2` is the fit of the calculated scattering of the map to the experimental data, `Rg` is the radius of gyration calculated directly from the electron density map, and `Support Volume` is the volume of the support region.

Electron density maps are written in CCP4/MRC format (credit Andrew Bruno). These files can be opened directly in some visualization programs such as [Chimera](http://www.rbvi.ucsf.edu/chimera/) and [PyMOL](https://www.pymol.org). In particular, the PyMOL "volume" function is well suited for displaying these maps with density information displayed as varying color and opacity.

Output files include:
```
output.mrc                 electron density map (MRC format)
output_support.mrc         final support volume formatted as unitary electron density map
output_stats_by_step.dat   statistics as a function of step number.
                           three columns: chi^2, Rg, support volume
output_map.fit             The fit of the calculated scattering profile to the
                           experimental data. Experimental data has been interpolated to the q values used for scaling intensities and I(0) has been scaled to the square of the number of electrons in the particle.
                           Columns are: q(data), I(data), error(data), I(calc)
output_*.png               If plotting is enabled, these are plots of the results.
output.log                 A log file containing parameters for the calculation
                           and summary of the results.
```

## Alignment, Averaging, and Resolution Estimation
The solutions are non-unique, meaning many different electron density maps will yield the same scattering profile. Different random starting points will return different results. Therefore, running the algorithm many times (>20) is strongly advised.

### denss-all
This should run on Mac, Linux and Windows systems (please email me with bugs), and requires no additional programs or modules to be installed (just the already required NumPy and SciPy modules). The built-in method is fully parallelized for taking advantage of multicore machines.

`denss.all.py` is the primary script for running the full pipeline of DENSS, including running multiple runs of DENSS (default = 20), aligning, selecting enantiomers, averaging, and estimating resolution. To run the defaults, which should be suitable for most applications, simply type:
```
$ denss-all -f 6lyz.out
```
If you would like to use multiple cores for parallel processing, simply add the -j option:
```
$ denss-all -f 6lyz.out -j 4
```
for example to run on 4 cores. All options available to denss can also be passed to `denss-all`. Some additional options exist as well. Type `denss-all -h` to view all of the options available.

Several helper scripts are also supplied for performing various tasks:
- `denss-align` - aligns electron density maps to a reference (MRC or PDB file)
- `denss-align2xyz` - aligns electron density maps to the XYZ axes
- `denss-align-and-average` - aligns and averages a set of electron density maps
- `denss-average` - averages a set of pre-aligned electron density maps
- `denss-calcfsc` - calculates the Fourier Shell Correlation curve between two
pre-aligned electron density maps, and estimates resolution.
- `denss-pdb2mrc` - calculates an electron density map from a PDB file and a corresponding scattering profile (including solvent terms).
- `denss-get-info` - prints basic information about an MRC file, such as box sizes, voxels, etc.
- `denss-mrc2sas` - calculates a solution scattering profile from an electron density map.
- `denss-mrcops` - performs basic operations on MRC file, such as resampling
an electron density map to have a new size or shape.
- `denss-refine` - runs `denss` but with the added requirement of an input density map with the `--rho` option. Refines the given density map, e.g. an averaged density map from `denss-all`, to fit the data.

### Results
`denss-all` will create a folder with the name given to the --output option (which is input file basename by default). If a folder of that name already exists, DENSS will create a new folder adding a number at the end and incrementing by one. The folder will contain all the output maps and files of each reconstruction, the aligned maps, the average map, and the FSC curve used to estimate resolution. The final resulting averaged map will have suffix avg.mrc.

## Miscellaneous
The combination of total real space box size (D x oversampling) divided by voxel size determines N. The number of grid points scales as N^3, so large N typically requires long compute times and lots of memory (tests show N>50 may start to slow things down noticeably). Preliminary tests have shown oversampling as low as 2 is often sufficient for accurate reconstruction. However, lesser oversampling also results in low sampling of scattering profile, so direct comparison with experimental data becomes more difficult. Note that D given by the user is only used to determine the size of the box and does not limit the the electron density of the object by default. If one wishes to impose D as a limit, enable --limit_dmax_on (off by default).

While the NumPy implementation of FFT is most efficient when N is a power of two, considerable performance gains can still be gained when N is not a power of two, and there is no requirement in DENSS for N to equal a power of two.

The electron density map is initially set to be random based on the random seed selected by the program. One can therefore exactly reproduce the results of a previous calculation by giving the random seed to the program with the `--seed` option and the same input parameters. The parameters of previous runs can all be found in the log file.

The `denss-rho2dat` file can be used for calculating scattering profiles from MRC formatted electron density maps. Currently, the input maps must be cubic (i.e. same length and shape on all sides). Type `denss.rho2dat.py -h` for more options.




