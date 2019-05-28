# DENSS: DENsity from Solution Scattering

Author: Thomas Grant | email: tgrant@hwi.buffalo.edu
#### [Nature Methods paper describing DENSS](https://www.nature.com/articles/nmeth.4581)
If you use DENSS in your work, please cite:

Grant, Thomas D. (2018). Ab initio electron density determination directly from
solution scattering data. Nature Methods. http://dx.doi.org/10.1038/nmeth.4581.

#### [DENSS.org (tdgrant.com/denss)](https://www.tdgrant.com/denss)
DENSS.org is the official home of DENSS. Packed with detailed instructions
for installing and running DENSS on your own computer. It also contains
useful tips and best practices.

#### DENSSWeb server: [denss.ccr.buffalo.edu](https://denss.ccr.buffalo.edu)
Try out DENSS without installing the code using the DENSSWeb server to
perform simple online calculations suitable for most cases. (N is limited
to 32 samples for efficiency to allow many users to try it out). For more
complex particle shapes, install DENSS and EMAN2 (see below). Thanks to
Andrew Bruno and the CCR for DENSSWeb!

#### New MEMBRANE mode
DENSS now has a new mode for membrane proteins. Membrane proteins are often
solubilized in detergents or lipid nanodiscs. The hydrophobic regions of these
molecules often have lesser scattering density than the bulk solvent, resulting in a
negative contrast relative to the solvent. The default setting for DENSS enforces a
positivity restraint that will not allow any density to be negative. While this is
appropriate for most standard biomolecules such as proteins and nucleic acids,
is it not appropriate for molecules containing regions of negative contrast. To
accommodate this scenario, there is a new MEMBRANE mode in addition to the
previously available FAST and SLOW modes. This mode disables the positivity
restraint and starts shrink-wrap immediately.

#### New symmetry averaging feature
A new feature in denss v1.4.6 allows for the use of symmetry if known. The
options for imposing symmetry are --ncs, --ncs_axis, and --ncs_steps. Currently
only symmetry along a single axis is supported, though multiple axes will be
supported in the future. Symmetry is imposed by first aligning the principal
axes of inertia with the XYZ axes (largest to smallest). Then symmetry averaging
is performed along the selected axis at the given step(s). More frequent steps
makes for stronger restraint, but consequently more bias. Can select a different
axis (in case the largest principal axis is not the symmetry axis).
Note that the averaging procedure is still symmetry agnostic.
Also, you may need to manually filter sets of maps in case the wrong
symmetry axis was chosen in some cases, then perform averaging separately
with the denss.align_and_average.py script.

#### New script for performing simple operations on MRC files
A new script, `denss.mrcops.py`, is included in v1.4.5 that includes handy
tools for resampling or reshaping an MRC formatted electron density map.

#### New refinement script
A new script called `denss.refine.py` is available for refining an averaged
electron density map. Final averaged maps from denss.all.py or superdenss are
unlikely to have scattering profiles matching the data (since they are an average
from many different maps). To potentially improve the results, one can input
the averaged map to denss.refine.py using the --rho_start option. `denss.refine.py`
works similarly to denss.py, and will take the same .out or .dat file used
in the original runs of denss.py. This script is still in testing, so use at your
own risk and let me know if you run into bugs or issues.

#### New Averaging Procedure (beta)
A new procedure for aligning and averaging electron density maps with DENSS is
now available with v1.4.3. The new procedure is written entirely in Python from
the ground up and no longer requires EMAN2 to be installed. The new procedure
can be accessed with a series of new scripts, named as denss.xxx.py where xxx
is the name of the script. denss.all.py acts as the old superdenss, running
twenty reconstructions, aligning and averaging all maps, including enantiomer
generation and selection. The new scripts are still in testing, so use at your
own risk. Old scripts using the EMAN2 averaging procedure are still available.
Manuals for the new scripts coming soon. Thanks to intern Nhan Nguyen for helping
to write the new code.

#### New interactive GUI for fitting data
A new script (denss.fit_data.py) is provided with DENSS v1.3.0 for calculating smooth
fits to experimental data using a convenient interactive GUI.

#### New script for calculating profiles from MRC files
A new script (denss.rho2dat.py) is provided with DENSS v1.3.0 for calculating
scattering profiles from MRC files.


### About DENSS
DENSS is an algorithm used for calculating ab initio electron density maps
directly from solution scattering data. DENSS implements a novel iterative
structure factor retrieval algorithm to cycle between real space density
and reciprocal space structure factors, applying appropriate restraints in
each domain to obtain a set of structure factors whose intensities are
consistent with experimental data and whose electron density is consistent
with expected real space properties of particles.

DENSS utilizes the NumPy Fast Fourier Transform for moving between real and
reciprocal space domains. Each domain is represented by a grid of points
(Cartesian), N x N x N. N is determined by the size of the system and the
desired resolution. The real space size of the box is determined by the
maximum dimension of the particle, D, and the desired sampling ratio. Larger
sampling ratio results in a larger real space box and therefore a higher
sampling in reciprocal space (i.e. distance between data points in q).
Smaller voxel size in real space corresponds to higher spatial resolution
and therefore to larger q values in reciprocal space.

The core functions are stored in the `saxstats.py` module. The actual script
to run DENSS is `denss.py`.

## Installation
DENSS can be installed by typing at the command prompt in the directory
where you downloaded DENSS:
```
python setup.py install
```

## Requirements
DENSS requires that Python 2.7, NumPy (minimum v1.10.0) and SciPy are installed. These packages are
are often installed by default, or are available for your operating system
using package managers such as PIP or [Anaconda](https://www.continuum.io/downloads).
The current code was built using the Anaconda package management system
on Mac OS X 10.11. If using Anaconda, install the Python 2.7 version.
Alternatively you can have two separate python environments under Anaconda
by installing the latest Python 3 version of Anaconda, and create a separate
Python 2.7 environment by typing at the command prompt:
```
conda create -n python2 python=2.7 anaconda
```
Afterwards you can enable the Python 2.7 environment by typing the following
during any terminal session:
```
source activate python2
```
Once the Python 2.7 environment is enabled, you can run denss.py in that terminal
shell.

## Input files
DENSS uses smooth fits to experimental scattering profiles (rather than the noisy
experimental data). Two file formats are currently acceptable: .dat files or .out files.
Files with .dat extensions are expected to be the smoothed (i.e. fitted) curve.

A script called `denss.fit_data.py` is provided which can be used to fit experimental data
with a smooth curve based on an extended version of Peter Moore's approach (Moore 1979)
using a trigonometric series. The denss.fit_data.py script includes a simple interactive
GUI for selecting Dmax and the smoothing factor alpha and displays the experimental
data, the smooth fit to the data, and the real space pair distribution function.
denss.fit_data.py will save a .dat file containing the smooth fit to the data which can
then be used as input to denss.py (see below). Additionally, useful parameters
calculated from the fit, such as the radius of gyration and Porod volume, are displayed.
The manuscript describing the mathematical derivation and the algorithm of this
new approach is currently in preparation.

`denss.fit_data.py` can be run simply from the command line as:
```
denss.fit_data.py -f <experimental_data.dat>
```
where <experimental_data.dat> is the noisy scattering profile, given as a three-column
ASCII text file with columns q, I, error. An interactive GUI will appear showing
the experimental scattering profile on the left along with the fit to the data,
and the associated pair distribution function (P(r)) on the right. Two interactive
sliders on the bottom left can be adjusted for Dmax (the maximum particle dimension)
and the alpha smoothing factor. See `denss.fit_data.py -h` for more options.

DENSS also accepts [GNOM](https://www.embl-hamburg.de/biosaxs/gnom.html)
.out files created by [ATSAS](https://www.embl-hamburg.de/biosaxs/software.html)
(credit for .out parsing - Jesse Hopkins).

DENSS uses the smoothed curve fit to the experimental data and extrapolated to
q = 0, i.e. I(0). You can also use any other smooth and extrapolated curve,
such as the output from FoXS or CRYSOL, as long as it is supplied in a three
column ASCII text file with columns q, I, error where q is given as 4 pi sin(theta)/lambda
in angstroms, I is scattering intensity, and error is the error on the intensity.

`lysozyme.out` is a GNOM .out file from real lysozyme data. `6lyz.dat` is a
simulated scattering profile from lysozyme PDB 6LYZ using FoXS. `6lyz.out` is
a GNOM .out file created from the `6lyz.dat` data file. Any of these files can
be used as input to DENSS for testing.

## Usage
DENSS can be run with basic defaults for an outfile:
```
denss.py -f <saxs.out>
```
In this case, DENSS uses the maximum dimension from the .out file. You can
override this maximum dimension by specifying the -d parameter. If you are
using a .dat file you must specify the -d parameter, as it is not contained
in the file:
```
denss.py -f <saxs.dat> -d <estimated maximum dimension>
```

For example, using the supplied lysozyme.out data, DENSS can be run with:
```
denss.py -f lysozyme.out
```
While using the 6lyz.dat data it can be run as:
```
denss.py -f 6lyz.dat -d 50.0
```
On Windows, depending on your setup you may need to type:
```
python C:\path\to\denss.py -f 6lyz.out
```

Options you may want to set are:
```
  -h, --help            show this help message and exit
  -f FILE, --file FILE  SAXS data file for input (either .dat or .out)
  -d DMAX, --dmax DMAX  Estimated maximum dimension (Default=100)
  -v VOXEL, --voxel VOXEL
                        Set desired voxel size, setting resolution of map
  -os OVERSAMPLING, --oversampling OVERSAMPLING
                        Sampling ratio (Default=3)
  -n NSAMPLES, --nsamples NSAMPLES
                        Number of samples, i.e. grid points, along a single
                        dimension. (Sets voxel size, overridden by --voxel.
                        Best optimization with n=power of 2. Default=64)
  --ne NE               Number of electrons in object (Default=10,000)
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
                        molecules. (default SLOW)
```
By default DENSS runs in SLOW mode, which is generally suitable for the vast
majority of particles, including those with complex shapes. You can override all
the default parameters set by the SLOW mode by explicitly setting any of the options.

Additional advanced options are can be seen by typing `denss.py -h`.

## Results
As the program runs, the current status will be printed to the screen like so:
```
Step  Chi2      Rg      Support Volume
----- --------- ------- --------------
 2259  1.31e+00  14.34        42135
```
Where `Step` represents the number of iterations so far, `Chi2` is the fit of
the calculated scattering of the map to the experimental data, `Rg` is the
radius of gyration calculated directly from the electron density map, and
`Support Volume` is the volume of the support region.

Electron density maps are written in CCP4/MRC format (credit Andrew Bruno) and
optionally as Xplor ASCII text format (with the --write_xplor option enabled).
These files can be opened directly in some visualization programs such as
[Chimera](http://www.rbvi.ucsf.edu/chimera/) and [PyMOL](https://www.pymol.org).
In particular, the PyMOL "volume" function is well suited for displaying these
maps with density information displayed as varying color and opacity.
Maps can be converted to other formats using tools such as the Situs map2map tool.

Output files include:
```
output.mrc                 electron density map (MRC format)
output_support.mrc         final support volume formatted as unitary electron density map
output_stats_by_step.dat   statistics as a function of step number.
                           three columns: chi^2, Rg, support volume
output_map.fit             The fit of the calculated scattering profile to the
                           experimental data. Experimental data has been interpolated to
                           the q values used for scaling intensities and I(0) has been
                           scaled to the square of the number of electrons in the particle.
                           Columns are: q(data), I(data), error(data), q(calc), I(calc)
output_*.png               If plotting is enabled, these are plots of the results.
output.log                 A log file containing parameters for the calculation
                           and summary of the results.
```

## Alignment, Averaging, and Resolution Estimation
The solutions are non-unique, meaning many different electron density maps will yield
the same scattering profile. Different random starting points will return different
results. Therefore, running the algorithm many times (>20) is strongly advised.

There now exists two options for performing alignment and averaging. The older option
uses EMAN2, the newer option is now built into the latest versions of DENSS (v1.4.1).

### New Built-in Method
The new built-in option should run on Mac, Linux and Windows systems (please email me with bugs),
and requires no additional programs or modules to be installed (just the already
required NumPy and SciPy modules). The built-in method is fully parallelized
for taking advantage of multicore machines.

`denss.all.py` is the primary script for running the full pipeline of DENSS,
including running multiple runs of DENSS (default = 20), aligning, selecting
enantiomers, averaging, and estimating resolution. To run the defaults, which
should be suitable for most applications, simply type:
```
$ denss.all.py -f 6lyz.out
```
If you would like to use multiple cores for parallel processing, simply add the -j option:
```
$ denss.all.py -f 6lyz.out -j 4
```
for example to run on 4 cores. All options available to denss.py can also be passed
to `denss.all.py`. Some additional options exist as well. Type `denss.all.py -h` to
view all of the options available.

Several helper scripts are also supplied for performing various tasks:
`denss.align.py` - aligns electron density maps to a reference (MRC or PDB file)
`denss.align2xyz.py` - aligns electron density maps to the XYZ axes
`denss.align_by_principal_axes.py` - aligns electron density maps to a reference
(MRC or PDB), but performs no minimization.
`denss.align_and_average.py` - aligns and averages a set of electron density maps
`denss.average.py` - averages a set of pre-aligned electron density maps
`denss.calcfsc.py` - calculates the Fourier Shell Correlation curve between two
pre-aligned electron density maps, and estimates resolution.
`denss.pdb2mrc.py` - calculates an electron density map from a PDB file.
`denss.get_info.py` - prints basic information about an MRC file, to be used
with denss.pdb2mrc.py, for example, to set box sizes, voxels, etc.
`denss.rho2dat.py` - calculates a solution scattering profile from an electron density map.
`denss.mrcops.py` - performs basic operations on MRC file, such as resampling
an electron density map to have a new size or shape.

### EMAN2 Method:
The older option for alignment and averaging requires installation of EMAN2.
Installing is pretty straightforward on Unix systems. Some users have noted difficulty
with installation of EMAN2 on Windows systems.

To install EMAN2, download the appropriate EMAN2 binary from [this page](https://cryoem.bcm.edu/cryoem/downloads/view_eman2_versions)
Then, change to the folder where you would like to install EMAN2 (such as your home folder)
and move the eman2 script you downloaded to that directory. Then execute the script.
For example:
```
$ cd $HOME
$ mv $HOME/Downloads/eman2.21.MacOS.sh .
$ bash eman2.21.MacOS.sh
```
But, obviously, use the correct filenames and paths for your platform.
If you have issues getting EMAN2 installed, see their [installation page](http://blake.bcm.edu/emanwiki/EMAN2/Install).

#### superdenss
A bash script is provided called `superdenss` that runs the EMAN2 pipeline automatically
in parallel assuming EMAN2 and gnu parallel are all installed. To run superdenss
with the default parameters for denss.py, type:
```
superdenss -f 6lyz.out
```
superdenss also takes its own options, as well as all of the options accepted by
denss.py. The following options are available for superdenss (accessible with the -h option):
```
 ------------------------------------------------------------------------------
 superdenss is a simple wrapper for denss that automates the process of
 generating multiple density reconstructions and averaging them with EMAN2.

 -f: filename of .out GNOM file or .dat solution scattering data
 -o: the output prefix to name the output directory and all the files.
 -i: input options for denss exactly as they would be given to denss, including
     dashed options. Enclose everything in quotes. Dont include --file or --output.
 -n: the number of reconstructions to run (default 20)
 -j: the number of cores to use for parallel processing (defaults to ncores - 1)
 -e: generate and select enantiomers (significantly increases runtime, default=no)
 -----------------------------------------------------------------------------
```
For example, to run superdenss while checking for the best enantiomers, type:
```
superdenss -f 6lyz.dat -e
```
If you would like to edit the parameters of denss.py, pass them into the -i option
of superdenss. This is admittedly a little awkward to use to pass the arguments correctly
as you must pass the denss.py options enclosed in double quotes after the -i option:
To run the above example using a .dat file (note the quotes in the command line):
```
superdenss -f 6lyz.dat -e -i " -d 50.0 " -o lysozyme
```
This will create a folder named "lysozyme" with all the output files for each of
the 20 individual runs of DENSS and a folder named spt_avg_01 which will contain the
final averaged density. You can add other denss.py options into the -i option in
in between the quotes.

## Miscellaneous
The combination of total real space box size (D x oversampling) divided by
voxel size determines N. The number of grid points scales as N^3, so large
N typically requires long compute times and lots of memory (tests show N>50 may
start to slow things down noticeably). Preliminary tests have shown oversampling
as low as 2 is often sufficient for accurate reconstruction. However, lesser
oversampling also results in low sampling of scattering profile, so direct
comparison with experimental data becomes more difficult. Note that D given by
the user is only used to determine the size of the box and does not limit the
the electron density of the object by default. If one wishes to impose D as a
limit, enable --limit_dmax_on (off by default).

While the NumPy implementation of FFT is most efficient when N is a power of two,
considerable performance gains can still be gained when N is not a power of two,
and there is no requirement in DENSS for N to equal a power of two.

The electron density map is initially set to be random based on the random seed
selected by the program. One can therefore exactly reproduce the results of a previous
calculation by giving the random seed to the program with the `--seed` option and
the same input parameters. The parameters of previous runs can all be found in the log file.

The `denss.rho2dat.py` file can be used for calculating scattering profiles from MRC formatted
electron density maps. Currently the input maps must be cubic (i.e. same length
and shape on all sides). Type `denss.rho2dat.py -h` for more options.




