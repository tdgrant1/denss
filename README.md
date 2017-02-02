# DENSS: DENsity from Solution Scattering

Author: Thomas Grant | email: tgrant@hwi.buffalo.edu

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
DENSS can be installed by typing at the command prompt:
```
python setup.py install
```
However, DENSS is just pure Python and can also just be run directly as
a script, provided the `saxstats.py` file is in the same directory as
`denss.py` that you're running from.

## Requirements
DENSS requires that Python 2.7, NumPy and SciPy are installed. These packages are 
are often installed by default, or are available for your operating system
using package managers such as PIP or Anaconda. 
(The current code was built using the Anaconda package management system 
on Mac OS X 10.11.)

## Input files
The only file required for using DENSS is the one dimensional solution
scattering profile, given as a three column ASCII text file with columns
q, I, error where q is given as 4 pi sin(theta)/lambda in angstroms, I is
scattering intensity, and error is the error on the intensity. To take
advantage of the oversampling typically granted by small angle scattering
data, one should first use a fitting program such as GNOM from ATSAS to fit
the experimental data with a smooth curve, and then extract the fitted data
(including the forward scattering intensity where q = 0, i.e. I(0)) from
the output and supply the smooth curve to DENSS. If using GNOM, a bash
script is provided (```gnom2dat```) to extract the smooth fit of the 
intensity (and add some missing error bars).

## Usage
DENSS can be run with basic defaults:
```
python denss.py -f <saxs.dat> -d <estimated maximum dimension> 
```
Additional options you may want to set are:
```
  -v VOXEL, --voxel VOXEL       Set desired voxel size (default 5 angstroms)
  --oversampling OVERSAMPLING   Sampling ratio (default 5)
  -n NE, --ne NE                Number of electrons in object (used only for scaling)
  -s STEPS, --steps STEPS       Maximum number of steps (iterations)
  -o OUTPUT, --output OUTPUT    Output map filename (default filename)
```
Additional advanced options are:
```
  --seed SEED           Random seed to initialize the map
  --rhostart RHOSTART   Filename of starting electron density in xplor format
  --rhobounds RHOBOUNDS RHOBOUNDS
                        Lower and upper bounds of number of electrons per grid
                        point in object
  --supportpdb SUPPORTPDB
                        PDB file used to create support
  --usedmax-on          Limit electron density to sphere of radius Dmax/2.
  --usedmax-off         Do not limit electron density to sphere of radius
                        Dmax/2. (Default)
  --recenter-on         Recenter electron density when updating support.
                        (default)
  --recenter-off        Do not recenter electron density when updating
                        support.
  --positivity-on       Enforce positivity restraint inside support. (default)
  --positivity-off      Do not enforce positivity restraint inside support.
  --extrapolate-on      Extrapolate data by Porod law to high resolution limit
                        of voxels. (default)
  --extrapolate-off     Do not extrapolate data by Porod law to high
                        resolution limit of voxels.
  --shrinkwrap-on       Turn shrinkwrap on (default)
  --shrinkwrap-off      Turn shrinkwrap off
  --shrinkwrap_sigma_start SHRINKWRAP_SIGMA_START
                        Starting sigma for Gaussian blurring, in voxels (default 3)
  --shrinkwrap_sigma_end SHRINKWRAP_SIGMA_END
                        Ending sigma for Gaussian blurring, in voxels (default 1.5)
  --shrinkwrap_sigma_decay SHRINKWRAP_SIGMA_DECAY
                        Rate of decay of sigma, fraction (default 0.99)
  --shrinkwrap_threshold_fraction SHRINKWRAP_THRESHOLD_FRACTION
                        Minimum threshold defining support, in fraction of
                        maximum density (default 0.20)
  --shrinkwrap_iter SHRINKWRAP_ITER
                        Number of iterations between updating support with
                        shrinkwrap (default 20)
  --shrinkwrap_minstep SHRINKWRAP_MINSTEP
                        First step to begin shrinkwrap (default 0)
  --chi_end_fraction CHI_END_FRACTION
                        Convergence criterion. Minimum threshold of chi2 std
                        dev, as a fraction of the median chi2 of last 100
                        steps. (default 0.001)
  --plot-on             Create simple plots of results (requires Matplotlib).
  --plot-off            Do not create simple plots of results. (Default)
  ```

## Results
As the program runs, the current status will be printed to the screen like so:
```
Step  Chi2      Rg      Support Volume
----- --------- ------- --------------
 2259  1.31e+00  20.34        62196   
```
Where `Step` represents the number of iterations so far, `Chi2` is the fit of
the calculated scattering of the map to the experimental data, `Rg` is the
radius of gyration calculated directly from the electron density map, and
`Support Volume` is the volume of the support region.

Electron density maps are written in Xplor ASCII text format. These files can
be opened directly in some visualization programs such as Chimera and PyMOL. 
Maps can be converted to other formats using tools such as the Situs map2map tool.

Output files include:
```
output.xplor               electron density map
output_support.xplor       final support volume formatted as unitary electron density map
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

## Miscellaneous
The combination of total real space box size (D x oversampling) divided by 
voxel size determines N. The number of grid points scales as N^3, so large 
N typically requires long compute times and lots of memory (tests show N>50 may
start to slow things down noticeably). Preliminary tests have shown oversampling
as low as 2 is often sufficient for accurate reconstruction. However, lesser
oversampling also results in low sampling of scattering profile, so direct
comparison with experimental data becomes more difficult.

The electron density map is initially set to be random based on the random seed
selected by the program. One can therefore exactly reproduce the results of a previous
calculation by giving the random seed to the program with the `--seed` option and
the same input parameters. The parameters of previous runs can all be found in the log file.
The solutions are non-unique, meaning many different electron density maps will yield
the same scattering profile. Therefore, running the algorithm many times (>=10) is
strongly advised. Subsequent alignment and averaging can be performed in Chimera
using the Fit in Map tool and `vop add`. Additionally, one can start with a
given electron density map, perhaps for refinement of an averaged map, using
the `--rhostart` option. 



