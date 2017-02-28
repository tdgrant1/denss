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
The only file required for using DENSS is the one dimensional solution
scattering profile, given as a three column ASCII text file with columns
q, I, error where q is given as 4 pi sin(theta)/lambda in angstroms, I is
scattering intensity, and error is the error on the intensity. To take
advantage of the oversampling typically granted by small angle scattering
data, one should first use a fitting program such as GNOM from 
[ATSAS](https://www.embl-hamburg.de/biosaxs/software.html) to fit
the experimental data with a smooth curve, and then extract the fitted data
(including the forward scattering intensity where q = 0, i.e. I(0)) from
the output and supply the smooth curve to DENSS. If using GNOM, a bash
script is provided (```gnom2dat```) to extract the smooth fit of the 
intensity (and add some missing error bars). To do so type
```
gnom2dat saxs.out
```

`6lyz.dat` is a simulated scattering profile from lysozyme PDB 6LYZ using
FoXS. This file can be used as input to DENSS for testing.

## Usage
DENSS can be run with basic defaults:
```
denss.py -f <saxs.dat> -d <estimated maximum dimension> 
```
or as a script:
```
python denss.py -f <saxs.dat> -d <estimated maximum dimension> 
```
For example, using the supplied 6lyz.dat data, DENSS can be run with:
```
denss.py -f 6lyz.dat -d 50.0
```
Additional options you may want to set are:
```
  -v VOXEL, --voxel VOXEL       Set desired voxel size (default 5 angstroms)
  --oversampling OVERSAMPLING   Sampling ratio (default 3)
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
  --plot-on             Create simple plots of results 
                        (requires Matplotlib, default is module exists).
  --plot-off            Do not create simple plots of results 
                        (Default if Matplotlib does not exist).
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
be opened directly in some visualization programs such as 
[Chimera](http://www.rbvi.ucsf.edu/chimera/) and [PyMOL](https://www.pymol.org). 
In particular, the PyMOL "volume" function is well suited for displaying these
maps with density information displayed as varying color and opacity.
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

## Alignment, Averaging, and Resolution Estimation
The solutions are non-unique, meaning many different electron density maps will yield
the same scattering profile. Different random starting points will return different
results. Therefore, running the algorithm many times (>=10) is strongly advised. 
Subsequent alignment and averaging can be performed in Chimera
using the Fit in Map tool and `vop add`. 
Additionally, one can start with a given electron density map, perhaps for refinement
of an averaged map, using the `--rhostart` option. 

For fully unsupervised (i.e. unbiased) alignment and averaging, and subsequent
estimation of resolution, the EMAN2 single particle tomography tool 
e2spt_classaverage.py is well suited for this task. Though, this takes
some patience. To use this tool, you must first install EMAN2. 
To calculate multiple reconstructions easily, you can use a for loop. E.g. in bash:
```
for i in {0..19}; do denss.py -f 6lyz.dat -d 50.0 -o lysozyme_${i} ; done
```
This will calculate 20 reconstructions with the output files numbered.
EMAN2 is supposed to work with .xplor files, but I haven't had success. 
So first convert the .xplor files to .mrc files (MRC/CCP4 formats are similar).
This can be done by opening the maps in Chimera, and clicking Save Map As... in 
the Volume Viewer. If you install Situs, this can be done on the command line in 
one step using the map2map tool:
```
for i lysozyme_*[0-9].xplor ; do map2map $i ${i%.*}.mrc <<< '2' ; done
```
Now we can combine the 20 maps into one 3D "stack" in EMAN2 format (.hdf).
```
e2buildstacks.py --stackname lysozyme_stack.hdf lysozyme_*[0-9].mrc
```
EMAN2 likes even numbers of samples in maps, whereas denss.py uses odd numbers,
so first we have to alter the maps by one sample in EMAN2. Look in your .log file
output from denss.py, or look at the header of your .xplor file, to find the 
the "Grid size", i.e. the number of voxels in each dimension. This should be an 
odd number calculated by denss.py. Subtract one from this number. So if the Grid
size is 31 x 31 x 31, then we want n=30. Use EMAN2 on our new .hdf file to convert:
```
e2proc3d.py lysozyme_stack.hdf lysozyme_stack_resized.hdf --clip 30
```
Now that we have our correctly formatted stack, we can run the alignment
and averaging. The e2spt_classaverage.py script does this all quite well
using default parameters. It will even output a gold standard Fourier
Shell Correlation curve, from which we can estimate resolution. Typing:
```
e2spt_classaverage.py --input lysozyme_stack_resized.hdf 
```
will run everything for you. The output will be in a folder named something
like spt_01. The averaged density will be named something like final_avg.hdf.
Fortunately, Chimera can open .hdf files by default. Then, if you'd like to
view it in PyMOL you can save it as an MRC formatted map in Chimera.

Resolution can be estimated from the FSC curve also written to the spt_01
directory as fsc_0.txt. Plot this in your favorite plotting program to
view. A good cutoff for FSC is 0.5, or even 0.143, but 0.5 is safe. Take the
reciprocal of the x axis position where FSC rises above 0.5, and that is your
estimated resolution.

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
limit, enable --usedmax-on (off by default). 

While the NumPy implementation of FFT is most efficient when N is a power of two,
considerable performance gains can still be gained when N is not a power of two,
and there is no requirement in DENSS for N to equal a power of two. 

The electron density map is initially set to be random based on the random seed
selected by the program. One can therefore exactly reproduce the results of a previous
calculation by giving the random seed to the program with the `--seed` option and
the same input parameters. The parameters of previous runs can all be found in the log file.



