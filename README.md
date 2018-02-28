# DENSS: DENsity from Solution Scattering

Author: Thomas Grant | email: tgrant@hwi.buffalo.edu
#### [Nature Methods paper describing DENSS](https://www.nature.com/articles/nmeth.4581)
If you use DENSS in your work, please cite:

Grant, Thomas D. (2018). Ab initio electron density determination directly from 
solution scattering data. Nature Methods. http://dx.doi.org/10.1038/nmeth.4581.

#### [DENSS.org is now live! (tdgrant.com/denss)](https://www.tdgrant.com/denss)
DENSS.org is the official home of DENSS. Packed with detailed instructions
for installing and running DENSS on your own computer. It also contains
useful tips and best practices.

#### New DENSSWeb server! [denss.ccr.buffalo.edu](https://denss.ccr.buffalo.edu)
Try out DENSS without installing the code using the DENSSWeb server to
perform simple online calculations suitable for most cases. (N is limited
to 32 samples for efficiency to allow many users to try it out). For more
complex particle shapes, install DENSS and EMAN2 (see below). Thanks to 
Andrew Bruno and the CCR for DENSSWeb! (If the link doesn't work and you 
are using Safari, try it out in another browser)

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
However, DENSS is just pure Python and can also just be run directly as
a script, provided the `saxstats.py` file is in the same directory as
`denss.py` that you're running from.

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
DENSS primarily accepts [GNOM](https://www.embl-hamburg.de/biosaxs/gnom.html)
.out files created by [ATSAS](https://www.embl-hamburg.de/biosaxs/software.html)
(credit for .out parsing - Jesse Hopkins).
DENSS uses the smoothed curve fit to the experimental data and extrapolated to
q = 0, i.e. I(0). You can also use any other smooth and  extrapolated curve,
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
You can also use DENSS as a script:
```
python denss.py -f <saxs.out>
```
For example, using the supplied lysozyme.out data, DENSS can be run with:
```
denss.py -f lysozyme.out
```
While using the 6lyz.dat data it can be run as:
```
denss.py -f 6lyz.dat -d 50.0
```
Options you may want to set are:
```
  -h, --help            show this help message and exit
  -f FILE, --file FILE  SAXS data file for input (either .dat or .out)
  -d DMAX, --dmax DMAX  Estimated maximum dimension
  -v VOXEL, --voxel VOXEL
                        Set desired voxel size, setting resolution of map
  -os OVERSAMPLING, --oversampling OVERSAMPLING
                        Sampling ratio
  -n NSAMPLES, --nsamples NSAMPLES
                        Number of samples, i.e. grid points, along a single
                        dimension. (Sets voxel size, overridden by --voxel.
                        Best optimization with n=power of 2)
  --ne NE               Number of electrons in object
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

Additional advanced options are:
```
  --seed SEED           Random seed to initialize the map
  --limit_dmax_on       Limit electron density to sphere of radius 0.6*Dmax
                        from center of object.
  --limit_dmax_off      Do not limit electron density to sphere of radius
                        0.6*Dmax from center of object. (default)
  --dmax_start_step DMAX_START_STEP
                        Starting step for limiting density to sphere of Dmax
                        (default=500)
  --recenter_on         Recenter electron density when updating support.
                        (default)
  --recenter_off        Do not recenter electron density when updating
                        support.
  --recenter_steps RECENTER_STEPS [RECENTER_STEPS ...]
                        List of steps to recenter electron density.
  -p_on, --positivity_on
                        Enforce positivity restraint inside support. (default)
  -p_off, --positivity_off
                        Do not enforce positivity restraint inside support.
  -e_on, --extrapolate_on
                        Extrapolate data by Porod law to high resolution limit
                        of voxels. (default)
  -e_off, --extrapolate_off
                        Do not extrapolate data by Porod law to high
                        resolution limit of voxels.
  -sw_on, --shrinkwrap_on
                        Turn shrinkwrap on (default)
  -sw_off, --shrinkwrap_off
                        Turn shrinkwrap off
  -sw_start SHRINKWRAP_SIGMA_START, --shrinkwrap_sigma_start SHRINKWRAP_SIGMA_START
                        Starting sigma for Gaussian blurring, in voxels
  -sw_end SHRINKWRAP_SIGMA_END, --shrinkwrap_sigma_end SHRINKWRAP_SIGMA_END
                        Ending sigma for Gaussian blurring, in voxels
  -sw_decay SHRINKWRAP_SIGMA_DECAY, --shrinkwrap_sigma_decay SHRINKWRAP_SIGMA_DECAY
                        Rate of decay of sigma, fraction
  -sw_threshold SHRINKWRAP_THRESHOLD_FRACTION, --shrinkwrap_threshold_fraction SHRINKWRAP_THRESHOLD_FRACTION
                        Minimum threshold defining support, in fraction of
                        maximum density
  -sw_iter SHRINKWRAP_ITER, --shrinkwrap_iter SHRINKWRAP_ITER
                        Number of iterations between updating support with
                        shrinkwrap
  -sw_minstep SHRINKWRAP_MINSTEP, --shrinkwrap_minstep SHRINKWRAP_MINSTEP
                        First step to begin shrinkwrap
  -ec_on, --enforce_connectivity_on
                        Enforce connectivity of support, i.e. remove extra
                        blobs (default)
  -ec_off, --enforce_connectivity_off
                        Do not enforce connectivity of support
  -ec_steps ENFORCE_CONNECTIVITY_STEPS [ENFORCE_CONNECTIVITY_STEPS ...], 
  --enforce_connectivity_steps ENFORCE_CONNECTIVITY_STEPS [ENFORCE_CONNECTIVITY_STEPS ...]
                        List of steps to enforce connectivity
  --chi_end_fraction CHI_END_FRACTION
                        Convergence criterion. Minimum threshold of chi2 std
                        dev, as a fraction of the median chi2 of last 100
                        steps.
  --write_xplor         Write out XPLOR map format (default only write MRC
                        format).
  --write_freq WRITE_FREQ
                        How often to write out current density map (in steps,
                        default 100).
  --cutout_on           When writing final map, cut out the particle to make
                        smaller files.
  --cutout_off          When writing final map, do not cut out the particle to
                        make smaller files (default).
  --plot_on             Create simple plots of results (requires Matplotlib,
                        default if module exists).
  --plot_off            Do not create simple plots of results. (Default if
                        Matplotlib does not exist)
```

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

For fully unsupervised (i.e. unbiased) alignment and averaging, and subsequent
estimation of resolution, the [EMAN2](http://blake.bcm.edu/emanwiki/EMAN2) single particle tomography tool
e2spt_classaverage.py is well suited for this task. Though, this takes
some patience. To use this tool, you must first install EMAN2. Recent updates
to EMAN2 have made this process quite easy. To begin, download the appropriate
EMAN2 binary from [this page](http://ncmi.bcm.tmc.edu/ncmi/software/software_details?selected_software=counter_222) 
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

Here I will describe the explicit sequence of commands used for averaging with EMAN2.
However, there is now a much more convenient and useful script called superdenss
included in the DENSS download that will run this process for you automatically.
This script also has some added utilities, most importantly the ability to
generate and select the best enantiomers for each reconstruction to use in the averaging.
Using the superdenss script is advised. The specific commands used in superdenss are
as follows:

To calculate multiple reconstructions easily, you can use a for loop. E.g. in bash:
```
for i in {0..19}; do denss.py -f 6lyz.dat -d 50.0 -o lysozyme_${i} ; done
```
This will calculate 20 reconstructions with the output files numbered.

The following step is only necessary if you choose to work with XPLOR maps.
EMAN2 is supposed to work with .xplor files, but I haven't had success.
So first convert the .xplor files to .mrc files (MRC/CCP4 formats are similar).
This can be done by opening the maps in Chimera, and clicking Save Map As... in
the Volume Viewer. If you install Situs, this can be done on the command line in
one step using the map2map tool:
```
for i in lysozyme_*[0-9].xplor ; do map2map $i ${i%.*}.mrc <<< '2' ; done
```

Now we can combine the 20 maps into one 3D "stack" in EMAN2 format (.hdf).
```
e2buildstacks.py --stackname lysozyme_stack.hdf lysozyme_*[0-9].mrc
```
Now that we have our correctly formatted stack, we can run the alignment
and averaging. The e2spt_classaverage.py script does this all quite well
using default parameters. It will even output a gold standard Fourier
Shell Correlation curve, from which we can estimate resolution. Typing:
```
e2spt_classaverage.py --input lysozyme_stack.hdf
```
will run everything for you. If you have multiple cores on your computer,
you can speed things up with:
```
e2spt_classaverage.py --input lysozyme_stack.hdf --parallel=thread:n
```
where n is the number of cores to use. Additionally, one can add the --keep and
--keepsig flags to filter out outlier volumes based on standard deviations of the
correlations.  The output will be in a folder named something like spt_01.
The averaged density will be named something like final_avg.hdf.
Fortunately, Chimera can open .hdf files by default. Then, if you would like to
view it in PyMOL you can save it as an MRC formatted map in Chimera or convert
it to MRC using the Situs map2map tool or using the EMAN2 e2proc3d.py program.

Resolution can be estimated from the FSC curve also written to the spt_01
directory as fsc_0.txt. Plot this in your favorite plotting program to
view. Take the reciprocal of the x axis position where FSC falls below 0.5,
and that is your estimated resolution.

### superdenss
A bash script is provided called `superdenss` that runs this pipeline automatically
in parallel assuming EMAN2 and gnu parallel are all installed. To run superdenss
with the default parameters for denss.py, type:
```
superdenss -f 6lyz.dat
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




