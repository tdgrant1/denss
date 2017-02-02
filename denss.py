#!/usr/bin/env python
#
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tgrant@hwi.buffalo.edu>
#    Copyright 2017 The Research Foundation for SUNY
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import saxstats as saxs
import numpy as np
import sys, argparse, os
import logging
import imp
try:
    imp.find_module('matplotlib')
    matplotlib_found = True
except ImportError:
    matplotlib_found = False

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="SAXS data file")
parser.add_argument("-d", "--dmax", default=100., type=float, help="Estimated maximum dimension")
parser.add_argument("-v", "--voxel", default=5., type=float, help="Set desired voxel size, setting resolution of map")
parser.add_argument("--oversampling", default=5., type=float, help="Sampling ratio")
parser.add_argument("-n", "--ne", default=10000, type=float, help="Number of electrons in object")
parser.add_argument("-s", "--steps", default=3000, type=int, help="Maximum number of steps (iterations)")
parser.add_argument("-o", "--output", help="Output map filename")
parser.add_argument("--seed", default=None, help="Random seed to initialize the map")
parser.add_argument("--rhostart", default=None, type=str, help="Filename of starting electron density in xplor format")
parser.add_argument("--rhobounds", default=None, nargs=2, type=float, help="Lower and upper bounds of number of electrons per grid point in object")
parser.add_argument("--supportpdb", default=None, type=str, help="PDB file used to create support")
parser.add_argument("--usedmax-on", dest="usedmax", action="store_true", help="Limit electron density to sphere of radius Dmax/2.")
parser.add_argument("--usedmax-off", dest="usedmax", action="store_false", help="Do not limit electron density to sphere of radius Dmax/2. (Default)")
parser.add_argument("--recenter-on", dest="recenter", action="store_true", help="Recenter electron density when updating support. (default)")
parser.add_argument("--recenter-off", dest="recenter", action="store_false", help="Do not recenter electron density when updating support.")
parser.add_argument("--positivity-on", dest="positivity", action="store_true", help="Enforce positivity restraint inside support. (default)")
parser.add_argument("--positivity-off", dest="positivity", action="store_false", help="Do not enforce positivity restraint inside support.")
parser.add_argument("--extrapolate-on", dest="extrapolate", action="store_true", help="Extrapolate data by Porod law to high resolution limit of voxels. (default)")
parser.add_argument("--extrapolate-off", dest="extrapolate", action="store_false", help="Do not extrapolate data by Porod law to high resolution limit of voxels.")
parser.add_argument("--shrinkwrap-on", dest="shrinkwrap", action="store_true", help="Turn shrinkwrap on (default)")
parser.add_argument("--shrinkwrap-off", dest="shrinkwrap", action="store_false", help="Turn shrinkwrap off")
parser.add_argument("--shrinkwrap_sigma_start", default=3, type=float, help="Starting sigma for Gaussian blurring, in voxels")
parser.add_argument("--shrinkwrap_sigma_end", default=1.5, type=float, help="Ending sigma for Gaussian blurring, in voxels")
parser.add_argument("--shrinkwrap_sigma_decay", default=0.99, type=float, help="Rate of decay of sigma, fraction")
parser.add_argument("--shrinkwrap_threshold_fraction", default=0.20, type=float, help="Minimum threshold defining support, in fraction of maximum density")
parser.add_argument("--shrinkwrap_iter", default=20, type=int, help="Number of iterations between updating support with shrinkwrap")
parser.add_argument("--shrinkwrap_minstep", default=0, type=int, help="First step to begin shrinkwrap")
parser.add_argument("--chi_end_fraction", default=0.001, type=float, help="Convergence criterion. Minimum threshold of chi2 std dev, as a fraction of the median chi2 of last 100 steps.")
parser.add_argument("--plot-on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
parser.add_argument("--plot-off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib doesnt exist)")
parser.set_defaults(usedmax=False)
parser.set_defaults(shrinkwrap=True)
parser.set_defaults(recenter=True)
parser.set_defaults(positivity=True)
parser.set_defaults(extrapolate=True)
if matplotlib_found:
    parser.set_defaults(plot=True)
else:
    parser.set_defaults(plot=False)
args = parser.parse_args()

logging.basicConfig(filename=args.output+'.log',level=logging.INFO,filemode='w',format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info('BEGIN')
logging.info('Data filename: %s', args.file)

if args.output is None:
    basename, ext = os.path.splitext(args.file)
    output = basename
else:
    output = args.output
logging.info('Output prefix: %s', output)

if args.supportpdb is not None:
    supportpdb = saxs.PDB(args.supportpdb)
else:
    supportpdb = None

Iq = np.loadtxt(args.file)
q = Iq[:,0]
I = Iq[:,1]
sigq = Iq[:,2]

logging.info('Maximum number of steps: %i', args.steps)
logging.info('q range of input data: %3.3f < q < %3.3f', q.min(), q.max())
logging.info('Maximum dimension: %3.3f', args.dmax)
logging.info('Sampling ratio: %3.3f', args.oversampling)
logging.info('Requested real space voxel size: %3.3f', args.voxel)
logging.info('Number of electrons: %3.3f', args.ne)
logging.info('Use Dmax: %s', args.usedmax)
logging.info('Recenter: %s', args.recenter)
logging.info('Positivity: %s', args.positivity)
logging.info('Extrapolate high q: %s', args.extrapolate)
logging.info('Shrinkwrap: %s', args.shrinkwrap)
logging.info('Shrinkwrap sigma start: %s', args.shrinkwrap_sigma_start)
logging.info('Shrinkwrap sigma end: %s', args.shrinkwrap_sigma_end)
logging.info('Shrinkwrap sigma decay: %s', args.shrinkwrap_sigma_decay)
logging.info('Shrinkwrap threshold fraction: %s', args.shrinkwrap_threshold_fraction)
logging.info('Shrinkwrap iterations: %s', args.shrinkwrap_iter)
logging.info('Shrinkwrap starting step: %s', args.shrinkwrap_minstep)
logging.info('Chi2 end fraction: %3.3e', args.chi_end_fraction)

qdata, Idata, sigqdata, qbinsc, Imean, chis, rg, supportV = saxs.denss(
    q=q,I=I,sigq=sigq,D=args.dmax,supportpdb=supportpdb,rhostart=args.rhostart,ne=args.ne,
    rhobounds=args.rhobounds,voxel=args.voxel,oversampling=args.oversampling,
    usedmax=args.usedmax,recenter=args.recenter,positivity=args.positivity,
    extrapolate=args.extrapolate,write=True,filename=output,steps=args.steps,seed=args.seed,
    shrinkwrap=args.shrinkwrap,shrinkwrap_sigma_start=args.shrinkwrap_sigma_start,
    shrinkwrap_sigma_end=args.shrinkwrap_sigma_end,shrinkwrap_sigma_decay=args.shrinkwrap_sigma_decay,
    shrinkwrap_threshold_fraction=args.shrinkwrap_threshold_fraction,shrinkwrap_iter=args.shrinkwrap_iter,
    shrinkwrap_minstep=args.shrinkwrap_minstep,chi_end_fraction=args.chi_end_fraction)

print output

fit = np.zeros(( max(len(qdata),len(qbinsc)),5))
fit[:len(qdata),0] = qdata
fit[:len(qdata),1] = Idata
fit[:len(qdata),2] = sigqdata
fit[:len(qbinsc),3] = qbinsc
fit[:len(qbinsc),4] = Imean

np.savetxt(output+'_map.fit',fit,delimiter=' ',fmt='%.5e')
np.savetxt(output+'_stats_by_step.dat',np.vstack((chis, rg, supportV)).T,delimiter=" ",fmt="%.5e")

if args.plot:
    import matplotlib.pyplot as plt
    from  matplotlib.colors import colorConverter as cc
    
    plt.errorbar(q, I*Idata[0]/I[0], fmt='k-', yerr=sigq, capsize=0, elinewidth=0.1, ecolor=cc.to_rgba('0',alpha=0.5))
    plt.plot(qdata,Idata,'r.')
    plt.plot(qbinsc, Imean, 'b.')
    plt.semilogy()
    plt.savefig(output+'_fit',ext='png',dpi=300)
    plt.close()

    plt.plot(chis[chis>0])
    plt.semilogy()
    plt.savefig(output+'_chis',ext='png',dpi=300)
    plt.close()

    plt.plot(rg[rg!=0])
    plt.savefig(output+'_rgs',ext='png',dpi=300)
    plt.close()

    plt.plot(supportV[supportV>0])
    plt.savefig(output+'_supportV',ext='png',dpi=300)
    plt.close()

logging.info('END')






