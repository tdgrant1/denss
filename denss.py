#!/usr/bin/env python
#
#    denss.py
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
    import matplotlib.pyplot as plt
    from  matplotlib.colors import colorConverter as cc
    import matplotlib.gridspec as gridspec
except ImportError:
    matplotlib_found = False

print saxs.__file__

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="SAXS data file for input (either .dat or .out)")
parser.add_argument("-d", "--dmax", default=None, type=float, help="Estimated maximum dimension")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Set desired voxel size, setting resolution of map")
parser.add_argument("-os","--oversampling", default=3., type=float, help="Sampling ratio")
parser.add_argument("-n", "--nsamples", default=None, type=int, help="Number of samples, i.e. grid points, along a single dimension. (Sets voxel size, overridden by --voxel. Best optimization with n=power of 2)")
parser.add_argument("--ne", default=10000, type=float, help="Number of electrons in object")
parser.add_argument("-s", "--steps", default=None, help="Maximum number of steps (iterations)")
parser.add_argument("-o", "--output", default=None, help="Output map filename")
parser.add_argument("--seed", default=None, help="Random seed to initialize the map")
parser.add_argument("--limit_dmax_on", dest="limit_dmax", action="store_true", help="Limit electron density to sphere of radius 0.6*Dmax from center of object.")
parser.add_argument("--limit_dmax_off", dest="limit_dmax", action="store_false", help="Do not limit electron density to sphere of radius 0.6*Dmax from center of object. (default)")
parser.add_argument("--dmax_start_step", default=500, type=int, help="Starting step for limiting density to sphere of Dmax (default=500)")
parser.add_argument("--recenter_on", dest="recenter", action="store_true", help="Recenter electron density when updating support. (default)")
parser.add_argument("--recenter_off", dest="recenter", action="store_false", help="Do not recenter electron density when updating support.")
parser.add_argument("--recenter_steps", default=None, type=int, help="List of steps to recenter electron density. (default=501,1001)")
parser.add_argument("--positivity_on", dest="positivity", action="store_true", help="Enforce positivity restraint inside support. (default)")
parser.add_argument("--positivity_off", dest="positivity", action="store_false", help="Do not enforce positivity restraint inside support.")
parser.add_argument("--extrapolate_on", dest="extrapolate", action="store_true", help="Extrapolate data by Porod law to high resolution limit of voxels. (default)")
parser.add_argument("--extrapolate_off", dest="extrapolate", action="store_false", help="Do not extrapolate data by Porod law to high resolution limit of voxels.")
parser.add_argument("--shrinkwrap_on", dest="shrinkwrap", action="store_true", help="Turn shrinkwrap on (default)")
parser.add_argument("--shrinkwrap_off", dest="shrinkwrap", action="store_false", help="Turn shrinkwrap off")
parser.add_argument("--shrinkwrap_sigma_start", default=3, type=float, help="Starting sigma for Gaussian blurring, in voxels")
parser.add_argument("--shrinkwrap_sigma_end", default=1.5, type=float, help="Ending sigma for Gaussian blurring, in voxels")
parser.add_argument("--shrinkwrap_sigma_decay", default=0.99, type=float, help="Rate of decay of sigma, fraction")
parser.add_argument("--shrinkwrap_threshold_fraction", default=0.20, type=float, help="Minimum threshold defining support, in fraction of maximum density")
parser.add_argument("--shrinkwrap_iter", default=20, type=int, help="Number of iterations between updating support with shrinkwrap")
parser.add_argument("--shrinkwrap_minstep", default=0, type=int, help="First step to begin shrinkwrap")
parser.add_argument("--enforce_connectivity_on", dest="enforce_connectivity", action="store_true", help="Enforce connectivity of support, i.e. remove extra blobs (default)")
parser.add_argument("--enforce_connectivity_off", dest="enforce_connectivity", action="store_false", help="Do not enforce connectivity of support")
parser.add_argument("--enforce_connectivity_steps", default=500, type=int, nargs='+', help="List of steps to enforce connectivity (Default=500, see enforce_connectivity)")
parser.add_argument("--chi_end_fraction", default=0.001, type=float, help="Convergence criterion. Minimum threshold of chi2 std dev, as a fraction of the median chi2 of last 100 steps.")
parser.add_argument("--write_xplor", default=False, action="store_true", help="Write out XPLOR map format (default only write MRC format).")
parser.add_argument("--write_freq", default=100, type=int, help="How often to write out current density map (in steps, default 100).")
parser.add_argument("--cutout_on", dest="cutout", action="store_true", help="When writing final map, cut out the particle to make smaller files (default)")
parser.add_argument("--cutout_off", dest="cutout", action="store_false", help="When writing final map, do not cut out the particle to make smaller files")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
parser.set_defaults(limit_dmax=False)
parser.set_defaults(shrinkwrap=True)
parser.set_defaults(recenter=True)
parser.set_defaults(positivity=True)
parser.set_defaults(extrapolate=True)
parser.set_defaults(enforce_connectivity=True)
parser.set_defaults(cutout=True)
if matplotlib_found:
    parser.set_defaults(plot=True)
else:
    parser.set_defaults(plot=False)
args = parser.parse_args()

if args.output is None:
    basename, ext = os.path.splitext(args.file)
    output = basename
else:
    output = args.output

logging.basicConfig(filename=output+'.log',level=logging.INFO,filemode='w',
    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info('BEGIN')
logging.info('Data filename: %s', args.file)
logging.info('Output prefix: %s', output)


q, I, sigq, dmax, isout = saxs.loadProfile(args.file)

if args.dmax is not None:
    dmax = args.dmax
elif dmax<=0:
    dmax = 100

if args.voxel is None and args.nsamples is None:
    voxel = 5.
elif args.voxel is None and args.nsamples is not None:
    voxel = dmax * args.oversampling / args.nsamples
else:
    voxel = args.voxel

if not isinstance(args.enforce_connectivity_steps, list):
    args.enforce_connectivity_steps = [ args.enforce_connectivity_steps ]

if not isinstance(args.recenter_steps, list):
    if args.recenter_steps is None:
        args.recenter_steps = [501, 601, 701, 801, 901, 1001]
    else:
        args.enforce_connectivity_steps = [ args.enforce_connectivity_steps ]

#logging.info('Maximum number of steps: %i', args.steps)
logging.info('q range of input data: %3.3f < q < %3.3f', q.min(), q.max())
logging.info('Maximum dimension: %3.3f', dmax)
logging.info('Sampling ratio: %3.3f', args.oversampling)
logging.info('Requested real space voxel size: %3.3f', voxel)
logging.info('Number of electrons: %3.3f', args.ne)
logging.info('Limit Dmax: %s', args.limit_dmax)
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
logging.info('Enforce connectivity: %s', args.enforce_connectivity)
logging.info('Enforce connectivity steps: %s', args.enforce_connectivity_steps)
logging.info('Chi2 end fraction: %3.3e', args.chi_end_fraction)

qdata, Idata, sigqdata, qbinsc, Imean, chis, rg, supportV = saxs.denss(q=q,I=I,
    sigq=sigq,D=dmax,ne=args.ne,voxel=voxel,oversampling=args.oversampling,
    limit_dmax=args.limit_dmax,dmax_start_step=args.dmax_start_step,
    recenter=args.recenter,recenter_steps=args.recenter_steps,
    positivity=args.positivity,extrapolate=args.extrapolate,write=True,
    filename=output,steps=args.steps,seed=args.seed,shrinkwrap=args.shrinkwrap,
    shrinkwrap_sigma_start=args.shrinkwrap_sigma_start,
    shrinkwrap_sigma_end=args.shrinkwrap_sigma_end,
    shrinkwrap_sigma_decay=args.shrinkwrap_sigma_decay,
    shrinkwrap_threshold_fraction=args.shrinkwrap_threshold_fraction,
    shrinkwrap_iter=args.shrinkwrap_iter,shrinkwrap_minstep=args.shrinkwrap_minstep,
    chi_end_fraction=args.chi_end_fraction,write_xplor_format=args.write_xplor,write_freq=args.write_freq,
    enforce_connectivity=args.enforce_connectivity,
    enforce_connectivity_steps=args.enforce_connectivity_steps,cutout=args.cutout)

print output

fit = np.zeros(( len(qbinsc),5 ))
fit[:len(qdata),0] = qdata
fit[:len(Idata),1] = Idata
fit[:len(sigqdata),2] = sigqdata
fit[:len(qbinsc),3] = qbinsc
fit[:len(Imean),4] = Imean
np.savetxt(output+'_map.fit',fit,delimiter=' ',fmt='%.5e', header='q(data),I(data),error(data),q(density),I(density)')
np.savetxt(output+'_stats_by_step.dat',np.vstack((chis, rg, supportV)).T,delimiter=" ",fmt="%.5e",header='Chi2 Rg SupportVolume')

if args.plot and matplotlib_found:
    f = plt.figure(figsize=[6,6])
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    ax0 = plt.subplot(gs[0])
    #handle sigq values whose error bounds would go negative and be missing on the log scale
    sigq2 = np.copy(sigq)
    sigq2[sigq>I] = I[sigq>I]*.999
    ax0.errorbar(q[q<=qdata[-1]], I[q<=qdata[-1]], fmt='k-', yerr=[sigq2[q<=qdata[-1]],sigq[q<=qdata[-1]]], capsize=0, elinewidth=0.1, ecolor=cc.to_rgba('0',alpha=0.5),label='Raw Data')
    ax0.plot(qdata, Idata, 'bo',alpha=0.5,label='Interpolated Data')
    ax0.plot(qbinsc,Imean,'r.',label='Scattering from Density')
    handles,labels = ax0.get_legend_handles_labels()
    handles = [handles[2], handles[0], handles[1]]
    labels = [labels[2], labels[0], labels[1]]
    ymin = np.min(np.hstack((I,Idata,Imean)))
    ymax = np.max(np.hstack((I,Idata,Imean)))
    ax0.set_ylim([0.5*ymin,1.5*ymax])
    ax0.legend(handles,labels)
    ax0.semilogy()
    ax0.set_ylabel('log I(q)')

    ax1 = plt.subplot(gs[1])
    ax1.plot(qdata, qdata*0, 'k--')
    ax1.plot(qdata, np.log10(Imean)-np.log10(Idata), 'ro-')
    ylim = ax1.get_ylim()
    ymax = np.max(np.abs(ylim))
    ax1.set_ylim([-ymax,ymax])
    ax1.yaxis.major.locator.set_params(nbins=5)
    ax1.set_ylabel('Residuals')
    ax1.set_xlabel(r'q ($\mathrm{\AA^{-1}}$)')
    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.savefig(output+'_fit',ext='png',dpi=150)
    plt.close()

    plt.plot(chis[chis>0])
    plt.xlabel('Step')
    plt.ylabel('$\chi^2$')
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(output+'_chis',ext='png',dpi=150)
    plt.close()

    plt.plot(rg[rg!=0])
    plt.xlabel('Step')
    plt.ylabel('Rg')
    plt.tight_layout()
    plt.savefig(output+'_rgs',ext='png',dpi=150)
    plt.close()

    plt.plot(supportV[supportV>0])
    plt.xlabel('Step')
    plt.ylabel('Support Volume ($\mathrm{\AA^{3}}$)')
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(output+'_supportV',ext='png',dpi=150)
    plt.close()

logging.info('END')






