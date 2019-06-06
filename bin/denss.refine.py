#!/usr/bin/env python
#
#    denss.refine.py
#    A tool for refining an electron density map from solution scattering data
#
#    Part of DENSS
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


from saxstats._version import __version__
import saxstats.saxstats as saxs
import saxstats.denssopts as dopts
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

#have to run parser twice, first just to get filename for loadProfile
#then have to run it after deciding what the correct dmax should be
#so that the voxel size, box size, nsamples, etc are set correctly
initparser = argparse.ArgumentParser(description=" A tool for refining an electron density map from solution scattering data", formatter_class=argparse.RawTextHelpFormatter)
initargs = dopts.parse_arguments(initparser, gnomdmax=None)

q, I, sigq, dmax, isout = saxs.loadProfile(initargs.file, units=initargs.units)

if not initargs.force_run:
    if min(q) != 0.0:
        print "CAUTION: Minimum q value = %f " % min(q)
        print "is not 0.0. It is STRONGLY recommended to include"
        print "I(q=0) in your given scattering profile. You can use"
        print "denss.fit_data.py to calculate a scattering profile fit"
        print "which will include I(q=0), or you can also use the GNOM"
        print "program from ATSAS to create a .out file."
        print
        print "If you are positive you would like to continue, "
        print "rerun with the --force_run option."
        sys.exit()


if dmax <= 0:
    dmax = None

parser = argparse.ArgumentParser(description="DENSS: DENsity from Solution Scattering.\n A tool for calculating an electron density map from solution scattering data", formatter_class=argparse.RawTextHelpFormatter)
args = dopts.parse_arguments(parser, gnomdmax=dmax)


if args.rho_start is None:
    print " denss.refine.py requires a .mrc file to be given to the --rho_start option."
    sys.exit()

basename, ext = os.path.splitext(args.rho_start)
args.output = basename + '_refine'

args.rho_start, rho_side = saxs.read_mrc(args.rho_start)

rho_nsamples = args.rho_start.shape[0]
rho_voxel = rho_side/rho_nsamples

args.side = args.dmax*args.oversampling

if (not np.isclose(rho_side, args.side) or
    not np.isclose(rho_voxel, args.voxel) or
    not np.isclose(rho_nsamples, args.nsamples)):
    print "rho_start density dimensions do not match given options."
    print "Oversampling and voxel size adjusted to match rho_start dimensions."

args.voxel = rho_voxel
args.oversampling = rho_side/args.dmax
args.nsamples = rho_nsamples


if __name__ == "__main__":

    logging.basicConfig(filename=args.output+'.log',level=logging.INFO,filemode='w',
        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', __version__)
    logging.info('Data filename: %s', args.file)
    logging.info('Output prefix: %s', args.output)
    logging.info('Mode: %s', args.mode)
    logging.info('Starting density: %s', basename+'.mrc')

    qdata, Idata, sigqdata, qbinsc, Imean, chis, rg, supportV, rho, side = saxs.denss(
        q=q,I=I,sigq=sigq,
        dmax=args.dmax,
        ne=args.ne,
        voxel=args.voxel,
        oversampling=args.oversampling,
        limit_dmax=args.limit_dmax,
        limit_dmax_steps=args.limit_dmax_steps,
        recenter=args.recenter,
        recenter_steps=args.recenter_steps,
        recenter_mode=args.recenter_mode,
        positivity=args.positivity,
        flatten_low_density=args.flatten_low_density,
        minimum_density=args.minimum_density,
        maximum_density=args.maximum_density,
        rho_start=args.rho_start,
        extrapolate=args.extrapolate,
        output=args.output,
        steps=args.steps,
        ncs=args.ncs,
        ncs_steps=args.ncs_steps,
        ncs_axis=args.ncs_axis,
        seed=args.seed,
        shrinkwrap=args.shrinkwrap,
        shrinkwrap_sigma_start=args.shrinkwrap_sigma_start,
        shrinkwrap_sigma_end=args.shrinkwrap_sigma_end,
        shrinkwrap_sigma_decay=args.shrinkwrap_sigma_decay,
        shrinkwrap_threshold_fraction=args.shrinkwrap_threshold_fraction,
        shrinkwrap_iter=args.shrinkwrap_iter,
        shrinkwrap_minstep=args.shrinkwrap_minstep,
        chi_end_fraction=args.chi_end_fraction,
        write_xplor_format=args.write_xplor_format,
        write_freq=args.write_freq,
        enforce_connectivity=args.enforce_connectivity,
        enforce_connectivity_steps=args.enforce_connectivity_steps,
        cutout=args.cutout)

    print args.output

    fit = np.zeros(( len(qbinsc),5 ))
    fit[:len(qdata),0] = qdata
    fit[:len(Idata),1] = Idata
    fit[:len(sigqdata),2] = sigqdata
    fit[:len(qbinsc),3] = qbinsc
    fit[:len(Imean),4] = Imean
    np.savetxt(args.output+'_map.fit',fit,delimiter=' ',fmt='%.5e', header='q(data),I(data),error(data),q(density),I(density)')
    np.savetxt(args.output+'_stats_by_step.dat',np.vstack((chis, rg, supportV)).T,delimiter=" ",fmt="%.5e",header='Chi2 Rg SupportVolume')

    if args.plot and matplotlib_found:
        f = plt.figure(figsize=[6,6])
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

        ax0 = plt.subplot(gs[0])
        #handle sigq values whose error bounds would go negative and be missing on the log scale
        sigq2 = np.copy(sigq)
        sigq2[sigq>I] = I[sigq>I]*.999
        ax0.errorbar(q[q<=qdata[-1]], I[q<=qdata[-1]], fmt='k-', yerr=[sigq2[q<=qdata[-1]],sigq[q<=qdata[-1]]], capsize=0, elinewidth=0.1, ecolor=cc.to_rgba('0',alpha=0.5),label='Supplied Data')
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
        ax0.set_ylabel('I(q)')

        ax1 = plt.subplot(gs[1])
        ax1.plot(qdata, qdata*0, 'k--')
        residuals = np.log10(Imean[np.in1d(qbinsc,qdata)])-np.log10(Idata)
        ax1.plot(qdata, residuals, 'ro-')
        ylim = ax1.get_ylim()
        ymax = np.max(np.abs(ylim))
        n = int(.9*len(residuals))
        ymax = np.max(np.abs(residuals[:-n]))
        ax1.set_ylim([-ymax,ymax])
        ax1.yaxis.major.locator.set_params(nbins=5)
        xlim = ax0.get_xlim()
        ax1.set_xlim(xlim)
        ax1.set_ylabel('Residuals')
        ax1.set_xlabel(r'q ($\mathrm{\AA^{-1}}$)')
        #plt.setp(ax0.get_xticklabels(), visible=False)
        plt.tight_layout()
        plt.savefig(args.output+'_fit.png',dpi=150)
        plt.close()

        plt.plot(chis[chis>0])
        plt.xlabel('Step')
        plt.ylabel('$\chi^2$')
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(args.output+'_chis.png',dpi=150)
        plt.close()

        plt.plot(rg[rg!=0])
        plt.xlabel('Step')
        plt.ylabel('Rg')
        plt.tight_layout()
        plt.savefig(args.output+'_rgs.png',dpi=150)
        plt.close()

        plt.plot(supportV[supportV>0])
        plt.xlabel('Step')
        plt.ylabel('Support Volume ($\mathrm{\AA^{3}}$)')
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(args.output+'_supportV.png',dpi=150)
        plt.close()

    logging.info('END')


