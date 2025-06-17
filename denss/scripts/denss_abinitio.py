#!/usr/bin/env python
#
#    denss_abinitio.py
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

from __future__ import print_function

import denss
from denss import options as dopts
import numpy as np
import sys, argparse, os
import logging


def main():
    parser = argparse.ArgumentParser(description="DENSS: DENsity from Solution Scattering.\n A tool for calculating an electron density map from solution scattering data", formatter_class=argparse.RawTextHelpFormatter)
    args = dopts.parse_arguments(parser)

    __spec__ = None
    my_logger = logging.getLogger()
    my_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %I:%M:%S %p')

    # h1 = logging.StreamHandler(sys.stdout)
    # h1.setLevel(logging.INFO)
    # h1.setFormatter(formatter)

    h2 = logging.FileHandler(args.output + '.log', mode='w')
    h2.setLevel(logging.INFO)
    h2.setFormatter(formatter)

    # my_logger.addHandler(h1)
    my_logger.addHandler(h2)

    my_logger.info('BEGIN')
    my_logger.info('Command: %s', ' '.join(sys.argv))
    my_logger.info('DENSS Version: %s', denss.__version__)
    my_logger.info('Data filename: %s', args.file)
    my_logger.info('Output prefix: %s', args.output)
    my_logger.info('Mode: %s', args.mode)

    qdata, Idata, sigqdata, qbinsc, Imean, chis, rg, supportV, rho, side, fit, final_chi2 = denss.reconstruct_abinitio_from_scattering_profile(
        q=args.q,
        I=args.I,
        sigq=args.sigq,
        dmax=args.dmax,
        qraw=args.qraw,
        Iraw=args.Iraw,
        sigqraw=args.sigqraw,
        ne=args.ne,
        voxel=args.voxel,
        oversampling=args.oversampling,
        recenter=args.recenter,
        recenter_steps=args.recenter_steps,
        recenter_mode=args.recenter_mode,
        positivity=args.positivity,
        positivity_steps=args.positivity_steps,
        extrapolate=args.extrapolate,
        output=args.output,
        steps=args.steps,
        ncs=args.ncs,
        ncs_steps=args.ncs_steps,
        ncs_axis=args.ncs_axis,
        ncs_type=args.ncs_type,
        seed=args.seed,
        support_start=args.support_start,
        shrinkwrap=args.shrinkwrap,
        shrinkwrap_old_method=args.shrinkwrap_old_method,
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
        enforce_connectivity_max_features=args.enforce_connectivity_max_features,
        cutout=args.cutout,
        quiet=args.quiet,
        DENSS_GPU=args.DENSS_GPU,
        my_logger=my_logger)

    print("\n%s"%args.output)

    if args.plot:
        import matplotlib.pyplot as plt
        from  matplotlib.colors import colorConverter as cc
        import matplotlib.gridspec as gridspec

        qraw = args.qraw
        Iraw = args.Iraw
        q = args.q
        I = args.I
        sigq = args.sigq

        f = plt.figure(figsize=[6,6])
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

        ax0 = plt.subplot(gs[0])
        ax0.errorbar(fit[:,0], fit[:,1], fmt='k.', yerr=fit[:,2], mec='none', mew=0, ms=5, alpha=0.3, capsize=0, elinewidth=0.1, ecolor=cc.to_rgba('0',alpha=0.5),label='Supplied Data',zorder=-1)
        ax0.plot(fit[:,0],fit[:,3],'r-',label=r'DENSS Map $\chi^2 = %.2f$'%final_chi2)

        handles,labels = ax0.get_legend_handles_labels()
        handles = [handles[1], handles[0] ]
        labels = [labels[1], labels[0] ]
        ax0.legend(handles,labels)
        ax0.semilogy()
        ax0.set_ylabel('I(q)')

        ax1 = plt.subplot(gs[1])
        ax1.plot(fit[:,0], fit[:,0]*0, 'k--')
        residuals = (fit[:,1]-fit[:,3])/fit[:,2]
        ax1.plot(fit[:,0], residuals, 'r.')
        ylim = ax1.get_ylim()
        ymax = np.max(np.abs(ylim))
        ymax = np.max(np.abs(residuals))
        ax1.set_ylim([-ymax,ymax])
        ax1.yaxis.major.locator.set_params(nbins=5)
        xlim = ax0.get_xlim()
        ax1.set_xlim(xlim)
        ax1.set_ylabel(r'$\Delta{I}/\sigma$')
        ax1.set_xlabel(r'q ($\mathrm{\AA^{-1}}$)')
        plt.tight_layout()
        plt.savefig(args.output+'_fit.png',dpi=150)
        plt.close()

        fig, host = plt.subplots(nrows=1, ncols=1)

        par1 = host.twinx()
        par2 = host.twinx()

        host.set_xlabel('Step')
        host.set_ylabel('$\chi^2$')
        par1.set_ylabel('Rg')
        par2.set_ylabel('Support Volume')

        color1 = plt.cm.viridis(0)
        color2 = plt.cm.viridis(0.5)
        color3 = plt.cm.viridis(.9)

        p1, = host.plot(chis[chis>0], color=color1,label="$\chi^2$")
        p2, = par1.plot(rg[rg!=0], color=color2, label="Rg")
        p3, = par2.plot(supportV[supportV>0], color=color3, label="Support Volume")

        host.semilogy()
        par2.semilogy()

        lns = [p1, p2, p3]
        host.legend(handles=lns, loc='best')

        # right, left, top, bottom
        par2.spines['right'].set_position(('outward', 60))

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())

        plt.savefig(args.output+'_stats_by_step.png', bbox_inches='tight',dpi=150)
        plt.close()

    logging.info('END')

if __name__ == "__main__":
    main()
