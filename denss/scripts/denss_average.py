#!/usr/bin/env python
#
#    denss_average.py
#    A tool for averaging multiple pre-aligned electron density maps.
#
#    Part of DENSS
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Authors: Thomas D. Grant, Nhan D. Nguyen
#    Email:  <tgrant@hwi.buffalo.edu>, <ndnguyen20@wabash.edu>
#    Copyright 2018 The Research Foundation for SUNY
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
import sys, os, argparse, logging
import numpy as np

import denss

def main():
    parser = argparse.ArgumentParser(description="A tool for averaging multiple pre-aligned electron density maps.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--files", type=str, nargs="+", help="List of MRC files")
    parser.add_argument("-o", "--output", type=str, help="Output filename prefix")
    args = parser.parse_args()

    if args.output is None:
        fname_nopath = os.path.basename(args.files[0])
        basename, ext = os.path.splitext(fname_nopath)
        output = basename
    else:
        output = args.output

    logging.basicConfig(filename=output+'_avg.log',level=logging.INFO,filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Command: %s', ' '.join(sys.argv))
    #logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', denss.__version__)

    rhosum = None
    n=1
    nmaps = len(args.files)
    rhos = []
    for file in args.files:
        sys.stdout.write("\r% 5i / % 5i" % (n, nmaps))
        sys.stdout.flush()
        n+=1
        rho, side = denss.read_mrc(file)
        rhos.append(rho)
        if rhosum is None:
            rhosum = rho
        else:
            rhosum += rho
    print()
    rhos = np.array(rhos)
    average_rho = rhosum / nmaps
    denss.write_mrc(average_rho, side, output+"_avg.mrc")
    print("%s_avg.mrc written." % output)

    #rather than compare two halves, average all fsc's to the reference
    fscs = []
    resns = []
    for calc_map in range(nmaps):
        fsc_map = denss.calc_fsc(rhos[calc_map],average_rho,side)
        fscs.append(fsc_map)
        resn_map = denss.fsc2res(fsc_map)
        resns.append(resn_map)

    fscs = np.array(fscs)
    resns = np.array(resns)
    fsc = np.mean(fscs,axis=0)
    resn, x, y, resx = denss.fsc2res(fsc, return_plot=True)
    resn_sd = np.std(resns)
    if np.min(fsc[:,1]) > 0.5:
        print("Resolution: < %.1f +- %.1f A (maximum possible)" % (resn,resn_sd))
    else:
        print("Resolution: %.1f +- %.1f A " % (resn,resn_sd))

    np.savetxt(output+'_fsc.dat',fsc,delimiter=" ",fmt="%.5e",header="1/resolution, FSC; Resolution=%.1f +- %.1f A" % (resn,resn_sd))

    logging.info('Resolution: %.1f '+ 'A', resn )
    logging.info('END')

    logging.info('END')


if __name__ == "__main__":
    main()