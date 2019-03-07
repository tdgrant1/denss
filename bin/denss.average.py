#!/usr/bin/env python
#
#    denss.average.py
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

import sys, os, argparse, logging
import numpy as np
from scipy import ndimage
from saxstats._version import __version__
import saxstats.saxstats as saxs

parser = argparse.ArgumentParser(description="A tool for averaging multiple pre-aligned electron density maps.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--files", type=str, nargs="+", help="List of MRC files")
parser.add_argument("-o", "--output", type=str, help="Output filename prefix")
args = parser.parse_args()

if __name__ == "__main__":

    if args.output is None:
        basename, ext = os.path.splitext(args.files[0])
        output = basename
    else:
        output = args.output

    logging.basicConfig(filename=output+'_avg.log',level=logging.INFO,filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', __version__)

    rhosum = None
    n=1
    nmaps = len(args.files)
    rhos = []
    for file in args.files:
        sys.stdout.write("\r% 5i / % 5i" % (n, nmaps))
        sys.stdout.flush()
        n+=1
        rho, side = saxs.read_mrc(file)
        rhos.append(rho)
        if rhosum is None:
            rhosum = rho
        else:
            rhosum += rho
    print
    rhos = np.array(rhos)
    average_rho = rhosum / nmaps
    saxs.write_mrc(average_rho,side, output+"_avg.mrc")
    print "%s_avg.mrc written." % output

    """
    #split maps into 2 halves--> enan, align, average independently with same refrho
    avg_rho1 = np.mean(aligned[::2],axis=0)
    avg_rho2 = np.mean(aligned[1::2],axis=0)
    fsc = saxs.calc_fsc(avg_rho1,avg_rho2,sides[0])
    np.savetxt(args.output+'_fsc.dat',fsc,delimiter=" ",fmt="%.5e",header="qbins, FSC")
    """
    #rather than compare two halves, average all fsc's to the reference
    fscs = []
    for map in range(nmaps):
        fscs.append(saxs.calc_fsc(rhos[map],average_rho,side))
    fscs = np.array(fscs)
    fsc = np.mean(fscs,axis=0)
    np.savetxt(args.output+'_fsc.dat',fsc,delimiter=" ",fmt="%.5e",header="1/resolution, FSC")
    x = np.linspace(fsc[0,0],fsc[-1,0],100)
    y = np.interp(x, fsc[:,0], fsc[:,1])
    resi = np.argmin(y>=0.5)
    resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])
    resn = round(float(1./resx),1)
    print "Resolution: %.1f" % resn, u'\u212B'.encode('utf-8')

    logging.info('Resolution: %.1f '+ u'\u212B'.encode('utf-8'), resn )
    logging.info('END')

    logging.info('END')

