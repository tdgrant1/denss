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

import os, argparse, logging
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
    logging.info('DENSS Version: %s', __version__)

    allrhos = []
    sides = []
    for file in args.files:
        rho, side = saxs.read_mrc(file)
        allrhos.append(rho)
        sides.append(side)

    nmaps = len(allrhos)
    allrhos = np.array(allrhos)
    sides = np.array(sides)
    average_rho = np.mean(allrhos,axis=0)
    saxs.write_mrc(average_rho,sides[0], output+"_average.mrc")
    logging.info('END')

