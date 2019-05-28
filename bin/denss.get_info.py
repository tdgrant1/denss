#!/usr/bin/env python
#
#    denss.get_info.py
#    Print some basic information about an MRC file.
#
#    Part of DENSS
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tgrant@hwi.buffalo.edu>
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

import os, sys, logging
import numpy as np
from scipy import ndimage
import argparse
from saxstats._version import __version__
import saxstats.saxstats as saxs

parser = argparse.ArgumentParser(description="Print some basic information about an MRC file.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-f", "--file", type=str, help="MRC filename.")
args = parser.parse_args()

if __name__ == "__main__":


    rho, (a,b,c) = saxs.read_mrc(args.file, returnABC=True)
    vx, vy, vz = np.array((a,b,c))/np.array(rho.shape)

    print " Grid size:   %i x %i x %i" % (rho.shape[0],rho.shape[1],rho.shape[2])
    print " Side length: %f x %f x %f" % (a,b,c)
    print " Voxel size:  %f x %f x %f" % (vx, vy, vz)
    print " Total number of electrons:  %f" % (rho.sum()*vx*vy*vz)













