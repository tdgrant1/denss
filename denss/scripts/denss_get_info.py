#!/usr/bin/env python
#
#    denss_get_info.py
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

from __future__ import print_function
import numpy as np
from scipy import ndimage
import argparse
import denss


def main():
    parser = argparse.ArgumentParser(description="Print some basic information about an MRC file.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-f", "--file", type=str, help="MRC filename.")
    args = parser.parse_args()


    rho, (a,b,c) = denss.read_mrc(args.file, returnABC=True)
    vx, vy, vz = np.array((a,b,c))/np.array(rho.shape)

    print(" Grid size:   %i x %i x %i" % (rho.shape[0],rho.shape[1],rho.shape[2]))
    print(" Side length: %f x %f x %f" % (a,b,c))
    print(" Voxel size:  %f x %f x %f" % (vx, vy, vz))
    print(" Voxel volume: %f" % (vx*vy*vz))
    print(" Total number of electrons:  %f" % (rho.sum()*vx*vy*vz))
    print(" Min/max density:  %f , %f" % (rho.min(),rho.max()))
    gridcenter = (np.array(rho.shape)-1.)/2.
    # com = gridcenter - np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    com = gridcenter - np.array(ndimage.center_of_mass(np.abs(rho)))
    print(" Center of mass in angstroms: [ %f %f %f ]" % (com[0]*vx,com[1]*vy,com[2]*vz))
    print(" Mean Density (all voxels): %3.5f" % np.mean(rho))
    print(" Std. Dev. of Density (all voxels): %3.5f" % np.std(rho))
    print(" RMSD of Density (all voxels): %3.5f" % np.sqrt(np.mean(np.square(rho))))
    idx = np.where(np.abs(rho)>0.01*rho.max())
    print(" Modified Mean Density (voxels >0.01*max): %3.5f" % np.mean(rho[idx]))
    print(" Modified Std. Dev. of Density (voxels >0.01*max): %3.5f" % np.std(rho[idx]))
    print(" Modified RMSD of Density (voxels >0.01*max): %3.5f" % np.sqrt(np.mean(np.square(rho[idx]))))


if __name__ == "__main__":
    main()











