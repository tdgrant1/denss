#!/usr/bin/env python
#
#    denss_hypred2mrc.py
#    A tool for calculating an electron density map from a HyPred pdb file.
#
#    Part of the DENSS package
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tdgrant@buffalo.edu>
#    Copyright 2017-Present The Research Foundation for SUNY
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

import os, copy, argparse
import numpy as np

import denss


def main():
    parser = argparse.ArgumentParser(description="A tool for converting hypred pdb files (with density in B-factor column) to MRC formatted electron density maps.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="Hypred model as a .pdb file for input.")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_hypred)")
    args = parser.parse_args()

    file = args.file
    fname_nopath = os.path.basename(file)
    basename, ext = os.path.splitext(fname_nopath)
    output = basename

    pdb = denss.PDB(file)
    prot = copy.deepcopy(pdb)
    h2o = copy.deepcopy(pdb)

    #separate protein and h2o atoms
    idx = np.where(h2o.resname=="HOH")
    prot.remove_atoms_from_object(idx)
    idx = np.where(h2o.resname!="HOH")
    h2o.remove_atoms_from_object(idx)

    #recenter both the protein and h2o (for some reason
    #they don't seem to be aligned in the hypred pdb output)
    # prot.coords -= prot.coords.mean(0)
    # h2o.coords -= h2o.coords.mean(0)

    #reset density to be contrast, so subtract 0.33 (not 0.334 apparently)
    prot.b -= 0.33
    h2o.b -= 0.33

    prot.x = prot.coords[:,0]
    prot.y = prot.coords[:,1]
    prot.z = prot.coords[:,2]

    h2o.x = h2o.coords[:,0]
    h2o.y = h2o.coords[:,1]
    h2o.z = h2o.coords[:,2]

    xmin, xmax = h2o.x.min(), h2o.x.max()
    ymin, ymax = h2o.y.min(), h2o.y.max()
    zmin, zmax = h2o.z.min(), h2o.z.max()

    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    xunique = np.unique(h2o.x)
    dx = xunique[1] - xunique[0]

    #get the minimum and maximum values to make a cube
    xxmin = np.min([xmin,ymin,zmin])
    xxmax = np.max([xmax,ymax,zmax])

    #create the density array
    x_ = np.arange(xxmin, xxmax, dx)
    x, y, z = np.meshgrid(x_,x_,x_,indexing='ij')
    rho_prot = np.zeros(x.shape)
    rho_h2o = np.zeros(x.shape)
    side = x_[-1]-x_[0] + dx

    #replace the zeros of the density with the b-factor column of the hypred pdb
    #since dx=0.5, the indices are simply twice the xyz coordinate values
    idx_x = (1/dx*h2o.x).astype(int)
    idx_y = (1/dx*h2o.y).astype(int)
    idx_z = (1/dx*h2o.z).astype(int)
    rho_h2o[idx_x, idx_y, idx_z] = h2o.b
    rho_h2o, shift = denss.center_rho_roll(rho_h2o, return_shift=True)
    denss.write_mrc(rho_h2o,side,output+'_H2O.mrc')

    #do the same for protein
    idx_x = (2*prot.x).astype(int)
    idx_y = (2*prot.y).astype(int)
    idx_z = (2*prot.z).astype(int)
    rho_prot[idx_x, idx_y, idx_z] = prot.b
    # rho_prot = denss.center_rho_roll(rho_prot)
    rho_prot = np.roll(np.roll(np.roll(rho_prot, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    denss.write_mrc(rho_prot,side,output+'_protein.mrc')


if __name__ == "__main__":
    main()







