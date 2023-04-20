#!/usr/bin/env python
#
#    saxstats.py
#    SAXStats
#    A collection of python functions useful for solution scattering
#
#    Tested using Anaconda / Python 2.7, 3.7
#
#    Author: Thomas D. Grant
#    Email:  <tdgrant@buffalo.edu>
#    Alt Email:  <tgrant@hwi.buffalo.edu>
#    Copyright 2017 - Present The Research Foundation for SUNY
#
#    Additional authors:
#    Nhan D. Nguyen
#    Jesse Hopkins
#    Andrew Bruno
#    Esther Gallmeier
#    Sarah Chamberlain
#    Stephen Moore
#    Jitendra Singh
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

from __future__ import print_function, division, unicode_literals
try:
    from builtins import object, range, map, zip, str
except ImportError:
    from __builtin__ import object, range, map, zip, str
from io import open

import sys
import re
import os
import json
import struct
import logging
from functools import partial
import multiprocessing
import datetime, time
from time import sleep
import warnings
import pickle

import numpy as np
from scipy import ndimage, interpolate, spatial, special, optimize, signal, stats, fft
from functools import reduce

from saxstats import protein_residues 

#load some dictionaries
from resources import resources
electrons = resources.electrons
atomic_volumes = resources.atomic_volumes
numH = resources.numH
volH = resources.volH
vdW = resources.vdW
radii_sf_dict = resources.radii_sf_dict
ffcoeff = resources.ffcoeff

#for implicit hydrogens, from distribution of corrected unique volumes
implicit_H_radius = 0.826377

try: 
    import numba as nb
    numba = True
    #suppress some unnecessary deprecation warnings
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
except:
    numba = False

try:
    import cupy as cp
    CUPY_LOADED = True
except ImportError:
    CUPY_LOADED = False

# try:
#     import pyfftw
#     pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
#     pyfftw.interfaces.cache.enable()
#     PYFFTW = True
#     # Try to load FFTW wisdom but don't panic if we can't
#     try:
#         with open("fft.wisdom", "rb") as the_file:
#             wisdom = pickle.load(the_file)
#             pyfftw.import_wisdom(wisdom)
#             print("pyfftw wisdom imported")
#     except FileNotFoundError:
#         print("Warning: pyfftw wisdom could not be imported")
# except:
#     PYFFTW = False

#disable pyfftw until we can make it more stable
#it works, but often results in nans randomly
PYFFTW = False


def myfftn(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.fft.fftn(x)
    else:
        if PYFFTW:
            return pyfftw.interfaces.numpy_fft.fftn(x)
        else:
            try:
                #try running the parallelized version of scipy fft
                return fft.fftn(x,workers=-1)
            except:
                #fall back to numpy
                return np.fft.fftn(x)

def myifftn(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.fft.ifftn(x)
    else:
        if PYFFTW:
            return pyfftw.interfaces.numpy_fft.ifftn(x)
        else:
            try:
                #try running the parallelized version of scipy fft
                return fft.ifftn(x,workers=-1)
            except:
                #fall back to numpy
                return np.fft.ifftn(x)

def myabs(x, out=None,DENSS_GPU=False):
    if DENSS_GPU:
        return cp.abs(x,out=out)
    else:
        return np.abs(x,out=out)

# @numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    #a faster way to calculate abs(x)**2, for calculating intensities
    # print(x.real.min(),x.real.max())
    # print(x.imag.min(),x.imag.max())
    re2 = (x.real)**2 
    im2 = (x.imag)**2
    # print("got here")
    # exit()
    # print(re2.max())
    # print(im2.max())
    _abs2 = re2 + im2
    return _abs2

# @numba.jit(nopython=True)
# def mybincount(x, weights):
#     result = np.zeros(x.max() + 1, int)
#     for i in x:
#         result[i] += weights[i]

def mybinmean(xravel,binsravel,xcount=None,DENSS_GPU=False):
    if DENSS_GPU:
        xsum = cp.bincount(binsravel, xravel)
        if xcount is None:
            xcount = cp.bincount(binsravel)
        return xsum/xcount
    else:
        xsum = np.bincount(binsravel, xravel)
        if xcount is None:
            xcount = np.bincount(binsravel)
        return xsum/xcount

def myones(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.ones(x)
    else:
        return np.ones(x)

def myzeros(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.zeros(x)
    else:
        return np.zeros(x)

def mysqrt(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.sqrt(x)
    else:
        return np.sqrt(x)

def mysum(x, out=None,DENSS_GPU=False):
    if DENSS_GPU:
        return cp.sum(x,out=out)
    else:
        return np.sum(x,out=out)

def myzeros_like(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.zeros_like(x)
    else:
        return np.zeros_like(x)

def mystd(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.std(x)
    else:
        return np.std(x)

def mymean(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.mean(x)
    else:
        return np.mean(x)

def mylog(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.log(x)
    else:
        return np.log(x)

def chi2(exp, calc, sig):
    """Return the chi2 discrepancy between experimental and calculated data"""
    return np.sum(np.square(exp - calc) / np.square(sig))

def rho2rg(rho,side=None,r=None,support=None,dx=None):
    """Calculate radius of gyration from an electron density map."""
    if side is None and r is None:
        print("Error: To calculate Rg, must provide either side or r parameters.")
        sys.exit()
    if side is not None and r is None:
        n = rho.shape[0]
        x_ = np.linspace(-side/2.,side/2.,n)
        x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
        r = np.sqrt(x**2 + y**2 + z**2)
    if support is None:
        support = np.ones_like(rho,dtype=bool)
    if dx is None:
        print("Error: To calculate Rg, must provide dx")
        sys.exit()
    gridcenter = (np.array(rho.shape)-1.)/2.
    com = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    rhocom = (gridcenter-com)*dx
    rg2 = np.sum(r[support]**2*rho[support])/np.sum(rho[support])
    rg2 = rg2 - np.linalg.norm(rhocom)**2
    rg = np.sign(rg2)*np.abs(rg2)**0.5
    return rg

def write_mrc(rho,side,filename="map.mrc"):
    """Write an MRC formatted electron density map.
       See here: http://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/#image
    """
    xs, ys, zs = rho.shape
    nxstart = -xs//2+1
    nystart = -ys//2+1
    nzstart = -zs//2+1
    side = np.atleast_1d(side)
    if len(side) == 1:
        a,b,c = side, side, side
    elif len(side) == 3:
        a,b,c = side
    else:
        print("Error. Argument 'side' must be float or 3-tuple")
    with open(filename, "wb") as fout:
        # NC, NR, NS, MODE = 2 (image : 32-bit reals)
        fout.write(struct.pack('<iiii', xs, ys, zs, 2))
        # NCSTART, NRSTART, NSSTART
        fout.write(struct.pack('<iii', nxstart, nystart, nzstart))
        # MX, MY, MZ
        fout.write(struct.pack('<iii', xs, ys, zs))
        # X length, Y, length, Z length
        fout.write(struct.pack('<fff', a, b, c))
        # Alpha, Beta, Gamma
        fout.write(struct.pack('<fff', 90.0, 90.0, 90.0))
        # MAPC, MAPR, MAPS
        fout.write(struct.pack('<iii', 1, 2, 3))
        # DMIN, DMAX, DMEAN
        fout.write(struct.pack('<fff', np.min(rho), np.max(rho), np.average(rho)))
        # ISPG, NSYMBT, mlLSKFLG
        fout.write(struct.pack('<iii', 1, 0, 0))
        # EXTRA
        fout.write(struct.pack('<'+'f'*12, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        for i in range(0, 12):
            fout.write(struct.pack('<f', 0.0))

        # XORIGIN, YORIGIN, ZORIGIN
        fout.write(struct.pack('<fff', 0.,0.,0. )) #nxstart*(a/xs), nystart*(b/ys), nzstart*(c/zs) ))
        # MAP
        fout.write('MAP '.encode())
        # MACHST (little endian)
        fout.write(struct.pack('<BBBB', 0x44, 0x41, 0x00, 0x00))
        # RMS (std)
        fout.write(struct.pack('<f', np.std(rho)))
        # NLABL
        fout.write(struct.pack('<i', 0))
        # LABEL(20,10) 10 80-character text labels
        for i in range(0, 800):
            fout.write(struct.pack('<B', 0x00))

        # Write out data
        s = struct.pack('=%sf' % rho.size, *rho.flatten('F'))
        fout.write(s)

def read_mrc(filename="map.mrc",returnABC=False,float64=True):
    """
        See MRC format at http://bio3d.colorado.edu/imod/doc/mrc_format.txt for offsets
    """
    with open(filename, 'rb') as fin:
        MRCdata=fin.read()
        nx = struct.unpack_from('<i',MRCdata, 0)[0]
        ny = struct.unpack_from('<i',MRCdata, 4)[0]
        nz = struct.unpack_from('<i',MRCdata, 8)[0]

        #side = struct.unpack_from('<f',MRCdata,40)[0]
        a, b, c = struct.unpack_from('<fff',MRCdata,40)
        side = a

        #header is 1024 bytes long. To read data, skip ahead to that point in the file
        fin.seek(1024, os.SEEK_SET)
        rho = np.fromfile(file=fin, dtype=np.dtype(np.float32)).reshape((nx,ny,nz),order='F')
        fin.close()
    if float64:
        rho = rho.astype(np.float64)
    if returnABC:
        return rho, (a,b,c)
    else:
        return rho, side

def write_xplor(rho,side,filename="map.xplor"):
    """Write an XPLOR formatted electron density map."""
    xs, ys, zs = rho.shape
    title_lines = ['REMARK FILENAME="'+filename+'"','REMARK DATE= '+str(datetime.datetime.today())]
    with open(filename,'w') as f:
        f.write("\n")
        f.write("%8d !NTITLE\n" % len(title_lines))
        for line in title_lines:
            f.write("%-264s\n" % line)
        #f.write("%8d%8d%8d%8d%8d%8d%8d%8d%8d\n" % (xs,0,xs-1,ys,0,ys-1,zs,0,zs-1))
        f.write("%8d%8d%8d%8d%8d%8d%8d%8d%8d\n" % (xs,-xs/2+1,xs/2,ys,-ys/2+1,ys/2,zs,-zs/2+1,zs/2))
        f.write("% -.5E% -.5E% -.5E% -.5E% -.5E% -.5E\n" % (side,side,side,90,90,90))
        f.write("ZYX\n")
        for k in range(zs):
            f.write("%8s\n" % k)
            for j in range(ys):
                for i in range(xs):
                    if (i+j*ys) % 6 == 5:
                        f.write("% -.5E\n" % rho[i,j,k])
                    else:
                        f.write("% -.5E" % rho[i,j,k])
            f.write("\n")
        f.write("    -9999\n")
        f.write("  %.4E  %.4E" % (np.average(rho), np.std(rho)))

def pad_rho(rho,newshape):
    """Pad rho with zeros to achieve new shape"""
    a = rho
    a_nx,a_ny,a_nz = a.shape
    b_nx,b_ny,b_nz = newshape
    padx1 = (b_nx-a_nx)//2
    padx2 = (b_nx-a_nx) - padx1
    pady1 = (b_ny-a_ny)//2
    pady2 = (b_ny-a_ny) - pady1
    padz1 = (b_nz-a_nz)//2
    padz2 = (b_nz-a_nz) - padz1
    #np.pad cannot take negative values, i.e. where the array will be cropped
    #however, can instead just use slicing to do the equivalent
    #but first need to identify which pad values are negative
    slcx1, slcx2, slcy1, slcy2, slcz1, slcz2 = None, None, None, None, None, None
    if padx1 < 0:
        slcx1 = -padx1
        padx1 = 0
    if padx2 < 0:
        slcx2 = padx2
        padx2 = 0
    if pady1 < 0:
        slcy1 = -pady1
        pady1 = 0
    if pady2 < 0:
        slcy2 = pady2
        pady2 = 0
    if padz1 < 0:
        slcz1 = -padz1
        padz1 = 0
    if padz2 < 0:
        slcz2 = padz2
        padz2 = 0
    a = np.pad(a,((padx1,padx2),(pady1,pady2),(padz1,padz2)),'constant')[
        slcx1:slcx2, slcy1:slcy2, slcz1:slcz2]
    return a

def zoom_rho(rho,vx,dx):
    """Resample rho to have new voxel size.

    rho - map to resample (3D array)
    vx - length of voxel of rho, float or tuple of three sides (a,b,c)
    dx - desired voxel size (only float allowed, assumes cubic grid desired)
    """
    vx = np.atleast_1d(vx)
    if len(vx) == 1:
        vx, vy, vz = vx, vx, vx
    elif len(vx) == 3:
        vx, vy, vz = vx
    else:
        print("Error. Argument 'vx' must be float or 3-tuple")
    dx = np.atleast_1d(dx)
    if len(dx) == 1:
        dx, dy, dz = dx, dx, dx
    elif len(dx) == 3:
        dx, dy, dz = dx
    else:
        print("Error. Argument 'vx' must be float or 3-tuple")
    #zoom factors
    zx, zy, zz = vx/dx, vy/dy, vz/dz
    newrho = ndimage.zoom(rho,(zx, zy, zz),order=1,mode="wrap")

    return newrho

def _fit_by_least_squares(radial, vectors, nmin=None,nmax=None):
    # This function fits a set of linearly combined vectors to a radial profile,
    # using a least-squares-based approach. The fit only takes into account the
    # range of radial bins defined by the xmin and xmax arguments.
    if nmin is None:
        nmin = 0
    if nmax is None:
        nmax = len(radial)
    a = np.nan_to_num(np.atleast_2d(vectors).T)
    b = np.nan_to_num(radial)
    a = a[nmin:nmax]
    b = b[nmin:nmax]
    coefficients, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    # coefficients, _ = optimize.nnls(a, b)
    return coefficients

def running_mean(x, N):
    # return ndimage.uniform_filter1d(x, N, mode='nearest', origin=-(N//2))[:-(N-1)]
    return np.convolve(x, np.ones(N)/N, mode='same')

def loadOutFile(filename):
    """Loads a GNOM .out file and returns q, Ireg, sqrt(Ireg), and a
    dictionary of miscellaneous results from GNOM. Taken from the BioXTAS
    RAW software package, used with permission under the GPL license."""

    five_col_fit = re.compile('\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*$')
    three_col_fit = re.compile('\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*$')
    two_col_fit = re.compile('\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s*$')

    results_fit = re.compile('\s*Current\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s*\d*[.]?\d*[+eE-]*\d*\s*$')

    te_fit = re.compile('\s*Total\s+[Ee]stimate\s*:\s+\d*[.]\d+\s*\(?[A-Za-z\s]+\)?\s*$')
    te_num_fit = re.compile('\d*[.]\d+')
    te_quality_fit = re.compile('[Aa][A-Za-z\s]+\)?\s*$')

    p_rg_fit = re.compile('\s*Real\s+space\s*\:?\s*Rg\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*\+-\s*\d*[.]\d+[+eE-]*\d*')
    q_rg_fit = re.compile('\s*Reciprocal\s+space\s*\:?\s*Rg\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*')

    p_i0_fit = re.compile('\s*Real\s+space\s*\:?[A-Za-z0-9\s\.,+-=]*\(0\)\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*\+-\s*\d*[.]\d+[+eE-]*\d*')
    q_i0_fit = re.compile('\s*Reciprocal\s+space\s*\:?[A-Za-z0-9\s\.,+-=]*\(0\)\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*')

    qfull = []
    qshort = []
    Jexp = []
    Jerr  = []
    Jreg = []
    Ireg = []

    R = []
    P = []
    Perr = []

    outfile = []

    #In case it returns NaN for either value, and they don't get picked up in the regular expression
    q_rg=None         #Reciprocal space Rg
    q_i0=None         #Reciprocal space I0


    with open(filename) as f:
        for line in f:
            twocol_match = two_col_fit.match(line)
            threecol_match = three_col_fit.match(line)
            fivecol_match = five_col_fit.match(line)
            results_match = results_fit.match(line)
            te_match = te_fit.match(line)
            p_rg_match = p_rg_fit.match(line)
            q_rg_match = q_rg_fit.match(line)
            p_i0_match = p_i0_fit.match(line)
            q_i0_match = q_i0_fit.match(line)

            outfile.append(line)

            if twocol_match:
                # print line
                found = twocol_match.group().split()

                qfull.append(float(found[0]))
                Ireg.append(float(found[1]))

            elif threecol_match:
                #print line
                found = threecol_match.group().split()

                R.append(float(found[0]))
                P.append(float(found[1]))
                Perr.append(float(found[2]))

            elif fivecol_match:
                #print line
                found = fivecol_match.group().split()

                qfull.append(float(found[0]))
                qshort.append(float(found[0]))
                Jexp.append(float(found[1]))
                Jerr.append(float(found[2]))
                Jreg.append(float(found[3]))
                Ireg.append(float(found[4]))

            elif results_match:
                found = results_match.group().split()
                Actual_DISCRP = float(found[1])
                Actual_OSCILL = float(found[2])
                Actual_STABIL = float(found[3])
                Actual_SYSDEV = float(found[4])
                Actual_POSITV = float(found[5])
                Actual_VALCEN = float(found[6])

                if len(found) == 8:
                    Actual_SMOOTH = float(found[7])
                else:
                    Actual_SMOOTH = -1

            elif te_match:
                te_num_search = te_num_fit.search(line)
                te_quality_search = te_quality_fit.search(line)

                TE_out = float(te_num_search.group().strip())
                quality = te_quality_search.group().strip().rstrip(')').strip()


            if p_rg_match:
                found = p_rg_match.group().split()
                try:
                    rg = float(found[-3])
                except:
                    rg = float(found[-2])
                try:
                    rger = float(found[-1])
                except:
                    rger = float(found[-1].strip('+-'))

            elif q_rg_match:
                found = q_rg_match.group().split()
                q_rg = float(found[-1])


            if p_i0_match:
                found = p_i0_match.group().split()
                i0 = float(found[-3])
                i0er = float(found[-1])

            elif q_i0_match:
                found = q_i0_match.group().split()
                q_i0 = float(found[-1])

    name = os.path.basename(filename)

    chisq = np.sum(np.square(np.array(Jexp)-np.array(Jreg))/np.square(Jerr))/(len(Jexp)-1) #DOF normalied chi squared

    results = { 'dmax'      : R[-1],        #Dmax
                'TE'        : TE_out,       #Total estimate
                'rg'        : rg,           #Real space Rg
                'rger'      : rger,         #Real space rg error
                'i0'        : i0,           #Real space I0
                'i0er'      : i0er,         #Real space I0 error
                'q_rg'      : q_rg,         #Reciprocal space Rg
                'q_i0'      : q_i0,         #Reciprocal space I0
                'quality'   : quality,      #Quality of GNOM out file
                'discrp'    : Actual_DISCRP,#DISCRIP, kind of chi squared (normalized by number of points, with a regularization parameter thing thrown in)
                'oscil'     : Actual_OSCILL,#Oscillation of solution
                'stabil'    : Actual_STABIL,#Stability of solution
                'sysdev'    : Actual_SYSDEV,#Systematic deviation of solution
                'positv'    : Actual_POSITV,#Relative norm of the positive part of P(r)
                'valcen'    : Actual_VALCEN,#Validity of the chosen interval in real space
                'smooth'    : Actual_SMOOTH,#Smoothness of the chosen interval? -1 indicates no real value, for versions of GNOM < 5.0 (ATSAS <2.8)
                'filename'  : name,         #GNOM filename
                'algorithm' : 'GNOM',       #Lets us know what algorithm was used to find the IFT
                'chisq'     : chisq         #Actual chi squared value
                    }

    #Jreg and Jerr are the raw data on the qfull axis
    Jerr = np.array(Jerr)
    prepend = np.zeros((len(Ireg)-len(Jerr)))
    prepend += np.mean(Jerr[:10])
    Jerr = np.concatenate((prepend,Jerr))
    Jreg = np.array(Jreg)
    Jreg = np.concatenate((prepend*0,Jreg))
    Jexp = np.array(Jexp)
    Jexp = np.concatenate((prepend*0,Jexp))

    return np.array(qfull), Jexp, Jerr, np.array(Ireg), results

def loadDatFile(filename):
    ''' Loads a Primus .dat format file. Taken from the BioXTAS RAW software package,
    used with permission under the GPL license.'''

    iq_pattern = re.compile('\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*')

    i = []
    q = []
    err = []

    with open(filename) as f:
        lines = f.readlines()

    comment = ''
    line = lines[0]
    j=0
    while line.split() and line.split()[0].strip()[0] == '#':
        comment = comment+line
        j = j+1
        line = lines[j]

    fileHeader = {'comment':comment}
    parameters = {'filename' : os.path.split(filename)[1],
                  'counters' : fileHeader}

    if comment.find('model_intensity') > -1:
        #FoXS file with a fit! has four data columns
        is_foxs_fit=True
        imodel = []
    else:
        is_foxs_fit = False

    for line in lines:
        iq_match = iq_pattern.match(line)

        if iq_match:
            if not is_foxs_fit:
                found = iq_match.group().split()
                q.append(float(found[0]))
                i.append(float(found[1]))
                err.append(float(found[2]))
            else:
                found = line.split()
                q.append(float(found[0]))
                i.append(float(found[1]))
                imodel.append(float(found[2]))
                err.append(float(found[3]))

    i = np.array(i)
    q = np.array(q)
    err = np.array(err)

    if is_foxs_fit:
        i = np.array(imodel)

    #Check to see if there is any header from RAW, and if so get that.
    header = []
    for j in range(len(lines)):
        if '### HEADER:' in lines[j]:
            header = lines[j+1:]

    hdict = None
    results = {}

    if len(header)>0:
        hdr_str = ''
        for each_line in header:
            hdr_str=hdr_str+each_line
        try:
            hdict = dict(json.loads(hdr_str))
        except Exception:
            hdict = {}

    if hdict:
        for each in hdict.keys():
            if each != 'filename':
                results[each] = hdict[each]

    if 'analysis' in results:
        if 'GNOM' in results['analysis']:
            results = results['analysis']['GNOM']

    return q, i, err, i, results

def loadFitFile(filename):
    ''' Loads a four column .fit format file (q, I, err, fit). Taken from the BioXTAS RAW software package,
    used with permission under the GPL license.'''

    iq_pattern = re.compile('\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*')

    i = []
    q = []
    err = []

    with open(filename) as f:
        lines = f.readlines()

    comment = ''
    line = lines[0]
    j=0
    while line.split() and line.split()[0].strip()[0] == '#':
        comment = comment+line
        j = j+1
        line = lines[j]

    fileHeader = {'comment':comment}
    parameters = {'filename' : os.path.split(filename)[1],
                  'counters' : fileHeader}

    imodel = []

    for line in lines:
        iq_match = iq_pattern.match(line)

        if iq_match:
            found = line.split()
            q.append(float(found[0]))
            i.append(float(found[1]))
            err.append(float(found[2]))
            imodel.append(float(found[3]))

    i = np.array(i)
    q = np.array(q)
    err = np.array(err)
    ifit = np.array(imodel)

    #grab some header info if available
    header = []
    for j in range(len(lines)):
        #If this is a _fit.dat or .fit file from DENSS, grab the header values beginning with hashtag #.
        if '# Parameter Values:' in lines[j]:
            header = lines[j+1:j+9]

    hdict = None
    results = {}

    if len(header)>0:
        hdr_str = '{'
        for each_line in header:
            line = each_line.split()
            hdr_str=hdr_str + "\""+line[1]+"\""+":"+line[3]+","
        hdr_str = hdr_str.rstrip(',')+"}"
        hdr_str = re.sub(r'\bnan\b', 'NaN', hdr_str)
        try:
            hdict = dict(json.loads(hdr_str))
        except Exception:
            hdict = {}

    if hdict:
        for each in hdict.keys():
            if each != 'filename':
                results[each] = hdict[each]

    if 'analysis' in results:
        if 'GNOM' in results['analysis']:
            results = results['analysis']['GNOM']

    return q, i, err, ifit, results

def loadOldFitFile(filename):
    ''' Loads a old denss _fit.dat format file. Taken from the BioXTAS RAW software package,
    used with permission under the GPL license.'''

    iq_pattern = re.compile('\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*')

    i = []
    q = []
    err = []

    with open(filename) as f:
        lines = f.readlines()

    comment = ''
    line = lines[0]
    j=0
    while line.split() and line.split()[0].strip()[0] == '#':
        comment = comment+line
        j = j+1
        line = lines[j]

    fileHeader = {'comment':comment}
    parameters = {'filename' : os.path.split(filename)[1],
                  'counters' : fileHeader}

    for line in lines:
        iq_match = iq_pattern.match(line)

        if iq_match:
            found = iq_match.group().split()
            q.append(float(found[0]))
            i.append(float(found[1]))
            err.append(float(found[2]))

    i = np.array(i)
    q = np.array(q)
    err = np.array(err)

    #If this is a _fit.dat file from DENSS, grab the header values.
    header = []
    for j in range(len(lines)):
        if '# Parameter Values:' in lines[j]:
            header = lines[j+1:j+9]

    hdict = None
    results = {}

    if len(header)>0:
        hdr_str = '{'
        for each_line in header:
            line = each_line.split()
            hdr_str=hdr_str + "\""+line[1]+"\""+":"+line[3]+","
        hdr_str = hdr_str.rstrip(',')+"}"
        try:
            hdict = dict(json.loads(hdr_str))
        except Exception:
            hdict = {}

    if hdict:
        for each in hdict.keys():
            if each != 'filename':
                results[each] = hdict[each]

    if 'analysis' in results:
        if 'GNOM' in results['analysis']:
            results = results['analysis']['GNOM']

    #just return i for ifit to be consistent with other functions' output
    return q, i, err, i, results

def loadProfile(fname, units="a"):
    """Determines which loading function to run, and then runs it."""

    if os.path.splitext(fname)[1] == '.out':
        q, I, Ierr, Ifit, results = loadOutFile(fname)
        isfit = True
    elif os.path.splitext(fname)[1] == '.fit':
        q, I, Ierr, Ifit, results = loadFitFile(fname)
        isfit = True
    elif "_fit.dat" in fname:
        q, I, Ierr, Ifit, results = loadOldFitFile(fname)
        isfit = True
    else:
        #Ifit here is just I, since it's just a data file
        q, I, Ierr, Ifit, results = loadDatFile(fname)
        isfit = False

    #keys = {key.lower().strip().translate(str.maketrans('','', '_ ')): key for key in list(results.keys())}
    keys = {key.lower().strip(): key for key in list(results.keys())}

    if 'dmax' in keys:
        dmax = float(results[keys['dmax']])
    else:
        dmax = -1.

    if units == "nm":
        #DENSS assumes 1/angstrom, so convert from 1/nm to 1/angstrom
        q /= 10
        if denss != -1:
            dmax *= 10
        print("Angular units converted from 1/nm to 1/angstrom")

    return q, I, Ierr, Ifit, dmax, isfit

def check_if_raw_data(Iq):
    """Check if an I(q) profile is a smooth fitted profile, or raw data.

    Iq - N x 3 numpy array, where N is the number of data  points, and the
        three columns are q, I, error.

    This performs a very simple check. It simply checks if there exists
    a q = 0 term. The Iq profile given should be first cleaned up by
    clean_up_data() function, which will remove I=0 and sig=0 data points.
    """
    #first, check if there is a q=0 value.
    if min(Iq[:,0]) > 1e-8:
        #allow for some floating point error
        return True
    else:
        return False

def clean_up_data(Iq):
    """Do a quick cleanup by removing zero intensities and zero errors.

    Iq - N x 3 numpy array, where N is the number of data  points, and the
    three columns are q, I, error.
    """
    return Iq[(~np.isclose(Iq[:,1],0))&(~np.isclose(Iq[:,2],0))]

def calc_rg_I0_by_guinier(Iq,nb=None,ne=None):
    """calculate Rg, I(0) by fitting Guinier equation to data.
    Use only desired q range in input arrays."""
    if nb is None:
        nb = 0
    if ne is None:
        ne = Iq.shape[0]
    while True:
        m, b = stats.linregress(Iq[nb:ne,0]**2,np.log(Iq[nb:ne,1]))[:2]
        if m < 0.0: 
            break
        else:
            #the slope should be negative
            #if the slope is positive, shift the 
            #region forward by one point and try again
            nb += 1
            ne += 1
            if nb>50:
                raise ValueError("Guinier estimation failed. Guinier region slope is positive.")
    rg = (-3*m)**(0.5)
    I0 = np.exp(b)
    return rg, I0

def calc_rg_by_guinier_first_2_points(q, I, DENSS_GPU=False):
    """calculate Rg using Guinier law, but only use the 
    first two data points. This is meant to be used with a 
    calculated scattering profile, such as Imean from denss()."""
    m = (mylog(I[1],DENSS_GPU)-mylog(I[0],DENSS_GPU))/(q[1]**2-q[0]**2)
    rg = (-3*m)**(0.5)
    return rg

def calc_rg_by_guinier_peak(Iq,exp=1,nb=0,ne=None):
    """roughly estimate Rg using the Guinier peak method.
    Use only desired q range in input arrays.
    exp - the exponent in q^exp * I(q)"""
    d = exp
    if ne is None:
        ne = Iq.shape[0]
    q = Iq[:,0] #[nb:ne,0]
    I = Iq[:,1] #[nb:ne,1]
    qdI = q**d * I
    try:
        #fit a quick quadratic for smoothness, ax^2 + bx + c
        a,b,c = np.polyfit(q,qdI,2)
        #get the peak position
        qpeak = -b/(2*a) 
    except:
        #if polyfit fails, just grab the maximum position
        qpeaki = np.argmax(qdI)
        qpeak = q[qpeaki]
    #calculate Rg from the peak position
    rg = (3.*d/2.)**0.5 / qpeak
    return rg

def estimate_dmax(Iq,dmax=None,clean_up=True):
    """Attempt to roughly estimate Dmax directly from data."""
    #first, clean up the data
    if clean_up:
        Iq = clean_up_data(Iq)
    q = Iq[:,0]
    I = Iq[:,1]
    if dmax is None:
        #first, estimate a very rough rg from the first 20 data points
        nmax = 20
        try:
            rg, I0 = calc_rg_I0_by_guinier(Iq,ne=nmax)
        except:
            rg = calc_rg_by_guinier_peak(Iq,exp=1,ne=100)
        #next, dmax is roughly 3.5*rg for most particles
        #so calculate P(r) using a larger dmax, say twice as large, so 7*rg
        D = 7*rg
    else:
        #allow user to give an initial estimate of Dmax
        #multiply by 2 to allow for enough large r values
        D = 2*dmax
    #create a calculated q range for Sasrec for low q out to q=0
    qmin = np.min(q)
    dq = (q.max()-q.min())/(q.size-1)
    nq = int(qmin/dq)
    qc = np.concatenate(([0.0],np.arange(nq)*dq+(qmin-nq*dq),q))
    #run Sasrec to perform IFT
    sasrec = Sasrec(Iq, D, qc=None, alpha=0.0, extrapolate=False)
    #now filter the P(r) curve for estimating Dmax better
    r, Pfilt, sigrfilt = filter_P(sasrec.r, sasrec.P, sasrec.Perr, qmax=Iq[:,0].max())
    #estimate D as the first position where P becomes less than 0.01*P.max(), after P.max()
    Pargmax = Pfilt.argmax()
    #catch cases where the P(r) plot goes largely negative at large r values,
    #as this indicates repulsion. Set the new Pargmax, which is really just an
    #identifier for where to begin searching for Dmax, to be any P value whose
    #absolute value is greater than at least 10% of Pfilt.max. The large 10% is to 
    #avoid issues with oscillations in P(r).
    above_idx = np.where((np.abs(Pfilt)>0.1*Pfilt.max())&(r>r[Pargmax]))
    Pargmax = np.max(above_idx)
    near_zero_idx = np.where((np.abs(Pfilt[Pargmax:])<(0.001*Pfilt.max())))[0]
    near_zero_idx += Pargmax
    D_idx = near_zero_idx[0]
    D = r[D_idx]
    sasrec.D = D
    sasrec.update()
    return D, sasrec

def filter_P(r,P,sigr=None,qmax=0.5,cutoff=0.75,qmin=0.0,cutoffmin=1.25):
    """Filter P(r) and sigr of oscillations."""
    npts = len(r)
    dr = (r.max()-r.min())/(r.size-1)
    fs = 1./dr
    nyq = fs*0.5
    fc = (cutoff*qmax/(2*np.pi))/nyq
    ntaps = npts//3
    if ntaps%2==0:
        ntaps -=1
    b = signal.firwin(ntaps, fc, window='hann')
    if qmin>0.0:
        fcmin = (cutoffmin*qmin/(2*np.pi))/nyq
        b = signal.firwin(ntaps, [fcmin,fc],pass_zero=False, window='hann')
    a = np.array([1])
    import warnings
    with warnings.catch_warnings():
        #theres a warning from filtfilt that is a bug in older scipy versions
        #we are just going to suppress that here.
        warnings.filterwarnings("ignore")
        Pfilt = signal.filtfilt(tuple(b),tuple(a),tuple(P),padlen=len(r)-1)
        r = np.arange(npts)/fs
        if sigr is not None:
            sigrfilt = signal.filtfilt(b, a, sigr,padlen=len(r)-1)/(2*np.pi)
            return r, Pfilt, sigrfilt
        else:
            return r, Pfilt

def denss(q, I, sigq, dmax, ne=None, voxel=5., oversampling=3., recenter=True, recenter_steps=None,
    recenter_mode="com", positivity=True, positivity_steps=None, extrapolate=True, output="map",
    steps=None, seed=None, rho_start=None, support_start=None, add_noise=None,
    shrinkwrap=True, shrinkwrap_old_method=False,shrinkwrap_sigma_start=3,
    shrinkwrap_sigma_end=1.5, shrinkwrap_sigma_decay=0.99, shrinkwrap_threshold_fraction=0.2,
    shrinkwrap_iter=20, shrinkwrap_minstep=100, chi_end_fraction=0.01,
    write_xplor_format=False, write_freq=100, enforce_connectivity=True,
    enforce_connectivity_steps=[500], enforce_connectivity_max_features=1, cutout=True, quiet=False, ncs=0,
    ncs_steps=[500],ncs_axis=1, ncs_type="cyclical",abort_event=None, my_logger=logging.getLogger(),
    path='.', gui=False, DENSS_GPU=False):
    """Calculate electron density from scattering data."""
    if abort_event is not None:
        if abort_event.is_set():
            my_logger.info('Aborted!')
            return []

    if DENSS_GPU and CUPY_LOADED:
        DENSS_GPU = True
    elif DENSS_GPU:
        if gui:
            my_logger.info("GPU option set, but CuPy failed to load")
        else:
            print("GPU option set, but CuPy failed to load")
        DENSS_GPU = False

    fprefix = os.path.join(path, output)

    D = dmax

    #Initialize variables

    side = oversampling*D
    halfside = side/2

    n = int(side/voxel)
    #want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n%2==1:
        n += 1
    #store n for later use if needed
    nbox = n

    dx = side/n
    dV = dx**3
    V = side**3
    x_ = np.linspace(-halfside,halfside,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)

    df = 1/side
    qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
    qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0]) - 1e-8 #subtract a tiny bit to deal with floating point error
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)

    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)

    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = mybinmean(qr.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)

    #allow for any range of q data
    qdata = qbinsc[np.where( (qbinsc>=q.min()) & (qbinsc<=q.max()) )]
    Idata = np.interp(qdata,q,I)

    if extrapolate:
        qextend = qbinsc[qbinsc>=qdata.max()]
        Iextend = qextend**-4
        Iextend = Iextend/Iextend[0] * Idata[-1]
        qdata = np.concatenate((qdata,qextend[1:]))
        Idata = np.concatenate((Idata,Iextend[1:]))

    #create list of qbin indices just in region of data for later F scaling
    qbin_args = np.in1d(qbinsc,qdata,assume_unique=True)
    qba = qbin_args #just for brevity when using it later
    #set qba bins outside of scaling region to false.
    #start with bins in corners
    # qba[qbinsc>qx_.max()] = False

    sigqdata = np.interp(qdata,q,sigq)

    scale_factor = ne**2 / Idata[0]
    Idata *= scale_factor
    sigqdata *= scale_factor
    I *= scale_factor
    sigq *= scale_factor

    if steps == 'None' or steps is None or np.int(steps) < 1:
        stepsarr = np.concatenate((enforce_connectivity_steps,[shrinkwrap_minstep]))
        maxec = np.max(stepsarr)
        steps = int(shrinkwrap_iter * (np.log(shrinkwrap_sigma_end/shrinkwrap_sigma_start)/np.log(shrinkwrap_sigma_decay)) + maxec)
        #add enough steps for convergence after shrinkwrap is finished
        #something like 7000 seems reasonable, likely will finish before that on its own
        #then just make a round number when using defaults
        steps += 7621
    else:
        steps = np.int(steps)

    Imean = np.zeros((len(qbins)))
    chi = np.zeros((steps+1))
    rg = np.zeros((steps+1))
    supportV = np.zeros((steps+1))
    if support_start is not None:
        support = np.copy(support_start)
    else:
        support = np.ones(x.shape,dtype=bool)

    if seed is None:
        #Have to reset the random seed to get a random in different from other processes
        prng = np.random.RandomState()
        seed = prng.randint(2**31-1)
    else:
        seed = int(seed)

    prng = np.random.RandomState(seed)

    if rho_start is not None:
        rho_start *= dV
        rho = np.copy(rho_start)
        if add_noise is not None:
            noise_factor = rho.max() * add_noise
            noise = prng.random_sample(size=x.shape)*noise_factor
            rho += noise
    else:
        rho = prng.random_sample(size=x.shape) #- 0.5
    newrho = np.zeros_like(rho)

    sigma = shrinkwrap_sigma_start

    #calculate the starting shrinkwrap volume as the volume of a sphere
    #of radius Dmax, i.e. much larger than the particle size
    swbyvol = True
    swV = V/2.0
    Vsphere_Dover2 = 4./3 * np.pi * (D/2.)**3
    swVend = Vsphere_Dover2
    swV_decay = 0.9
    first_time_swdensity = True
    threshold = shrinkwrap_threshold_fraction
    #erode will make take five outer edge pixels of the support, like a shell,
    #and will make sure no negative density is in that region
    #this is to counter an artifact that occurs when allowing for negative density
    #as the negative density often appears immediately next to positive density
    #at the edges of the object. This ensures (i.e. biases) only real negative density
    #in the interior of the object (i.e. more than five pixels from the support boundary)
    #thus we only need this on when in membrane mode, i.e. when positivity=False
    if shrinkwrap_old_method or positivity:
        erode = False
    else:
        erode = True
        erosion_width = 5

    my_logger.info('q range of input data: %3.3f < q < %3.3f', q.min(), q.max())
    my_logger.info('Maximum dimension: %3.3f', D)
    my_logger.info('Sampling ratio: %3.3f', oversampling)
    my_logger.info('Requested real space voxel size: %3.3f', voxel)
    my_logger.info('Number of electrons: %3.3f', ne)
    my_logger.info('Recenter: %s', recenter)
    my_logger.info('Recenter Steps: %s', recenter_steps)
    my_logger.info('Recenter Mode: %s', recenter_mode)
    my_logger.info('NCS: %s', ncs)
    my_logger.info('NCS Steps: %s', ncs_steps)
    my_logger.info('NCS Axis: %s', ncs_axis)
    my_logger.info('Positivity: %s', positivity)
    my_logger.info('Positivity Steps: %s', positivity_steps)
    my_logger.info('Extrapolate high q: %s', extrapolate)
    my_logger.info('Shrinkwrap: %s', shrinkwrap)
    my_logger.info('Shrinkwrap Old Method: %s', shrinkwrap_old_method)
    my_logger.info('Shrinkwrap sigma start (angstroms): %s', shrinkwrap_sigma_start*dx)
    my_logger.info('Shrinkwrap sigma end (angstroms): %s', shrinkwrap_sigma_end*dx)
    my_logger.info('Shrinkwrap sigma start (voxels): %s', shrinkwrap_sigma_start)
    my_logger.info('Shrinkwrap sigma end (voxels): %s', shrinkwrap_sigma_end)
    my_logger.info('Shrinkwrap sigma decay: %s', shrinkwrap_sigma_decay)
    my_logger.info('Shrinkwrap threshold fraction: %s', shrinkwrap_threshold_fraction)
    my_logger.info('Shrinkwrap iterations: %s', shrinkwrap_iter)
    my_logger.info('Shrinkwrap starting step: %s', shrinkwrap_minstep)
    my_logger.info('Enforce connectivity: %s', enforce_connectivity)
    my_logger.info('Enforce connectivity steps: %s', enforce_connectivity_steps)
    my_logger.info('Chi2 end fraction: %3.3e', chi_end_fraction)
    my_logger.info('Maximum number of steps: %i', steps)
    my_logger.info('Grid size (voxels): %i x %i x %i', n, n, n)
    my_logger.info('Real space box width (angstroms): %3.3f', side)
    my_logger.info('Real space box range (angstroms): %3.3f < x < %3.3f', x_.min(), x_.max())
    my_logger.info('Real space box volume (angstroms^3): %3.3f', V)
    my_logger.info('Real space voxel size (angstroms): %3.3f', dx)
    my_logger.info('Real space voxel volume (angstroms^3): %3.3f', dV)
    my_logger.info('Reciprocal space box width (angstroms^(-1)): %3.3f', qx_.max()-qx_.min())
    my_logger.info('Reciprocal space box range (angstroms^(-1)): %3.3f < qx < %3.3f', qx_.min(), qx_.max())
    my_logger.info('Maximum q vector (diagonal) (angstroms^(-1)): %3.3f', qr.max())
    my_logger.info('Number of q shells: %i', nbins)
    my_logger.info('Width of q shells (angstroms^(-1)): %3.3f', qstep)
    my_logger.info('Random seed: %i', seed)

    if not quiet:
        if gui:
            my_logger.info("\n Step     Chi2     Rg    Support Volume")
            my_logger.info(" ----- --------- ------- --------------")
        else:
            print("\n Step     Chi2     Rg    Support Volume")
            print(" ----- --------- ------- --------------")

    if PYFFTW:
        a = np.copy(rho)
        rho = pyfftw.empty_aligned(a.shape, dtype='complex64')
        rho[:] = a
        rhoprime = pyfftw.empty_aligned(a.shape, dtype='complex64')
        newrho = pyfftw.empty_aligned(a.shape, dtype='complex64')
        try:
            # Try to plan our transforms with the wisdom we have already
            fftw_object = pyfftw.FFTW(rho,
                                      rhoprime,
                                      direction="FFTW_FORWARD",
                                      flags=("FFTW_WISDOM_ONLY",))
        except RuntimeError as e:
            # If we don't have enough wisdom, print a warning and proceed.
            print(e)
            start = time.perf_counter()
            fftw_object = pyfftw.FFTW(rho,
                                      rhoprime,
                                      direction="FFTW_FORWARD",
                                      flags=("FFTW_MEASURE",))
            print("Generating wisdom took {}s".format(time.perf_counter() - start))
            with open("fft.wisdom", "wb") as the_file:
                wisdom = pyfftw.export_wisdom()
                pickle.dump(wisdom, the_file)

    if DENSS_GPU:
        rho = cp.array(rho)
        qbin_labels = cp.array(qbin_labels)
        qbins = cp.array(qbins)
        Idata = cp.array(Idata)
        qbin_args = cp.array(qbin_args)
        sigqdata = cp.array(sigqdata)
        support = cp.array(support)
        chi = cp.array(chi)
        supportV = cp.array(supportV)
        Imean = cp.array(Imean)
        newrho = cp.array(newrho)
        qblravel = cp.array(qblravel)
        xcount = cp.array(xcount)

    for j in range(steps):
        if abort_event is not None:
            if abort_event.is_set():
                my_logger.info('Aborted!')
                return []

        F = myfftn(rho, DENSS_GPU=DENSS_GPU)

        #sometimes, when using denss.refine.py with non-random starting rho,
        #the resulting Fs result in zeros in some locations and the algorithm to break
        #here just make those values to be 1e-16 to be non-zero
        F[np.abs(F)==0] = 1e-16

        #APPLY RECIPROCAL SPACE RESTRAINTS
        #calculate spherical average of intensities from 3D Fs
        #for some reason, sometimes this fails
        try:
            I3D = abs2(F)
        except:
            I3D = myabs(F,DENSS_GPU=DENSS_GPU)**2
        Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=DENSS_GPU)

        #scale Fs to match data
        factors = mysqrt(Idata/Imean, DENSS_GPU=DENSS_GPU)
        #do not scale bins outside of desired range
        #so set those factors to 1.0
        factors[~qba] = 1.0
        F *= factors[qbin_labels]

        chi[j] = mysum(((Imean[qba]-Idata[qba])/sigqdata[qba])**2, DENSS_GPU=DENSS_GPU)/Idata[qba].size

        #APPLY REAL SPACE RESTRAINTS
        rhoprime = myifftn(F, DENSS_GPU=DENSS_GPU).real

        # use Guinier's law to approximate quickly
        rg[j] = calc_rg_by_guinier_first_2_points(qbinsc, Imean, DENSS_GPU=DENSS_GPU)

        #Error Reduction
        newrho *= 0
        newrho[support] = rhoprime[support]

        if not DENSS_GPU and j%write_freq == 0:
            if write_xplor_format:
                write_xplor(rhoprime/dV, side, fprefix+"_current.xplor")
            write_mrc(rhoprime/dV, side, fprefix+"_current.mrc")

        # enforce positivity by making all negative density points zero.
        if positivity and j in positivity_steps:
            newrho[newrho<0] = 0.0

        #apply non-crystallographic symmetry averaging
        if ncs != 0 and j in ncs_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
            newrho = align2xyz(newrho)
            if DENSS_GPU:
                newrho = cp.array(newrho)

        if ncs != 0 and j in [stepi+1 for stepi in ncs_steps]:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
            if ncs_axis == 1:
                axes=(1,2) #longest
                axes2=(0,1) #shortest
            if ncs_axis == 2:
                axes=(0,2) #middle
                axes2=(0,1) #shortest
            if ncs_axis == 3:
                axes=(0,1) #shortest
                axes2=(1,2) #longest
            degrees = 360./ncs
            newrho_total = np.copy(newrho)
            if ncs_type == "dihedral":
                #first, rotate original about perpendicular axis by 180
                #then apply n-fold cyclical rotation
                d2fold = ndimage.rotate(newrho,180,axes=axes2,reshape=False)
                newrhosym = np.copy(newrho) + d2fold
                newrhosym /= 2.0
                newrho_total = np.copy(newrhosym)
            else:
                newrhosym = np.copy(newrho)
            for nrot in range(1,ncs):
                sym = ndimage.rotate(newrhosym,degrees*nrot,axes=axes,reshape=False)
                newrho_total += np.copy(sym)
            newrho = newrho_total / ncs

            #run shrinkwrap after ncs averaging to get new support
            if shrinkwrap_old_method:
                #run the old method
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho,absv=absv,sigma=sigma,threshold=threshold,recenter=recenter,recenter_mode=recenter_mode)
            else:
                swN = int(swV/dV)
                #end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho,absv=True,sigma=sigma,N=swN,recenter=recenter,recenter_mode=recenter_mode)
                    swV *= swV_decay
                else:
                    threshold = shrinkwrap_threshold_fraction
                    if first_time_swdensity:
                        if not quiet:
                            if gui:
                                my_logger.info("switched to shrinkwrap by density threshold = %.4f" %threshold)
                            else:
                                print("\nswitched to shrinkwrap by density threshold = %.4f" %threshold)
                        first_time_swdensity = False
                    newrho, support = shrinkwrap_by_density_value(newrho,absv=True,sigma=sigma,threshold=threshold,recenter=recenter,recenter_mode=recenter_mode)


            if DENSS_GPU:
                newrho = cp.array(newrho)

        if recenter and j in recenter_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            #cannot run center_rho_roll() function since we want to also recenter the support
            #perhaps we should fix this in the future to clean it up
            if recenter_mode == "max":
                rhocom = np.unravel_index(newrho.argmax(), newrho.shape)
            else:
                rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(newrho)))
            gridcenter = (np.array(newrho.shape)-1.)/2.
            shift = gridcenter-rhocom
            shift = np.rint(shift).astype(int)
            newrho = np.roll(np.roll(np.roll(newrho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
            support = np.roll(np.roll(np.roll(support, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        #update support using shrinkwrap method
        if shrinkwrap and j >= shrinkwrap_minstep and j%shrinkwrap_iter==1:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            if shrinkwrap_old_method:
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho,absv=absv,sigma=sigma,threshold=threshold,recenter=recenter,recenter_mode=recenter_mode)
            else:
                swN = int(swV/dV)
                #end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho,absv=True,sigma=sigma,N=swN,recenter=recenter,recenter_mode=recenter_mode)
                    swV *= swV_decay
                else:
                    threshold = shrinkwrap_threshold_fraction
                    if first_time_swdensity:
                        if not quiet:
                            if gui:
                                my_logger.info("switched to shrinkwrap by density threshold = %.4f" %threshold)
                            else:
                                print("\nswitched to shrinkwrap by density threshold = %.4f" %threshold)
                        first_time_swdensity = False
                    newrho, support = shrinkwrap_by_density_value(newrho,absv=True,sigma=sigma,threshold=threshold,recenter=recenter,recenter_mode=recenter_mode)

            if sigma > shrinkwrap_sigma_end:
                sigma = shrinkwrap_sigma_decay*sigma

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        #run erode when shrinkwrap is run
        if erode and shrinkwrap and j > shrinkwrap_minstep and j%shrinkwrap_iter==1:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            #eroded is the region of the support _not_ including the boundary pixels
            #so it is the entire interior. erode_region is _just_ the boundary pixels
            eroded = ndimage.binary_erosion(support,np.ones((erosion_width,erosion_width,erosion_width)))
            #get just boundary voxels, i.e. where support=True and eroded=False
            erode_region = np.logical_and(support,~eroded)
            #set all negative density in boundary pixels to zero.
            newrho[(newrho<0)&(erode_region)] = 0

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        if enforce_connectivity and j in enforce_connectivity_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)

            #first run shrinkwrap to define the features
            if shrinkwrap_old_method:
                #run the old method
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho,absv=absv,sigma=sigma,threshold=threshold,recenter=recenter,recenter_mode=recenter_mode)
            else:
                #end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                swN = int(swV/dV)
                if swbyvol and swV>swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho,absv=True,sigma=sigma,N=swN,recenter=recenter,recenter_mode=recenter_mode)
                else:
                    newrho, support = shrinkwrap_by_density_value(newrho,absv=True,sigma=sigma,threshold=threshold,recenter=recenter,recenter_mode=recenter_mode)

            #label the support into separate segments based on a 3x3x3 grid
            struct = ndimage.generate_binary_structure(3, 3)
            labeled_support, num_features = ndimage.label(support, structure=struct)
            sums = np.zeros((num_features))
            num_features_to_keep = np.min([num_features,enforce_connectivity_max_features])
            if not quiet:
                if not gui:
                    print("EC: %d -> %d " % (num_features,num_features_to_keep))

            #find the feature with the greatest number of electrons
            for feature in range(num_features+1):
                sums[feature-1] = np.sum(newrho[labeled_support==feature])
            big_feature = np.argmax(sums)+1
            #order the indices of the features in descending order based on their sum/total density
            sums_order = np.argsort(sums)[::-1]
            sums_sorted = sums[sums_order]
            #now grab the actual feature numbers (rather than the indices)
            features_sorted = sums_order + 1

            #remove features from the support that are not the primary feature
            # support[labeled_support != big_feature] = False
            #reset support to zeros everywhere
            #then progressively add in regions of support up to num_features_to_keep
            support *= False
            for feature in range(num_features_to_keep):
                support[labeled_support == features_sorted[feature]] = True

            #clean up density based on new support
            newrho[~support] = 0

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        supportV[j] = mysum(support, DENSS_GPU=DENSS_GPU)*dV

        if not quiet:
            if gui:
                my_logger.info("% 5i % 4.2e % 3.2f       % 5i          ", j, chi[j], rg[j], supportV[j])
            else:
                sys.stdout.write("\r% 5i % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
                sys.stdout.flush()

        #occasionally report progress in logger
        if j%500==0 and not gui:
            my_logger.info('Step % 5i: % 4.2e % 3.2f       % 5i          ', j, chi[j], rg[j], supportV[j])


        if j > 101 + shrinkwrap_minstep:
            if DENSS_GPU:
                lesser = mystd(chi[j-100:j], DENSS_GPU=DENSS_GPU).get() < chi_end_fraction * mymean(chi[j-100:j], DENSS_GPU=DENSS_GPU).get()
            else:
                lesser = mystd(chi[j-100:j], DENSS_GPU=DENSS_GPU) < chi_end_fraction * mymean(chi[j-100:j], DENSS_GPU=DENSS_GPU)
            if lesser:
                break

        rho = newrho

    #convert back to numpy outside of for loop
    if DENSS_GPU:
        rho = cp.asnumpy(rho)
        qbin_labels = cp.asnumpy(qbin_labels)
        qbin_args = cp.asnumpy(qbin_args)
        sigqdata = cp.asnumpy(sigqdata)
        Imean = cp.asnumpy(Imean)
        chi = cp.asnumpy(chi)
        qbins = cp.asnumpy(qbins)
        Idata = cp.asnumpy(Idata)
        support = cp.asnumpy(support)
        supportV = cp.asnumpy(supportV)
        Idata = cp.asnumpy(Idata)
        newrho = cp.asnumpy(newrho)
        qblravel = cp.asnumpy(qblravel)
        xcount = cp.asnumpy(xcount)

    F = myfftn(rho)
    #calculate spherical average intensity from 3D Fs
    I3D = abs2(F)
    Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)

    #scale Fs to match data
    factors = np.sqrt(Idata/Imean)
    factors[~qba] = 1.0
    F *= factors[qbin_labels]
    rho = myifftn(F).real

    #negative images yield the same scattering, so flip the image
    #to have more positive than negative values if necessary
    #to make sure averaging is done properly
    #whether theres actually more positive than negative values
    #is ambiguous, but this ensures all maps are at least likely
    #the same designation when averaging
    if np.sum(np.abs(rho[rho<0])) > np.sum(rho[rho>0]):
        rho *= -1

    #scale total number of electrons
    if ne is not None:
        rho *= ne / np.sum(rho)

    rg[j+1] = calc_rg_by_guinier_first_2_points(qbinsc, Imean)
    supportV[j+1] = supportV[j]

    #change rho to be the electron density in e-/angstroms^3, rather than number of electrons,
    #which is what the FFT assumes
    rho /= dV
    my_logger.info('FINISHED DENSITY REFINEMENT')


    if cutout:
        #here were going to cut rho out of the large real space box
        #to the voxels that contain the particle
        #use D to estimate particle size
        #assume the particle is in the center of the box
        #calculate how many voxels needed to contain particle of size D
        #use bigger than D to make sure we don't crop actual particle in case its larger than expected
        #lets clip it to a maximum of 2*D to be safe
        nD = int(2*D/dx)+1
        #make sure final box will still have even samples
        if nD%2==1:
            nD += 1

        nmin = nbox//2 - nD//2
        nmax = nbox//2 + nD//2 + 2
        #create new rho array containing only the particle
        newrho = rho[nmin:nmax,nmin:nmax,nmin:nmax]
        rho = newrho
        #do the same for the support
        newsupport = support[nmin:nmax,nmin:nmax,nmin:nmax]
        support = newsupport
        #update side to new size of box
        side = dx * (nmax-nmin)

    if write_xplor_format:
        write_xplor(rho,side,fprefix+".xplor")
        write_xplor(np.ones_like(rho)*support, side, fprefix+"_support.xplor")

    write_mrc(rho,side,fprefix+".mrc")
    write_mrc(np.ones_like(rho)*support,side, fprefix+"_support.mrc")

    #Write some more output files
    fit = np.zeros(( len(qbinsc),4 ))
    fit[:len(qdata),0] = qdata
    fit[:len(Idata),1] = Idata
    fit[:len(sigqdata),2] = sigqdata
    fit[:len(Imean),3] = Imean
    np.savetxt(fprefix+'_map.fit', fit, delimiter=' ', fmt='%.5e'.encode('ascii'),
        header='q(data),I(data),error(data),I(density)')

    np.savetxt(fprefix+'_stats_by_step.dat',np.vstack((chi, rg, supportV)).T,
        delimiter=" ", fmt="%.5e".encode('ascii'), header='Chi2 Rg SupportVolume')

    my_logger.info('Number of steps: %i', j)
    my_logger.info('Final Chi2: %.3e', chi[j])
    my_logger.info('Final Rg: %3.3f', rg[j+1])
    my_logger.info('Final Support Volume: %3.3f', supportV[j+1])
    my_logger.info('Mean Density (all voxels): %3.5f', np.mean(rho))
    my_logger.info('Std. Dev. of Density (all voxels): %3.5f', np.std(rho))
    my_logger.info('RMSD of Density (all voxels): %3.5f', np.sqrt(np.mean(np.square(rho))))
    idx = np.where(np.abs(rho)>0.01*rho.max())
    my_logger.info('Modified Mean Density (voxels >0.01*max): %3.5f', np.mean(rho[idx]))
    my_logger.info('Modified Std. Dev. of Density (voxels >0.01*max): %3.5f', np.std(rho[idx]))
    my_logger.info('Modified RMSD of Density (voxels >0.01*max): %3.5f', np.sqrt(np.mean(np.square(rho[idx]))))
    # my_logger.info('END')

    #return original unscaled values of Idata (and therefore Imean) for comparison with real data
    Idata /= scale_factor
    sigqdata /= scale_factor
    Imean /= scale_factor
    I /= scale_factor
    sigq /= scale_factor

    return qdata, Idata, sigqdata, qbinsc, Imean, chi, rg, supportV, rho, side

def shrinkwrap_by_density_value(rho,absv=True,sigma=3.0,threshold=0.2,recenter=True,recenter_mode="com"):
    """Create support using shrinkwrap method based on threshold as fraction of maximum density

    rho - electron density; numpy array
    absv - boolean, whether or not to take the absolute value of the density
    sigma - sigma, in pixels, for gaussian filter
    threshold - fraction of maximum gaussian filtered density (0 to 1)
    recenter - boolean, whether or not to recenter the density prior to calculating support
    recenter_mode - either com (center of mass) or max (maximum density value)
    """
    if recenter:
        rho = center_rho_roll(rho, recenter_mode)

    if absv:
        tmp = np.abs(rho)
    else:
        tmp = rho
    rho_blurred = ndimage.filters.gaussian_filter(tmp,sigma=sigma,mode='wrap')

    support = np.zeros(rho.shape,dtype=bool)
    support[rho_blurred >= threshold*rho_blurred.max()] = True

    return rho, support

def shrinkwrap_by_volume(rho,N,absv=True,sigma=3.0,recenter=True,recenter_mode="com"):
    """Create support using shrinkwrap method based on threshold as fraction of maximum density

    rho - electron density; numpy array
    absv - boolean, whether or not to take the absolute value of the density
    sigma - sigma, in pixels, for gaussian filter
    N - set the threshold such that N voxels are in the support (must precalculate this based on volume)
    recenter - boolean, whether or not to recenter the density prior to calculating support
    recenter_mode - either com (center of mass) or max (maximum density value)
    """
    if recenter:
        rho = center_rho_roll(rho, recenter_mode)

    if absv:
        tmp = np.abs(rho)
    else:
        tmp = rho
    rho_blurred = ndimage.filters.gaussian_filter(tmp,sigma=sigma,mode='wrap')

    #grab the N largest values of the array
    idx = largest_indices(rho_blurred, N)
    support = np.zeros(rho.shape,dtype=bool)
    support[idx] = True
    #now, calculate the threshold that would correspond to the by_density_value method
    threshold = np.min(rho_blurred[idx])/rho_blurred.max()

    return rho, support, threshold

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    n = x.size
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return np.vstack((vals, ecdf)).T

def find_nearest_i(array,value):
    """Return the index of the array item nearest to specified value"""
    return (np.abs(array-value)).argmin()

def center_rho(rho, centering="com", return_shift=False, maxfirst=True, iterations=1):
    """Move electron density map so its center of mass aligns with the center of the grid

    centering - which part of the density to center on. By default, center on the
                center of mass ("com"). Can also center on maximum density value ("max").
    """
    ne_rho= np.sum((rho))
    gridcenter = (np.array(rho.shape)-1.)/2.
    total_shift = np.zeros(3)
    if maxfirst:
        #sometimes the density crosses the box boundary, meaning
        #the center of mass calculation becomes an issue
        #first roughly center using the maximum density value (by
        #rolling to avoid interpolation artifacts). Then perform
        #the center of mass translation.
        rho, shift = center_rho_roll(rho, recenter_mode="max", return_shift=True)
        total_shift += shift.astype(float)
    for i in range(iterations):
        if centering == "max":
            rhocom = np.unravel_index(rho.argmax(), rho.shape)
        else:
            rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
        shift = gridcenter-rhocom
        rho = ndimage.interpolation.shift(rho,shift,order=3,mode='wrap')
        rho = rho*ne_rho/np.sum(rho)
        total_shift += shift
    if return_shift:
        return rho, total_shift
    else:
        return rho

def center_rho_roll(rho, recenter_mode="com", maxfirst=True, return_shift=False):
    """Move electron density map so its center of mass aligns with the center of the grid

    rho - electron density array
    recenter_mode - a string either com (center of mass) or max (maximum density)
    """
    total_shift = np.zeros(3,dtype=int)
    gridcenter = (np.array(rho.shape)-1.)/2.
    if maxfirst:
        #sometimes the density crosses the box boundary, meaning
        #the center of mass calculation becomes an issue
        #first roughly center using the maximum density value (by
        #rolling to avoid interpolation artifacts). Then perform
        #the center of mass translation.
        rhoargmax = np.unravel_index(np.abs(rho).argmax(), rho.shape)
        shift = gridcenter - rhoargmax
        shift = np.rint(shift).astype(int)
        rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
        total_shift += shift
    if recenter_mode == "max":
        rhocom = np.unravel_index(np.abs(rho).argmax(), rho.shape)
    else:
        rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    shift = gridcenter-rhocom
    shift = np.rint(shift).astype(int)
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    total_shift += shift
    if return_shift:
        return rho, total_shift
    else:
        return rho

def euler_grid_search(refrho, movrho, topn=1, abort_event=None):
    """Simple grid search on uniformly sampled sphere to optimize alignment.
        Return the topn candidate maps (default=1, i.e. the best candidate)."""
    #taken from https://stackoverflow.com/a/44164075/2836338

    #the euler angles search implicitly assumes the object is located
    #at the center of the grid, which may not be the case
    #first translate both refrho and movrho to center of grid, then
    #calculate optimal coarse rotations, the translate back
    gridcenter = (np.array(refrho.shape)-1.)/2.
    refrhocom = np.array(ndimage.measurements.center_of_mass(np.abs(refrho)))
    movrhocom = np.array(ndimage.measurements.center_of_mass(np.abs(movrho)))
    refshift = gridcenter-refrhocom
    movshift = gridcenter-movrhocom
    refrhocen = ndimage.interpolation.shift(refrho,refshift,order=3,mode='wrap')
    movrhocen = ndimage.interpolation.shift(movrho,movshift,order=3,mode='wrap')

    num_pts = 100 #~20 degrees between points
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    scores = np.zeros(num_pts)
    refrho2 = ndimage.gaussian_filter(refrhocen, sigma=1.0, mode='wrap')
    movrho2 = ndimage.gaussian_filter(movrhocen, sigma=1.0, mode='wrap')
    n = refrho2.shape[0]
    b,e = (int(n/4),int(3*n/4))
    refrho3 = refrho2[b:e,b:e,b:e]
    movrho3 = movrho2[b:e,b:e,b:e]

    for i in range(num_pts):
        scores[i] = -minimize_rho_score(T=[phi[i],theta[i],0,0,0,0],
                                        refrho=refrho3,movrho=movrho3
                                        )

        if abort_event is not None:
            if abort_event.is_set():
                return None, None

    best_pt = largest_indices(scores, topn)
    best_scores = scores[best_pt]
    movrhos = np.zeros((topn,movrho.shape[0],movrho.shape[1],movrho.shape[2]))

    for i in range(topn):
        movrhos[i] = transform_rho(movrho, T=[phi[best_pt[0][i]],theta[best_pt[0][i]],0,0,0,0])
        #now that the top five rotations are calculated, move each one back
        #to the same center of mass as the original refrho, i.e. by -refrhoshift
        movrhos[i] = ndimage.interpolation.shift(movrhos[i],-refshift,order=3,mode='wrap')

        if abort_event is not None:
            if abort_event.is_set():
                return movrhos, best_scores

    return movrhos, best_scores

def largest_indices(a, n):
    """Returns the n largest indices from a numpy array."""
    flat = a.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, a.shape)

def coarse_then_fine_alignment(refrho, movrho, coarse=True, topn=1,
    abort_event=None):
    """Course alignment followed by fine alignment.
        Select the topn candidates from the grid search
        and minimize each, selecting the best fine alignment.
        """
    if coarse:
        movrhos, scores = euler_grid_search(refrho, movrho, topn=topn,
            abort_event=abort_event)
    else:
        movrhos = movrho[np.newaxis,...]

    if abort_event is not None:
        if abort_event.is_set():
            return None, None

    for i in range(movrhos.shape[0]):
        movrhos[i], scores[i] = minimize_rho(refrho, movrhos[i])

        if abort_event is not None:
            if abort_event.is_set():
                return None, None

    best_i = np.argmax(scores)
    movrho = movrhos[best_i]
    score = scores[best_i]
    return movrho, score

def minimize_rho(refrho, movrho, T = np.zeros(6)):
    """Optimize superposition of electron density maps. Move movrho to refrho."""
    bounds = np.zeros(12).reshape(6,2)
    bounds[:3,0] = -20*np.pi
    bounds[:3,1] = 20*np.pi
    bounds[3:,0] = -5
    bounds[3:,1] = 5
    save_movrho = np.copy(movrho)
    save_refrho = np.copy(refrho)

    #first translate both to center
    #then afterwards translate back by -refshift
    gridcenter = (np.array(refrho.shape)-1.)/2.
    refrhocom = np.array(ndimage.measurements.center_of_mass(np.abs(refrho)))
    movrhocom = np.array(ndimage.measurements.center_of_mass(np.abs(movrho)))
    refshift = gridcenter-refrhocom
    movshift = gridcenter-movrhocom
    refrho = ndimage.interpolation.shift(refrho,refshift,order=3,mode='wrap')
    movrho = ndimage.interpolation.shift(movrho,movshift,order=3,mode='wrap')

    #for alignment only, run a low-pass filter to remove noise
    refrho2 = ndimage.gaussian_filter(refrho, sigma=1.0, mode='wrap')
    movrho2 = ndimage.gaussian_filter(movrho, sigma=1.0, mode='wrap')
    n = refrho2.shape[0]
    #to speed it up crop out the solvent
    b,e = (int(n/4),int(3*n/4))
    refrho3 = refrho2[b:e,b:e,b:e]
    movrho3 = movrho2[b:e,b:e,b:e]
    result = optimize.fmin_l_bfgs_b(minimize_rho_score, T, factr= 0.1,
        maxiter=100, maxfun=200, epsilon=0.05,
        args=(refrho3,movrho3), approx_grad=True)
    Topt = result[0]
    newrho = transform_rho(movrho, Topt)
    #now move newrho back by -refshift
    newrho = ndimage.interpolation.shift(newrho,-refshift,order=3,mode='wrap')
    finalscore = -1.*rho_overlap_score(save_refrho,newrho)
    return newrho, finalscore

def minimize_rho_score(T, refrho, movrho):
    """Scoring function for superposition of electron density maps.

        refrho - fixed, reference rho
        movrho - moving rho
        T - 6-element list containing alpha, beta, gamma, Tx, Ty, Tz in that order
        to move movrho by.
        """
    newrho = transform_rho(movrho, T)
    score = rho_overlap_score(refrho,newrho)
    return score

def rho_overlap_score(rho1,rho2, threshold=None):
    """Scoring function for superposition of electron density maps."""
    if threshold is None:
        n=np.sum(rho1*rho2)
        d=np.sum(rho1**2)**0.5*np.sum(rho2**2)**0.5
    else:
        #if there's a threshold, base it on only one map, then use
        #those indices for both maps to ensure the same pixels are compared
        idx = np.where(np.abs(rho1)>threshold*np.abs(rho1).max())
        n=np.sum(rho1[idx]*rho2[idx])
        d=np.sum(rho1[idx]**2)**0.5*np.sum(rho2[idx]**2)**0.5
    score = n/d
    #-score for least squares minimization, i.e. want to minimize, not maximize score
    return -score

def transform_rho(rho, T, order=1):
    """ Rotate and translate electron density map by T vector.

        T = [alpha, beta, gamma, x, y, z], angles in radians
        order = interpolation order (0-5)
    """
    ne_rho= np.sum((rho))
    R = euler2matrix(T[0],T[1],T[2])
    c_in = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    c_out = (np.array(rho.shape)-1.)/2.
    offset = c_in-c_out.dot(R)
    offset += T[3:]
    rho = ndimage.interpolation.affine_transform(rho,R.T, order=order,
        offset=offset, output=np.float64, mode='wrap')
    rho *= ne_rho/np.sum(rho)
    return rho

def euler2matrix(alpha=0.0,beta=0.0,gamma=0.0):
    """Convert Euler angles alpha, beta, gamma to a standard rotation matrix.

        alpha - yaw, counterclockwise rotation about z-axis, upper-left quadrant
        beta - pitch, counterclockwise rotation about y-axis, four-corners
        gamma - roll, counterclockwise rotation about x-axis, lower-right quadrant
        all angles given in radians

        """
    R = []
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    cosb = np.cos(beta)
    sinb = np.sin(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    R.append(np.array(
        [[cosa, -sina, 0],
        [sina, cosa, 0],
        [0, 0, 1]]))
    R.append(np.array(
        [[cosb, 0, sinb],
        [0, 1, 0],
        [-sinb, 0, cosb]]))
    R.append(np.array(
        [[1, 0, 0],
        [0, cosg, -sing],
        [0, sing, cosg]]))
    return reduce(np.dot,R[::-1])

def inertia_tensor(rho,side):
    """Calculate the moment of inertia tensor for the given electron density map."""
    halfside = side/2.
    n = rho.shape[0]
    x_ = np.linspace(-halfside,halfside,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
    Ixx = np.sum((y**2 + z**2)*rho)
    Iyy = np.sum((x**2 + z**2)*rho)
    Izz = np.sum((x**2 + y**2)*rho)
    Ixy = -np.sum(x*y*rho)
    Iyz = -np.sum(y*z*rho)
    Ixz = -np.sum(x*z*rho)
    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    return I

def principal_axes(I):
    """Calculate the principal inertia axes and order them Ia < Ib < Ic."""
    w,v = np.linalg.eigh(I)
    return w,v

def principal_axis_alignment(refrho,movrho):
    """ Align movrho principal axes to refrho."""
    side = 1.0
    ne_movrho = np.sum((movrho))
    #first center refrho and movrho, save refrho shift
    rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(refrho)))
    gridcenter = (np.array(refrho.shape)-1.)/2.
    shift = gridcenter-rhocom
    refrho = ndimage.interpolation.shift(refrho,shift,order=3,mode='wrap')
    #calculate, save and perform rotation of refrho to xyz for later
    refI = inertia_tensor(refrho, side)
    refw,refv = principal_axes(refI)
    refR = refv.T
    refrho = align2xyz(refrho)
    #align movrho to xyz too
    #check for best enantiomer, eigh is ambiguous in sign
    movrho = align2xyz(movrho)
    enans = generate_enantiomers(movrho)
    scores = np.zeros(enans.shape[0])
    for i in range(enans.shape[0]):
        scores[i] = -rho_overlap_score(refrho,enans[i])
    movrho = enans[np.argmax(scores)]
    #now rotate movrho by the inverse of the refrho rotation
    R = np.linalg.inv(refR)
    c_in = np.array(ndimage.measurements.center_of_mass(np.abs(movrho)))
    c_out = (np.array(movrho.shape)-1.)/2.
    offset=c_in-c_out.dot(R)
    movrho = ndimage.interpolation.affine_transform(movrho,R.T,order=3,offset=offset,mode='wrap')
    #now shift it back to where refrho was originally
    movrho = ndimage.interpolation.shift(movrho,-shift,order=3,mode='wrap')
    movrho *= ne_movrho/np.sum(movrho)
    return movrho

def align2xyz(rho, return_transform=False):
    """ Align rho such that principal axes align with XYZ axes."""
    side = 1.0
    ne_rho = np.sum(rho)
    #shift refrho to the center
    rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    gridcenter = (np.array(rho.shape)-1.)/2.
    shift = gridcenter-rhocom
    rho = ndimage.interpolation.shift(rho,shift,order=3,mode='wrap')
    #calculate, save and perform rotation of refrho to xyz for later
    I = inertia_tensor(rho, side)
    w,v = principal_axes(I)
    R = v.T
    refR = np.copy(R)
    refshift = np.copy(shift)
    #apparently need to run this a few times to get good alignment
    #maybe due to interpolation artifacts?
    for i in range(3):
        I = inertia_tensor(rho, side)
        w,v = np.linalg.eigh(I) #principal axes
        R = v.T #rotation matrix
        c_in = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
        c_out = (np.array(rho.shape)-1.)/2.
        offset=c_in-c_out.dot(R)
        rho = ndimage.interpolation.affine_transform(rho, R.T, order=3,
            offset=offset, mode='wrap')
    #also need to run recentering a few times
    for i in range(3):
        rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
        shift = gridcenter-rhocom
        rho = ndimage.interpolation.shift(rho,shift,order=3,mode='wrap')
    rho *= ne_rho/np.sum(rho)
    if return_transform:
        return rho, refR, refshift
    else:
        return rho

def generate_enantiomers(rho):
    """ Generate all enantiomers of given density map.
        Output maps are original, and flipped over z.
        """
    rho_zflip = rho[:,:,::-1]
    enans = np.array([rho,rho_zflip])
    return enans

def align(refrho, movrho, coarse=True, abort_event=None):
    """ Align second electron density map to the first."""
    if abort_event is not None:
        if abort_event.is_set():
            return None, None

    try:
        sleep(1)
        ne_rho = np.sum((movrho))
        movrho, score = coarse_then_fine_alignment(refrho=refrho, movrho=movrho, coarse=coarse, topn=1,
            abort_event=abort_event)

        if movrho is not None:
            movrho *= ne_rho/np.sum(movrho)

        return movrho, score

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

def select_best_enantiomer(refrho, rho, abort_event=None):
    """ Generate, align and select the enantiomer that best fits the reference map."""
    #translate refrho to center in case not already centered
    #just use roll to approximate translation to avoid interpolation, since
    #fine adjustments and interpolation will happen during alignment step

    try:
        sleep(1)
        c_refrho = center_rho_roll(refrho)
        #center rho in case it is not centered. use roll to get approximate location
        #and avoid interpolation
        c_rho = center_rho_roll(rho)
        #generate an array of the enantiomers
        enans = generate_enantiomers(c_rho)
        #allow for abort
        if abort_event is not None:
            if abort_event.is_set():
                return None, None

        #align each enantiomer and store the aligned maps and scores in results list
        results = [align(c_refrho, enan, abort_event=abort_event) for enan in enans]

        #now select the best enantiomer
        #rather than return the aligned and therefore interpolated enantiomer,
        #instead just return the original enantiomer, flipped from the original map
        #then no interpolation has taken place. So just dont overwrite enans essentially.
        #enans = np.array([results[k][0] for k in range(len(results))])
        enans_scores = np.array([results[k][1] for k in range(len(results))])
        best_i = np.argmax(enans_scores)
        best_enan, best_score = enans[best_i], enans_scores[best_i]
        return best_enan, best_score

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

def select_best_enantiomers(rhos, refrho=None, cores=1, avg_queue=None,
    abort_event=None, single_proc=False):
    """ Select the best enantiomer from each map in the set (or a single map).
        refrho should not be binary averaged from the original
        denss maps, since that would likely lose handedness.
        By default, refrho will be set to the first map."""
    if rhos.ndim == 3:
        rhos = rhos[np.newaxis,...]
    if refrho is None:
        refrho = rhos[0]

    #in parallel, select the best enantiomer for each rho
    if not single_proc:
        pool = multiprocessing.Pool(cores)
        try:
            mapfunc = partial(select_best_enantiomer, refrho)
            results = pool.map(mapfunc, rhos)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
            sys.exit(1)
            raise

    else:
        results = [select_best_enantiomer(refrho=refrho, rho=rho, abort_event=abort_event) for rho in rhos]

    best_enans = np.array([results[k][0] for k in range(len(results))])
    best_scores = np.array([results[k][1] for k in range(len(results))])

    return best_enans, best_scores

def align_multiple(refrho, rhos, cores=1, abort_event=None, single_proc=False):
    """ Align multiple (or a single) maps to the reference."""
    if rhos.ndim == 3:
        rhos = rhos[np.newaxis,...]
    #first, center all the rhos, then shift them to where refrho is
    cen_refrho, refshift = center_rho_roll(refrho, return_shift=True)
    shift = -refshift
    for i in range(rhos.shape[0]):
        rhos[i] = center_rho_roll(rhos[i])
        ne_rho = np.sum(rhos[i])
        #now shift each rho back to where refrho was originally
        #rhos[i] = ndimage.interpolation.shift(rhos[i],-refshift,order=3,mode='wrap')
        rhos[i] = np.roll(np.roll(np.roll(rhos[i], shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
        rhos[i] *= ne_rho/np.sum(rhos[i])

    if abort_event is not None:
        if abort_event.is_set():
            return None, None

    if not single_proc:
        pool = multiprocessing.Pool(cores)
        try:
            mapfunc = partial(align, refrho)
            results = pool.map(mapfunc, rhos)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
            sys.exit(1)
            raise
    else:
        results = [align(refrho, rho, abort_event=abort_event) for rho in rhos]

    rhos = np.array([results[i][0] for i in range(len(results))])
    scores = np.array([results[i][1] for i in range(len(results))])

    return rhos, scores

def average_two(rho1, rho2, abort_event=None):
    """ Align two electron density maps and return the average."""
    rho2, score = align(rho1, rho2, abort_event=abort_event)
    average_rho = (rho1+rho2)/2
    return average_rho

def multi_average_two(niter, **kwargs):
    """ Wrapper script for averaging two maps for multiprocessing."""
    try:
        sleep(1)
        return average_two(kwargs['rho1'][niter],kwargs['rho2'][niter],abort_event=kwargs['abort_event'])
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

def average_pairs(rhos, cores=1, abort_event=None, single_proc=False):
    """ Average pairs of electron density maps, second half to first half."""
    #create even/odd pairs, odds are the references
    rho_args = {'rho1':rhos[::2], 'rho2':rhos[1::2], 'abort_event': abort_event}

    if not single_proc:
        pool = multiprocessing.Pool(cores)
        try:
            mapfunc = partial(multi_average_two, **rho_args)
            average_rhos = pool.map(mapfunc, list(range(rhos.shape[0]//2)))
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
            sys.exit(1)
            raise
    else:
        average_rhos = [multi_average_two(niter, **rho_args) for niter in
            range(rhos.shape[0]//2)]

    return np.array(average_rhos)

def binary_average(rhos, cores=1, abort_event=None, single_proc=False):
    """ Generate a reference electron density map using binary averaging."""
    twos = 2**np.arange(20)
    nmaps = np.max(twos[twos<=rhos.shape[0]])
    #eight maps should be enough for the reference
    nmaps = np.max([nmaps,8])
    levels = int(np.log2(nmaps))-1
    rhos = rhos[:nmaps]
    for level in range(levels):
        rhos = average_pairs(rhos, cores, abort_event=abort_event,
            single_proc=single_proc)
    refrho = center_rho_roll(rhos[0])
    return refrho

def calc_fsc(rho1, rho2, side):
    """ Calculate the Fourier Shell Correlation between two electron density maps."""
    df = 1.0/side
    n = rho1.shape[0]
    qx_ = np.fft.fftfreq(n)*n*df
    qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qx_max = qx.max()
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    F1 = np.fft.fftn(rho1)
    F2 = np.fft.fftn(rho2)
    numerator = ndimage.sum(np.real(F1*np.conj(F2)), labels=qbin_labels,
        index=np.arange(0,qbin_labels.max()+1))
    term1 = ndimage.sum(np.abs(F1)**2, labels=qbin_labels,
        index=np.arange(0,qbin_labels.max()+1))
    term2 = ndimage.sum(np.abs(F2)**2, labels=qbin_labels,
        index=np.arange(0,qbin_labels.max()+1))
    denominator = (term1*term2)**0.5
    FSC = numerator/denominator
    qidx = np.where(qbins<qx_max)
    return  np.vstack((qbins[qidx],FSC[qidx])).T

def fsc2res(fsc, cutoff=0.5, return_plot=False):
    """Calculate resolution from the FSC curve using the cutoff given.

    fsc - an Nx2 array, where the first column is the x axis given as
          as 1/resolution (angstrom).
    cutoff - the fsc value at which to estimate resolution, default=0.5.
    return_plot - return additional arrays for plotting (x, y, resx)
    """
    x = np.linspace(fsc[0,0],fsc[-1,0],1000)
    y = np.interp(x, fsc[:,0], fsc[:,1])
    if np.min(fsc[:,1]) > 0.5:
        #if the fsc curve never falls below zero, then
        #set the resolution to be the maximum resolution
        #value sampled by the fsc curve
        resx = np.max(fsc[:,0])
        resn = float(1./resx)
        #print("Resolution: < %.1f A (maximum possible)" % resn)
    else:
        idx  = np.where(y>=0.5)
        #resi = np.argmin(y>=0.5)
        #resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])
        resx = np.max(x[idx])
        resn = float(1./resx)
        #print("Resolution: %.1f A" % resn)
    if return_plot:
        return resn, x, y, resx
    else:
        return resn

class Sasrec(object):
    def __init__(self, Iq, D, qc=None, r=None, nr=None, alpha=0.0, ne=2, extrapolate=True):
        self.Iq = Iq
        self.q = Iq[:,0]
        self.I = Iq[:,1]
        self.Ierr = Iq[:,2]
        self.q.clip(1e-10)
        self.I[np.abs(self.I)<1e-10] = 1e-10
        self.Ierr.clip(1e-10)
        self.q_data = np.copy(self.q)
        self.I_data = np.copy(self.I)
        self.Ierr_data = np.copy(self.Ierr)
        if qc is None:
            #self.qc = self.q
            self.create_lowq()
        else:
            self.qc = qc
        if extrapolate:
            self.extrapolate()
        self.D = D
        self.qmin = np.min(self.q)
        self.qmax = np.max(self.q)
        self.nq = len(self.q)
        self.qi = np.arange(self.nq)
        if r is not None:
            self.r = r
            self.nr = len(self.r)
        elif nr is not None:
            self.nr = nr
            self.r = np.linspace(0,self.D,self.nr)
        else:
            self.nr = self.nq
            self.r = np.linspace(0,self.D,self.nr)
        self.alpha = alpha
        self.ne = ne
        self.update()

    def update(self):
        #self.r = np.linspace(0,self.D,self.nr)
        self.ri = np.arange(self.nr)
        self.n = self.shannon_channels(self.qmax,self.D) + self.ne
        self.Ni = np.arange(self.n)
        self.N = self.Ni + 1
        self.Mi = np.copy(self.Ni)
        self.M = np.copy(self.N)
        self.qn = np.pi/self.D * self.N
        self.In = np.zeros((self.nq))
        self.Inerr = np.zeros((self.nq))
        self.B_data = self.Bt(q=self.q_data)
        self.B = self.Bt()
        #Bc is for the calculated q values in
        #cases where qc is not equal to q.
        self.Bc = self.Bt(q=self.qc)
        self.S = self.St()
        self.Y = self.Yt()
        self.C = self.Ct2()
        self.Cinv = np.linalg.inv(self.C)
        self.In = np.linalg.solve(self.C,self.Y)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.Inerr = np.diagonal(self.Cinv)**(0.5)
        self.Ic = self.Ish2Iq()
        self.Icerr = self.Icerrt()
        self.P = self.Ish2P()
        self.Perr = self.Perrt()
        self.I0 = self.Ish2I0()
        self.I0err = self.I0errf()
        self.F = self.Ft()
        self.rg = self.Ish2rg()
        self.E = self.Et()
        self.rgerr = self.rgerrf()
        self.avgr = self.Ish2avgr()
        self.avgrerr = self.avgrerrf()
        self.Q = self.Ish2Q()
        self.Qerr = self.Qerrf()
        self.Vp = self.Ish2Vp()
        self.Vperr = self.Vperrf()
        self.mwVp = self.Ish2mwVp()
        self.mwVperr = self.mwVperrf()
        self.Vc = self.Ish2Vc()
        self.Vcerr = self.Vcerrf()
        self.Qr = self.Ish2Qr()
        self.mwVc = self.Ish2mwVc()
        self.mwVcerr = self.mwVcerrf()
        self.lc = self.Ish2lc()
        self.lcerr = self.lcerrf()

    def create_lowq(self):
        """Create a calculated q range for Sasrec for low q out to q=0.
        Just the q values, not any extrapolation of intensities."""
        dq = (self.q.max()-self.q.min())/(self.q.size-1)
        nq = int(self.q.min()/dq)
        self.qc = np.concatenate(([0.0],np.arange(nq)*dq+(self.q.min()-nq*dq),self.q))

    def extrapolate(self):
        """Extrapolate to high q values"""
        #create a range of 1001 data points from 1*qmax to 3*qmax
        qe = np.linspace(1.0*self.q[-1],3.0*self.q[-1],1001)
        qe = qe[qe>self.q[-1]]
        qce = np.linspace(1.0*self.qc[-1],3.0*self.q[-1],1001)
        qce = qce[qce>self.qc[-1]]
        #extrapolated intensities can be anything, since they will
        #have infinite errors and thus no impact on the calculation
        #of the fit, so just make them a constant
        Ie = np.ones_like(qe)
        #set infinite error bars so that the actual intensities don't matter
        Ierre = Ie*np.inf
        self.q = np.hstack((self.q, qe))
        self.I = np.hstack((self.I, Ie))
        self.Ierr = np.hstack((self.Ierr, Ierre))
        self.qc = np.hstack((self.qc, qce))

    def optimize_alpha(self):
        """Scan alpha values to find optimal alpha"""
        ideal_chi2 = self.calc_chi2()
        al = []
        chi2 = []
        #here, alphas are actually the exponents, since the range can
        #vary from 10^-20 upwards of 10^20. This should cover nearly all likely values
        alphas = np.arange(-20,20.)
        i = 0
        nalphas = len(alphas)
        for alpha in alphas:
            i += 1
            sys.stdout.write("\rScanning alphas... {:.0%} complete".format(i*1./nalphas))
            sys.stdout.flush()
            try:
                self.alpha = 10.**alpha
                self.update()
            except:
                continue
            chi2value = self.calc_chi2()
            al.append(alpha)
            chi2.append(chi2value)
        al = np.array(al)
        chi2 = np.array(chi2)
        print()
        #find optimal alpha value based on where chi2 begins to rise, to 10% above the ideal chi2
        #interpolate between tested alphas to find more precise value
        #x = np.linspace(alphas[0],alphas[-1],1000)
        x = np.linspace(al[0],al[-1],1000)
        y = np.interp(x, al, chi2)
        chif = 1.1
        #take the maximum alpha value (x) where the chi2 just starts to rise above ideal
        try:
            ali = np.argmax(x[y<=chif*ideal_chi2])
        except:
            #if it fails, it may mean that the lowest alpha value of 10^-20 is still too large, so just take that.
            ali = 0
        #set the optimal alpha to be 10^alpha, since we were actually using exponents
        #also interpolate between the two neighboring alpha values, to get closer to the chif*ideal_chi2
        opt_alpha_exponent = np.interp(chif*ideal_chi2,[y[ali],y[ali-1]],[x[ali],x[ali-1]])
        #print(opt_alpha_exponent)
        opt_alpha = 10.0**(opt_alpha_exponent)
        self.alpha = opt_alpha
        self.update()
        return self.alpha

    def calc_chi2(self):
        Ish = self.In
        Bn = self.B_data
        #calculate Ic at experimental q vales for chi2 calculation
        Ic_qe = 2*np.einsum('n,nq->q',Ish,Bn)
        chi2 = (1./(self.nq-self.n-1.))*np.sum(1/(self.Ierr_data**2)*(self.I_data-Ic_qe)**2)
        return chi2

    def estimate_Vp_etal(self):
        """Estimate Porod volume using modified method based on oversmoothing.

        Oversmooth the P(r) curve with a high alpha. This helps to remove shape 
        scattering that distorts Porod assumptions. """
        #how much to oversmooth by, i.e. multiply alpha times this factor
        oversmoothing = 1.0e1
        #use a different qmax to limit effects of shape scattering.
        #use 8/Rg as the new qmax, but be sure to keep these effects
        #separate from the rest of sasrec, as it is only used for estimating
        #porod volume.
        qmax = 8./self.rg
        if np.isnan(qmax):
            qmax = 8./(self.D/3.5)
        Iq = np.vstack((self.q,self.I,self.Ierr)).T
        sasrec4vp = Sasrec(Iq[self.q<qmax], self.D, alpha=self.alpha*oversmoothing, extrapolate=self.extrapolate)
        self.Q = sasrec4vp.Q
        self.Qerr = sasrec4vp.Qerr
        self.Vp = sasrec4vp.Vp
        self.Vperr = sasrec4vp.Vperr
        self.mwVp = sasrec4vp.mwVp
        self.mwVperr = sasrec4vp.mwVperr
        self.Vc = sasrec4vp.Vc
        self.Vcerr = sasrec4vp.Vcerr
        self.Qr = sasrec4vp.Qr
        self.mwVc = sasrec4vp.mwVc
        self.mwVcerr = sasrec4vp.mwVcerr
        self.lc = sasrec4vp.lc
        self.lcerr = sasrec4vp.lcerr

    def shannon_channels(self, D, qmax=0.5, qmin=0.0):
        """Return the number of Shannon channels given a q range and maximum particle dimension"""
        width = np.pi / D
        num_channels = int((qmax-qmin) / width)
        return num_channels

    def Bt(self,q=None):
        N = self.N[:, None]
        if q is None:
            q = self.q
        else:
            q = q
        D = self.D
        #catch cases where qD==nPi, not often, but possible
        x = (N*np.pi)**2-(q*D)**2
        y =  np.where(x==0,(N*np.pi)**2,x)
        #B = (N*np.pi)**2/((N*np.pi)**2-(q*D)**2) * np.sinc(q*D/np.pi) * (-1)**(N+1)
        B = (N*np.pi)**2/y * np.sinc(q*D/np.pi) * (-1)**(N+1)
        return B

    def St(self):
        N = self.N[:,None]
        r = self.r
        D = self.D
        S = r/(2*D**2) * N * np.sin(N*np.pi*r/D)
        return S

    def Yt(self):
        """Return the values of Y, an m-length vector."""
        I = self.I
        Ierr = self.Ierr
        Bm = self.B
        Y = np.einsum('q, nq->n', I/Ierr**2, Bm)
        return Y

    def Ct(self):
        """Return the values of C, a m x n variance-covariance matrix"""
        Ierr = self.Ierr
        Bm = self.B
        Bn = self.B
        C = 2*np.einsum('ij,kj->ik', Bm/Ierr**2, Bn)
        return C

    def Gmn(self):
        """Return the mxn matrix of coefficients for the integral of (2nd deriv of P(r))**2 used for smoothing"""
        M = self.M
        N = self.N
        D = self.D
        gmn = np.zeros((self.n,self.n))
        mm, nn = np.meshgrid(M,N,indexing='ij')
        #two cases, one where m!=n, one where m==n. Do both separately.
        idx = np.where(mm!=nn)
        gmn[idx] = np.pi**2/(2*D**5) * (mm[idx]*nn[idx])**2 * (mm[idx]**4+nn[idx]**4)/(mm[idx]**2-nn[idx]**2)**2 * (-1)**(mm[idx]+nn[idx])
        idx = np.where(mm==nn)
        gmn[idx] = nn[idx]**4*np.pi**2/(48*D**5) * (2*nn[idx]**2*np.pi**2 + 33)
        return gmn

    def Ct2(self):
        """Return the values of C, a m x n variance-covariance matrix while smoothing P(r)"""
        n = self.n
        Ierr = self.Ierr
        Bm = self.B
        Bn = self.B
        alpha = self.alpha
        gmn = self.Gmn()
        return alpha * gmn + 2*np.einsum('ij,kj->ik', Bm/Ierr**2, Bn)

    def Ish2Iq(self):
        """Calculate I(q) from intensities at Shannon points."""
        Ish = self.In
        Bn = self.Bc
        I = 2*np.einsum('n,nq->q',Ish,Bn)
        return I

    def Ish2P(self):
        """Calculate P(r) from intensities at Shannon points."""
        Ish = self.In
        Sn = self.S
        P = np.einsum('n,nr->r',Ish,Sn)
        return P

    def Icerrt(self):
        """Return the standard errors on I_c(q)."""
        Bn = self.Bc
        Bm = self.Bc
        err2 = 2 * np.einsum('nq,mq,nm->q', Bn, Bm, self.Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            err = err2**(.5)
        return err

    def Perrt(self):
        """Return the standard errors on P(r)."""
        Sn = self.S
        Sm = self.S
        err2 = np.einsum('nr,mr,nm->r', Sn, Sm, self.Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            err = err2**(.5)
        return err

    def Ish2I0(self):
        """Calculate I0 from Shannon intensities"""
        N = self.N
        Ish = self.In
        I0 = 2 * np.sum(Ish*(-1)**(N+1))
        return I0

    def I0errf(self):
        """Calculate error on I0 from Shannon intensities from inverse C variance-covariance matrix"""
        N = self.N
        M = self.M
        Cinv = self.Cinv
        s2 = 2*np.einsum('n,m,nm->',(-1)**(N),(-1)**M,Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s = s2**(0.5)
        return s

    def Ft(self):
        """Calculate Fn function, for use in Rg calculation"""
        N = self.N
        F = (1-6/(N*np.pi)**2)*(-1)**(N+1)
        return F

    def Ish2rg(self):
        """Calculate Rg from Shannon intensities"""
        N = self.N
        Ish = self.In
        D = self.D
        I0 = self.I0
        F = self.F
        summation = np.sum(Ish*F)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rg2 = D**2/I0 * summation
            rg = np.sqrt(rg2)
        return rg

    def rgerrfold(self):
        """Calculate error on Rg from Shannon intensities from inverse C variance-covariance matrix"""
        Ish = self.In
        D = self.D
        Cinv = self.Cinv
        rg = self.rg
        I0 = self.I0
        Fn = self.F
        Fm = self.F
        s2 = np.einsum('n,m,nm->',Fn,Fm,Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s = D**2/(I0*rg)*s2**(0.5)
        return s

    def rgerrf(self):
        """Calculate error on Rg from Shannon intensities from inverse C variance-covariance matrix"""
        Ish = self.In
        D = self.D
        Cinv = self.Cinv
        rg = self.rg
        I0 = self.I0
        Fn = self.F
        Fm = self.F
        s2 = np.einsum('n,m,nm->',Fn,Fm,Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rgerr = D**2/(I0*rg)*s2**(0.5)
        return rgerr

    def Et(self):
        """Calculate En function, for use in ravg calculation"""
        N = self.N
        E = ((-1)**N-1)/(N*np.pi)**2 - (-1)**N/2.
        return E

    def Ish2avgr(self):
        """Calculate average vector length r from Shannon intensities"""
        Ish = self.In
        I0 = self.I0
        D = self.D
        E = self.E
        avgr = 4*D/I0 * np.sum(Ish * E)
        return avgr

    def avgrerrf(self):
        """Calculate error on Rg from Shannon intensities from inverse C variance-covariance matrix"""
        D = self.D
        Cinv = self.Cinv
        I0 = self.I0
        En = self.E
        Em = self.E
        s2 = np.einsum('n,m,nm->',En,Em,Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            avgrerr = 4*D/I0 * s2**(0.5)
        return avgrerr

    def Ish2Q(self):
        """Calculate Porod Invariant Q from Shannon intensities"""
        D = self.D
        N = self.N
        Ish = self.In
        Q = (np.pi/D)**3 * np.sum(Ish*N**2)
        return Q

    def Qerrf(self):
        """Calculate error on Q from Shannon intensities from inverse C variance-covariance matrix"""
        D = self.D
        Cinv = self.Cinv
        N = self.N
        M = self.M
        s2 = np.einsum('n,m,nm->', N**2, M**2,Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s = (np.pi/D)**3 * s2**(0.5)
        return s

    def gamma0(self):
        """Calculate gamma at r=0. gamma is P(r)/4*pi*r^2"""
        Ish = self.In
        D = self.D
        Q = self.Q
        return 1/(8*np.pi**3) * Q

    def Ish2Vp(self):
        """Calculate Porod Volume from Shannon intensities"""
        Q = self.Q
        I0 = self.I0
        Vp = 2*np.pi**2 * I0/Q
        return Vp

    def Vperrf(self):
        """Calculate error on Vp from Shannon intensities from inverse C variance-covariance matrix"""
        I0 = self.I0
        Q = self.Q
        I0s = self.I0err
        Qs = self.Qerr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            Vperr2 = (2*np.pi/Q)**2*(I0s)**2 + (2*np.pi*I0/Q**2)**2*Qs**2
            Vperr = Vperr2**(0.5)
        return Vperr

    def Ish2mwVp(self):
        """Calculate molecular weight via Porod Volume from Shannon intensities"""
        Vp = self.Vp
        mw = Vp/1.66
        return mw

    def mwVperrf(self):
        """Calculate error on mwVp from Shannon intensities from inverse C variance-covariance matrix"""
        Vps = self.Vperr
        return Vps/1.66

    def Ish2Vc(self):
        """Calculate Volume of Correlation from Shannon intensities"""
        Ish = self.In
        N = self.N
        I0 = self.I0
        D = self.D
        area_qIq = 2*np.pi/D**2 * np.sum(N * Ish * special.sici(N*np.pi)[0])
        Vc = I0/area_qIq
        return Vc

    def Vcerrf(self):
        """Calculate error on Vc from Shannon intensities from inverse C variance-covariance matrix"""
        I0 = self.I0
        Vc = self.Vc
        N = self.N
        M = self.M
        D = self.D
        Cinv = self.Cinv
        Sin = special.sici(N*np.pi)[0]
        Sim = special.sici(M*np.pi)[0]
        s2 = np.einsum('n,m,nm->', N*Sin, M*Sim,Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            Vcerr = (2*np.pi*Vc**2/(D**2*I0)) * s2**(0.5)
        return Vcerr

    def Ish2Qr(self):
        """Calculate Rambo Invariant Qr (Vc^2/Rg) from Shannon intensities"""
        Vc = self.Vc
        Rg = self.rg
        Qr = Vc**2/Rg
        return Qr

    def Ish2mwVc(self,RNA=False):
        """Calculate molecular weight via the Volume of Correlation from Shannon intensities"""
        Qr = self.Qr
        if RNA:
            mw = (Qr/0.00934)**(0.808)
        else:
            mw = (Qr/0.1231)**(1.00)
        return mw

    def mwVcerrf(self):
        Vc = self.Vc
        Rg = self.rg
        Vcs = self.Vcerr
        Rgs = self.rgerr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            mwVcs = Vc/(0.1231*Rg) * (4*Vcs**2 + (Vc/Rg*Rgs)**2)**(0.5)
        return mwVcs

    def Ish2lc(self):
        """Calculate length of correlation from Shannon intensities"""
        Vp = self.Vp
        Vc = self.Vc
        lc = Vp/(2*np.pi*Vc)
        return lc

    def lcerrf(self):
        """Calculate error on lc from Shannon intensities from inverse C variance-covariance matrix"""
        Vp = self.Vp
        Vc = self.Vc
        Vps = self.Vperr
        Vcs = self.Vcerr
        s2 = Vps**2 + (Vp/Vc)**2*Vcs**2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            lcerr = 1/(2*np.pi*Vc) * s2**(0.5)
        return lcerr

class PDB(object):
    """Load pdb file."""
    def __init__(self, filename=None, natoms=None, ignore_waters=False):
        if isinstance(filename, int):
            #if a user gives no keyword argument, but just an integer,
            #assume the user means the argument is to be interpreted
            #as natoms, rather than filename
            natoms = filename
            filename = None
        if filename is not None:
            self.read_pdb(filename, ignore_waters=ignore_waters)
        elif natoms is not None:
            self.generate_pdb_from_defaults(natoms)
        self.rij = None
        self.radius = None
        self.unique_radius = None
        self.unique_volume = None

    def read_pdb(self, filename, ignore_waters=False):
        self.natoms = 0
        with open(filename) as f:
            for line in f:
                if line[0:6] == "ENDMDL":
                    break
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                if ignore_waters and ((line[17:20]=="HOH") or (line[17:20]=="TIP")):
                    continue
                self.natoms += 1
        self.atomnum = np.zeros((self.natoms),dtype=int)
        self.atomname = np.zeros((self.natoms),dtype=np.dtype((np.str,3)))
        self.atomalt = np.zeros((self.natoms),dtype=np.dtype((np.str,1)))
        self.resname = np.zeros((self.natoms),dtype=np.dtype((np.str,3)))
        self.resnum = np.zeros((self.natoms),dtype=int)
        self.chain = np.zeros((self.natoms),dtype=np.dtype((np.str,1)))
        self.coords = np.zeros((self.natoms, 3))
        self.occupancy = np.zeros((self.natoms))
        self.b = np.zeros((self.natoms))
        self.atomtype = np.zeros((self.natoms),dtype=np.dtype((np.str,2)))
        self.charge = np.zeros((self.natoms),dtype=np.dtype((np.str,2)))
        self.nelectrons = np.zeros((self.natoms),dtype=int)
        self.vdW = np.zeros(self.natoms)
        self.numH = np.zeros(self.natoms, dtype=int)
        self.exvolHradius = np.zeros(self.natoms)
        with open(filename) as f:
            atom = 0
            for line in f:
                if line[0:6] == "ENDMDL":
                    break
                if line[0:6] == "CRYST1":
                    cryst = line.split()
                    self.cella = float(cryst[1])
                    self.cellb = float(cryst[2])
                    self.cellc = float(cryst[3])
                    self.cellalpha = float(cryst[4])
                    self.cellbeta = float(cryst[5])
                    self.cellgamma = float(cryst[6])
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                if ignore_waters and ((line[17:20]=="HOH") or (line[17:20]=="TIP")):
                    continue
                try:
                    self.atomnum[atom] = int(line[6:11])
                except ValueError as e:
                    self.atomnum[atom] = int(line[6:11],36)
                self.atomname[atom] = line[12:16].split()[0]
                self.atomalt[atom] = line[16]
                self.resname[atom] = line[17:20]
                try:
                    self.resnum[atom] = int(line[22:26])
                except ValueError as e:
                    self.resnum[atom] = int(line[22:26],36)
                self.chain[atom] = line[21]
                self.coords[atom, 0] = float(line[30:38])
                self.coords[atom, 1] = float(line[38:46])
                self.coords[atom, 2] = float(line[46:54])
                self.occupancy[atom] = float(line[54:60])
                self.b[atom] = float(line[60:66])
                atomtype = line[76:78].strip()
                if len(atomtype) == 2:
                    atomtype0 = atomtype[0].upper()
                    atomtype1 = atomtype[1].lower()
                    atomtype = atomtype0 + atomtype1
                if len(atomtype) == 0:
                    #if atomtype column is not in pdb file, set to first
                    #character of atomname
                    atomtype = self.atomname[atom][0]
                self.atomtype[atom] = atomtype
                self.charge[atom] = line[78:80].strip('\n')
                self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(),6)
                if len(self.atomtype[atom])==1:
                    atomtype = self.atomtype[atom][0].upper()
                else:
                    atomtype = self.atomtype[atom][0].upper() + self.atomtype[atom][1].lower()
                try:
                    dr = vdW[atomtype]
                except:
                    try:
                        dr = vdW[atomtype[0]]
                    except:
                        #default to carbon
                        dr = vdW['C']
                self.vdW[atom] = dr
                atom += 1

    def generate_pdb_from_defaults(self, natoms):
        self.natoms = natoms
        #simple array of incrementing integers, starting from 1
        self.atomnum = np.arange((self.natoms),dtype=int)+1
        #all carbon atoms by default
        self.atomname = np.full((self.natoms),"C",dtype=np.dtype((np.str,3)))
        #no alternate conformations by default
        self.atomalt = np.zeros((self.natoms),dtype=np.dtype((np.str,1)))
        #all Alanines by default
        self.resname = np.full((self.natoms),"ALA",dtype=np.dtype((np.str,3)))
        #each atom belongs to a new residue by default
        self.resnum = np.arange((self.natoms),dtype=int)
        #chain A by default
        self.chain = np.full((self.natoms),"A",dtype=np.dtype((np.str,1)))
        #all atoms at (0,0,0) by default
        self.coords = np.zeros((self.natoms, 3))
        #all atoms 1.0 occupancy by default
        self.occupancy = np.ones((self.natoms))
        #all atoms 20 A^2 by default
        self.b = np.ones((self.natoms))*20.0
        #all atom types carbon by default
        self.atomtype = np.full((self.natoms),"C",dtype=np.dtype((np.str,2)))
        #all atoms neutral by default
        self.charge = np.zeros((self.natoms),dtype=np.dtype((np.str,2)))
        #all atoms carbon so have six electrons by default
        self.nelectrons = np.ones((self.natoms),dtype=int)*6
        self.radius = np.zeros(self.natoms)
        self.vdW = np.zeros(self.natoms)
        self.unique_volume = np.zeros(self.natoms)
        self.unique_radius = np.zeros(self.natoms)
        #set a variable with H radius to be used for exvol radii optimization
        #set a variable for number of hydrogens bonded to atoms
        # self.exvolHradius = implicit_H_radius
        self.unique_exvolHradius = np.zeros(self.natoms)
        self.implicitH = False
        self.numH = np.zeros((self.natoms), dtype=int)
        #for CRYST1 card, use default defined by PDB, but 100 A side
        self.cella = 100.0
        self.cellb = 100.0
        self.cellc = 100.0
        self.cellalpha = 90.0
        self.cellbeta = 90.0
        self.cellgamma = 90.0

    def calculate_unique_volume(self,n=16,use_b=False,atomidx=None):
        """Generate volumes and radii for each atom of a pdb by accounting for overlapping sphere volumes,
        i.e., each radius is set to the value that yields a volume of a sphere equal to the
        corrected volume of the sphere after subtracting spherical caps from bonded atoms."""
        #first, for each atom, find all atoms closer than the sum of the two vdW radii
        ns = np.array([8,16,32])
        corrections = np.array([1.53,1.19,1.06]) #correction for n=8 voxels (1.19 for n=16, 1.06 for n=32)
        correction = np.interp(n,ns,corrections) #a rough approximation.
        # print("Calculating unique atomic volumes...")
        if atomidx is None:
            atomidx = range(self.natoms)
        for i in atomidx:
            # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,self.natoms))
            # sys.stdout.flush()
            #for each atom, make a box of voxels around it
            ra = self.vdW[i] #ra is the radius of the main atom
            if use_b:
                ra += B2u(self.b[i])
            side = 2*ra
            #n = 8 #yields somewhere around 0.2 A voxel spacing depending on atom size
            dx = side/n
            dV = dx**3
            x_ = np.linspace(-side/2,side/2,n)
            x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
            minigrid = np.zeros(x.shape,dtype=np.bool_)
            shift = np.ones(3)*dx/2.
            #create a column stack of coordinates for the minigrid
            xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
            #for simplicity assume the atom is at the center of the minigrid, (0,0,0),
            #therefore we need to subtract the vector shift (i.e. the coordinates
            #of the atom) from each of the neighboring atoms, so grab those coordinates
            p = np.copy(self.coords[i])
            #calculate all distances from the atom to the minigrid points
            center = np.zeros(3)
            xa, ya, za = center
            dist = spatial.distance.cdist(center[None,:], xyz)[0].reshape(n,n,n)
            #now, any elements of minigrid that have a dist less than ra make true
            minigrid[dist<=ra] = True
            #grab atoms nearby this atom just based on xyz coordinates
            #first, recenter all coordinates in this frame
            coordstmp = self.coords - p
            #next, get all atoms whose x, y, and z coordinates are within the nearby box 
            #of length 4 A (more than the sum of two atoms vdW radii, with the limit being about 2.5 A)
            bl = 5.0
            idx_close = np.where(
                (coordstmp[:,0]>=xa-bl/2)&(coordstmp[:,0]<=xa+bl/2)&
                (coordstmp[:,1]>=ya-bl/2)&(coordstmp[:,1]<=ya+bl/2)&
                (coordstmp[:,2]>=za-bl/2)&(coordstmp[:,2]<=za+bl/2)
                )[0]
            idx_close=idx_close[idx_close!=i] #ignore this atom
            nclose = len(idx_close)
            for j in range(nclose):
                #get index of next closest atom
                idx_j = idx_close[j]
                #get the coordinates of the  neighboring atom, and shift using the same vector p as the main atom
                cb = self.coords[idx_j] - p #center of neighboring atom in new coordinate frame
                xb,yb,zb = cb
                rb = self.vdW[idx_j]
                if use_b:
                    rb += B2u(self.b[idx_j])
                a,b,c,d = equation_of_plane_from_sphere_intersection(xa,ya,za,ra,xb,yb,zb,rb)
                normal = np.array([a,b,c]) #definition of normal to a plane
                #for each grid point, calculate the distance to the plane in the direction of the vector normal
                #if the distance is positive, then that gridpoint is beyond the plane
                #we can calculate the center of the circle which lies on the plane, so thats a good point to use
                circle_center = center_of_circle_from_sphere_intersection(xa,ya,za,ra,xb,yb,zb,rb,a,b,c,d)
                xyz_minus_cc = xyz - circle_center
                #calculate distance matrix to neighbor
                dist2neighbor = spatial.distance.cdist(cb[None,:], xyz)[0].reshape(n,n,n)
                overlapping_voxels = np.zeros(n**3,dtype=bool)
                overlapping_voxels[minigrid.ravel() & np.ravel(dist2neighbor<=rb)] = True
                #calculate the distance to the plane for each minigrid voxel
                #there may be a way to vectorize this if its too slow
                noverlap = overlapping_voxels.sum()
                # print(noverlap, overlapping_voxels.size)
                d2plane = np.zeros(x.size)
                for k in range(n**3):
                    if overlapping_voxels[k]:
                        d2plane[k] = np.dot(normal,xyz_minus_cc[k,:])
                d2plane = d2plane.reshape(n,n,n)
                #all voxels with a positive d2plane value are _beyond_ the plane
                minigrid[d2plane>0] = False
            #add up all the remaining voxels in the minigrid to get the volume
            #also correct for limited voxel size
            self.unique_volume[i] = minigrid.sum()*dV * correction

    def lookup_unique_volume(self):
        self.unique_volume = np.zeros(self.natoms)
        for i in range(self.natoms):
            notfound = False
            if (self.resname[i] in atomic_volumes.keys()):
                if (self.atomname[i] in atomic_volumes[self.resname[i]].keys()):
                    self.unique_volume[i] = atomic_volumes[self.resname[i]][self.atomname[i]]
                else:
                    notfound = True
            else:
                notfound = True
            if notfound:
                print("%s:%s not found in volumes dictionary. Calculating unique volume."%(self.resname[i],self.atomname[i]))
                # print("Setting volume to ALA:CA.")
                # self.unique_volume[i] = atomic_volumes['ALA']['CA']
                self.calculate_unique_volume(atomidx=[i])

    def add_ImplicitH_old(self):
        bondlist_resNorm = protein_residues.normal
        bondlist_resCterm = protein_residues.c_terminal
        bondlist_resNterm = protein_residues.n_terminal

        if 'H' in self.atomtype:
            self.remove_by_atomtype('H')

        for i in range(len(self.atomname)):
            H_count = 0
            res = self.resname[i]
            atom = self.atomname[i]
            resnum = self.resnum[i]

            #If first residue, read bonds from resNterm
            #If not first residue and not last residue, read from resNorm
            try:
                if resnum == 1:
                    Hbond_count = bondlist_resNterm[res]['numH']
                else:
                    Hbond_count = bondlist_resNorm[res]['numH']
            except:
                Hbond_count = 0

            #For each atom, atom should be a key in "numH", so now just look up value 
            # associated with atom
            try:
                H_count = Hbond_count[atom]
            except:
                #This except could be more complex and use bond lengths to count potential number 
                #of hydrogens based on atom type and number of bonds
                print("atom ", atom, " not in ", res, " list. setting numH to 0.")
                H_count = 0

            #Add number of hydrogens for the atom to a pdb object so it can
            #be carried with pdb class
            self.numH[i] = H_count
            self.nelectrons[i]+=H_count

    def add_ImplicitH(self):
        if 'H' in self.atomtype:
            self.remove_by_atomtype('H')

        for i in range(len(self.atomname)):
            res = self.resname[i]
            atom = self.atomname[i]

            #For each atom, atom should be a key in "numH", so now just look up value 
            # associated with atom
            try:
                H_count = numH[res][atom] #the number of H attached
                H_mean_volume = volH[res][atom] #the average volume of each H attached
            except:
                print("atom ", atom, " not in ", res, " list. setting numH to 0.")
                H_count = 0
                H_mean_volume = 0

            #Add number of hydrogens for the atom to a pdb object so it can
            #be carried with pdb class
            self.numH[i] = H_count #the number of H attached
            self.unique_exvolHradius[i] = sphere_radius_from_volume(H_mean_volume)
            self.nelectrons[i] += H_count

    def remove_waters(self):
        idx = np.where((self.resname=="HOH") | (self.resname=="TIP"))
        self.remove_atoms_from_object(idx)

    def remove_by_atomtype(self, atomtype):
        idx = np.where((self.atomtype==atomtype))
        self.remove_atoms_from_object(idx)

    def remove_by_atomname(self, atomname):
        idx = np.where((self.atomname==atomname))
        self.remove_atoms_from_object(idx)

    def remove_by_atomnum(self, atomnum):
        idx = np.where((self.atomnum==atomnum))
        self.remove_atoms_from_object(idx)

    def remove_by_resname(self, resname):
        idx = np.where((self.resname==resname))
        self.remove_atoms_from_object(idx)

    def remove_by_resnum(self, resnum):
        idx = np.where((self.resnum==resnum))
        self.remove_atoms_from_object(idx)

    def remove_by_chain(self, chain):
        idx = np.where((self.chain==chain))
        self.remove_atoms_from_object(idx)

    def remove_atomalt(self):
        idx = np.where((self.atomalt!=' ') & (self.atomalt!='A'))
        self.remove_atoms_from_object(idx)

    def remove_atoms_from_object(self, idx):
        mask = np.ones(self.natoms, dtype=bool)
        mask[idx] = False
        self.atomnum = self.atomnum[mask]
        self.atomname = self.atomname[mask]
        self.atomalt = self.atomalt[mask]
        self.resname = self.resname[mask]
        self.resnum = self.resnum[mask]
        self.chain = self.chain[mask]
        self.coords = self.coords[mask]
        self.occupancy = self.occupancy[mask]
        self.b = self.b[mask]
        self.atomtype = self.atomtype[mask]
        self.charge = self.charge[mask]
        self.nelectrons = self.nelectrons[mask]
        self.natoms = len(self.atomnum)
        if self.radius is not None:
            self.radius = self.radius[mask]
        self.vdW = self.vdW[mask]
        self.numH = self.numH[mask]
        if self.unique_radius is not None:
            self.unique_radius = self.unique_radius[mask]
        if self.unique_volume is not None:
            self.unique_volume = self.unique_volume[mask]
        if self.unique_exvolHradius is not None:
            self.unique_exvolHradius = self.unique_exvolHradius[mask]

    def write(self, filename):
        """Write PDB file format using pdb object as input."""
        records = []
        anum,rc = (np.unique(self.atomnum,return_counts=True))
        if np.any(rc>1):
            #in case default atom numbers are repeated, just renumber them
            self_numbering=True
        else:
            self_numbering=False
        for i in range(self.natoms):
            if self_numbering:
                atomnum = '%5i' % ((i+1)%99999)
            else:
                atomnum = '%5i' % (self.atomnum[i]%99999)
            atomname = '%3s' % self.atomname[i]
            atomalt = '%1s' % self.atomalt[i]
            resnum = '%4i' % (self.resnum[i]%9999)
            resname = '%3s' % self.resname[i]
            chain = '%1s' % self.chain[i]
            x = '%8.3f' % self.coords[i,0]
            y = '%8.3f' % self.coords[i,1]
            z = '%8.3f' % self.coords[i,2]
            o = '% 6.2f' % self.occupancy[i]
            b = '%6.2f' % self.b[i]
            atomtype = '%2s' % self.atomtype[i]
            charge = '%2s' % self.charge[i]
            records.append(['ATOM  ' + atomnum + '  ' + atomname + ' ' + resname + ' ' + chain + resnum + '    ' + x + y + z + o + b + '          ' + atomtype + charge])
        np.savetxt(filename, records, fmt='%80s'.encode('ascii'))

def sphere_volume_from_radius(R):
    V_sphere = 4*np.pi/3 * R**3
    return V_sphere

def sphere_radius_from_volume(V):
    R_sphere = (3*V/(4*np.pi))**(1./3)
    return R_sphere

def cap_heights(r1,r2,d):
    """Calculate the heights h1, h2 of spherical caps from overlapping spheres of radii r1, r2 a distance d apart"""
    h1 = (r2-r1+d)*(r2+r1-d)/(2*d)
    h2 = (r1-r2+d)*(r1+r2-d)/(2*d)
    return h1, h2

def spherical_cap_volume(R,h):
    #sphere of radius R, cap of height h
    V_cap = 1./3 * np.pi * h**2 * (3*R-h)
    return V_cap

def equation_of_plane_from_sphere_intersection(x1,y1,z1,r1,x2,y2,z2,r2):
    """Calculate coefficients a,b,c,d of equation of a plane (ax+by+cz+d=0) formed by the
    intersection of two spheres with centers (x1,y1,z1), (x2,y2,z2) and radii r1,r2.
    from: http://ambrnet.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm"""
    a = 2*(x2-x1)
    b = 2*(y2-y1)
    c = 2*(z2-z1)
    d = x1**2 - x2**2 + y1**2 - y2**2 + z1**2 - z2**2 - r1**2 + r2**2
    return a,b,c,d

def center_of_circle_from_sphere_intersection(x1,y1,z1,r1,x2,y2,z2,r2,a,b,c,d):
    """Calculate the center of the circle formed by the intersection of two spheres"""
    # print(a*(x1-x2), b*(y1-y2), c*(z1-z2))
    # print((a*(x1-x2) + b*(y1-y2) +c*(z1-z2)))
    # print((x1*a + y1*b + z1*c + d))
    t = (x1*a + y1*b + z1*c + d) / (a*(x1-x2) + b*(y1-y2) +c*(z1-z2))
    xc = x1 + t*(x2-x1)
    yc = y1 + t*(y2-y1)
    zc = z1 + t*(z2-z1)
    return (xc,yc,zc)

class PDB2MRC(object):
    def __init__(self, 
        pdb,
        ignore_waters=True,
        explicitH=True,
        modifiable_atom_types=None,
        center_coords=True,
        radii_sf=None,
        recalculate_atomic_volumes=False,
        exvol_type='gaussian',
        use_b=False,
        resolution=None,
        voxel=None,
        side=None,
        nsamples=None,
        rho0=0.334,
        shell_contrast=0.03,
        shell_mrcfile=None,
        shell_type='gaussian',
        Icalc_interpolation=True,
        data_filename=None,
        data_units="a",
        n1=None,
        n2=None,
        penalty_weight=0.0,
        penalty_weights=[1.0,0.0],
        fit_rho0=True,
        fit_shell=True,
        fit_all=True,
        min_method='Nelder-Mead',
        min_opts='{"adaptive": True}',
        ):
        self.pdb = pdb
        self.ignore_waters = ignore_waters
        self.explicitH = explicitH
        if self.explicitH is None:
            #only use explicitH if H exists in the pdb file
            #for atoms that are not waters
            if 'H' not in self.pdb.atomtype: #[pdb.resname!="HOH"]:
                self.explicitH = False
            else:
                self.explicitH = True
        if not self.explicitH:
            self.pdb.add_ImplicitH()
            print("Implicit hydrogens used")
        #add a line here that will delete alternate conformations if they exist
        if 'B' in self.pdb.atomalt:
            self.pdb.remove_atomalt()
        if modifiable_atom_types is None:
            self.modifiable_atom_types = ['H', 'C', 'N', 'O']
        else:
            self.modifiable_atom_types = modifiable_atom_types
        self.center_coords = center_coords
        if self.center_coords:
            self.pdb.coords -= self.pdb.coords.mean(axis=0)
        if recalculate_atomic_volumes:
            print("Calculating unique atomic volumes...")
            self.pdb.unique_volume = np.zeros(self.pdb.natoms)
            self.pdb.calculate_unique_volume()
        elif self.pdb.unique_volume is None:
            print("Looking up unique atomic volumes...")
            self.pdb.lookup_unique_volume()
        self.pdb.unique_radius = sphere_radius_from_volume(self.pdb.unique_volume)
        if radii_sf is None:
            self.radii_sf = np.ones(len(self.modifiable_atom_types))
            for i in range(len(self.modifiable_atom_types)):
                if self.modifiable_atom_types[i] in radii_sf_dict.keys():
                    self.radii_sf[i] = radii_sf_dict[self.modifiable_atom_types[i]]
                else:
                    self.radii_sf[i] = 1.0
        else:
            self.radii_sf = radii_sf
        self.exvol_type = exvol_type
        self.use_b = use_b
        if not self.use_b:
            self.pdb.b *= 0
        #calculate some optimal grid values
        self.optimal_side = estimate_side_from_pdb(self.pdb)
        self.optimal_voxel = 1.0
        self.optimal_nsamples = np.ceil(self.optimal_side/self.optimal_voxel).astype(int)
        self.nsamples_limit = 256
        self.resolution = resolution
        self.voxel=voxel
        self.side=side
        self.nsamples=nsamples
        self.rho0 = rho0
        self.shell_contrast = shell_contrast
        self.shell_mrcfile = shell_mrcfile
        self.shell_type = shell_type
        self.Icalc_interpolation=Icalc_interpolation
        self.data_filename = data_filename
        self.data_units = data_units
        self.n1 = n1
        self.n2 = n2
        self.penalty_weight = penalty_weight
        self.penalty_weights = penalty_weights
        self.fit_rho0 = fit_rho0
        self.fit_shell = fit_shell
        self.fit_all = fit_all
        self.fit_params = False #start with no fitting
        self.min_method = min_method
        self.min_opts = min_opts
        self.param_names = ['rho0', 'shell_contrast']
        self.params = np.array([self.rho0, self.shell_contrast])

    def scale_radii(self, radii_sf=None):
        """Set all the modifiable atom type radii in the pdb"""
        if self.pdb.radius is None:
            self.pdb.radius = np.copy(self.pdb.unique_radius)
        if radii_sf is None:
            radii_sf = self.radii_sf
        for i in range(len(self.modifiable_atom_types)):
            if not self.explicitH:
                if self.modifiable_atom_types[i]=='H':
                    self.pdb.exvolHradius = radii_sf[i] * self.pdb.unique_exvolHradius 
                else:
                    self.pdb.radius[self.pdb.atomtype==self.modifiable_atom_types[i]] = radii_sf[i] * self.pdb.unique_radius[self.pdb.atomtype==self.modifiable_atom_types[i]]
            else:
                self.pdb.exvolHradius = np.zeros(self.pdb.natoms)
                self.pdb.radius[self.pdb.atomtype==self.modifiable_atom_types[i]] = radii_sf[i] * self.pdb.unique_radius[self.pdb.atomtype==self.modifiable_atom_types[i]]

    def calculate_average_radii(self):
        self.mean_radius = np.ones(len(self.modifiable_atom_types))
        for i in range(len(self.modifiable_atom_types)):
            #try using a scale factor for radii instead
            if self.modifiable_atom_types[i]=='H' and not self.explicitH:
                self.mean_radius[i] = self.pdb.exvolHradius[i]
            else:
                self.mean_radius[i] = self.pdb.radius[self.pdb.atomtype==self.modifiable_atom_types[i]].mean()

    def make_grids(self):
        optimal_side = self.optimal_side
        optimal_nsamples = self.optimal_nsamples
        optimal_voxel = self.optimal_voxel
        optimal_nsamples = self.optimal_nsamples
        nsamples_limit = self.nsamples_limit
        voxel = self.voxel
        side = self.side
        nsamples = self.nsamples
        if voxel is not None and nsamples is not None and side is not None:
            #if v, n, s are all given, side and nsamples dominates
            side = optimal_side
            nsamples = nsamples
            voxel = side / nsamples
        elif voxel is not None and nsamples is not None and side is None:
            #if v and n given, voxel and nsamples dominates
            voxel = voxel
            nsamples = nsamples
            side = voxel * nsamples
        elif voxel is not None and nsamples is None and side is not None:
            #if v and s are given, adjust voxel to match nearest integer value of n
            voxel = voxel
            side = side
            nsamples = np.ceil(side/voxel).astype(int)
            voxel = side / nsamples
        elif voxel is not None and nsamples is None and side is None:
            #if v is given, voxel thus dominates, so estimate side, calculate nsamples.
            voxel = voxel
            nsamples = np.ceil(optimal_side/voxel).astype(int)
            side = voxel * nsamples
            #if n > 256, adjust side length
            if nsamples > nsamples_limit:
                nsamples = nsamples_limit
                side = voxel * nsamples
        elif voxel is None and nsamples is not None and side is not None:
            #if n and s are given, set voxel size based on those
            nsamples = nsamples
            side = side
            voxel = side / nsamples
        elif voxel is None and nsamples is not None and side is None:
            #if n is given, set side, adjust voxel.
            nsamples = nsamples
            side = optimal_side
            voxel = side / nsamples
        elif voxel is None and nsamples is None and side is not None:
            #if s is given, set voxel, adjust nsamples, reset voxel if necessary
            side = side
            voxel = optimal_voxel
            nsamples = np.ceil(side/voxel).astype(int)
            if nsamples > nsamples_limit:
                nsamples = nsamples_limit
            voxel = side / nsamples
        elif voxel is None and nsamples is None and side is None:
            #if none given, set side and voxel, adjust nsamples, reset voxel if necessary
            side = optimal_side
            voxel = optimal_voxel
            nsamples = np.ceil(side/voxel).astype(int)
            if nsamples > nsamples_limit:
                nsamples = nsamples_limit
            voxel = side / nsamples

        #make some warnings for certain cases
        side_small_warning = """
            Side length may be too small and may result in undersampling errors."""
        side_way_too_small_warning = """
            Disabling interpolation of I_calc due to severe undersampling."""
        voxel_big_warning = """
            Voxel size is greater than 1 A. This may lead to less accurate I(q) estimates at high q."""
        nsamples_warning = """
            To avoid long computation times and excessive memory requirements, the number of voxels
            has been limited to 256 and the voxel size has been set to {v:.2f},
            which may be too large and lead to less accurate I(q) estimates at high q.""".format(v=voxel)
        optimal_values_warning = """
            To ensure the highest accuracy, manually set the -s option to {os:.2f} and
            the -v option to {ov:.2f}, which will set -n option to {on:d} and thus 
            may take a long time to calculate and use a lot of memory.
            If that requires too much computation, set -s first, and -n to the 
            limit you prefer (n=512 may approach an upper limit for many computers).
            """.format(os=optimal_side, ov=optimal_voxel, on=optimal_nsamples)

        warn = False
        if side < optimal_side:
            print(side_small_warning)
            warn = True
        if side < 2/3*optimal_side:
            print(side_way_too_small_warning)
            self.Icalc_interpolation = False
            warn = True
        if voxel > optimal_voxel:
            print(voxel_big_warning)
            warn = True
        if nsamples < optimal_nsamples:
            print(nsamples_warning)
            warn = True
        if warn:
            print(optimal_values_warning)

        #make the real space grid
        halfside = side/2
        n = int(side/voxel)
        #want n to be even for speed/memory optimization with the FFT, 
        #ideally a power of 2, but wont enforce that
        if n%2==1: n += 1
        dx = side/n
        dV = dx**3
        x_ = np.linspace(-halfside,halfside,n)
        x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
        xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))

        #make the reciprocal space grid
        df = 1/side
        qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
        qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
        qr = np.sqrt(qx**2+qy**2+qz**2)
        qmax = np.max(qr)
        qstep = np.min(qr[qr>0]) - 1e-8
        nbins = int(qmax/qstep)
        qbins = np.linspace(0,nbins*qstep,nbins+1)
        #create an array labeling each voxel according to which qbin it belongs
        qbin_labels = np.searchsorted(qbins,qr,"right")
        qbin_labels -= 1
        qblravel = qbin_labels.ravel()
        xcount = np.bincount(qblravel)
        #create modified qbins and put qbins in center of bin rather than at left edge of bin.
        qbinsc = mybinmean(qr.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
        q_calc = np.copy(qbinsc)

        #make attributes for all that
        self.halfside = halfside
        self.side = side
        self.n = n
        self.dx = dx
        self.dV = dV
        self.x_ = x_
        self.x, self.y, self.z = x,y,z
        self.xyz = xyz
        self.df = df
        self.qx_ = qx_
        self.qx, self.qy, self.qz = qx, qy, qz
        self.qr = qr
        self.qmax = qmax
        self.qstep = qstep
        self.nbins = nbins
        self.qbins = qbins
        self.qbinsc = qbinsc
        self.q_calc = q_calc
        self.qblravel = qblravel
        self.xcount = xcount

        #save an array of indices containing only desired q range for speed
        qmax4calc = self.qx_.max()*1.1
        self.qidx = np.where((self.qr<=qmax4calc))

    def calculate_resolution(self):
        if self.dx is None:
            #if make_grids has not been run yet, run it
            self.make_grids()
        if self.resolution is None and not self.use_b:
            self.resolution = 0.30 * self.dx #this helps with voxel sampling issues
        else:
            self.resolution = 0.0

    def calculate_invacuo_density(self):
        print('Calculating in vacuo density...')
        self.rho_invacuo, self.support = pdb2map_multigauss(self.pdb,
            x=self.x,y=self.y,z=self.z,
            resolution=self.resolution,
            use_b=self.use_b,
            ignore_waters=self.ignore_waters)
        print('Finished in vacuo density.')

    def calculate_excluded_volume(self, quiet=False):
        if not quiet: print('Calculating excluded volume...')
        if self.exvol_type == "gaussian":
            #generate excluded volume assuming gaussian dummy atoms
            #this function outputs in electron count units
            self.rho_exvol, self.supportexvol = pdb2map_simple_gauss_by_radius(self.pdb,
                self.x,self.y,self.z,
                rho0=self.rho0,
                ignore_waters=self.ignore_waters)
        elif self.exvol_type == "flat":
            #generate excluded volume assuming flat solvent
            if not self.pdb.explicitH:
                v = 4*np.pi/3*self.pdb.vdW**3 + self.pdb.numH*4/3*np.pi*vdW['H']**3
                radjusted = sphere_radius_from_volume(v)
            else:
                radjusted = self.pdb.vdW
            self.supportexvol = pdb2support_fast(self.pdb,self.x,self.y,self.z,radius=radjusted,probe=B2u(self.pdb.b))
            #estimate excluded volume electron count based on unique volumes of atoms
            v = np.sum(4/3*np.pi*self.pdb.radius**3)
            ne = v * self.rho0
            #blur the exvol to have gaussian-like edges
            sigma = 1.0/self.dx #best exvol sigma to match water molecule exvol thing is 1 A
            self.rho_exvol = ndimage.gaussian_filter(self.supportexvol*1.0,sigma=sigma,mode='wrap')
            self.rho_exvol *= ne/self.rho_exvol.sum() #put in electron count units
        if not quiet: print('Finished excluded volume.')

    def calculate_hydration_shell(self):
        print('Calculating hydration shell...')
        #calculate the volume of a shell of water diameter
        #this assumes a single layer of hexagonally packed water molecules on the surface
        self.r_water = r_water = 1.4 
        uniform_shell = calc_uniform_shell(self.pdb,self.x,self.y,self.z,thickness=self.r_water)
        self.water_shell_idx = water_shell_idx = uniform_shell.astype(bool)
        V_shell = water_shell_idx.sum() * self.dV
        N_H2O_in_shell = 2/(3**0.5) * V_shell / (2*r_water)**3
        V_H2O = 4/3*np.pi*r_water**3
        V_H2O_in_shell = N_H2O_in_shell * V_H2O

        if self.shell_mrcfile is not None:
            #allow user to provide mrc filename to read in a custom shell
            rho_shell, sidex = read_mrc(self.shell_mrcfile)
            rho_shell *= self.dV #assume mrc file is in units of density, convert to electron count
            print(sidex, self.side)
            if (sidex != self.side) or (rho_shell.shape[0] != self.x.shape[0]):
                print("Error: shell_mrcfile does not match grid.")
                print("Use denss.mrcops.py to resample onto the desired grid.")
                exit()
        elif self.shell_type == "gaussian":
            #the default is gaussian type shell
            #generate initial hydration shell
            thickness = max(1.0,self.dx) #in angstroms
            #calculate euclidean distance transform of grid to water shell center
            #where the water shell center is the surface of the protein plus a water radius
            protein_idx = pdb2support_fast(self.pdb,self.x,self.y,self.z,radius=self.pdb.vdW,probe=0)
            protein_rw_idx = calc_uniform_shell(self.pdb,self.x,self.y,self.z,thickness=self.r_water,distance=self.r_water).astype(bool)
            print('Calculating dist transform...')
            dist = ndimage.distance_transform_edt(~protein_rw_idx)
            #look at only the voxels near the shell for efficiency
            rho_shell = np.zeros(self.x.shape)
            print('Calculating shell values...')
            rho_shell[dist<2*r_water] = realspace_formfactor(element='O',r=dist[dist<2*r_water],B=u2B(0.5))
            #zero out any voxels overlapping protein atoms
            rho_shell[protein_idx] = 0.0
            #estimate initial shell scale based on contrast using mean density
            shell_mean_density = np.mean(rho_shell[water_shell_idx]) / self.dV
            #scale the mean density of the invacuo shell to match the desired mean density
            rho_shell *= self.shell_contrast / shell_mean_density
            #shell should still be in electron count units
        elif self.shell_type == "uniform":
            rho_shell = water_shell_idx * (self.shell_contrast)
            rho_shell *= self.dV #convert to electron units
        else:
            print("Error: no valid shell_type given. Disabling hydration shell.")
            rho_shell = self.x*0.0
        self.rho_shell = rho_shell
        print('Finished hydration shell.')

    def calculate_structure_factors(self):
        print('Calculating structure factors...')
        #F_invacuo
        self.F_invacuo = myfftn(self.rho_invacuo)
        #perform B-factor sharpening to correct for B-factor sampling workaround
        if self.resolution > 2:
            Bsharp = -u2B(2)
        else:
            Bsharp = -u2B(self.resolution)
        self.F_invacuo *= np.exp(-(Bsharp)*(self.qr/(4*np.pi))**2)

        #exvol F_exvol
        self.F_exvol = myfftn(self.rho_exvol)

        #shell invacuo F_shell
        self.F_shell = myfftn(self.rho_shell)
        print('Finished structure factors...')

    def load_data(self, filename=None):
        print('Loading data...')
        if filename is None and self.data_filename is None:
            print("ERROR: No data filename given.")
        elif filename is None:
            fn = self.data_filename
        else:
            fn = filename
            self.data_filename = filename
        if self.data_filename is not None:
            Iq_exp = np.genfromtxt(fn, invalid_raise = False, usecols=(0,1,2))
            if len(Iq_exp.shape) < 2:
                print("Invalid data format. Data file must have 3 columns: q, I, errors.")
                exit()
            if Iq_exp.shape[1] < 3:
                print("Not enough columns (data must have 3 columns: q, I, errors).")
                exit()
            Iq_exp = Iq_exp[~np.isnan(Iq_exp).any(axis = 1)]
            #get rid of any data points equal to zero in the intensities or errors columns
            idx = np.where((Iq_exp[:,1]!=0)&(Iq_exp[:,2]!=0))
            Iq_exp = Iq_exp[idx]
            if self.data_units == "nm":
                Iq_exp[:,0] *= 0.1
            Iq_exp_orig = np.copy(Iq_exp)
            if self.n1 is None:
                self.n1 = 0
            if self.n2 is None:
                self.n2 = len(Iq_exp[:,0])
            self.Iq_exp = Iq_exp[self.n1:self.n2]
            self.q_exp = self.Iq_exp[:,0]
            self.I_exp = self.Iq_exp[:,1]
            self.sigq_exp = self.Iq_exp[:,2]
        else:
            self.Iq_exp = None
            self.q_exp = None
            self.I_exp = None
            self.sigq_exp = None
            self.fit_params = False

        #save an array of indices containing only desired q range for speed
        if self.data_filename:
            qmax4calc = self.q_exp.max()*1.1
        else:
            qmax4calc = self.qx_.max()*1.1
        self.qidx = np.where((self.qr<=qmax4calc))
        print('Data loaded.')

    def minimize_parameters(self, fit_radii=False):

        if fit_radii:
            self.param_names += self.modifiable_atom_types

        #generate a set of bounds
        self.bounds = np.zeros((len(self.param_names),2))

        #don't bother fitting if none is requested (default)
        if self.fit_rho0:
            self.bounds[0,0] = 0
            self.bounds[0,1] = np.inf
            self.fit_params = True
        else:
            self.bounds[0,0] = self.rho0
            self.bounds[0,1] = self.rho0
        if self.fit_shell:
            self.bounds[1,0] = -np.inf
            self.bounds[1,1] = np.inf
            self.fit_params = True
        else:
            self.bounds[1,0] = self.shell_contrast
            self.bounds[1,1] = self.shell_contrast

        if fit_radii:
            #radii_sf, i.e. radii scale factors
            self.bounds[2:,0] = 0
            self.bounds[2:,1] = np.inf
            self.params = np.append(self.params, self.radii_sf)
            self.penalty_weights = np.append(self.penalty_weights, np.ones(len(self.param_names[2:])))

        if not self.fit_all:
            #disable all fitting if requested
            self.fit_params = False

        self.params_guess = self.params
        self.params_target = self.params_guess

        if self.fit_params:
            print('Optimizing parameters...')
            print(["scale_factor"], self.param_names, ["penalty"], ["chi2"])
            print("-"*100)
            results = optimize.minimize(self.calc_score_with_modified_params, self.params_guess,
                bounds = self.bounds,
                method=self.min_method,
                options=eval(self.min_opts),
                # method='L-BFGS-B', options={'eps':0.001},
                )
            self.optimized_params = results.x
            self.optimized_chi2 = results.fun
            print('Finished minimizing parameters.')
        else:
            self.calc_score_with_modified_params(self.params)
            self.optimized_params = self.params_guess
            self.optimized_chi2 = self.chi2
        self.params = self.optimized_params 

    def calc_score_with_modified_params(self, params):
        self.calc_I_with_modified_params(params)
        self.chi2, self.exp_scale_factor = calc_chi2(self.Iq_exp, self.Iq_calc,interpolation=self.Icalc_interpolation,return_sf=True)
        self.calc_penalty(params)
        self.score = self.chi2 + self.penalty
        if self.fit_params:
            print("%.5e"%self.exp_scale_factor, ' '.join("%.5e"%param for param in params), "%.3f"%self.penalty, "%.3f"%self.chi2)
        return self.score

    def calc_penalty(self, params):
        """Calculates a penalty using quadratic loss function
        for parameters dependent on a target value for each parameter.
        """
        nparams = len(params)
        params_weights = np.ones(nparams) #note, different than penalty_weights
        params_target = self.params_target
        penalty_weight = self.penalty_weight
        penalty_weights = self.penalty_weights
        #set the individual parameter penalty weights
        #to be 1/params_target, so that each penalty 
        #is weighted as a fraction of the target rather than an
        #absolute number.
        for i in range(nparams):
            if params_target[i] != 0:
                params_weights[i] = 1/params_target[i]
        #multiply each weight be the desired individual penalty weight
        if penalty_weights is not None:
            params_weights *= penalty_weights
        #use quadratic loss function
        penalty = 1/nparams * np.sum((params_weights * (params - params_target))**2)
        penalty *= penalty_weight
        self.penalty = penalty

    def calc_F_with_modified_params(self, params):
        """Calculates structure factor sum from set of parameters"""
        #sf_ex is ratio of params[0] to initial rho0
        if self.rho0 != 0:
            sf_ex = params[0] / self.rho0
        else:
            sf_ex = 1.0
        #sf_sh is ratio of params[1] to initial shell_contrast
        if self.shell_contrast != 0:
            sf_sh = params[1] / self.shell_contrast
        else:
            sf_sh = 1.0
        self.F = self.F_invacuo*0
        self.F[self.qidx] = self.F_invacuo[self.qidx] - sf_ex * self.F_exvol[self.qidx] + sf_sh * self.F_shell[self.qidx]

    def calc_I_with_modified_params(self,params):
        """Calculates intensity profile for optimization of parameters"""
        if len(params)>2:
            #more params means we want to scale radii also
            #which means we must first recalculate the excluded volume density
            #in real space
            self.scale_radii(radii_sf = params[2:])
            self.calculate_excluded_volume(quiet=True)
            #and recalculate the F_exvol
            self.F_exvol = myfftn(self.rho_exvol)
        self.calc_F_with_modified_params(params)
        self.I3D = abs2(self.F)
        self.I_calc = mybinmean(self.I3D.ravel(), self.qblravel, xcount=self.xcount)
        self.Iq_calc = np.vstack((self.qbinsc, self.I_calc, self.I_calc*.01 + self.I_calc[0]*0.002)).T

    def calc_rho_with_modified_params(self,params):
        """Calculates electron density map for protein in solution. Includes the excluded volume and
        hydration shell calculations."""
        #sf_ex is ratio of params[0] to initial rho0
        sf_ex = params[0] / self.rho0
        #add hydration shell to density
        sf_sh = params[1] / self.shell_contrast
        self.rho_insolvent = self.rho_invacuo - sf_ex * self.rho_exvol + sf_sh * self.rho_shell

class PDB2SAS(object):
    """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.

    pdb - a saxstats PDB file object.
    q - q values to use for calculations (optional).
    """
    def __init__(self, pdb, q=None, numba=True):
        self.pdb = pdb
        if q is None:
            q = np.linspace(0,0.5,101)
        self.q = q
        self.calc_I()

    def calc_form_factors(self, B=0.0):
        """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.

        B - B-factors (i.e. Debye-Waller/temperature factors) of atoms (default=0.0)
        """
        B = np.atleast_1d(B)
        if B.shape[0] == 1:
            B = np.ones(self.pdb.natoms) * B
        self.ff = np.zeros((self.pdb.natoms,len(self.q)))
        for i in range(self.pdb.natoms):
            try:
                self.ff[i,:] = formfactor(self.pdb.atomtype[i],q=self.q,B=B[i])
            except Exception as e:
                print("pdb.atomtype unknown for atom %d"%i)
                print("attempting to use pdb.atomname instead")
                print(e)
                try:
                    self.ff[i,:] = formfactor(self.pdb.atomname[i][0],q=self.q,B=B[i])
                except Exception as e:
                    print("pdb.atomname unknown for atom %d"%i)
                    print("Defaulting to Carbon form factor.")
                    print(e)
                    self.ff[i,:] = formfactor("C",q=self.q,B=B[i])

    def calc_debye(self, natoms_limit=1000):
        """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
        """
        if self.pdb.natoms > natoms_limit:
            print("Error: Too many atoms. This function is not suitable for large macromolecules over %i atoms"%natoms_limit)
            #if natoms is too large, sinc lookup table has huge memory requirements
        else:
            if self.pdb.rij is None:
                self.pdb.rij = spatial.distance.squareform(spatial.distance.pdist(self.pdb.coords[:,:3]))
            s = np.sinc(self.q * self.pdb.rij[...,None]/np.pi)
            self.I = np.einsum('iq,jq,ijq->q',self.ff,self.ff,s)

    def calc_I(self, numba=True):
        self.calc_form_factors()
        if numba:
            try:
                #try numba function first
                self.I = calc_debye_numba(self.pdb.coords, self.q, self.ff)
            except:
                print("numba failed. Calculating debye slowly.")
                self.calc_debye()
        else:
            self.calc_debye()

if numba:
    @nb.njit(fastmath=True,parallel=True,error_model="numpy",cache=True)
    def numba_cdist(A,B):
        assert A.shape[1]==B.shape[1]
        C=np.empty((A.shape[0],B.shape[0]),A.dtype)
        
        #workaround to get the right datatype for acc
        init_val_arr=np.zeros(1,A.dtype)
        init_val=init_val_arr[0]
        
        for i in nb.prange(A.shape[0]):
            for j in range(B.shape[0]):
                acc=init_val
                for k in range(A.shape[1]):
                    acc+=(A[i,k]-B[j,k])**2
                C[i,j]=np.sqrt(acc)
        return C

    @nb.njit(fastmath=True,parallel=True,error_model="numpy",cache=True)
    def calc_debye_numba(coords, q, ff):
        """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
        This function is intended to be used with the numba njit decorator for speed.
        coords - Nx3 array of coordinates of atoms (like pdb.coords)
        q - q values to use for calculations.
        ff - an array of form factors calculated for each atom in a pdb object. q's much match q array.
        """
        nr = coords.shape[0]
        nq = q.shape[0]
        I = np.empty(nq, np.float64)
        ff_T = np.ascontiguousarray(ff.T)
        for qi in nb.prange(nq):
            acc=0
            for ri in nb.prange(nr):
                for rj in nb.prange(nr):
                    #loop through all atoms, and add the debye formulism for the pair
                    qri_j = q[qi] * np.linalg.norm(coords[ri]-coords[rj])
                    if qri_j != 0:
                        s = np.sin(qri_j)/(qri_j)
                    else:
                        s = 1.0
                    acc += ff_T[qi,ri]*ff_T[qi,rj]*s
            I[qi]=acc
        return I

def pdb2map_simple_gauss_by_radius(pdb,x,y,z,cutoff=3.0,rho0=0.334,ignore_waters=True):
    """Simple isotropic single gaussian sum at coordinate locations.

    This function only calculates the values at
    grid points near the atom for speed.

    pdb - instance of PDB class (required, must have pdb.radius attribute)
    x,y,z - meshgrids for x, y, and z (required)
    cutoff - maximum distance from atom to calculate density
    rho0 - average bulk solvent density used for excluded volume estimation (0.334 for water)
    """
    side = x[-1,0,0] - x[0,0,0]
    halfside = side/2
    n = x.shape[0]
    dx = side/n
    dV = dx**3
    V = side**3
    x_ = x[:,0,0]
    shift = np.ones(3)*dx/2.
    # print("\n Calculate density map from PDB... ")
    values = np.zeros(x.shape)
    support = np.zeros(x.shape,dtype=bool)
    # cutoff = max(cutoff,2*resolution)
    cutoffs = 2*pdb.vdW
    gxmin = x.min()
    gxmax = x.max()
    gymin = y.min()
    gymax = y.max()
    gzmin = z.min()
    gzmax = z.max()
    for i in range(pdb.coords.shape[0]):
        if ignore_waters and pdb.resname[i]=="HOH":
            continue
        if rho0 == 0:
            continue
        # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        # sys.stdout.flush()
        #this will cut out the grid points that are near the atom
        #first, get the min and max distances for each dimension
        #also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i] # for convenience, store up x,y,z coordinates of atom
        #ignore atoms whose coordinates are outside the box limits
        if (
            (xa < gxmin) or
            (xa > gxmax) or
            (ya < gymin) or
            (ya > gymax) or
            (za < gzmin) or
            (za > gzmax)
           ):
           # print()
           print("Atom %d outside boundary of cell ignored."%i)
           continue
        cutoff = cutoffs[i]
        xmin = int(np.floor((xa-cutoff)/dx)) + n//2
        xmax = int(np.ceil((xa+cutoff)/dx)) + n//2
        ymin = int(np.floor((ya-cutoff)/dx)) + n//2
        ymax = int(np.ceil((ya+cutoff)/dx)) + n//2
        zmin = int(np.floor((za-cutoff)/dx)) + n//2
        zmax = int(np.ceil((za+cutoff)/dx)) + n//2
        #handle edges
        xmin = max([xmin,0])
        xmax = min([xmax,n])
        ymin = max([ymin,0])
        ymax = min([ymax,n])
        zmin = max([zmin,0])
        zmax = min([zmax,n])
        #now lets create a slice object for convenience
        slc = np.s_[xmin:xmax,ymin:ymax,zmin:zmax]
        nx = xmax-xmin
        ny = ymax-ymin
        nz = zmax-zmin
        #now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(),y[slc].ravel(),z[slc].ravel()))
        dist = spatial.distance.cdist(pdb.coords[None,i]-shift, xyz)

        V = (4*np.pi/3)*pdb.radius[i]**3 + pdb.numH[i]*(4*np.pi/3)*pdb.exvolHradius[i]**3
        tmpvalues = realspace_gaussian_formfactor(r=dist, rho0=rho0, V=V, radius=None)

        #rescale total number of electrons by expected number of electrons
        if np.sum(tmpvalues)>1e-8:
            ne_total = rho0*V
            tmpvalues *= ne_total / np.sum(tmpvalues)

        values[slc] += tmpvalues.reshape(nx,ny,nz)
        support[slc] = True
    return values, support

def pdb2map_multigauss(pdb,x,y,z,cutoff=3.0,resolution=None,use_b=False,ignore_waters=True):
    """5-term gaussian sum at coordinate locations using Cromer-Mann coefficients.

    This function only calculates the values at
    grid points near the atom for speed.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    sigma - width of Gaussian, i.e. effectively resolution
    cutoff - maximum distance from atom to calculate density
    resolution - desired resolution of density map, calculated as a B-factor
    corresonding to atomic displacement equal to resolution.
    """
    side = x[-1,0,0] - x[0,0,0]
    halfside = side/2
    n = x.shape[0]
    dx = side/n
    dV = dx**3
    V = side**3
    x_ = x[:,0,0]
    shift = np.ones(3)*dx/2.
    # print("\n Calculate density map from PDB... ")
    values = np.zeros(x.shape)
    support = np.zeros(x.shape,dtype=bool)
    if resolution is None:
        resolution = 0.0
    cutoff = max(cutoff,2*resolution)
    #convert resolution to B-factor for form factor calculation
    #set resolution equal to atomic displacement
    if use_b:
        u = B2u(pdb.b)
    else:
        u = np.zeros(pdb.natoms)
    u += resolution
    B = u2B(u)
    gxmin = x.min()
    gxmax = x.max()
    gymin = y.min()
    gymax = y.max()
    gzmin = z.min()
    gzmax = z.max()
    for i in range(pdb.coords.shape[0]):
        if ignore_waters and pdb.resname[i]=="HOH":
            continue
        # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        # sys.stdout.flush()
        #this will cut out the grid points that are near the atom
        #first, get the min and max distances for each dimension
        #also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i] # for convenience, store up x,y,z coordinates of atom
        #ignore atoms whose coordinates are outside the box limits
        if (
            (xa < gxmin) or
            (xa > gxmax) or
            (ya < gymin) or
            (ya > gymax) or
            (za < gzmin) or
            (za > gzmax)
           ):
           # print()
           print("Atom %d outside boundary of cell ignored."%i)
           continue
        xmin = int(np.floor((xa-cutoff)/dx)) + n//2
        xmax = int(np.ceil((xa+cutoff)/dx)) + n//2
        ymin = int(np.floor((ya-cutoff)/dx)) + n//2
        ymax = int(np.ceil((ya+cutoff)/dx)) + n//2
        zmin = int(np.floor((za-cutoff)/dx)) + n//2
        zmax = int(np.ceil((za+cutoff)/dx)) + n//2
        #handle edges
        xmin = max([xmin,0])
        xmax = min([xmax,n])
        ymin = max([ymin,0])
        ymax = min([ymax,n])
        zmin = max([zmin,0])
        zmax = min([zmax,n])
        #now lets create a slice object for convenience
        slc = np.s_[xmin:xmax,ymin:ymax,zmin:zmax]
        nx = xmax-xmin
        ny = ymax-ymin
        nz = zmax-zmin
        #now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(),y[slc].ravel(),z[slc].ravel()))
        dist = spatial.distance.cdist(pdb.coords[None,i]-shift, xyz)[0]
        try:
            element = pdb.atomtype[i]
            ffcoeff[element]
        except:
            try:
                element = pdb.atomname[i][0].upper()+pdb.atomname[i][1].lower()
                ffcoeff[element]
            except:
                try:
                    element = pdb.atomname[i][0]
                    ffcoeff[element]
                except:
                    print("Atom type %s or name not recognized for atom # %s"
                           % (pdb.atomtype[i],
                              pdb.atomname[i][0].upper()+pdb.atomname[i][1].lower(),
                              i))
                    print("Using default form factor for Carbon")
                    element = 'C'
                    ffcoeff[element]


        if pdb.numH[i] > 0:
            Va = V_without_impH = sphere_volume_from_radius(pdb.radius[i])
            Vb = V_with_impH = V_without_impH + pdb.numH[i]*sphere_volume_from_radius(pdb.exvolHradius[i])
            ra = sphere_radius_from_volume(Va)
            rb = sphere_radius_from_volume(Vb)
            Ba = u2B(ra*2)
            Bb = u2B(rb*2)
            Bdiff = Bb - Ba
        else:
            Bdiff = 0.0
        tmpvalues = realspace_formfactor(element=element,r=dist,B=B[i]+Bdiff)
        #rescale total number of electrons by expected number of electrons
        #pdb.nelectrons is already corrected with the number of electrons including hydrogens
        if np.sum(tmpvalues)>1e-8:
            ne_total = pdb.nelectrons[i]
            tmpvalues *= ne_total / tmpvalues.sum()

        values[slc] += tmpvalues.reshape(nx,ny,nz)
        support[slc] = True
    return values, support

def pdb2F_multigauss(pdb,qx,qy,qz,qr=None,radii=None,B=None):
    """Calculate structure factors F from pdb coordinates.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    radii - float or list of radii of atoms in pdb (optional, uses spherical form factor rather than Kromer-Mann)
    """
    radii = np.atleast_1d(radii)
    if qr is None:
        qr = (qx**2+qy**2+qz**2)**0.5
    n = qr.shape[0]
    F = np.zeros(qr.shape,dtype=complex)
    if B is None:
        B = np.zeros(pdb.natoms)
    if radii[0] is None:
        useradii = False
    else:
        useradii = True
        radii = np.ones(radii.size)*radii
    for i in range(pdb.natoms):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.natoms))
        sys.stdout.flush()
        Fatom = formfactor(element=pdb.atomtype[i],q=qr,B=B[i]) * np.exp(-1j * (qx*pdb.coords[i,0] + qy*pdb.coords[i,1] + qz*pdb.coords[i,2]))
        ne_total = electrons[pdb.atomtype[i]] + pdb.numH[i]
        Fatom *= ne_total/Fatom[0,0,0].real
        F += Fatom
    return F

def pdb2F_simple_gauss_by_radius(pdb,qx,qy,qz,qr=None,rho0=0.334,radii=None,B=None):
    """Calculate structure factors F from pdb coordinates.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    radii - float or list of radii of atoms in pdb (optional, uses spherical form factor rather than Kromer-Mann)
    """
    radii = np.atleast_1d(radii)
    if qr is None:
        qr = (qx**2+qy**2+qz**2)**0.5
    n = qr.shape[0]
    F = np.zeros(qr.shape,dtype=complex)
    if B is None:
        B = np.zeros(pdb.natoms)
    if radii[0] is None:
        useradii = False
    else:
        useradii = True
        radii = np.ones(radii.size)*radii
    for i in range(pdb.natoms):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.natoms))
        sys.stdout.flush()
        V = (4*np.pi/3)*pdb.radius[i]**3 + pdb.numH[i]*(4*np.pi/3)*pdb.exvolHradius**3
        Fatom = reciprocalspace_gaussian_formfactor(q=qr,rho0=rho0,V=V) * np.exp(-1j * (qx*pdb.coords[i,0] + qy*pdb.coords[i,1] + qz*pdb.coords[i,2]))
        ne_total = rho0 * V
        Fatom *= ne_total/Fatom[0,0,0].real
        F += Fatom
    return F

def pdb2map_FFT(pdb,x,y,z,radii=None,restrict=True):
    """Calculate electron density from pdb coordinates by FFT of Fs.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    radii - float or list of radii of atoms in pdb (optional, uses spherical form factor rather than Kromer-Mann)
    """
    radii = np.atleast_1d(radii)
    side = x[-1,0,0] - x[0,0,0]
    halfside = side/2
    n = x.shape[0]
    dx = side/n
    dV = dx**3
    V = side**3
    x_ = x[:,0,0]
    df = 1/side
    qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
    qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)
    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = mybinmean(qr.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
    F = np.zeros(qr.shape,dtype=complex)
    natoms = pdb.coords.shape[0]
    if radii[0] is None:
        useradii = False
    else:
        useradii = True
        radii = np.ones(radii.size)*radii
    for i in range(pdb.natoms):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.natoms))
        sys.stdout.flush()
        if useradii:
            F += sphere(q=qr, R=radii[i], I0=pdb.nelectrons[i],amp=True) * np.exp(-1j * (qx*pdb.coords[i,0] + qy*pdb.coords[i,1] + qz*pdb.coords[i,2]))
        else:
            F += formfactor(element=pdb.atomtype[i],q=qr) * np.exp(-1j * (qx*pdb.coords[i,0] + qy*pdb.coords[i,1] + qz*pdb.coords[i,2]))
    I3D = abs2(F)
    Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
    rho = myifftn(F).real
    # rho[rho<0] = 0
    #need to shift rho to center of grid, since FFT is offset by half a grid length
    shift = [n//2-1,n//2-1,n//2-1]
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    if restrict:
        xyz = np.column_stack([x.flat,y.flat,z.flat])
        pdb.coords -= np.ones(3)*dx/2.
        pdbidx = pdb2support(pdb, xyz=xyz,probe=0.0)
        rho[~pdbidx] = 0.0
    return rho, pdbidx

def pdb2support_fast(pdb,x,y,z,radius=None,probe=0.0):
    """Return a boolean 3D density map with support from PDB coordinates"""

    support = np.zeros(x.shape,dtype=np.bool_)
    n = x.shape[0]
    side = x.max()-x.min()
    dx = side/n
    shift = np.ones(3)*dx/2.

    if radius is None:
        radius = pdb.vdW

    radius = np.atleast_1d(radius)
    if len(radius) != pdb.natoms:
        print("Error: radius argument does not have same length as pdb.")
        exit()

    dr = radius + probe

    natoms = pdb.natoms
    for i in range(natoms):
        #sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        #sys.stdout.flush()
        #if a grid point of env is within the desired distance, dr, of
        #the atom coordinate, add it to env
        #to save memory, only run the distance matrix one atom at a time
        #and will only look at grid points within a box of size dr near the atom
        #this will cut out the grid points that are near the atom
        #first, get the min and max distances for each dimension
        #also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i] # for convenience, store up x,y,z coordinates of atom
        xmin = int(np.floor((xa-dr[i])/dx)) + n//2
        xmax = int(np.ceil((xa+dr[i])/dx)) + n//2
        ymin = int(np.floor((ya-dr[i])/dx)) + n//2
        ymax = int(np.ceil((ya+dr[i])/dx)) + n//2
        zmin = int(np.floor((za-dr[i])/dx)) + n//2
        zmax = int(np.ceil((za+dr[i])/dx)) + n//2
        #handle edges
        xmin = max([xmin,0])
        xmax = min([xmax,n])
        ymin = max([ymin,0])
        ymax = min([ymax,n])
        zmin = max([zmin,0])
        zmax = min([zmax,n])
        #now lets create a slice object for convenience
        slc = np.s_[xmin:xmax,ymin:ymax,zmin:zmax]
        nx = xmax-xmin
        ny = ymax-ymin
        nz = zmax-zmin
        #now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(),y[slc].ravel(),z[slc].ravel()))
        #now calculate all distances from the atom to the minigrid points
        dist = spatial.distance.cdist(pdb.coords[None,i]-shift, xyz)
        #now, add any grid points within dr of atom to the env grid
        #first, create a dummy array to hold booleans of size dist.size
        tmpenv = np.zeros(dist.shape,dtype=np.bool_)
        #now, any elements that have a dist less than dr make true
        tmpenv[dist<=dr[i]] = True
        #now reshape for inserting into env
        tmpenv = tmpenv.reshape(nx,ny,nz)
        support[slc] += tmpenv
    return support

def u2B(u):
    """Calculate B-factor from atomic displacement, u"""
    return np.sign(u) * 8 * np.pi**2 * u**2

def B2u(B):
    """Calculate atomic displacement, u, from B-factor"""
    return np.sign(B)*(np.abs(B)/(8*np.pi**2))**0.5

def v2B(v):
    """Calculate B-factor from atomic volume displacement, v"""
    u = sphere_radius_from_volume(v)
    return np.sign(u) * 8 * np.pi**2 * u**2

def sphere(R, q=np.linspace(0,0.5,501), I0=1.,amp=False):
    """Calculate the scattering of a uniform sphere."""
    q = np.atleast_1d(q)
    a = np.where(q==0.0,1.0,(3 * (np.sin(q*R)-q*R*np.cos(q*R))/(q*R)**3))
    if np.isnan(a).any():
        a[np.where(np.isnan(a))] = 1.
    if amp:
        return I0 * a
    else:
        return I0 * a**2

def formfactor(element, q=(np.arange(500)+1)/1000.,B=None):
    """Calculate atomic form factors"""
    if B is None:
        B = 0.0
    q = np.atleast_1d(q)
    ff = np.zeros(q.shape)
    for i in range(4):
        # print(ffcoeff[element]['a'][i])
        ff += ffcoeff[element]['a'][i] * np.exp(-ffcoeff[element]['b'][i]*(q/(4*np.pi))**2)
    # ff += ffcoeff[element]['c']
    ff *= np.exp(-B* (q / (4*np.pi))**2)
    # print(ff[q==0])
    return ff

def realspace_formfactor(element, r=(np.arange(501))/1000., B=None):
    """Calculate real space atomic form factors"""
    if B is None:
        B = 0.0
    r = np.atleast_1d(r)
    ff = np.zeros(r.shape)
    for i in range(4):
        ai = ffcoeff[element]['a'][i]
        bi = ffcoeff[element]['b'][i]
        ff += (4*np.pi/(bi+B))**(3/2.)* ai * np.exp(-4 * np.pi**2 * r**2 /(bi+B))
    # i = np.where((r==0))
    # ff += signal.unit_impulse(r.shape, i) * ffcoeff[element]['c']
    return ff

def reciprocalspace_gaussian_formfactor(q=np.linspace(0,0.5,501), rho0=0.334, V=None, radius=None):
    """Calculate reciprocal space atomic form factors assuming an isotropic gaussian sphere (for excluded volume)."""
    if (V is None) and (radius is None):
        print("Error: either radius or volume of atom must be given.")
        exit()
    elif V is None:
        #calculate volume from radius assuming sphere
        V = (4*np.pi/3)*radius**3
    ff = rho0 * V * np.exp(-q**2*V**(2./3)/(4*np.pi))
    return ff

def realspace_gaussian_formfactor(r=np.linspace(-3,3,101), rho0=0.334, V=None, radius=None):
    """Calculate real space atomic form factors assuming an isotropic gaussian sphere (for excluded volume)."""
    if (V is None) and (radius is None):
        print("Error: either radius or volume of atom must be given.")
        exit()
    elif V is None:
        #calculate volume from radius assuming sphere
        V = (4*np.pi/3)*radius**3
    if V <= 0:
        ff = r*0
    else:
        ff = rho0 * np.exp(-np.pi*r**2/V**(2./3))
    return ff

def estimate_side_from_pdb(pdb):
    #roughly estimate maximum dimension
    #calculate max distance along x, y, z
    #take the maximum of the three
    #triple that value to set the default side
    #i.e. set oversampling to 3, like in denss
    if pdb.rij is not None:
        #if pdb.rij has already been calculated
        #then just take the Dmax from that
        D = np.max(pdb.rij)
    else:
        #if pdb.rij has not been calculated,
        #rather than calculating the whole distance
        #matrix, which can be slow and memory intensive
        #for large models, just approximate the maximum
        #length as the max of the range of x, y, or z 
        #values of the coordinates.
        xmin = np.min(pdb.coords[:,0]) - 1.7
        xmax = np.max(pdb.coords[:,0]) + 1.7
        ymin = np.min(pdb.coords[:,1]) - 1.7
        ymax = np.max(pdb.coords[:,1]) + 1.7
        zmin = np.min(pdb.coords[:,2]) - 1.7
        zmax = np.max(pdb.coords[:,2]) + 1.7
        wx = xmax-xmin
        wy = ymax-ymin
        wz = zmax-zmin
        D = np.max([wx,wy,wz])
    side = 3*D
    return side

def calc_chi2(Iq_exp, Iq_calc, scale=True, offset=True, interpolation=True,return_sf=False,return_fit=False):
    """Calculates a final score comparing experimental vs calculated intensity profiles
    for optimization of parameters defined in pdb2mrc. Score includes the chi2 and penalty 
    due to parameter variation from target parameter.

    Iq_exp - Experimental data, q, I, sigq (required)
    Iq_calc - calculated scattering profile's q, I, and sigq (required)
    scale (boolean) - Scale I_calc to I_exp
    """
    q_exp = Iq_exp[:,0]
    I_exp = Iq_exp[:,1]
    sigq_exp = Iq_exp[:,2]
    q_calc = Iq_calc[:,0]
    I_calc = Iq_calc[:,1]
    if interpolation:
        I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')
        I_calc = I_calc_interpolator(q_exp)
    else:
        #if interpolation of (coarse) calculated profile is disabled, we still need to at least
        #put the experimental data on the correct grid for comparison, so regrid the exp arrays
        #with simple 1D linear interpolation
        I_exp = np.interp(q_calc, q_exp, I_exp)
        sigq_exp = np.interp(q_calc, q_exp, sigq_exp)
        q_exp = np.copy(q_calc)

    if scale and offset:
        calc = np.vstack((I_calc/sigq_exp, np.ones(len(sigq_exp))*1/sigq_exp))
        exp = I_exp/sigq_exp
        exp_scale_factor, offset = _fit_by_least_squares(exp, calc)
    elif scale:
        exp_scale_factor = _fit_by_least_squares(I_calc/sigq_exp,I_exp/sigq_exp)
        offset = 0.0
    else:
        exp_scale_factor = 1.0
        offset = 0.0
    I_calc *= exp_scale_factor
    I_calc += offset
    chi2 = 1/len(q_exp) * np.sum(((I_exp-I_calc)/sigq_exp)**2)
    fit = np.vstack((q_exp,I_exp,sigq_exp,I_calc)).T
    if return_sf and return_fit:
        return chi2, exp_scale_factor, offset, fit
    elif return_sf and not return_fit:
        return chi2, exp_scale_factor
    elif not return_sf and return_fit:
        return chi2, fit
    else:
        return chi2

def calc_uniform_shell(pdb,x,y,z,thickness,distance=1.4):
    """create a one angstrom uniform layer around the particle

    Centered one water molecule radius away from the particle surface,
    #which means add the radius of a water molecule (1.4 A) to the radius of
    #the pdb surface atom (say 1.7 A), for a total of 1.4+1.7 from the pdb coordinates
    #since that is the center of the shell, and we want 1 A thick shell before blurring,
    #subtract 0.5 A from the inner support, and add 0.5 A for the outer support,
    #then subtract the inner support from the outer support

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    thickness - thickness of the shell (required)
    """
    r_water = 1.4
    inner_support = pdb2support_fast(pdb,x,y,z,radius=pdb.vdW,probe=distance-thickness/2)
    outer_support = pdb2support_fast(pdb,x,y,z,radius=pdb.vdW,probe=distance+thickness/2)
    shell_idx = outer_support
    shell_idx[inner_support] = False
    shell = shell_idx * 1.0
    return shell

def denss_3DFs(rho_start, dmax, ne=None, voxel=5., oversampling=3., positivity=True,
        output="map", steps=2001, seed=None, shrinkwrap=True, shrinkwrap_sigma_start=3,
        shrinkwrap_sigma_end=1.5, shrinkwrap_sigma_decay=0.99, shrinkwrap_threshold_fraction=0.2,
        shrinkwrap_iter=20, shrinkwrap_minstep=50, write_freq=100,support=None,
        enforce_connectivity=True, enforce_connectivity_steps=[6000],quiet=False):
    """Calculate electron density from starting map by refining phases only."""
    D = dmax
    side = oversampling*D
    halfside = side/2
    n = int(side/voxel)
    #want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n%2==1: n += 1
    #store n for later use if needed
    nbox = n
    dx = side/n
    dV = dx**3
    V = side**3
    x_ = np.linspace(-halfside,halfside,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)
    df = 1/side
    qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
    qz_ = np.fft.rfftfreq(x_.size)*n*df*2*np.pi
    qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)
    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    qbinsc[1:] += qstep/2.
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    if steps == 'None' or steps is None or steps < 1:
        steps = int(shrinkwrap_iter * (np.log(shrinkwrap_sigma_end/shrinkwrap_sigma_start)/np.log(shrinkwrap_sigma_decay)) + shrinkwrap_minstep)
        steps += 3000
    else:
        steps = np.int(steps)
    Imean = np.zeros((steps+1,len(qbins)))
    chi = np.zeros((steps+1))
    rg = np.zeros((steps+1))
    supportV = np.zeros((steps+1))
    chibest = np.inf
    usesupport = True
    if support is None:
        support = np.ones(x.shape,dtype=bool)
    else:
        support = support.astype(bool)
    update_support = True
    sigma = shrinkwrap_sigma_start

    rho = rho_start
    F = np.fft.fftn(rho)
    Amp = np.abs(F)

    if not quiet:
        print("\n Step     Chi2     Rg    Support Volume")
        print(" ----- --------- ------- --------------")

    for j in range(steps):
        F = np.fft.fftn(rho)
        #APPLY RECIPROCAL SPACE RESTRAINTS
        #calculate spherical average of intensities from 3D Fs
        I3D = np.abs(F)**2
        Imean[j] = ndimage.mean(I3D, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
        #scale Fs to match data
        F *= Amp/np.abs(F)
        chi[j] = 1.0
        #APPLY REAL SPACE RESTRAINTS
        rhoprime = np.fft.ifftn(F,rho.shape)
        rhoprime = rhoprime.real
        if j%write_freq == 0:
            write_mrc(rhoprime/dV,side,output+"_current.mrc")
        rg[j] = rho2rg(rhoprime,r=r,support=support,dx=dx)
        newrho = np.zeros_like(rho)
        #Error Reduction
        newrho[support] = rhoprime[support]
        newrho[~support] = 0.0
        #enforce positivity by making all negative density points zero.
        if positivity:
            netmp = np.sum(newrho)
            newrho[newrho<0] = 0.0
            if np.sum(newrho) != 0:
                newrho *= netmp / np.sum(newrho)
        supportV[j] = np.sum(support)*dV

        if not quiet:
            sys.stdout.write("\r% 5i % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
            sys.stdout.flush()

        rho = newrho

    if not quiet:
        print()

    F = np.fft.fftn(rho)
    #calculate spherical average intensity from 3D Fs
    Imean[j+1] = ndimage.mean(np.abs(F)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    #scale Fs to match data
    F *= Amp/np.abs(F)
    rho = np.fft.ifftn(F,rho.shape)
    rho = rho.real

    #scale total number of electrons
    if ne is not None:
        rho *= ne / np.sum(rho)

    rg[j+1] = rho2rg(rho=rho,r=r,support=support,dx=dx)
    supportV[j+1] = supportV[j]

    #change rho to be the electron density in e-/angstroms^3, rather than number of electrons,
    #which is what the FFT assumes
    rho /= dV

    return rho


# electrons = {'H': 1,'HE': 2,'He': 2,'LI': 3,'Li': 3,'BE': 4,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'NE': 10,'Ne': 10,'NA': 11,'Na': 11,'MG': 12,'Mg': 12,'AL': 13,'Al': 13,'SI': 14,'Si': 14,'P': 15,'S': 16,'CL': 17,'Cl': 17,'AR': 18,'Ar': 18,'K': 19,'CA': 20,'Ca': 20,'SC': 21,'Sc': 21,'TI': 22,'Ti': 22,'V': 23,'CR': 24,'Cr': 24,'MN': 25,'Mn': 25,'FE': 26,'Fe': 26,'CO': 27,'Co': 27,'NI': 28,'Ni': 28,'CU': 29,'Cu': 29,'ZN': 30,'Zn': 30,'GA': 31,'Ga': 31,'GE': 32,'Ge': 32,'AS': 33,'As': 33,'SE': 34,'Se': 34,'Se': 34,'Se': 34,'BR': 35,'Br': 35,'KR': 36,'Kr': 36,'RB': 37,'Rb': 37,'SR': 38,'Sr': 38,'Y': 39,'ZR': 40,'Zr': 40,'NB': 41,'Nb': 41,'MO': 42,'Mo': 42,'TC': 43,'Tc': 43,'RU': 44,'Ru': 44,'RH': 45,'Rh': 45,'PD': 46,'Pd': 46,'AG': 47,'Ag': 47,'CD': 48,'Cd': 48,'IN': 49,'In': 49,'SN': 50,'Sn': 50,'SB': 51,'Sb': 51,'TE': 52,'Te': 52,'I': 53,'XE': 54,'Xe': 54,'CS': 55,'Cs': 55,'BA': 56,'Ba': 56,'LA': 57,'La': 57,'CE': 58,'Ce': 58,'PR': 59,'Pr': 59,'ND': 60,'Nd': 60,'PM': 61,'Pm': 61,'SM': 62,'Sm': 62,'EU': 63,'Eu': 63,'GD': 64,'Gd': 64,'TB': 65,'Tb': 65,'DY': 66,'Dy': 66,'HO': 67,'Ho': 67,'ER': 68,'Er': 68,'TM': 69,'Tm': 69,'YB': 70,'Yb': 70,'LU': 71,'Lu': 71,'HF': 72,'Hf': 72,'TA': 73,'Ta': 73,'W': 74,'RE': 75,'Re': 75,'OS': 76,'Os': 76,'IR': 77,'Ir': 77,'PT': 78,'Pt': 78,'AU': 79,'Au': 79,'HG': 80,'Hg': 80,'TL': 81,'Tl': 81,'PB': 82,'Pb': 82,'BI': 83,'Bi': 83,'PO': 84,'Po': 84,'AT': 85,'At': 85,'RN': 86,'Rn': 86,'FR': 87,'Fr': 87,'RA': 88,'Ra': 88,'AC': 89,'Ac': 89,'TH': 90,'Th': 90,'PA': 91,'Pa': 91,'U': 92,'NP': 93,'Np': 93,'PU': 94,'Pu': 94,'AM': 95,'Am': 95,'CM': 96,'Cm': 96,'BK': 97,'Bk': 97,'CF': 98,'Cf': 98,'ES': 99,'Es': 99,'FM': 100,'Fm': 100,'MD': 101,'Md': 101,'NO': 102,'No': 102,'LR': 103,'Lr': 103,'RF': 104,'Rf': 104,'DB': 105,'Db': 105,'SG': 106,'Sg': 106,'BH': 107,'Bh': 107,'HS': 108,'Hs': 108,'MT': 109,'Mt': 109}

# vdW = {
#      "H":  1.20 ,
#      "D":  1.20 ,
#      "He": 1.40 ,
#      "Li": 1.82 ,
#      "Be": 0.63 ,
#      "B":  1.75 ,
#      "C":  1.70 ,
#      "N":  1.55 ,
#      "O":  1.52 ,
#      "F":  1.47 ,
#      "Ne": 1.54 ,
#      "Na": 2.27 ,
#      "Mg": 1.73 ,
#      "Al": 1.50 ,
#      "Si": 2.10 ,
#      "P":  1.90 ,
#      "S":  1.80 ,
#      "Cl": 1.75 ,
#      "Ar": 1.88 ,
#      "K":  2.75 ,
#      "Ca": 1.95 ,
#      "Sc": 1.32 ,
#      "Ti": 1.95 ,
#      "V":  1.06 ,
#      "Cr": 1.13 ,
#      "Mn": 1.19 ,
#      "Fe": 1.26 ,
#      "Co": 1.13 ,
#      "Ni": 1.63 ,
#      "Cu": 1.40 ,
#      "Zn": 1.39 ,
#      "Ga": 1.87 ,
#      "Ge": 1.48 ,
#      "As": 0.83 ,
#      "Se": 1.90 ,
#      "Br": 1.85 ,
#      "Kr": 2.02 ,
#      "Rb": 2.65 ,
#      "Sr": 2.02 ,
#      "Y":  1.61 ,
#      "Zr": 1.42 ,
#      "Nb": 1.33 ,
#      "Mo": 1.75 ,
#      "Tc": 2.00 ,
#      "Ru": 1.20 ,
#      "Rh": 1.22 ,
#      "Pd": 1.63 ,
#      "Ag": 1.72 ,
#      "Cd": 1.58 ,
#      "In": 1.93 ,
#      "Sn": 2.17 ,
#      "Sb": 1.12 ,
#      "Te": 1.26 ,
#      "I":  1.98 ,
#      "Xe": 2.16 ,
#      "Cs": 3.01 ,
#      "Ba": 2.41 ,
#      "La": 1.83 ,
#      "Ce": 1.86 ,
#      "Pr": 1.62 ,
#      "Nd": 1.79 ,
#      "Pm": 1.76 ,
#      "Sm": 1.74 ,
#      "Eu": 1.96 ,
#      "Gd": 1.69 ,
#      "Tb": 1.66 ,
#      "Dy": 1.63 ,
#      "Ho": 1.61 ,
#      "Er": 1.59 ,
#      "Tm": 1.57 ,
#      "Yb": 1.54 ,
#      "Lu": 1.53 ,
#      "Hf": 1.40 ,
#      "Ta": 1.22 ,
#      "W":  1.26 ,
#      "Re": 1.30 ,
#      "Os": 1.58 ,
#      "Ir": 1.22 ,
#      "Pt": 1.72 ,
#      "Au": 1.66 ,
#      "Hg": 1.55 ,
#      "Tl": 1.96 ,
#      "Pb": 2.02 ,
#      "Bi": 1.73 ,
#      "Po": 1.21 ,
#      "At": 1.12 ,
#      "Rn": 2.30 ,
#      "Fr": 3.24 ,
#      "Ra": 2.57 ,
#      "Ac": 2.12 ,
#      "Th": 1.84 ,
#      "Pa": 1.60 ,
#      "U":  1.75 ,
#      "Np": 1.71 ,
#      "Pu": 1.67 ,
#      "Am": 1.66 ,
#      "Cm": 1.65 ,
#      "Bk": 1.64 ,
#      "Cf": 1.63 ,
#      "Es": 1.62 ,
#      "Fm": 1.61 ,
#      "Md": 1.60 ,
#      "No": 1.59 ,
#      "Lr": 1.58 ,
#       }

# radii_sf_dict = {'H':1.10113e+00, 
#                  'C':1.24599e+00, 
#                  'N':1.02375e+00, 
#                  'O':1.05142e+00,
#                  }

# atomic_volumes = {
#  ' CA': {'CA': 24.851127348632804},
#  ' DC': {"C1'": 9.563317626953124,
#          'C2': 10.676659082031248,
#          "C2'": 10.573889101562498,
#          "C3'": 8.712610566406248,
#          'C4': 9.611847895507811,
#          "C4'": 8.706901123046874,
#          'C5': 10.66238547363281,
#          "C5'": 11.293278964843747,
#          'C6': 11.387484780273436,
#          "H1'": 2.2721934374999995,
#          "H2'": 2.3319351562499997,
#          "H3'": 2.2330349999999997,
#          "H4'": 2.3444859374999996,
#          'H41': 2.3836443749999994,
#          'H42': 2.419790624999999,
#          'H5': 2.1286124999999996,
#          "H5'": 2.3520164062499997,
#          'H6': 2.0141493749999997,
#          'HO3': 3.0162037499999994,
#          'N1': 4.236671497802735,
#          'N3': 6.709866350708008,
#          'N4': 8.427903720092775,
#          'O2': 9.153918515,
#          "O3'": 7.1970286675,
#          "O4'": 7.14193375,
#          "O5'": 6.21960402,
#          'OP1': 9.227378405,
#          'OP2': 8.8151868,
#          'P': 17.38455079101562},
#  ' DG': {"C1'": 9.526206245117185,
#          'C2': 10.348366088867186,
#          "C2'": 10.476828564453124,
#          "C3'": 8.635533081054687,
#          'C4': 9.471966533203123,
#          "C4'": 8.655516132812497,
#          'C5': 8.698336958007811,
#          "C5'": 11.301843129882812,
#          'C6': 9.80596896972656,
#          'C8': 12.392346811523437,
#          'H1': 2.2982990624999995,
#          "H1'": 2.3083396874999993,
#          "H2'": 2.2601446874999995,
#          'H21': 2.9077649999999995,
#          'H22': 2.2561284374999993,
#          "H3'": 2.2671731249999993,
#          "H4'": 2.3444859375,
#          "H5'": 2.3269148437499996,
#          'H8': 2.0613403124999996,
#          'HO5': 2.9921062499999995,
#          'N1': 6.108336893920899,
#          'N2': 8.42141239501953,
#          'N3': 7.363326408081056,
#          'N7': 7.486661584472657,
#          'N9': 4.191232222290039,
#          "O3'": 6.464470320000001,
#          "O4'": 7.154177065,
#          "O5'": 6.8542158475,
#          'O6': 9.180445697500002,
#          'OP1': 9.250504666666666,
#          'OP2': 8.951223633333333,
#          'P': 17.42971927083333},
#  ' MG': {'MG': 13.562483594160154},
#  ' MN': {'MN': 4.942847818398437},
#  'ACE': {'C': 9.843080351562499,
#          'CH3': 13.097463066406247,
#          'H1': 2.3414737499999996,
#          'H2': 2.3414737499999996,
#          'H3': 2.3495062499999997,
#          'O': 9.557947910000001},
#  'ALA': {'C': 9.782161344866072,
#          'CA': 8.845305432477677,
#          'CB': 13.007407041515814,
#          'H': 2.530625075392038,
#          'H1': 2.9057568749999994,
#          'H2': 3.04833375,
#          'H3': 2.9860818749999996,
#          'HA': 2.3171754374999995,
#          'HB1': 2.3470475213414628,
#          'HB2': 2.343202696646341,
#          'HB3': 2.336130178353658,
#          'N': 6.15379694998605,
#          'O': 9.449752084285715},
#  'ARG': {'C': 9.75829353465082,
#          'CA': 8.66752601551669,
#          'CB': 10.305345485727438,
#          'CD': 11.104742764311078,
#          'CG': 10.31731838592752,
#          'CZ': 10.483496527500568,
#          'H': 2.5147292814232896,
#          'HA': 2.29716137245841,
#          'HB2': 2.297236368613138,
#          'HB3': 2.290867534215328,
#          'HD2': 2.2518506557377047,
#          'HD3': 2.2826931147540983,
#          'HE': 2.6838004311131383,
#          'HG2': 2.2662822477272724,
#          'HG3': 2.2938337227272725,
#          'HH1': 2.663250130018248,
#          'HH2': 2.6736975102645983,
#          'N': 6.125203140275713,
#          'NE': 6.126187052546712,
#          'NH1': 8.335556329087307,
#          'NH2': 8.294365633732037,
#          'O': 9.42438035892791},
#  'ASN': {'C': 9.765317095081178,
#          'CA': 8.762942977697275,
#          'CB': 10.457775336328123,
#          'CG': 9.962584907394934,
#          'H': 2.6003176588983044,
#          'HA': 2.2827102749999995,
#          'HB2': 2.3314331249999998,
#          'HB3': 2.3226389224137924,
#          'HD2': 2.69267057650862,
#          'N': 6.182690973111797,
#          'ND2': 8.28526866532249,
#          'O': 9.405647455251396,
#          'OD1': 9.59240275623563},
#  'ASP': {'C': 9.77802319280879,
#          'CA': 8.751966759058556,
#          'CB': 10.466455551743147,
#          'CG': 10.047980162790008,
#          'H': 2.5747201502225514,
#          'HA': 2.3005574493243235,
#          'HB2': 2.2972767443181814,
#          'HB3': 2.3168285795454544,
#          'N': 6.0875017308519865,
#          'O': 9.411064460311573,
#          'OD1': 9.427129606479514,
#          'OD2': 9.5444845985129,
#          'OXT': 9.4681636},
#  'CYS': {'C': 9.716675931140985,
#          'CA': 8.735182784338662,
#          'CB': 10.754024252513322,
#          'H': 2.592442674418604,
#          'HA': 2.319462209302325,
#          'HB2': 2.3084097383720925,
#          'HB3': 2.3299542732558134,
#          'HG': 2.6972509090909087,
#          'N': 6.1020468502452765,
#          'O': 9.429377284263566,
#          'SG': 18.884944556686047},
#  'DOM': {'C1': 10.859361269531247,
#          "C1'": 10.825104609374998,
#          'C2': 9.785985917968748,
#          "C2'": 12.298140996093748,
#          'C3': 9.64895927734375,
#          "C3'": 9.900174785156247,
#          'C4': 10.014363652343748,
#          "C4'": 10.060039199218748,
#          'C5': 10.014363652343748,
#          "C5'": 9.917303115234372,
#          'C6': 13.451448554687497,
#          "C6'": 13.177395273437497,
#          'O1': 7.419448890000001,
#          "O1'": 10.42314217,
#          'O2': 9.753840949999999,
#          'O3': 10.34968228,
#          "O3'": 10.39865554,
#          'O4': 10.60271079,
#          'O5': 7.403124470000001,
#          "O5'": 6.7644315375,
#          'O6': 10.545575320000001,
#          "O6'": 9.94157178},
#  'DTT': {'C1': 12.800572011718748,
#          'C2': 10.322673593749998,
#          'C3': 10.276998046874999,
#          'C4': 13.051787519531247,
#          'O2': 10.51292648,
#          'O3': 10.292546810000001,
#          'S1': 18.21771,
#          'S4': 18.2990390625},
#  'GLN': {'C': 9.713547355346678,
#          'CA': 8.663978343505859,
#          'CB': 10.337915768432614,
#          'CD': 10.038073880047401,
#          'CG': 10.457647400807584,
#          'H': 2.585380253906249,
#          'HA': 2.289872109374999,
#          'HB2': 2.317628957865168,
#          'HB3': 2.31688888483146,
#          'HE2': 2.795350613764044,
#          'HG2': 2.308630752808988,
#          'HG3': 2.296239042134831,
#          'N': 6.023543960151672,
#          'NE2': 8.414896757867364,
#          'O': 9.396908235468752,
#          'OE1': 9.607232984764046},
#  'GLU': {'C': 9.74631094754227,
#          'CA': 8.701385768413596,
#          'CB': 10.31779577799479,
#          'CD': 10.008751529041634,
#          'CG': 10.410992553738522,
#          'H': 2.566093624291784,
#          'HA': 2.316665157577903,
#          'HB2': 2.2829374735169488,
#          'HB3': 2.266191753177966,
#          'HG2': 2.2862217879971585,
#          'HG3': 2.288703419744318,
#          'N': 6.094342847516545,
#          'O': 9.403582714645893,
#          'OE1': 9.459224588764204,
#          'OE2': 9.499444342301137,
#          'OXT': 9.794652},
#  'GLY': {'C': 9.874891312199962,
#          'CA': 11.243842100777785,
#          'H': 2.5695134499999996,
#          'H1': 3.0523499999999992,
#          'H2': 2.9720249999999995,
#          'H3': 2.2571324999999995,
#          'HA2': 2.327485097858198,
#          'HA3': 2.3228756056129978,
#          'N': 6.150554477816743,
#          'O': 9.452755469453471,
#          'OXT': 9.675483734000002},
#  'HEM': {'C1A': 8.746867226562498,
#          'C1B': 8.632678359374998,
#          'C1C': 8.689772792968748,
#          'C1D': 8.697385384114583,
#          'C2A': 8.099796979166666,
#          'C2B': 8.1987606640625,
#          'C2C': 8.236823619791666,
#          'C2D': 8.1987606640625,
#          'C3A': 8.194954368489583,
#          'C3B': 8.046508841145831,
#          'C3C': 8.271080279947915,
#          'C3D': 8.115022161458333,
#          'C4A': 8.682160201822915,
#          'C4B': 8.697385384114583,
#          'C4C': 8.73544833984375,
#          'C4D': 8.701191679687499,
#          'CAA': 10.764203880208333,
#          'CAB': 10.676659082031248,
#          'CAC': 10.623370944010416,
#          'CAD': 10.825104609374998,
#          'CBA': 10.802266835937496,
#          'CBB': 14.151806940104164,
#          'CBC': 14.064262141927081,
#          'CBD': 10.292223229166664,
#          'CGA': 10.181840657552081,
#          'CGD': 9.900174785156247,
#          'CHA': 10.50156948567708,
#          'CHB': 10.410218391927081,
#          'CHC': 10.49395689453125,
#          'CHD': 10.51298837239583,
#          'CMA': 13.200233046874997,
#          'CMB': 13.325840800781249,
#          'CMC': 13.165976386718746,
#          'CMD': 13.196426751302079,
#          'FE': 4.29441396890625,
#          'HAA': 2.3126906249999997,
#          'HAB': 1.6895024999999997,
#          'HAC': 2.04159375,
#          'HAD': 2.340804375,
#          'HBA': 2.3461593749999996,
#          'HBB': 2.3562,
#          'HBC': 2.2417368749999995,
#          'HBD': 2.3394656249999994,
#          'HHA': 2.0737237499999996,
#          'HHB': 2.0764012499999995,
#          'HHC': 2.1446775,
#          'HHD': 2.1245962499999993,
#          'HMA': 2.3191612499999996,
#          'HMB': 2.3535224999999995,
#          'HMC': 2.2928325,
#          'HMD': 2.3566462499999994,
#          'NA': 7.076986846516927,
#          'NB': 7.137572547200521,
#          'NC': 7.033711346028646,
#          'ND': 7.120262347005209,
#          'O1A': 9.699426216666666,
#          'O1D': 9.620524853333333,
#          'O2A': 9.538902753333334,
#          'O2D': 9.555227173333334},
#  'HIS': {'C': 9.760781320738635,
#          'CA': 8.713607123792611,
#          'CB': 9.436239548168372,
#          'CD2': 11.464407956345015,
#          'CE1': 12.458969843156144,
#          'CG': 8.54661235846442,
#          'H': 2.5903493293795616,
#          'HA': 2.280090845454545,
#          'HB2': 2.207472111486486,
#          'HB3': 2.197282233952702,
#          'HD2': 2.0512725506756753,
#          'HE1': 2.1083955658783777,
#          'N': 6.175712914044745,
#          'ND1': 7.011859167628933,
#          'NE2': 7.151159494876348,
#          'O': 9.353388087018182},
#  'HOH': {'O': 13.86359336490238,
#          'H01': 3.60257625,
#          'H02': 3.60257625,
#          'H': 3.60257625},
#  'ILE': {'C': 9.679370401766537,
#          'CA': 8.722930728629722,
#          'CB': 8.09065921732088,
#          'CD1': 13.068716218722681,
#          'CG1': 10.360244126625325,
#          'CG2': 12.922985671437935,
#          'H': 2.527421984536082,
#          'HA': 2.296987134146341,
#          'HB': 2.3209290865384613,
#          'HD1': 2.3105186669580413,
#          'HG1': 2.2957292242132863,
#          'HG2': 2.3004733610139856,
#          'N': 6.104522403929407,
#          'O': 9.336446286735395},
#  'LEU': {'C': 9.752386096429063,
#          'CA': 8.699696108683627,
#          'CB': 10.227351152672151,
#          'CD1': 12.93372801547556,
#          'CD2': 12.901198411139761,
#          'CG': 8.076515589430187,
#          'H': 2.5252473982300883,
#          'HA': 2.284674023230088,
#          'HB2': 2.3268951912811384,
#          'HB3': 2.32640209297153,
#          'HD1': 2.336796462411032,
#          'HD2': 2.324889448398576,
#          'HG': 2.319030630560498,
#          'N': 6.150566888954906,
#          'O': 9.365984287911505,
#          'OXT': 9.582434540000001},
#  'LYS': {'C': 9.753291175774029,
#          'CA': 8.700537362703614,
#          'CB': 10.474490010162013,
#          'CD': 10.54907117435874,
#          'CE': 11.34303749282447,
#          'CG': 10.461770452386986,
#          'H': 2.464449983766233,
#          'H1': 2.9539518749999996,
#          'H2': 2.7390824999999994,
#          'H3': 2.8575618749999996,
#          'HA': 2.3074174375,
#          'HB2': 2.3236765505725185,
#          'HB3': 2.3102864957061064,
#          'HD2': 2.2991966036345772,
#          'HD3': 2.327113091846758,
#          'HE2': 2.3038292504882807,
#          'HE3': 2.300879816894531,
#          'HG2': 2.303454797687861,
#          'HG3': 2.308755628612716,
#          'HZ1': 2.759061774902343,
#          'HZ2': 2.884020490722656,
#          'HZ3': 2.826514379882812,
#          'N': 6.154192126185054,
#          'NZ': 7.565267474031449,
#          'O': 9.398882882218114,
#          'OXT': 9.810976420000001},
#  'MET': {'C': 9.731806944813828,
#          'CA': 8.672887417927193,
#          'CB': 10.311983572140955,
#          'CE': 13.113639822591145,
#          'CG': 10.454466577962236,
#          'H': 2.4827512499999993,
#          'H1': 2.6788387499999993,
#          'H2': 1.4458499999999999,
#          'H3': 1.8956699999999995,
#          'HA': 2.3146417819148932,
#          'HB2': 2.2784060742187493,
#          'HB3': 2.2820876367187495,
#          'HE1': 2.3104733203124996,
#          'HE2': 2.1906551953124995,
#          'HE3': 2.2993867968749995,
#          'HG2': 2.2588896093749997,
#          'HG3': 2.2646629687499993,
#          'N': 6.102858399710148,
#          'O': 9.418061523723404,
#          'SD': 17.462380451660156},
#  'MPD': {'C1': 15.678131464843748,
#          'C2': 6.999777558593748,
#          'C3': 11.613007792968748,
#          'C4': 9.83166146484375,
#          'C5': 15.837995878906247,
#          'CM': 15.278470429687497,
#          'O2': 9.90892294,
#          'O4': 10.35784449},
#  'MQD': {'C1': 16.020698066406247,
#          'C2': 6.674339287109373,
#          'C3': 12.115438808593748,
#          'C4': 10.031491982421873,
#          'C5': 16.03782639648437,
#          'CM': 13.188814160156248,
#          'O2': 10.14562703,
#          'O4': 10.288465705,
#          'O6': 10.206843605},
#  'MRD': {'C1': 15.940765859374997,
#          'C2': 6.691467617187499,
#          'C3': 12.086891591796874,
#          'C4': 10.002944765625,
#          'C5': 16.032116953124998,
#          'CM': 15.935056416015623,
#          'O2': 10.492520955,
#          'O4': 10.239492445},
#  'PHE': {'C': 9.738857058238635,
#          'CA': 8.725716334117541,
#          'CB': 10.374215724627293,
#          'CD1': 10.597460198000286,
#          'CD2': 10.605107709288989,
#          'CE1': 10.626321696082998,
#          'CE2': 10.621083674652377,
#          'CG': 7.852501257391412,
#          'CZ': 10.626007414797161,
#          'H': 2.5909650255681815,
#          'HA': 2.2906902952981647,
#          'HB2': 2.34100395928899,
#          'HB3': 2.339963050458715,
#          'HD1': 2.119741745986238,
#          'HD2': 2.1249278669724765,
#          'HE1': 2.125406869266055,
#          'HE2': 2.124633096330275,
#          'HZ': 2.128676981077981,
#          'N': 6.101943922257857,
#          'O': 9.382701551204546},
#  'PRO': {'C': 9.759032896395594,
#          'CA': 8.84240771706321,
#          'CB': 10.488171325260415,
#          'CD': 11.277406712304685,
#          'CG': 10.54595089205729,
#          'HA': 2.3008694624999992,
#          'HB2': 2.3357037375,
#          'HB3': 2.3274704249999996,
#          'HD2': 2.3180188499999996,
#          'HD3': 2.3023287,
#          'HG2': 2.3293446749999998,
#          'HG3': 2.3323300874999995,
#          'N': 4.365092950550427,
#          'O': 9.4183158175},
#  'SER': {'C': 9.752820395876734,
#          'CA': 8.772293280989581,
#          'CB': 11.38593580572975,
#          'H': 2.5641346499999997,
#          'H1': 3.0393373499999994,
#          'H2': 2.9415014999999993,
#          'H3': 3.013794,
#          'HA': 2.331361084641255,
#          'HB2': 2.3288756922645737,
#          'HB3': 2.325849997197309,
#          'HG': 2.8465795086206893,
#          'N': 6.156391933018664,
#          'O': 9.500721748777778,
#          'OG': 9.092427426210762},
#  'SO4': {'O1': 9.41919034,
#          'O2': 9.90892294,
#          'O3': 9.65589443,
#          'O4': 10.21908692,
#          'S': 13.419295312500001},
#  'THR': {'C': 9.721674747968747,
#          'CA': 8.694309897291665,
#          'CB': 8.921713219999997,
#          'CG2': 12.991922102760418,
#          'H': 2.5172141399999997,
#          'HA': 2.2985909099999997,
#          'HB': 2.3228597699999995,
#          'HG1': 2.711580644044321,
#          'HG2': 2.2964417699999995,
#          'N': 6.084535368652345,
#          'O': 9.32868775552,
#          'OG1': 9.097076884560002},
#  'TRP': {'C': 9.770334168069773,
#          'CA': 8.749525070884966,
#          'CB': 10.435878074151402,
#          'CD1': 11.6671490662042,
#          'CD2': 7.896258604694234,
#          'CE2': 8.622834491514006,
#          'CE3': 10.692310831930227,
#          'CG': 8.051791716897897,
#          'CH2': 10.583339214709053,
#          'CZ2': 10.631869483263737,
#          'CZ3': 10.68443573764143,
#          'H': 2.6258657974137933,
#          'HA': 2.3053274999999998,
#          'HB2': 2.3131522629310344,
#          'HB3': 2.3281093318965516,
#          'HD1': 2.122830484913793,
#          'HE1': 2.684655387931034,
#          'HE3': 2.123799924568965,
#          'HH2': 2.114105528017241,
#          'HZ2': 2.1201299030172414,
#          'HZ3': 2.102853103448275,
#          'N': 6.10363627921269,
#          'NE1': 6.20533370536015,
#          'O': 9.496872062758621},
#  'TRS': {'C': 6.6115354101562485,
#          'C1': 13.405773007812497,
#          'C2': 13.165976386718748,
#          'C3': 13.234489707031248,
#          'N': 11.606489230957033,
#          'O1': 10.63535963,
#          'O2': 10.333357860000001,
#          'O3': 9.97422062},
#  'TYR': {'C': 9.739037565249301,
#          'CA': 8.738648537395502,
#          'CB': 10.412206393436502,
#          'CD1': 10.577162030876789,
#          'CD2': 10.621310210738454,
#          'CE1': 10.600436194889529,
#          'CE2': 10.617637256730195,
#          'CG': 7.8861959145476686,
#          'CZ': 8.773523417533339,
#          'H': 2.6038218511146494,
#          'HA': 2.2851822969745217,
#          'HB2': 2.3412307285031844,
#          'HB3': 2.330333132961783,
#          'HD1': 2.095523204617834,
#          'HD2': 2.129635748407643,
#          'HE1': 2.1243916003184706,
#          'HE2': 2.117778857484076,
#          'HH': 2.5823969274193543,
#          'N': 6.1496966955022145,
#          'O': 9.324441118821657,
#          'OH': 8.96967092111465},
#  'VAL': {'C': 9.652430618906248,
#          'CA': 8.720261220507812,
#          'CB': 8.006159038479712,
#          'CG1': 12.868325607713267,
#          'CG2': 12.83558839619298,
#          'H': 2.504135883233532,
#          'HA': 2.3135232621951216,
#          'HB': 2.2977160584677416,
#          'HG1': 2.3037701234879027,
#          'HG2': 2.319316897681451,
#          'N': 6.104580580478516,
#          'O': 9.377367175959998,
#          'OXT': 9.73751653},
#  'ZN ': {'ZN': 7.871119170332029}}




# # form factors taken from http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
# ffcoeff_old = {
#     "H": {"a": [0.489918, 0.262003, 0.196767, 0.049879], "b": [20.6593, 7.74039, 49.5519, 2.20159], "c": 0.001305},
#     "H1-": {"a": [0.897661, 0.565616, 0.415815, 0.116973], "b": [53.1368, 15.187, 186.576, 3.56709], "c": 0.002389},
#     "He": {"a": [0.8734, 0.6309, 0.3112, 0.178], "b": [9.1037, 3.3568, 22.9276, 0.9821], "c": 0.0064},
#     "Li": {"a": [1.1282, 0.7508, 0.6175, 0.4653], "b": [3.9546, 1.0524, 85.3905, 168.261], "c": 0.0377},
#     "Li1+": {"a": [0.6968, 0.7888, 0.3414, 0.1563], "b": [4.6237, 1.9557, 0.6316, 10.0953], "c": 0.0167},
#     "Be": {"a": [1.5919, 1.1278, 0.5391, 0.7029], "b": [43.6427, 1.8623, 103.483, 0.542], "c": 0.0385},
#     "Be2+": {"a": [6.2603, 0.8849, 0.7993, 0.1647], "b": [0.0027, 0.8313, 2.2758, 5.1146], "c": -6.1092},
#     "B": {"a": [2.0545, 1.3326, 1.0979, 0.7068], "b": [23.2185, 1.021, 60.3498, 0.1403], "c": -0.1932},
#     "C": {"a": [2.31, 1.02, 1.5886, 0.865], "b": [20.8439, 10.2075, 0.5687, 51.6512], "c": 0.2156},
#     "Cval": {"a": [2.26069, 1.56165, 1.05075, 0.839259], "b": [22.6907, 0.656665, 9.75618, 55.5949], "c": 0.286977},
#     "N": {"a": [12.2126, 3.1322, 2.0125, 1.1663], "b": [0.0057, 9.8933, 28.9975, 0.5826], "c": -11.529},
#     "O": {"a": [3.0485, 2.2868, 1.5463, 0.867], "b": [13.2771, 5.7011, 0.3239, 32.9089], "c": 0.2508},
#     "O1-": {"a": [4.1916, 1.63969, 1.52673, -20.307], "b": [12.8573, 4.17236, 47.0179, -0.01404], "c": 21.9412},
#     "F": {"a": [3.5392, 2.6412, 1.517, 1.0243], "b": [10.2825, 4.2944, 0.2615, 26.1476], "c": 0.2776},
#     "F1-": {"a": [3.6322, 3.51057, 1.26064, 0.940706], "b": [5.27756, 14.7353, 0.442258, 47.3437], "c": 0.653396},
#     "Ne": {"a": [3.9553, 3.1125, 1.4546, 1.1251], "b": [8.4042, 3.4262, 0.2306, 21.7184], "c": 0.3515},
#     "Na": {"a": [4.7626, 3.1736, 1.2674, 1.1128], "b": [3.285, 8.8422, 0.3136, 129.424], "c": 0.676},
#     "Na1+": {"a": [3.2565, 3.9362, 1.3998, 1.0032], "b": [2.6671, 6.1153, 0.2001, 14.039], "c": 0.404},
#     "Mg": {"a": [5.4204, 2.1735, 1.2269, 2.3073], "b": [2.8275, 79.2611, 0.3808, 7.1937], "c": 0.8584},
#     "Mg2+": {"a": [3.4988, 3.8378, 1.3284, 0.8497], "b": [2.1676, 4.7542, 0.185, 10.1411], "c": 0.4853},
#     "Al": {"a": [6.4202, 1.9002, 1.5936, 1.9646], "b": [3.0387, 0.7426, 31.5472, 85.0886], "c": 1.1151},
#     "Al3+": {"a": [4.17448, 3.3876, 1.20296, 0.528137], "b": [1.93816, 4.14553, 0.228753, 8.28524], "c": 0.706786},
#     "Siv": {"a": [6.2915, 3.0353, 1.9891, 1.541], "b": [2.4386, 32.3337, 0.6785, 81.6937], "c": 1.1407},
#     "Sival": {"a": [5.66269, 3.07164, 2.62446, 1.3932], "b": [2.6652, 38.6634, 0.916946, 93.5458], "c": 1.24707},
#     "Si4+": {"a": [4.43918, 3.20345, 1.19453, 0.41653], "b": [1.64167, 3.43757, 0.2149, 6.65365], "c": 0.746297},
#     "P": {"a": [6.4345, 4.1791, 1.78, 1.4908], "b": [1.9067, 27.157, 0.526, 68.1645], "c": 1.1149},
#     "S": {"a": [6.9053, 5.2034, 1.4379, 1.5863], "b": [1.4679, 22.2151, 0.2536, 56.172], "c": 0.8669},
#     "Cl": {"a": [11.4604, 7.1964, 6.2556, 1.6455], "b": [0.0104, 1.1662, 18.5194, 47.7784], "c": -9.5574},
#     "Cl1-": {"a": [18.2915, 7.2084, 6.5337, 2.3386], "b": [0.0066, 1.1717, 19.5424, 60.4486], "c": -16.378},
#     "Ar": {"a": [7.4845, 6.7723, 0.6539, 1.6442], "b": [0.9072, 14.8407, 43.8983, 33.3929], "c": 1.4445},
#     "K": {"a": [8.2186, 7.4398, 1.0519, 0.8659], "b": [12.7949, 0.7748, 213.187, 41.6841], "c": 1.4228},
#     "K1+": {"a": [7.9578, 7.4917, 6.359, 1.1915], "b": [12.6331, 0.7674, -0.002, 31.9128], "c": -4.9978},
#     "Ca": {"a": [8.6266, 7.3873, 1.5899, 1.0211], "b": [10.4421, 0.6599, 85.7484, 178.437], "c": 1.3751},
#     "Ca2+": {"a": [15.6348, 7.9518, 8.4372, 0.8537], "b": [-0.0074, 0.6089, 10.3116, 25.9905], "c": -14.875},
#     "Sc": {"a": [9.189, 7.3679, 1.6409, 1.468], "b": [9.0213, 0.5729, 136.108, 51.3531], "c": 1.3329},
#     "Sc3+": {"a": [13.4008, 8.0273, 1.65943, 1.57936], "b": [0.29854, 7.9629, -0.28604, 16.0662], "c": -6.6667},
#     "Ti": {"a": [9.7595, 7.3558, 1.6991, 1.9021], "b": [7.8508, 0.5, 35.6338, 116.105], "c": 1.2807},
#     "Ti2+": {"a": [9.11423, 7.62174, 2.2793, 0.087899], "b": [7.5243, 0.457585, 19.5361, 61.6558], "c": 0.897155},
#     "Ti3+": {"a": [17.7344, 8.73816, 5.25691, 1.92134], "b": [0.22061, 7.04716, -0.15762, 15.9768], "c": -14.652},
#     "Ti4+": {"a": [19.5114, 8.23473, 2.01341, 1.5208], "b": [0.178847, 6.67018, -0.29263, 12.9464], "c": -13.28},
#     "V": {"a": [10.2971, 7.3511, 2.0703, 2.0571], "b": [6.8657, 0.4385, 26.8938, 102.478], "c": 1.2199},
#     "V2+": {"a": [10.106, 7.3541, 2.2884, 0.0223], "b": [6.8818, 0.4409, 20.3004, 115.122], "c": 1.2298},
#     "V3+": {"a": [9.43141, 7.7419, 2.15343, 0.016865], "b": [6.39535, 0.383349, 15.1908, 63.969], "c": 0.656565},
#     "V5+": {"a": [15.6887, 8.14208, 2.03081, -9.576], "b": [0.679003, 5.40135, 9.97278, 0.940464], "c": 1.7143},
#     "Cr": {"a": [10.6406, 7.3537, 3.324, 1.4922], "b": [6.1038, 0.392, 20.2626, 98.7399], "c": 1.1832},
#     "Cr2+": {"a": [9.54034, 7.7509, 3.58274, 0.509107], "b": [5.66078, 0.344261, 13.3075, 32.4224], "c": 0.616898},
#     "Cr3+": {"a": [9.6809, 7.81136, 2.87603, 0.113575], "b": [5.59463, 0.334393, 12.8288, 32.8761], "c": 0.518275},
#     "Mn": {"a": [11.2819, 7.3573, 3.0193, 2.2441], "b": [5.3409, 0.3432, 17.8674, 83.7543], "c": 1.0896},
#     "Mn2+": {"a": [10.8061, 7.362, 3.5268, 0.2184], "b": [5.2796, 0.3435, 14.343, 41.3235], "c": 1.0874},
#     "Mn3+": {"a": [9.84521, 7.87194, 3.56531, 0.323613], "b": [4.91797, 0.294393, 10.8171, 24.1281], "c": 0.393974},
#     "Mn4+": {"a": [9.96253, 7.97057, 2.76067, 0.054447], "b": [4.8485, 0.283303, 10.4852, 27.573], "c": 0.251877},
#     "Fe": {"a": [11.7695, 7.3573, 3.5222, 2.3045], "b": [4.7611, 0.3072, 15.3535, 76.8805], "c": 1.0369},
#     "Fe2+": {"a": [11.0424, 7.374, 4.1346, 0.4399], "b": [4.6538, 0.3053, 12.0546, 31.2809], "c": 1.0097},
#     "Fe3+": {"a": [11.1764, 7.3863, 3.3948, 0.0724], "b": [4.6147, 0.3005, 11.6729, 38.5566], "c": 0.9707},
#     "Co": {"a": [12.2841, 7.3409, 4.0034, 2.3488], "b": [4.2791, 0.2784, 13.5359, 71.1692], "c": 1.0118},
#     "Co2+": {"a": [11.2296, 7.3883, 4.7393, 0.7108], "b": [4.1231, 0.2726, 10.2443, 25.6466], "c": 0.9324},
#     "Co3+": {"a": [10.338, 7.88173, 4.76795, 0.725591], "b": [3.90969, 0.238668, 8.35583, 18.3491], "c": 0.286667},
#     "Ni": {"a": [12.8376, 7.292, 4.4438, 2.38], "b": [3.8785, 0.2565, 12.1763, 66.3421], "c": 1.0341},
#     "Ni2+": {"a": [11.4166, 7.4005, 5.3442, 0.9773], "b": [3.6766, 0.2449, 8.873, 22.1626], "c": 0.8614},
#     "Ni3+": {"a": [10.7806, 7.75868, 5.22746, 0.847114], "b": [3.5477, 0.22314, 7.64468, 16.9673], "c": 0.386044},
#     "Cu": {"a": [13.338, 7.1676, 5.6158, 1.6735], "b": [3.5828, 0.247, 11.3966, 64.8126], "c": 1.191},
#     "Cu1+": {"a": [11.9475, 7.3573, 6.2455, 1.5578], "b": [3.3669, 0.2274, 8.6625, 25.8487], "c": 0.89},
#     "Cu2+": {"a": [11.8168, 7.11181, 5.78135, 1.14523], "b": [3.37484, 0.244078, 7.9876, 19.897], "c": 1.14431},
#     "Zn": {"a": [14.0743, 7.0318, 5.1652, 2.41], "b": [3.2655, 0.2333, 10.3163, 58.7097], "c": 1.3041},
#     "Zn2+": {"a": [11.9719, 7.3862, 6.4668, 1.394], "b": [2.9946, 0.2031, 7.0826, 18.0995], "c": 0.7807},
#     "Ga": {"a": [15.2354, 6.7006, 4.3591, 2.9623], "b": [3.0669, 0.2412, 10.7805, 61.4135], "c": 1.7189},
#     "Ga3+": {"a": [12.692, 6.69883, 6.06692, 1.0066], "b": [2.81262, 0.22789, 6.36441, 14.4122], "c": 1.53545},
#     "Ge": {"a": [16.0816, 6.3747, 3.7068, 3.683], "b": [2.8509, 0.2516, 11.4468, 54.7625], "c": 2.1313},
#     "Ge4+": {"a": [12.9172, 6.70003, 6.06791, 0.859041], "b": [2.53718, 0.205855, 5.47913, 11.603], "c": 1.45572},
#     "As": {"a": [16.6723, 6.0701, 3.4313, 4.2779], "b": [2.6345, 0.2647, 12.9479, 47.7972], "c": 2.531},
#     "Se": {"a": [17.0006, 5.8196, 3.9731, 4.3543], "b": [2.4098, 0.2726, 15.2372, 43.8163], "c": 2.8409},
#     "Br": {"a": [17.1789, 5.2358, 5.6377, 3.9851], "b": [2.1723, 16.5796, 0.2609, 41.4328], "c": 2.9557},
#     "Br1-": {"a": [17.1718, 6.3338, 5.5754, 3.7272], "b": [2.2059, 19.3345, 0.2871, 58.1535], "c": 3.1776},
#     "Kr": {"a": [17.3555, 6.7286, 5.5493, 3.5375], "b": [1.9384, 16.5623, 0.2261, 39.3972], "c": 2.825},
#     "Rb": {"a": [17.1784, 9.6435, 5.1399, 1.5292], "b": [1.7888, 17.3151, 0.2748, 164.934], "c": 3.4873},
#     "Rb1+": {"a": [17.5816, 7.6598, 5.8981, 2.7817], "b": [1.7139, 14.7957, 0.1603, 31.2087], "c": 2.0782},
#     "Sr": {"a": [17.5663, 9.8184, 5.422, 2.6694], "b": [1.5564, 14.0988, 0.1664, 132.376], "c": 2.5064},
#     "Sr2+": {"a": [18.0874, 8.1373, 2.5654, -34.193], "b": [1.4907, 12.6963, 24.5651, -0.0138], "c": 41.4025},
#     "Y": {"a": [17.776, 10.2946, 5.72629, 3.26588], "b": [1.4029, 12.8006, 0.125599, 104.354], "c": 1.91213},
#     "Y3+": {"a": [17.9268, 9.1531, 1.76795, -33.108], "b": [1.35417, 11.2145, 22.6599, -0.01319], "c": 40.2602},
#     "Zr": {"a": [17.8765, 10.948, 5.41732, 3.65721], "b": [1.27618, 11.916, 0.117622, 87.6627], "c": 2.06929},
#     "Zr4+": {"a": [18.1668, 10.0562, 1.01118, -2.6479], "b": [1.2148, 10.1483, 21.6054, -0.10276], "c": 9.41454},
#     "Nb": {"a": [17.6142, 12.0144, 4.04183, 3.53346], "b": [1.18865, 11.766, 0.204785, 69.7957], "c": 3.75591},
#     "Nb3+": {"a": [19.8812, 18.0653, 11.0177, 1.94715], "b": [0.019175, 1.13305, 10.1621, 28.3389], "c": -12.912},
#     "Nb5+": {"a": [17.9163, 13.3417, 10.799, 0.337905], "b": [1.12446, 0.028781, 9.28206, 25.7228], "c": -6.3934},
#     "Mo": {"a": [3.7025, 17.2356, 12.8876, 3.7429], "b": [0.2772, 1.0958, 11.004, 61.6584], "c": 4.3875},
#     "Mo3+": {"a": [21.1664, 18.2017, 11.7423, 2.30951], "b": [0.014734, 1.03031, 9.53659, 26.6307], "c": -14.421},
#     "Mo5+": {"a": [21.0149, 18.0992, 11.4632, 0.740625], "b": [0.014345, 1.02238, 8.78809, 23.3452], "c": -14.316},
#     "Mo6+": {"a": [17.8871, 11.175, 6.57891, 0], "b": [1.03649, 8.48061, 0.058881, 0], "c": 0.344941},
#     "Tc": {"a": [19.1301, 11.0948, 4.64901, 2.71263], "b": [0.864132, 8.14487, 21.5707, 86.8472], "c": 5.40428},
#     "Ru": {"a": [19.2674, 12.9182, 4.86337, 1.56756], "b": [0.80852, 8.43467, 24.7997, 94.2928], "c": 5.37874},
#     "Ru3+": {"a": [18.5638, 13.2885, 9.32602, 3.00964], "b": [0.847329, 8.37164, 0.017662, 22.887], "c": -3.1892},
#     "Ru4+": {"a": [18.5003, 13.1787, 4.71304, 2.18535], "b": [0.844582, 8.12534, 0.36495, 20.8504], "c": 1.42357},
#     "Rh": {"a": [19.2957, 14.3501, 4.73425, 1.28918], "b": [0.751536, 8.21758, 25.8749, 98.6062], "c": 5.328},
#     "Rh3+": {"a": [18.8785, 14.1259, 3.32515, -6.1989], "b": [0.764252, 7.84438, 21.2487, -0.01036], "c": 11.8678},
#     "Rh4+": {"a": [18.8545, 13.9806, 2.53464, -5.6526], "b": [0.760825, 7.62436, 19.3317, -0.0102], "c": 11.2835},
#     "Pd": {"a": [19.3319, 15.5017, 5.29537, 0.605844], "b": [0.698655, 7.98929, 25.2052, 76.8986], "c": 5.26593},
#     "Pd2+": {"a": [19.1701, 15.2096, 4.32234, 0], "b": [0.696219, 7.55573, 22.5057, 0], "c": 5.2916},
#     "Pd4+": {"a": [19.2493, 14.79, 2.89289, -7.9492], "b": [0.683839, 7.14833, 17.9144, 0.005127], "c": 13.0174},
#     "Ag": {"a": [19.2808, 16.6885, 4.8045, 1.0463], "b": [0.6446, 7.4726, 24.6605, 99.8156], "c": 5.179},
#     "Ag1+": {"a": [19.1812, 15.9719, 5.27475, 0.357534], "b": [0.646179, 7.19123, 21.7326, 66.1147], "c": 5.21572},
#     "Ag2+": {"a": [19.1643, 16.2456, 4.3709, 0], "b": [0.645643, 7.18544, 21.4072, 0], "c": 5.21404},
#     "Cd": {"a": [19.2214, 17.6444, 4.461, 1.6029], "b": [0.5946, 6.9089, 24.7008, 87.4825], "c": 5.0694},
#     "Cd2+": {"a": [19.1514, 17.2535, 4.47128, 0], "b": [0.597922, 6.80639, 20.2521, 0], "c": 5.11937},
#     "In": {"a": [19.1624, 18.5596, 4.2948, 2.0396], "b": [0.5476, 6.3776, 25.8499, 92.8029], "c": 4.9391},
#     "In3+": {"a": [19.1045, 18.1108, 3.78897, 0], "b": [0.551522, 6.3247, 17.3595, 0], "c": 4.99635},
#     "Sn": {"a": [19.1889, 19.1005, 4.4585, 2.4663], "b": [5.8303, 0.5031, 26.8909, 83.9571], "c": 4.7821},
#     "Sn2+": {"a": [19.1094, 19.0548, 4.5648, 0.487], "b": [0.5036, 5.8378, 23.3752, 62.2061], "c": 4.7861},
#     "Sn4+": {"a": [18.9333, 19.7131, 3.4182, 0.0193], "b": [5.764, 0.4655, 14.0049, -0.7583], "c": 3.9182},
#     "Sb": {"a": [19.6418, 19.0455, 5.0371, 2.6827], "b": [5.3034, 0.4607, 27.9074, 75.2825], "c": 4.5909},
#     "Sb3+": {"a": [18.9755, 18.933, 5.10789, 0.288753], "b": [0.467196, 5.22126, 19.5902, 55.5113], "c": 4.69626},
#     "Sb5+": {"a": [19.8685, 19.0302, 2.41253, 0], "b": [5.44853, 0.467973, 14.1259, 0], "c": 4.69263},
#     "Te": {"a": [19.9644, 19.0138, 6.14487, 2.5239], "b": [4.81742, 0.420885, 28.5284, 70.8403], "c": 4.352},
#     "I": {"a": [20.1472, 18.9949, 7.5138, 2.2735], "b": [4.347, 0.3814, 27.766, 66.8776], "c": 4.0712},
#     "I1-": {"a": [20.2332, 18.997, 7.8069, 2.8868], "b": [4.3579, 0.3815, 29.5259, 84.9304], "c": 4.0714},
#     "Xe": {"a": [20.2933, 19.0298, 8.9767, 1.99], "b": [3.9282, 0.344, 26.4659, 64.2658], "c": 3.7118},
#     "Cs": {"a": [20.3892, 19.1062, 10.662, 1.4953], "b": [3.569, 0.3107, 24.3879, 213.904], "c": 3.3352},
#     "Cs1+": {"a": [20.3524, 19.1278, 10.2821, 0.9615], "b": [3.552, 0.3086, 23.7128, 59.4565], "c": 3.2791},
#     "Ba": {"a": [20.3361, 19.297, 10.888, 2.6959], "b": [3.216, 0.2756, 20.2073, 167.202], "c": 2.7731},
#     "Ba2+": {"a": [20.1807, 19.1136, 10.9054, 0.77634], "b": [3.21367, 0.28331, 20.0558, 51.746], "c": 3.02902},
#     "La": {"a": [20.578, 19.599, 11.3727, 3.28719], "b": [2.94817, 0.244475, 18.7726, 133.124], "c": 2.14678},
#     "La3+": {"a": [20.2489, 19.3763, 11.6323, 0.336048], "b": [2.9207, 0.250698, 17.8211, 54.9453], "c": 2.4086},
#     "Ce": {"a": [21.1671, 19.7695, 11.8513, 3.33049], "b": [2.81219, 0.226836, 17.6083, 127.113], "c": 1.86264},
#     "Ce3+": {"a": [20.8036, 19.559, 11.9369, 0.612376], "b": [2.77691, 0.23154, 16.5408, 43.1692], "c": 2.09013},
#     "Ce4+": {"a": [20.3235, 19.8186, 12.1233, 0.144583], "b": [2.65941, 0.21885, 15.7992, 62.2355], "c": 1.5918},
#     "Pr": {"a": [22.044, 19.6697, 12.3856, 2.82428], "b": [2.77393, 0.222087, 16.7669, 143.644], "c": 2.0583},
#     "Pr3+": {"a": [21.3727, 19.7491, 12.1329, 0.97518], "b": [2.6452, 0.214299, 15.323, 36.4065], "c": 1.77132},
#     "Pr4+": {"a": [20.9413, 20.0539, 12.4668, 0.296689], "b": [2.54467, 0.202481, 14.8137, 45.4643], "c": 1.24285},
#     "Nd": {"a": [22.6845, 19.6847, 12.774, 2.85137], "b": [2.66248, 0.210628, 15.885, 137.903], "c": 1.98486},
#     "Nd3+": {"a": [21.961, 19.9339, 12.12, 1.51031], "b": [2.52722, 0.199237, 14.1783, 30.8717], "c": 1.47588},
#     "Pm": {"a": [23.3405, 19.6095, 13.1235, 2.87516], "b": [2.5627, 0.202088, 15.1009, 132.721], "c": 2.02876},
#     "Pm3+": {"a": [22.5527, 20.1108, 12.0671, 2.07492], "b": [2.4174, 0.185769, 13.1275, 27.4491], "c": 1.19499},
#     "Sm": {"a": [24.0042, 19.4258, 13.4396, 2.89604], "b": [2.47274, 0.196451, 14.3996, 128.007], "c": 2.20963},
#     "Sm3+": {"a": [23.1504, 20.2599, 11.9202, 2.71488], "b": [2.31641, 0.174081, 12.1571, 24.8242], "c": 0.954586},
#     "Eu": {"a": [24.6274, 19.0886, 13.7603, 2.9227], "b": [2.3879, 0.1942, 13.7546, 123.174], "c": 2.5745},
#     "Eu2+": {"a": [24.0063, 19.9504, 11.8034, 3.87243], "b": [2.27783, 0.17353, 11.6096, 26.5156], "c": 1.36389},
#     "Eu3+": {"a": [23.7497, 20.3745, 11.8509, 3.26503], "b": [2.22258, 0.16394, 11.311, 22.9966], "c": 0.759344},
#     "Gd": {"a": [25.0709, 19.0798, 13.8518, 3.54545], "b": [2.25341, 0.181951, 12.9331, 101.398], "c": 2.4196},
#     "Gd3+": {"a": [24.3466, 20.4208, 11.8708, 3.7149], "b": [2.13553, 0.155525, 10.5782, 21.7029], "c": 0.645089},
#     "Tb": {"a": [25.8976, 18.2185, 14.3167, 2.95354], "b": [2.24256, 0.196143, 12.6648, 115.362], "c": 3.58324},
#     "Tb3+": {"a": [24.9559, 20.3271, 12.2471, 3.773], "b": [2.05601, 0.149525, 10.0499, 21.2773], "c": 0.691967},
#     "Dy": {"a": [26.507, 17.6383, 14.5596, 2.96577], "b": [2.1802, 0.202172, 12.1899, 111.874], "c": 4.29728},
#     "Dy3+": {"a": [25.5395, 20.2861, 11.9812, 4.50073], "b": [1.9804, 0.143384, 9.34972, 19.581], "c": 0.68969},
#     "Ho": {"a": [26.9049, 17.294, 14.5583, 3.63837], "b": [2.07051, 0.19794, 11.4407, 92.6566], "c": 4.56796},
#     "Ho3+": {"a": [26.1296, 20.0994, 11.9788, 4.93676], "b": [1.91072, 0.139358, 8.80018, 18.5908], "c": 0.852795},
#     "Er": {"a": [27.6563, 16.4285, 14.9779, 2.98233], "b": [2.07356, 0.223545, 11.3604, 105.703], "c": 5.92046},
#     "Er3+": {"a": [26.722, 19.7748, 12.1506, 5.17379], "b": [1.84659, 0.13729, 8.36225, 17.8974], "c": 1.17613},
#     "Tm": {"a": [28.1819, 15.8851, 15.1542, 2.98706], "b": [2.02859, 0.238849, 10.9975, 102.961], "c": 6.75621},
#     "Tm3+": {"a": [27.3083, 19.332, 12.3339, 5.38348], "b": [1.78711, 0.136974, 7.96778, 17.2922], "c": 1.63929},
#     "Yb": {"a": [28.6641, 15.4345, 15.3087, 2.98963], "b": [1.9889, 0.257119, 10.6647, 100.417], "c": 7.56672},
#     "Yb2+": {"a": [28.1209, 17.6817, 13.3335, 5.14657], "b": [1.78503, 0.15997, 8.18304, 20.39], "c": 3.70983},
#     "Yb3+": {"a": [27.8917, 18.7614, 12.6072, 5.47647], "b": [1.73272, 0.13879, 7.64412, 16.8153], "c": 2.26001},
#     "Lu": {"a": [28.9476, 15.2208, 15.1, 3.71601], "b": [1.90182, 9.98519, 0.261033, 84.3298], "c": 7.97628},
#     "Lu3+": {"a": [28.4628, 18.121, 12.8429, 5.59415], "b": [1.68216, 0.142292, 7.33727, 16.3535], "c": 2.97573},
#     "Hf": {"a": [29.144, 15.1726, 14.7586, 4.30013], "b": [1.83262, 9.5999, 0.275116, 72.029], "c": 8.58154},
#     "Hf4+": {"a": [28.8131, 18.4601, 12.7285, 5.59927], "b": [1.59136, 0.128903, 6.76232, 14.0366], "c": 2.39699},
#     "Ta": {"a": [29.2024, 15.2293, 14.5135, 4.76492], "b": [1.77333, 9.37046, 0.295977, 63.3644], "c": 9.24354},
#     "Ta5+": {"a": [29.1587, 18.8407, 12.8268, 5.38695], "b": [1.50711, 0.116741, 6.31524, 12.4244], "c": 1.78555},
#     "W": {"a": [29.0818, 15.43, 14.4327, 5.11982], "b": [1.72029, 9.2259, 0.321703, 57.056], "c": 9.8875},
#     "W6+": {"a": [29.4936, 19.3763, 13.0544, 5.06412], "b": [1.42755, 0.104621, 5.93667, 11.1972], "c": 1.01074},
#     "Re": {"a": [28.7621, 15.7189, 14.5564, 5.44174], "b": [1.67191, 9.09227, 0.3505, 52.0861], "c": 10.472},
#     "Os": {"a": [28.1894, 16.155, 14.9305, 5.67589], "b": [1.62903, 8.97948, 0.382661, 48.1647], "c": 11.0005},
#     "Os4+": {"a": [30.419, 15.2637, 14.7458, 5.06795], "b": [1.37113, 6.84706, 0.165191, 18.003], "c": 6.49804},
#     "Ir": {"a": [27.3049, 16.7296, 15.6115, 5.83377], "b": [1.59279, 8.86553, 0.417916, 45.0011], "c": 11.4722},
#     "Ir3+": {"a": [30.4156, 15.862, 13.6145, 5.82008], "b": [1.34323, 7.10909, 0.204633, 20.3254], "c": 8.27903},
#     "Ir4+": {"a": [30.7058, 15.5512, 14.2326, 5.53672], "b": [1.30923, 6.71983, 0.167252, 17.4911], "c": 6.96824},
#     "Pt": {"a": [27.0059, 17.7639, 15.7131, 5.7837], "b": [1.51293, 8.81174, 0.424593, 38.6103], "c": 11.6883},
#     "Pt2+": {"a": [29.8429, 16.7224, 13.2153, 6.35234], "b": [1.32927, 7.38979, 0.263297, 22.9426], "c": 9.85329},
#     "Pt4+": {"a": [30.9612, 15.9829, 13.7348, 5.92034], "b": [1.24813, 6.60834, 0.16864, 16.9392], "c": 7.39534},
#     "Au": {"a": [16.8819, 18.5913, 25.5582, 5.86], "b": [0.4611, 8.6216, 1.4826, 36.3956], "c": 12.0658},
#     "Au1+": {"a": [28.0109, 17.8204, 14.3359, 6.58077], "b": [1.35321, 7.7395, 0.356752, 26.4043], "c": 11.2299},
#     "Au3+": {"a": [30.6886, 16.9029, 12.7801, 6.52354], "b": [1.2199, 6.82872, 0.212867, 18.659], "c": 9.0968},
#     "Hg": {"a": [20.6809, 19.0417, 21.6575, 5.9676], "b": [0.545, 8.4484, 1.5729, 38.3246], "c": 12.6089},
#     "Hg1+": {"a": [25.0853, 18.4973, 16.8883, 6.48216], "b": [1.39507, 7.65105, 0.443378, 28.2262], "c": 12.0205},
#     "Hg2+": {"a": [29.5641, 18.06, 12.8374, 6.89912], "b": [1.21152, 7.05639, 0.284738, 20.7482], "c": 10.6268},
#     "Tl": {"a": [27.5446, 19.1584, 15.538, 5.52593], "b": [0.65515, 8.70751, 1.96347, 45.8149], "c": 13.1746},
#     "Tl1+": {"a": [21.3985, 20.4723, 18.7478, 6.82847], "b": [1.4711, 0.517394, 7.43463, 28.8482], "c": 12.5258},
#     "Tl3+": {"a": [30.8695, 18.3481, 11.9328, 7.00574], "b": [1.1008, 6.53852, 0.219074, 17.2114], "c": 9.8027},
#     "Pb": {"a": [31.0617, 13.0637, 18.442, 5.9696], "b": [0.6902, 2.3576, 8.618, 47.2579], "c": 13.4118},
#     "Pb2+": {"a": [21.7886, 19.5682, 19.1406, 7.01107], "b": [1.3366, 0.488383, 6.7727, 23.8132], "c": 12.4734},
#     "Pb4+": {"a": [32.1244, 18.8003, 12.0175, 6.96886], "b": [1.00566, 6.10926, 0.147041, 14.714], "c": 8.08428},
#     "Bi": {"a": [33.3689, 12.951, 16.5877, 6.4692], "b": [0.704, 2.9238, 8.7937, 48.0093], "c": 13.5782},
#     "Bi3+": {"a": [21.8053, 19.5026, 19.1053, 7.10295], "b": [1.2356, 6.24149, 0.469999, 20.3185], "c": 12.4711},
#     "Bi5+": {"a": [33.5364, 25.0946, 19.2497, 6.91555], "b": [0.91654, 0.39042, 5.71414, 12.8285], "c": -6.7994},
#     "Po": {"a": [34.6726, 15.4733, 13.1138, 7.02588], "b": [0.700999, 3.55078, 9.55642, 47.0045], "c": 13.677},
#     "At": {"a": [35.3163, 19.0211, 9.49887, 7.42518], "b": [0.68587, 3.97458, 11.3824, 45.4715], "c": 13.7108},
#     "Rn": {"a": [35.5631, 21.2816, 8.0037, 7.4433], "b": [0.6631, 4.0691, 14.0422, 44.2473], "c": 13.6905},
#     "Fr": {"a": [35.9299, 23.0547, 12.1439, 2.11253], "b": [0.646453, 4.17619, 23.1052, 150.645], "c": 13.7247},
#     "Ra": {"a": [35.763, 22.9064, 12.4739, 3.21097], "b": [0.616341, 3.87135, 19.9887, 142.325], "c": 13.6211},
#     "Ra2+": {"a": [35.215, 21.67, 7.91342, 7.65078], "b": [0.604909, 3.5767, 12.601, 29.8436], "c": 13.5431},
#     "Ac": {"a": [35.6597, 23.1032, 12.5977, 4.08655], "b": [0.589092, 3.65155, 18.599, 117.02], "c": 13.5266},
#     "Ac3+": {"a": [35.1736, 22.1112, 8.19216, 7.05545], "b": [0.579689, 3.41437, 12.9187, 25.9443], "c": 13.4637},
#     "Th": {"a": [35.5645, 23.4219, 12.7473, 4.80703], "b": [0.563359, 3.46204, 17.8309, 99.1722], "c": 13.4314},
#     "Th4+": {"a": [35.1007, 22.4418, 9.78554, 5.29444], "b": [0.555054, 3.24498, 13.4661, 23.9533], "c": 13.376},
#     "Pa": {"a": [35.8847, 23.2948, 14.1891, 4.17287], "b": [0.547751, 3.41519, 16.9235, 105.251], "c": 13.4287},
#     "U": {"a": [36.0228, 23.4128, 14.9491, 4.188], "b": [0.5293, 3.3253, 16.0927, 100.613], "c": 13.3966},
#     "U3+": {"a": [35.5747, 22.5259, 12.2165, 5.37073], "b": [0.52048, 3.12293, 12.7148, 26.3394], "c": 13.3092},
#     "U4+": {"a": [35.3715, 22.5326, 12.0291, 4.7984], "b": [0.516598, 3.05053, 12.5723, 23.4582], "c": 13.2671},
#     "U6+": {"a": [34.8509, 22.7584, 14.0099, 1.21457], "b": [0.507079, 2.8903, 13.1767, 25.2017], "c": 13.1665},
#     "Np": {"a": [36.1874, 23.5964, 15.6402, 4.1855], "b": [0.511929, 3.25396, 15.3622, 97.4908], "c": 13.3573},
#     "Np3+": {"a": [35.7074, 22.613, 12.9898, 5.43227], "b": [0.502322, 3.03807, 12.1449, 25.4928], "c": 13.2544},
#     "Np4+": {"a": [35.5103, 22.5787, 12.7766, 4.92159], "b": [0.498626, 2.96627, 11.9484, 22.7502], "c": 13.2116},
#     "Np6+": {"a": [35.0136, 22.7286, 14.3884, 1.75669], "b": [0.48981, 2.81099, 12.33, 22.6581], "c": 13.113},
#     "Pu": {"a": [36.5254, 23.8083, 16.7707, 3.47947], "b": [0.499384, 3.26371, 14.9455, 105.98], "c": 13.3812},
#     "Pu3+": {"a": [35.84, 22.7169, 13.5807, 5.66016], "b": [0.484938, 2.96118, 11.5331, 24.3992], "c": 13.1991},
#     "Pu4+": {"a": [35.6493, 22.646, 13.3595, 5.18831], "b": [0.481422, 2.8902, 11.316, 21.8301], "c": 13.1555},
#     "Pu6+": {"a": [35.1736, 22.7181, 14.7635, 2.28678], "b": [0.473204, 2.73848, 11.553, 20.9303], "c": 13.0582},
#     "Am": {"a": [36.6706, 24.0992, 17.3415, 3.49331], "b": [0.483629, 3.20647, 14.3136, 102.273], "c": 13.3592},
#     "Cm": {"a": [36.6488, 24.4096, 17.399, 4.21665], "b": [0.465154, 3.08997, 13.4346, 88.4834], "c": 13.2887},
#     "Bk": {"a": [36.7881, 24.7736, 17.8919, 4.23284], "b": [0.451018, 3.04619, 12.8946, 86.003], "c": 13.2754},
#     "Cf": {"a": [36.9185, 25.1995, 18.3317, 4.24391], "b": [0.437533, 3.00775, 12.4044, 83.7881], "c": 13.2674},
# }


# #new coefficients obtained by fitting four gaussian sum _without_ any c offset to standard cromer-mann coefficients to enable better behavior in real space.
# ffcoeff = {
# 'H': {'a': [0.48964434865164597, 0.25795506418162684, 0.19862808650933614, 0.05398838736648019], 'b': [20.659384838283998, 7.740482800259867, 49.551905131184874, 2.2010683273966882]} ,
# 'H1-': {'a': [0.9071753985631006, 0.5547447306202674, 0.4115769050296404, 0.12447711090054299], 'b': [53.136705476073885, 15.188432591359481, 186.575978405005, 3.5617171474702056]} ,
# 'He': {'a': [0.8402922443501193, 0.6770417172598773, 0.3232320715891434, 0.15997466322213108], 'b': [9.09424670699855, 3.3772384290126336, 22.929576163282476, 0.7981420961478064]} ,
# 'Li': {'a': [1.343609673471987, 0.5581460903914761, 0.6658627628946067, 0.4297009293162577], 'b': [3.4186798862221264, 0.7097939424452019, 85.3914626034448, 168.2600051988492]} ,
# 'Li1+': {'a': [0.6521325007637485, 0.8713921608933449, 0.30384786383519363, 0.17325108516049692], 'b': [4.60746938127854, 1.943068012377627, 0.4989788685624173, 10.098249257464639]} ,
# 'Be': {'a': [1.5994303672876535, 1.2478102584629966, 0.5339734402314658, 0.6185783589344618], 'b': [43.644988718536744, 1.7575840483943501, 103.48264477722779, 0.4201575607825604]} ,
# 'Be2+': {'a': [1.4925894596616573, -2.997317044757223, 2.9716243406311458, 0.5325085179518985], 'b': [0.5007632237312274, 0.7724419338026846, 1.0230595840990484, 3.590969161148091]} ,
# 'B': {'a': [2.0574620263921792, 1.257240860265411, 1.0956144348175705, 0.5880097963382053], 'b': [23.218442527004388, 1.0411009520183316, 60.3496687850438, 0.2505910912714547]} ,
# 'C': {'a': [3.049502465329157, 0.9469490189353504, 0.9205181585530644, 1.0895088896753868], 'b': [16.122830991045564, 0.879328843206143, 0.2466316869059288, 51.88436420676904]} ,
# 'Cval': {'a': [2.932013507490663, 0.8160532965401153, 1.1234353890187063, 1.137903084035025], 'b': [16.51213525878479, 0.19986107169540498, 0.9733924016132077, 55.68201812702491]} ,
# 'N': {'a': [5.551459140691302, 2.480555683008789, 2.773120764746902, -3.8187500178972695], 'b': [0.3145270000398675, 7.917446957141198, 24.43261890668131, 0.3145232450476127]} ,
# 'O': {'a': [3.2844165713667874, 2.1863494590744375, 1.734158144526276, 0.7918335350536696], 'b': [13.427747243117944, 5.212997411416355, 0.24282907358098274, 32.87575302824983]} ,
# 'O1-': {'a': [4.224882908837335, 1.831296845810794, 2.9124978314912977, -0.00047209932300866167], 'b': [7.76577712362189, 0.27544797766020385, 31.29055593973186, -1.4660236391560162]} ,
# 'F': {'a': [3.6762412015418953, 2.60065420705191, 1.7420151081741362, 0.9787401742292346], 'b': [10.364707057431389, 4.079237346883059, 0.19463190393042593, 26.12658850187785]} ,
# 'F1-': {'a': [2.482295216093523, 4.736356610988728, 1.6761532518219728, 1.110174052761956], 'b': [3.7134925956696763, 11.81776769187783, 0.17403637609617464, 47.59964081720466]} ,
# 'Ne': {'a': [4.055730269122589, 3.0998667969555798, 1.7505299617295156, 1.0916156990940862], 'b': [8.454980638330753, 3.3007667744262896, 0.16047469144754145, 21.7022092253482]} ,
# 'Na': {'a': [4.367451307758355, 3.723708903958624, 1.77879968238579, 1.1243459810203416], 'b': [2.9482062298952902, 8.200827604895427, 0.13827661121196047, 129.44511395457567]} ,
# 'Na1+': {'a': [3.2799886235669637, 3.9596293759294316, 1.7525359468423036, 1.0082777045790943], 'b': [2.609617328989359, 6.1176923173324145, 0.13371642453893953, 14.035869144602884]} ,
# 'Mg': {'a': [4.818076047335444, 2.1875633144226194, 1.7944008622352252, 3.189046687147888], 'b': [2.434499912312549, 79.40242393599029, 0.1197273318089302, 6.279937730139619]} ,
# 'Mg2+': {'a': [3.519572807270986, 3.8786505321226734, 1.7527677662942893, 0.8493666825150271], 'b': [2.117126429542827, 4.757392456792007, 0.11312219259845034, 10.132102208317802]} ,
# 'Al': {'a': [7.131647526944289, 2.052237395760297, 1.6777005913633178, 2.145059987171763], 'b': [2.605480932115646, 0.14405362103885988, 23.900259857804425, 87.0731462362444]} ,
# 'Al3+': {'a': [4.215201503680869, 3.4664591461366374, 1.787953514807232, 0.5309487717025967], 'b': [1.8687554144251337, 4.141629472459154, 0.10335255253614334, 8.270426349821584]} ,
# 'Siv': {'a': [7.177842532089736, 2.979233663655885, 2.115396326947422, 1.732368108082893], 'b': [2.1471528723641295, 29.156303522037696, 0.13610466792249196, 82.09299264982448]} ,
# 'Sival': {'a': [7.283489866937556, 2.9446452003186616, 2.0712098516533555, 1.711456507844776], 'b': [2.1602180093474126, 32.93110363598472, 0.12538274411646538, 94.159406410391]} ,
# 'Si4+': {'a': [4.479210923307321, 3.308079396828919, 1.8042959299452555, 0.4085515358831025], 'b': [1.581533581234293, 3.437288224831375, 0.09236081995334067, 6.628264673006444]} ,
# 'P': {'a': [7.138789042249852, 4.142606814307699, 2.128293691849062, 1.5929377824602673], 'b': [1.7548326959642166, 26.16163488546785, 0.12230615790259426, 68.26204711582082]} ,
# 'S': {'a': [7.1088513543231375, 5.215028849933757, 2.091806378705691, 1.583611951864076], 'b': [1.4377552638137123, 22.18043778213814, 0.10418335440325559, 56.17384058069765]} ,
# 'Cl': {'a': [1.9675809012785415, 7.134250887705102, 6.233287662565015, 1.6648772411362953], 'b': [0.07802936441900629, 1.1734538042863885, 18.504210586649002, 47.40001481268868]} ,
# 'Cl1-': {'a': [1.9797087751903877, 7.141842567575879, 6.4621750349341704, 2.4083590826533365], 'b': [0.07935371271621547, 1.1783215880574136, 19.42441234878529, 59.166678815987005]} ,
# 'Ar': {'a': [2.134311333383945, 7.048364017604981, -42.54265350919579, 51.266219019124605], 'b': [0.08816736534588678, 1.0403197028738291, 18.756223989671735, 18.756228496362894]} ,
# 'K': {'a': [7.128272700184678, 1.84675077855445, 1.3047710718253969, 8.754974969411487], 'b': [0.8346440074564205, 0.05336834294143281, 213.59269919690686, 13.834219672415935]} ,
# 'K1+': {'a': [7.944603928601062, 7.491903004891839, 1.3604015189795347, 1.2051140678974082], 'b': [12.621630459473565, 0.7671925307800991, -0.009314751010806187, 31.72580083414482]} ,
# 'Ca': {'a': [7.293370149744261, 1.4918644094208173, 8.695226611602525, 2.501556322788237], 'b': [0.6722854061968923, 0.015079185750225878, 10.574975039109164, 117.64743359674988]} ,
# 'Ca2+': {'a': [0.854464166648321, 7.847870117074988, 8.187136693755233, 1.110267478701466], 'b': [-0.09393101327901535, 0.6104018757577403, 10.156644775059668, 22.917226121946168]} ,
# 'Sc': {'a': [6.927716637509347, 1.8406189781185702, 2.8136124472016806, 9.388668559325467], 'b': [0.6152464249726448, 0.0545188013823816, 94.8780054779866, 9.319601027722328]} ,
# 'Sc3+': {'a': [0.0817349659497468, 12.607511351012718, -4.064516754545099, 9.338805092158417], 'b': [-0.6945002189745337, 0.4727673635973685, 0.4725265268626642, 9.101032168568663]} ,
# 'Ti': {'a': [5.798010397584866, 2.986715517050453, 10.143729517004187, 3.0321993153415723], 'b': [0.6191507176029577, 0.1299204448311216, 8.329940725940265, 80.3754663832968]} ,
# 'Ti2+': {'a': [2.2735088158146275, 6.717072996397934, 10.460700433558953, 0.5835022648484749], 'b': [1.1509867160518434, 0.2952971935099805, 8.931984896745671, 61.26058897167574]} ,
# 'Ti3+': {'a': [8.407411200151689, 8.833401429274844, 0.03204659115183811, 1.7244847417346636], 'b': [0.39734049188184956, 7.221690045271849, -0.8228198626145534, 16.28463077354178]} ,
# 'Ti4+': {'a': [8.476658138704183, 14.441920812371531, 0.03208255824326217, -4.978879154891721], 'b': [0.4024551482238033, 7.5664191322581065, -0.8750173641735599, 7.56641736169824]} ,
# 'V': {'a': [3.3194328838976803, 5.564007129352879, 10.879782505212269, 3.189706659771217], 'b': [0.8004604109968732, 0.2170743259814505, 7.556847000507465, 70.33440719348388]} ,
# 'V2+': {'a': [2.428302951323674, 7.410525470212022, 10.822124925456993, 0.38622798931619795], 'b': [1.9443349727202766, 0.27967991987671936, 8.834247919473418, 115.00508996707138]} ,
# 'V3+': {'a': [1.8483337959280466, 7.910957332616291, 10.011513171648161, 0.24829388842407749], 'b': [2.6239568396117297, 0.3104100357943613, 8.032333631198057, 63.81787797852419]} ,
# 'V5+': {'a': [12.014180750919468, 9.774468510524875, 9.536559159536635, -13.346321661732093], 'b': [-0.03174470437607606, 0.33650729809417274, 6.424674366316235, -0.005825976728445994]} ,
# 'Cr': {'a': [7.8741948280687355, 8.02821825507154, 6.380182817714588, 1.733734242205387], 'b': [4.665025971179984, 0.28051188026144186, 13.073238958909593, 99.23633042114646]} ,
# 'Cr2+': {'a': [3.9944671521822, 8.060536959370573, 9.034600589461204, 0.9212279820454514], 'b': [3.7494763581047357, 0.28931995688802403, 8.427580952099639, 32.50859056032136]} ,
# 'Cr3+': {'a': [3.9803720028685063, 8.079385646943878, 8.527881463755337, 0.4211649162780186], 'b': [3.8447094204450867, 0.2895904227833914, 7.971672197671396, 32.721953281312516]} ,
# 'Mn': {'a': [9.679809368496745, 8.070655740259923, 4.864372125200564, 2.391372058969052], 'b': [4.578312394600616, 0.2576607566059864, 12.961828784388233, 84.42449085089338]} ,
# 'Mn2+': {'a': [5.834072311965402, 7.983497001746192, 8.576602006733218, 0.6229893450380755], 'b': [3.6556191713972668, 0.25395236854452957, 8.691222344728285, 41.25369125784355]} ,
# 'Mn3+': {'a': [9.627068289952064, 8.154810243348713, 4.037035391015142, 0.17755809941086245], 'b': [4.741558048475204, 0.2678652729655168, 10.990005922395714, 24.101840727795633]} ,
# 'Mn4+': {'a': [9.833496900137092, 8.154802732652835, 3.0168085539475245, -0.007119000195157985], 'b': [4.74508972274041, 0.2670155695812446, 10.597117602191526, 27.570564917908712]} ,
# 'Fe': {'a': [10.688900319565898, 8.088685887682288, 4.820323476012463, 2.4014580387467794], 'b': [4.2825433607661125, 0.23701813386013074, 12.503475815560247, 77.24478277789902]} ,
# 'Fe2+': {'a': [7.52163739958737, 8.027526091498078, 7.6690618564183435, 0.7918982713887573], 'b': [3.6913933367545257, 0.23465530530429618, 8.46673608197085, 31.322835831678848]} ,
# 'Fe3+': {'a': [7.709749721783281, 8.022725556732716, 7.000073235342508, 0.2767741909384679], 'b': [3.7148102943753054, 0.2337650772854649, 8.172612519331967, 38.5042229226976]} ,
# 'Co': {'a': [11.499293816512356, 8.09146785252594, 4.986773996024937, 2.4177540443964842], 'b': [3.955799054905932, 0.2182472013789286, 11.762488158071305, 71.38290050598879]} ,
# 'Co2+': {'a': [8.737363288806632, 8.039921557827999, 7.250154466599938, 0.9786787316026032], 'b': [3.5287464192658953, 0.2163103314764206, 8.060544279970765, 25.763634385128736]} ,
# 'Co3+': {'a': [10.290952438757389, 8.10618776705137, 4.9501359083873115, 0.651154890957391], 'b': [3.8500242471870987, 0.2239448468652033, 8.45942482884542, 18.31978823121851]} ,
# 'Ni': {'a': [12.228560624787853, 8.086439780457194, 5.244069898673469, 2.4332622490013924], 'b': [3.641884069277759, 0.2013763900123453, 10.97344889054884, 66.47979770670938]} ,
# 'Ni2+': {'a': [9.663739003815504, 8.036055871377453, 7.1439447698249765, 1.1598515951663935], 'b': [3.3011032101014557, 0.1994774540660824, 7.575597205980667, 22.314955718079155]} ,
# 'Ni3+': {'a': [10.753799063560493, 8.06850344666784, 5.406794686850915, 0.7692174285281526], 'b': [3.490932875312508, 0.2050907943630945, 7.746508829674841, 16.932459861874786]} ,
# 'Cu': {'a': [12.809376984231838, 8.098010632746204, 6.357765183035302, 1.725122630314828], 'b': [3.37492912785031, 0.18740577737899483, 10.526955852134309, 64.86680684817746]} ,
# 'Cu1+': {'a': [11.016697483289493, 8.0460899812387, 7.292783621193453, 1.645365380086967], 'b': [3.1435261714060605, 0.18532821819319567, 7.944401588153332, 25.988681725155512]} ,
# 'Cu2+': {'a': [9.677874798834255, 7.951204176450752, 8.032430252455619, 1.3419653500156055], 'b': [2.9580156405229387, 0.18455372567416234, 6.745585408932257, 20.07910374097473]} ,
# 'Zn': {'a': [13.609647713903893, 8.069746100746938, 5.856791667902585, 2.453106395886765], 'b': [3.0948668481635084, 0.1728941891912288, 9.547948888213368, 58.78824499323322]} ,
# 'Zn2+': {'a': [11.981604305278022, 8.036994731438769, 6.649176457941015, 1.3297639070473135], 'b': [2.9353619709575365, 0.17176674096039274, 7.150594105752649, 18.063972564784844]} ,
# 'Ga': {'a': [14.891152516727383, 8.068375948334818, 5.009789220033591, 3.011852155999751], 'b': [2.9011908179266426, 0.16097568008517638, 9.80459484875494, 61.5217013121947]} ,
# 'Ga3+': {'a': [10.26428482365009, 7.846556968401461, 8.664270313815388, 1.227688075991066], 'b': [2.445742520452998, 0.1556373854373596, 5.371470968887742, 14.545736206772403]} ,
# 'Ge': {'a': [15.916069364354462, 8.060104064902745, 4.26416150337424, 3.7430009906335346], 'b': [2.695421101450001, 0.15006368031238693, 10.242480718074383, 54.960551218032435]} ,
# 'Ge4+': {'a': [12.960979135805113, 7.900503059711587, 6.433384423499361, 0.7024180390606878], 'b': [2.453518685708239, 0.14764665594710813, 5.607313169883798, 11.495773441485255]} ,
# 'As': {'a': [16.546046440254102, 8.025859803876386, 3.709618117665297, 4.695204913379802], 'b': [2.4685838748852316, 0.13932675514449963, 10.17066097718369, 45.16465982217951]} ,
# 'Se': {'a': [17.322347906266007, 8.035659386649552, 4.181630304523794, 4.45538558070083], 'b': [2.295018054339401, 0.13134701551535444, 13.958070846499647, 44.16138539990943]} ,
# 'Br': {'a': [17.598199482462583, 5.303045987858294, 7.977840732705101, 4.119119436227171], 'b': [2.085289104906673, 15.66629268701866, 0.12172380749665068, 41.63799635316099]} ,
# 'Br1-': {'a': [17.716706588540525, 6.396745813643356, 8.019334033097135, 3.8590133847192356], 'b': [2.106663613342509, 18.425351013871694, 0.12327637670136245, 58.26418914597132]} ,
# 'Kr': {'a': [17.736237964281305, 6.733419721135826, 7.878437543564538, 3.6509502411805768], 'b': [1.882521647850299, 16.07503935024332, 0.1114692716170516, 39.478219164922734]} ,
# 'Rb': {'a': [17.86427210314376, 9.728713237341527, 7.8355761204620284, 1.5531529301401938], 'b': [1.7194415198809374, 17.08235241528327, 0.10431798153572124, 164.93128372439642]} ,
# 'Rb1+': {'a': [17.77803654883815, 7.65399052526326, 7.734332623204072, 2.833775378642999], 'b': [1.6914552007200476, 14.643361278632849, 0.10057923206820207, 31.23108965166537]} ,
# 'Sr': {'a': [17.833200657321598, 9.85029936926106, 7.623132493957599, 2.676887918076099], 'b': [1.5336253471267733, 14.036115829061147, 0.09185781687226496, 132.36953801892722]} ,
# 'Sr2+': {'a': [7.5576376280521345, 8.728591071759498, 1.8906361614298586, 17.823633215040545], 'b': [0.0893493405104573, 13.241071189167167, 26.81767231319672, 1.520862659880473]} ,
# 'Y': {'a': [17.925549529791752, 10.30913274963786, 7.470923147563964, 3.2697162477600665], 'b': [1.3922518363683456, 12.775111436619321, 0.08258558768988813, 104.34088422438373]} ,
# 'Y3+': {'a': [7.686583103886185, 51.28964009715699, 10.553948044972422, -33.56874993409134], 'b': [0.08968069403772926, 1.4157962532938413, 12.869781336073437, 1.4156457739154786]} ,
# 'Zr': {'a': [18.02947856216866, 10.959340266967383, 7.3189557352086965, 3.6608014858556817], 'b': [1.267144098312377, 11.897317210552744, 0.07393449005858094, 87.63891494848176]} ,
# 'Zr4+': {'a': [17.724450449800077, 9.328389919452698, 1.6897128793915053, 7.254196712575971], 'b': [1.2426109659717193, 9.967521758373682, 17.725765933571477, 0.07144483381497152]} ,
# 'Nb': {'a': [18.186394435483777, 12.040504915935923, 7.1881609106371105, 3.5453869019221615], 'b': [1.1617794569749893, 11.720318125113405, 0.06661139711431247, 69.73056591736231]} ,
# 'Nb3+': {'a': [7.0880514205453, 17.93305220963883, 10.777264909817958, 2.1970230173186267], 'b': [0.06300048579087202, 1.136702113278468, 10.025273586350396, 26.4760608308937]} ,
# 'Nb5+': {'a': [7.05459699772521, 17.817676656044288, 10.76954933457027, 0.35911914157140296], 'b': [0.06204440916193552, 1.129419992006384, 9.27881053724674, 24.645401617526986]} ,
# 'Mo': {'a': [6.98924374895165, 18.289186255416038, 12.919989236877681, 3.7584219590883823], 'b': [0.05773531836099571, 1.056410060557956, 10.954181341406386, 61.58476361233627]} ,
# 'Mo3+': {'a': [6.826693523854915, 18.102528983384722, 11.523083068329322, 2.5430091026986688], 'b': [0.05222105951778729, 1.0315833084024293, 9.418202480596367, 25.235470558884792]} ,
# 'Mo5+': {'a': [6.786874356813757, 18.001628841076297, 11.297558634821002, 0.9138176780550937], 'b': [0.05104716293002873, 1.0246614305730266, 8.71368899375687, 21.00151626411173]} ,
# 'Mo6+': {'a': [17.896013692287035, 11.175609326677826, 6.7493376384118315, 0.1649732771785508], 'b': [1.0361014721761084, 8.480243995537196, 0.05632867184829788, 0.015796839735227674]} ,
# 'Tc': {'a': [6.096734719862069, 18.833925258053807, 13.606487722628122, 4.406861701872697], 'b': [0.025598280364449038, 0.9175484395543193, 9.66279992154511, 60.046913913611924]} ,
# 'Ru': {'a': [5.994561155996193, 18.93567855915201, 14.823045439115054, 4.18527989861619], 'b': [0.022216614030365338, 0.8490374076607612, 9.46969394478458, 48.68361979245615]} ,
# 'Ru3+': {'a': [18.541601448484673, 13.312995515349357, 6.1631735301613615, 2.981355524651754], 'b': [0.8483649327780811, 8.384095948280406, 0.02828023674034836, 23.001940355101304]} ,
# 'Ru4+': {'a': [20.919028802285496, 13.233315558669844, 3.684640054209721, 2.1649591772711196], 'b': [0.8030780473811314, 8.111931515730669, 0.11751830636099739, 21.055011916322563]} ,
# 'Rh': {'a': [5.993696498167812, 18.880516303116227, 15.868281563235248, 4.195262437602287], 'b': [0.023189837626479068, 0.7892431942662751, 8.988091508779519, 45.518367970180336]} ,
# 'Rh3+': {'a': [12.473247481612963, 16.86992009490624, 16.21075504039705, -3.699547329210125], 'b': [0.10131670361036627, 0.9613688439831227, 9.957680774895453, 0.1011115270635728]} ,
# 'Rh4+': {'a': [11.838166828283663, 17.376583744609615, 15.596948777806794, -3.9116264029764394], 'b': [0.07983733554353012, 0.900742388547325, 9.130578416958032, 0.07997826133848704]} ,
# 'Pd': {'a': [5.585122455105748, 19.11565442669995, 16.33896667355388, 4.934005879811926], 'b': [0.011051064920156861, 0.7142769532749469, 8.31424190970012, 31.619191725135053]} ,
# 'Pd2+': {'a': [19.170716935639273, 15.209319247572717, 4.322778976414755, 5.290821508519311], 'b': [0.6961852354043655, 7.555507391835464, 22.504583104340806, -2.858360398734787e-05]} ,
# 'Pd4+': {'a': [10.154619272413669, 16.940541038310272, 16.6168324599612, -1.818090099420638], 'b': [0.0882613276456976, 0.8563799652930173, 8.672291845857146, 0.08815941128014315]} ,
# 'Ag': {'a': [6.200419878651503, 18.515775652778405, 17.993617387496748, 4.22492300390107], 'b': [0.03247796522315408, 0.6881256632610405, 8.03377231153247, 40.00772998129887]} ,
# 'Ag1+': {'a': [5.492798082124939, 18.982347669555033, 16.606630566882366, 4.902366952240362], 'b': [0.009240748912280394, 0.6581528718468714, 7.398063465690764, 25.263521787505038]} ,
# 'Ag2+': {'a': [19.164897094678665, 16.24531698443506, 4.3712986896657755, 5.2133222774326775], 'b': [0.645615815560667, 7.185286221407857, 21.406101640114798, -2.5352283513989697e-05]} ,
# 'Cd': {'a': [6.684763915704815, 17.933995966750835, 18.793232974298558, 4.524022623440256], 'b': [0.047612347291735975, 0.6551081780008208, 7.438714071254825, 44.00612231946612]} ,
# 'Cd2+': {'a': [19.1528551390452, 17.253097564932425, 4.471952207950897, 5.117639315817369], 'b': [0.5978622949749756, 6.806109349056603, 20.250595515926822, -5.897362882625187e-05]} ,
# 'In': {'a': [7.646229713690459, 16.895999084388986, 19.493874779352403, 4.8822626007371674], 'b': [0.07167142598290258, 0.6381716171985667, 6.893014144454027, 49.25939296022993]} ,
# 'In3+': {'a': [19.107370644651628, 18.10993977791651, 3.790255094367243, 4.993047778615268], 'b': [0.5514232195465032, 6.3242814651361545, 17.35693391733262, -0.00011021183957204235]} ,
# 'Sn': {'a': [15.944188548989132, 8.420970540542418, 19.840611887610837, 5.716612980300875], 'b': [0.6117747304064486, 0.08773789790339957, 6.256651649612374, 48.0138705072425]} ,
# 'Sn2+': {'a': [5.5057903682299045, 18.49430023908842, 19.372635872614403, 4.610498870825351], 'b': [0.021421813749264267, 0.5237094089338343, 5.9556831033006565, 27.2776123071188]} ,
# 'Sn4+': {'a': [18.787731147414878, 21.64210387054202, 3.6507452794990796, 1.9208632834022978], 'b': [5.702758972972682, 0.4280446731396779, 13.67848379588595, -0.1477658506811732]} ,
# 'Sb': {'a': [14.912926877860817, 9.222324672712686, 20.04599059373399, 6.748509113528009], 'b': [0.5842821262098832, 0.10162718429922638, 5.6409380506341185, 44.31670958548569]} ,
# 'Sb3+': {'a': [5.436934011536769, 18.33067086723167, 19.203438598731857, 5.016812850804455], 'b': [0.021191172143032556, 0.4858345858388681, 5.311247117209016, 21.64675384279665]} ,
# 'Sb5+': {'a': [19.86884197595894, 19.03336603156193, 2.412488433358039, 4.689174074577019], 'b': [5.448437604642272, 0.4678930768268693, 14.126723170411257, -0.00010928089749756589]} ,
# 'Te': {'a': [13.444293410464944, 10.454966466263755, 20.17472594426033, 7.858879893950705], 'b': [0.5693196209449595, 0.11954406995642837, 5.095649520146917, 40.237063663414844]} ,
# 'I': {'a': [10.868743489342876, 12.83630122916029, 20.180280530192785, 9.05025525191156], 'b': [0.5901142344364548, 0.14616647324662135, 4.601099173381122, 36.11276928305425]} ,
# 'I1-': {'a': [7.826191959313713, 16.450701025568613, 20.012580443648126, 9.58939225131849], 'b': [0.8093990924490041, 0.18488879489634585, 4.793489978094477, 42.55472287082024]} ,
# 'Xe': {'a': [7.484944723940286, 16.108456500821987, 20.08554387925215, 10.257545429219183], 'b': [0.6819562814577115, 0.17481885128253288, 4.182301200436235, 32.581518935073156]} ,
# 'Cs': {'a': [20.778896936356997, 21.399750252124672, 11.202659959114532, 1.624184393061515], 'b': [3.284694895908563, 0.22532056304473003, 22.798928094859768, 213.91131120654234]} ,
# 'Cs1+': {'a': [20.55982957260471, 21.35638551694399, 10.454357845587635, 1.6529853483322146], 'b': [3.243182677769799, 0.2246299715673662, 20.847286185055037, 59.52636885166727]} ,
# 'Ba': {'a': [20.622148119582846, 21.294181289111954, 11.302462071229492, 2.782421846168423], 'b': [3.027215676334376, 0.21311801661288926, 19.29046836898035, 167.20869458574015]} ,
# 'Ba2+': {'a': [20.38893939674827, 21.23724277505373, 11.120190307822249, 1.273314698195083], 'b': [2.9795344252789215, 0.2126007022692342, 18.28273952919527, 51.78133468555797]} ,
# 'La': {'a': [20.804093998491467, 21.220171501160248, 11.619774026499877, 3.346303936542478], 'b': [2.836959952577033, 0.20208489565596802, 18.255797559901822, 133.12785110867748]} ,
# 'La3+': {'a': [20.453339003394234, 21.15510510208706, 11.852851524184576, 0.5492307776668323], 'b': [2.780965565376222, 0.20150338293279974, 16.99895051643849, 54.94790483702286]} ,
# 'Ce': {'a': [21.351628666547644, 21.218274002314608, 12.042745667752657, 3.373228380017131], 'b': [2.732134150763417, 0.1932332057976821, 17.25857882228192, 127.11381571549278]} ,
# 'Ce3+': {'a': [20.95770490841373, 21.15188354617442, 12.06237150246465, 0.8361384236233754], 'b': [2.6746509522161035, 0.19265746523114172, 15.904131307034614, 43.1776683196551]} ,
# 'Ce4+': {'a': [20.46780658785507, 21.050692482112346, 12.272875800137761, 0.21436649977640013], 'b': [2.59053252866421, 0.19073658470100335, 15.483558103865061, 62.235311432155044]} ,
# 'Pr': {'a': [22.232289357104968, 21.292333148730588, 12.599871648233327, 2.862268445960792], 'b': [2.693803909281057, 0.1864002957654481, 16.44196727411959, 143.64311853937306]} ,
# 'Pr3+': {'a': [21.48834917137474, 21.136121251490938, 12.202385072588186, 1.178647371928732], 'b': [2.5721014792817463, 0.1841953572020822, 14.856269470209613, 36.42217743314406]} ,
# 'Pr4+': {'a': [21.038301249114365, 21.040693184637654, 12.555573073166693, 0.36963206681154914], 'b': [2.4988535515898023, 0.18243667426544635, 14.584616503425261, 45.46443752062808]} ,
# 'Nd': {'a': [22.860093697660574, 21.278339750056404, 12.962042276474051, 2.8830434407097902], 'b': [2.596556334283178, 0.17843612127560346, 15.629012372472031, 137.90087541048754]} ,
# 'Nd3+': {'a': [22.045104741826865, 21.115860179035383, 12.150198967802003, 1.6929163620100862], 'b': [2.4752790562906832, 0.1761973082984479, 13.83369647749148, 30.90012743124577]} ,
# 'Pm': {'a': [23.516718265530972, 21.260930299087423, 13.300107511159672, 2.9033053721740787], 'b': [2.5044199872795176, 0.17092829513032773, 14.884043898776243, 132.71801628268963]} ,
# 'Pm3+': {'a': [22.614213035300804, 21.08668262262865, 12.086885051108618, 2.2147546764301476], 'b': [2.3815567171516716, 0.16857171576308796, 12.890340723433617, 27.482589937285102]} ,
# 'Sm': {'a': [24.195949760216834, 21.241697390412433, 13.618354494734005, 2.9226884251749516], 'b': [2.416986538588082, 0.16386741601112575, 14.201210390654035, 128.0026275934193]} ,
# 'Sm3+': {'a': [23.218059983359883, 21.057743069792117, 11.969154929101158, 2.754744131357956], 'b': [2.2945178585488186, 0.16142278559142087, 12.064442722960177, 24.829222408477147]} ,
# 'Eu': {'a': [24.85374648309956, 21.213613325803614, 13.958363942197979, 2.9515195512485017], 'b': [2.329381255932067, 0.15708986458845103, 13.552801349925566, 123.16767505640108]} ,
# 'Eu2+': {'a': [24.0890671513384, 21.092251672287084, 11.880834387348129, 3.934813508157573], 'b': [2.2477259867902117, 0.15571776061546067, 11.465732054998245, 26.525198090869484]} ,
# 'Eu3+': {'a': [23.796719726752272, 21.016805919213553, 11.886162947720237, 3.2997744315472977], 'b': [2.207195231174613, 0.15453834657750679, 11.240954120612608, 23.00361719428446]} ,
# 'Gd': {'a': [25.2814185997105, 21.106061122259135, 14.012223597491934, 3.5707710477975936], 'b': [2.2075468589521816, 0.1495029992876105, 12.777842480418922, 101.38765688513575]} ,
# 'Gd3+': {'a': [24.381276621444933, 20.971620393931826, 11.897973427376451, 3.747552486305546], 'b': [2.123832569892709, 0.1480063050555474, 10.520433990481896, 21.71204160270305]} ,
# 'Tb': {'a': [26.21142667806698, 21.17005012186444, 14.564101950769615, 3.0201266154220576], 'b': [2.170956913982665, 0.1449679489827454, 12.381705948587939, 112.45833333640589]} ,
# 'Tb3+': {'a': [24.988044943446795, 20.921319812961073, 12.273420099644742, 3.8128583133553366], 'b': [2.0444984802816495, 0.14182054236584518, 9.9880455107844, 21.289528265698877]} ,
# 'Dy': {'a': [26.91133582297724, 21.154138174356508, 14.856219275412181, 3.0417947692928666], 'b': [2.0985200876617944, 0.13952815530236076, 11.878459514702051, 108.67140388777987]} ,
# 'Dy3+': {'a': [25.564217832582813, 20.881116363708387, 12.00889697497743, 4.5437854141030085], 'b': [1.9696464210243518, 0.13605619048644613, 9.286375563067452, 19.598520108952133]} ,
# 'Ho': {'a': [27.376984412611606, 21.05431646316108, 14.857668395289394, 3.6792341411221487], 'b': [1.9962634583696501, 0.1332110801481831, 11.215438136419634, 92.62076693188642]} ,
# 'Ho3+': {'a': [26.150042369258347, 20.835504890890796, 12.016961650444753, 4.99618984014991], 'b': [1.897787818968079, 0.13057403859539174, 8.71699846029907, 18.617632420960373]} ,
# 'Er': {'a': [28.331474748224093, 21.13959515593194, 15.406215228806182, 3.0822526564340866], 'b': [1.9655869087981321, 0.1298369977344657, 10.977210158466955, 101.75618804162296]} ,
# 'Er3+': {'a': [26.77491834197368, 20.79976217460133, 12.192865850343411, 5.230318946603022], 'b': [1.8323544288807225, 0.12553974013458918, 8.291427407238599, 17.892094286530423]} ,
# 'Tm': {'a': [29.093351406947022, 21.160934380835286, 15.671732174457567, 3.0461566814242946], 'b': [1.9092287945754616, 0.12580459435699043, 10.66255566219408, 102.92188659033714]} ,
# 'Tm3+': {'a': [27.38511129124567, 20.76085592743977, 12.388833586642825, 5.462807575523996], 'b': [1.7687359091007433, 0.12074711149707033, 7.8784004846984965, 17.274206531234004]} ,
# 'Yb': {'a': [29.76783760033415, 21.155772685895244, 15.913499842673373, 3.118734345001671], 'b': [1.8472432769832943, 0.12165573303545035, 10.196795513245357, 95.60546695804919]} ,
# 'Yb2+': {'a': [28.259706819608194, 20.808819661495843, 13.125179566890662, 5.796898561538107], 'b': [1.7334591020411518, 0.1173321622406408, 7.77048627776402, 19.530091689717764]} ,
# 'Yb3+': {'a': [28.004620399778858, 20.726308140573117, 12.675003035401955, 5.59157856506838], 'b': [1.7088228542820003, 0.11627312028764382, 7.529540908194306, 16.77437400781091]} ,
# 'Lu': {'a': [30.190606889259936, 15.846306604396084, 21.056257342034176, 3.8600762276032277], 'b': [1.7602246663185135, 9.52826189426161, 0.11652006538366007, 80.95304159941352]} ,
# 'Lu3+': {'a': [28.540819288883586, 20.67542304405619, 12.563427544273216, 6.2158376108649085], 'b': [1.6469579487564097, 0.11183219860122598, 7.026694055074942, 15.824787810404143]} ,
# 'Hf': {'a': [30.664800529674537, 15.831911254590995, 20.974709048614887, 4.477730904147435], 'b': [1.6817031343555595, 9.114401069402001, 0.11188409411240202, 69.20524402650419]} ,
# 'Hf4+': {'a': [28.933403353682937, 20.563276471449583, 12.774718656273098, 5.7270682430808275], 'b': [1.57104476024411, 0.10687918080224937, 6.666761254293958, 13.993823309037207]} ,
# 'Ta': {'a': [31.136723527428696, 15.920472413597443, 20.900574527571955, 4.987629028331005], 'b': [1.6080064622882575, 8.842398856890476, 0.10759203074165964, 60.7637076356592]} ,
# 'Ta5+': {'a': [29.245758734384186, 20.428011473717078, 12.85288640765996, 5.4723494967563635], 'b': [1.4946245744547273, 0.10189868932243736, 6.259131194489869, 12.402981803620309]} ,
# 'W': {'a': [31.577112820370758, 16.142244719817846, 20.82700190036082, 5.396558777219381], 'b': [1.537287810663225, 8.653603635020888, 0.10354273640293733, 54.53077708104311]} ,
# 'W6+': {'a': [29.5410859247549, 20.28647465957908, 13.065361454843337, 5.1063779634657065], 'b': [1.4217840207089896, 0.09709501381204079, 5.912131221289429, 11.188876638442817]} ,
# 'Re': {'a': [31.97149113432179, 16.44378389014609, 20.748440228266094, 5.777917055318315], 'b': [1.4687734032283828, 8.480963729949352, 0.09965101599722397, 49.61079032248328]} ,
# 'Os': {'a': [32.31910265768036, 16.882500358474424, 20.662762784803874, 6.076828914917404], 'b': [1.4022122973287576, 8.33637428857383, 0.09588700000867463, 45.691183315346905]} ,
# 'Os4+': {'a': [31.114326463985083, 15.16716727007314, 20.154146414673168, 5.55706137955854], 'b': [1.3220155406851877, 6.595172013558815, 0.08982487670918048, 17.3768116854278]} ,
# 'Ir': {'a': [32.617649413509405, 17.455546421591578, 20.565744824251073, 6.302388258614014], 'b': [1.3372924276349412, 8.199500915074758, 0.09219412925560784, 42.50222326326727]} ,
# 'Ir3+': {'a': [31.72458852782645, 15.849314535474273, 20.097665759653236, 6.3173273895617], 'b': [1.2729877651218446, 6.814253290661127, 0.08662391882860124, 19.63790928832728]} ,
# 'Ir4+': {'a': [31.49875318879973, 15.29751699309206, 19.98118007997426, 6.213879245847549], 'b': [1.2584832583500443, 6.421313506515947, 0.085271228995903, 16.699100408473043]} ,
# 'Pt': {'a': [32.96432928925383, 18.26083878972203, 20.36054046989642, 6.359548655712784], 'b': [1.2685496817352055, 8.187794756342075, 0.08738585806906068, 36.28765289254246]} ,
# 'Pt2+': {'a': [32.23203949701645, 16.784258403836652, 20.059365934295343, 6.907078903915291], 'b': [1.2254359291803811, 7.02930233030864, 0.08384670316117493, 22.07235354309194]} ,
# 'Pt4+': {'a': [31.847104228712418, 15.48523902310183, 19.770754381603666, 6.886329460839567], 'b': [1.195335648318536, 6.243285609685901, 0.08050799998403217, 15.922897968063221]} ,
# 'Au': {'a': [20.206506183557714, 19.07265591176942, 33.168216582200465, 6.500003685912872], 'b': [0.08347434015883237, 8.002954958839846, 1.2049901583396525, 34.089209054753844]} ,
# 'Au1+': {'a': [32.65046117737725, 18.010010378861036, 20.077148008866697, 7.234653005058141], 'b': [1.1822037274132289, 7.272200284738968, 0.0819173371788424, 25.154224577312483]} ,
# 'Au3+': {'a': [32.39771193014597, 16.83083029014533, 19.749174165833622, 7.012284375282704], 'b': [1.1536036690550835, 6.586994728211824, 0.07813202712517017, 18.1281446285586]} ,
# 'Hg': {'a': [20.18829184437555, 19.84247081000301, 33.24675775858782, 6.666512165242892], 'b': [0.08135383334367963, 7.7271528906139935, 1.1526295272116451, 35.57875003628969]} ,
# 'Hg1+': {'a': [32.79093465281916, 18.91797075733871, 20.06338474240891, 7.193764734042212], 'b': [1.1330378095681466, 7.103999181147633, 0.07981698150507942, 26.65271963883946]} ,
# 'Hg2+': {'a': [32.79799136134202, 18.044252045471247, 19.674934006025023, 7.467269330149924], 'b': [1.1072187593828857, 6.756031827331673, 0.07534064713473054, 20.04536437630334]} ,
# 'Tl': {'a': [20.33194223071459, 20.9070890723771, 33.284148713073506, 6.396563220168269], 'b': [0.081199427806335, 7.612525265983518, 1.1126062352265023, 40.96748850608644]} ,
# 'Tl1+': {'a': [32.85257166592103, 20.032616562288535, 19.473987269623624, 7.60445419233929], 'b': [1.0837958676401729, 0.07765539974100971, 6.803411927934827, 27.115909954479783]} ,
# 'Tl3+': {'a': [32.96890328672284, 18.243106381662543, 19.29685509544275, 7.4484539128628295], 'b': [1.0403835585607422, 6.357386120424931, 0.06934034893047368, 16.81272545449983]} ,
# 'Pb': {'a': [20.28631272483533, 33.22776030638617, 21.41011705587274, 6.996205392377766], 'b': [0.07899366775958007, 1.0605253388515676, 7.17593190081012, 41.651474630076436]} ,
# 'Pb2+': {'a': [32.93500051525247, 19.638466187180484, 19.670894527123075, 7.731215820853677], 'b': [1.0162016172183594, 0.07161789998685739, 6.295572440886542, 22.640174127708054]} ,
# 'Pb4+': {'a': [33.15434014052007, 18.699389821007607, 18.91250705963444, 7.228494495700378], 'b': [0.978721615467574, 6.02556898997778, 0.06355101777568761, 14.531025667313903]} ,
# 'Bi': {'a': [20.108428670965036, 33.1864353136083, 21.86071149475039, 7.76179334471166], 'b': [0.07546264862796209, 1.0045666296149989, 6.6973553176684595, 41.42170101496954]} ,
# 'Bi3+': {'a': [33.03144934350525, 19.91011388682681, 19.26336706470245, 7.777816704742979], 'b': [0.9553164086496952, 5.865477960917584, 0.06606600423084817, 19.465446153501645]} ,
# 'Bi5+': {'a': [38.56224766071817, 17.90764930397914, 17.54880535052353, 4.054682911169283], 'b': [0.8841513530149444, 0.884151603631334, 8.879770497948725, 8.879755922838646]} ,
# 'Po': {'a': [19.695007042116135, 33.22276795369767, 22.20173653990443, 8.794288559272518], 'b': [0.06958364126947765, 0.9418625033188421, 6.164271571511025, 39.40071699385926]} ,
# 'At': {'a': [19.16423321298916, 33.291155302646416, 22.46895494087548, 9.98745741435394], 'b': [0.06274368695582329, 0.8789621140916205, 5.637753480578911, 36.66664923214832]} ,
# 'Rn': {'a': [18.754345686501075, 33.271002751555265, 22.650443978694845, 11.23633379645956], 'b': [0.057505094080891946, 0.8247619003757612, 5.1677870417669185, 33.96657782272902]} ,
# 'Fr': {'a': [19.28790956179968, 33.01918267737003, 22.902568536286957, 11.396878989360188], 'b': [0.0616204382525962, 0.8118414777229082, 5.0424520127681305, 33.67248609497652]} ,
# 'Ra': {'a': [24.986132865820668, 30.388539922375575, 21.964797917314737, 10.203674190698699], 'b': [0.11209111951722556, 1.009302659555598, 5.835806708395191, 40.49000531505381]} ,
# 'Ra2+': {'a': [18.48844461919266, 32.7110412721449, 22.31291697820626, 12.438512010815295], 'b': [0.05305687633790193, 0.7422699954496351, 4.301707675555792, 23.70544916780889]} ,
# 'Ac': {'a': [28.355287347669762, 29.28634105022964, 21.09666305091714, 9.896091927110797], 'b': [0.13699867303100935, 1.1555443377007573, 6.385817774674845, 46.128050447013514]} ,
# 'Ac3+': {'a': [17.802427090481203, 32.80896883222214, 22.329184803708046, 13.024101142552796], 'b': [0.04590188029667876, 0.6897781477453503, 3.917697682196261, 20.465239179115848]} ,
# 'Th': {'a': [30.31923668038516, 28.760420130707484, 20.27949720869558, 10.33831834413739], 'b': [0.14883811425901744, 1.2394619079536477, 6.5561713897928655, 46.96264095639936]} ,
# 'Th4+': {'a': [17.154848803978847, 32.90677139585852, 22.359047163324494, 13.553131600457675], 'b': [0.039421902613470346, 0.6429131597277128, 3.5897252648035534, 17.969670421054715]} ,
# 'Pa': {'a': [31.98559639043836, 28.4423399146534, 20.71620659165121, 9.545490786331202], 'b': [0.1578037399796097, 1.3099957173813332, 7.105987017313643, 45.879659347159325]} ,
# 'U': {'a': [33.575876599037315, 28.228143932558112, 20.90732859276934, 9.01539381334477], 'b': [0.1654099726295301, 1.3883322841092274, 7.569607713853529, 47.12530552317038]} ,
# 'U3+': {'a': [21.064605925491744, 30.796391077048234, 22.27098229979784, 14.822053635753363], 'b': [0.07169427403836304, 0.6973218090265524, 3.760209909702603, 18.186092789081602]} ,
# 'U4+': {'a': [19.27518181139392, 31.617259590052946, 22.242947986659726, 14.833695714342834], 'b': [0.05713495364250147, 0.6463505008785546, 3.4955826296183634, 16.553145923765836]} ,
# 'U6+': {'a': [16.064135720083556, 32.99957603685267, 22.426766491355043, 14.494111786121831], 'b': [0.02926772281235677, 0.5643782057695792, 3.0568808152039875, 14.353416117307892]} ,
# 'Np': {'a': [34.96395456347403, 28.059406891536742, 21.33019800856106, 8.40491893349928], 'b': [0.17118732377997417, 1.4576859429636262, 7.984749092453818, 49.05310451045045]} ,
# 'Np3+': {'a': [22.510938126602294, 29.86908562317578, 22.293189239310447, 15.279860898064282], 'b': [0.08143987133147122, 0.7097336354849217, 3.7471497837775773, 17.477888339718973]} ,
# 'Np4+': {'a': [20.66905288838749, 30.732798892372262, 22.215510158412624, 15.35022088983083], 'b': [0.06736773652419717, 0.6561039968981242, 3.4798103278950174, 15.918644872765011]} ,
# 'Np6+': {'a': [17.202396833738216, 32.329357347670424, 22.293449627695413, 15.158099830472246], 'b': [0.03924770045847983, 0.5680053456966027, 3.029706677172008, 13.762233936591125]} ,
# 'Pu': {'a': [35.99194908162117, 27.617022217554332, 22.500447208120534, 7.632537371711295], 'b': [0.1740140868002542, 1.4857096084384125, 8.14885843937281, 47.37530288900378]} ,
# 'Pu3+': {'a': [24.043505351696517, 28.873011993966305, 22.378014216259572, 15.659083466456156], 'b': [0.09083293397429283, 0.7254188698635026, 3.7460257392320306, 16.832313898090675]} ,
# 'Pu4+': {'a': [22.189388168486335, 29.742292201419804, 22.239588797998703, 15.79574556171176], 'b': [0.07750351606462184, 0.6692870487191902, 3.4755933447800462, 15.329977212904776]} ,
# 'Pu6+': {'a': [18.65747781159686, 31.414218927040633, 22.16151606629879, 15.748768950058261], 'b': [0.05079413105421273, 0.5777744375181861, 3.0193432557750395, 13.21554201291412]} ,
# 'Am': {'a': [37.130113384584625, 27.509660830943172, 23.1119489883673, 7.023638280462391], 'b': [0.17726712217475304, 1.54389271801707, 8.414666503784138, 49.985384082630276]} ,
# 'Cm': {'a': [38.02710267378455, 27.57881656487396, 23.15680820572782, 7.064756670009228], 'b': [0.17851105618357427, 1.5884470495492526, 8.46612722887841, 53.38527579039251]} ,
# 'Bk': {'a': [38.846839479387704, 27.459132272908132, 23.769257332996418, 6.767054483757886], 'b': [0.17916006545142038, 1.6194479696021395, 8.457528306298578, 54.31449974191302]} ,
# 'Cf': {'a': [39.60540285385262, 27.37601578275248, 24.352702792083782, 6.519981168216521], 'b': [0.1792753694574756, 1.6480625541705138, 8.39892193233561, 54.99145248549249]} ,
# }



