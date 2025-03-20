#!/usr/bin/env python
#
#    core.py
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

# load some dictionaries
from denss.resources import resources

electrons = resources.electrons
atomic_volumes = resources.atomic_volumes
numH = resources.numH
volH = resources.volH
vdW = resources.vdW
radii_sf_dict = resources.radii_sf_dict
ffcoeff = resources.ffcoeff

# for implicit hydrogens, from distribution of corrected unique volumes
implicit_H_radius = 0.826377

try:
    import numba as nb

    HAS_NUMBA = True
    # suppress some unnecessary deprecation warnings
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
except:
    HAS_NUMBA = False
HAS_NUMBA = False
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

# disable pyfftw until we can make it more stable
# it works, but often results in nans randomly
PYFFTW = False


def myfftn(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.fft.fftn(x)
    else:
        if PYFFTW:
            return pyfftw.interfaces.numpy_fft.fftn(x)
        else:
            try:
                # try running the parallelized version of scipy fft
                return fft.fftn(x, workers=-1)
            except:
                # fall back to numpy
                return np.fft.fftn(x)


def myrfftn(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.fft.rfftn(x)
    else:
        if PYFFTW:
            return pyfftw.interfaces.numpy_fft.rfftn(x)
        else:
            try:
                # try running the parallelized version of scipy fft
                return fft.rfftn(x, workers=-1)
            except:
                # fall back to numpy
                return np.fft.rfftn(x)


def myifftn(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.fft.ifftn(x)
    else:
        if PYFFTW:
            return pyfftw.interfaces.numpy_fft.ifftn(x)
        else:
            try:
                # try running the parallelized version of scipy fft
                return fft.ifftn(x, workers=-1)
            except:
                # fall back to numpy
                return np.fft.ifftn(x)


def myirfftn(x, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.fft.irfftn(x)
    else:
        if PYFFTW:
            return pyfftw.interfaces.numpy_fft.irfftn(x)
        else:
            try:
                # try running the parallelized version of scipy fft
                return fft.irfftn(x, workers=-1)
            except:
                # fall back to numpy
                return np.fft.irfftn(x)


def myabs(x, out=None, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.abs(x, out=out)
    else:
        return np.abs(x, out=out)


# @numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    # a faster way to calculate abs(x)**2, for calculating intensities
    re2 = (x.real) ** 2
    im2 = (x.imag) ** 2
    _abs2 = re2 + im2
    return _abs2


def abs2_fast(x, out=None):
    if out is None:
        out = np.empty_like(x, dtype=np.float64)

    # Use the original abs2 implementation
    np.add((x.real) ** 2, (x.imag) ** 2, out=out)
    return out

def mybinmean_optimized(xravel, binsravel, xcount=None, out=None):
    # Calculate sums binned by bin index
    xsum = np.bincount(binsravel, xravel)

    # Use provided bin counts or calculate them
    if xcount is None:
        xcount = np.bincount(binsravel)

    # Use provided output array or create new one
    if out is None:
        out = np.empty_like(xsum, dtype=float)

    # Compute means in-place
    np.divide(xsum, xcount, out=out)
    return out


def mybinmean(xravel, binsravel, xcount=None, DENSS_GPU=False):
    if DENSS_GPU:
        xsum = cp.bincount(binsravel, xravel)
        if xcount is None:
            xcount = cp.bincount(binsravel)
        return xsum / xcount
    else:
        xsum = np.bincount(binsravel, xravel)
        if xcount is None:
            xcount = np.bincount(binsravel)
        return xsum / xcount


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


def mysum(x, out=None, DENSS_GPU=False):
    if DENSS_GPU:
        return cp.sum(x, out=out)
    else:
        return np.sum(x, out=out)


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


def rho2rg(rho, side=None, r=None, support=None, dx=None):
    """Calculate radius of gyration from an electron density map."""
    if side is None and r is None:
        print("Error: To calculate Rg, must provide either side or r parameters.")
        sys.exit()
    if side is not None and r is None:
        n = rho.shape[0]
        x_ = np.linspace(-side / 2., side / 2., n)
        x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if support is None:
        support = np.ones_like(rho, dtype=bool)
    if dx is None:
        print("Error: To calculate Rg, must provide dx")
        sys.exit()
    gridcenter = grid_center(rho)  # (np.array(rho.shape)-1.)/2.
    com = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    rhocom = (gridcenter - com) * dx
    rg2 = np.sum(r[support] ** 2 * rho[support]) / np.sum(rho[support])
    rg2 = rg2 - np.linalg.norm(rhocom) ** 2
    rg = np.sign(rg2) * np.abs(rg2) ** 0.5
    return rg


def write_mrc(rho, side, filename="map.mrc"):
    """Write an MRC formatted electron density map.
       See here: http://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/#image
    """
    xs, ys, zs = rho.shape
    nxstart = -xs // 2
    nystart = -ys // 2
    nzstart = -zs // 2
    side = np.atleast_1d(side)
    if len(side) == 1:
        a, b, c = side, side, side
    elif len(side) == 3:
        a, b, c = side
    else:
        print("Error. Argument 'side' must be float or 3-tuple")
        return None
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
        fout.write(struct.pack('<' + 'f' * 12, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        for i in range(0, 12):
            fout.write(struct.pack('<f', 0.0))

        # CCP4 format does not use the ORIGIN header records.
        # To keep map on correct origin in programs such as COOT, use NCSTART, etc header fields
        # XORIGIN, YORIGIN, ZORIGIN
        fout.write(struct.pack('<fff', 0., 0., 0.))  # nxstart*(a/xs), nystart*(b/ys), nzstart*(c/zs) ))
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


def read_mrc(filename="map.mrc", returnABC=False, float64=True):
    """
        See MRC format at http://bio3d.colorado.edu/imod/doc/mrc_format.txt for offsets
    """
    with open(filename, 'rb') as fin:
        MRCdata = fin.read()
        nx = struct.unpack_from('<i', MRCdata, 0)[0]
        ny = struct.unpack_from('<i', MRCdata, 4)[0]
        nz = struct.unpack_from('<i', MRCdata, 8)[0]

        # side = struct.unpack_from('<f',MRCdata,40)[0]
        a, b, c = struct.unpack_from('<fff', MRCdata, 40)
        side = a

        # header is 1024 bytes long. To read data, skip ahead to that point in the file
        fin.seek(1024, os.SEEK_SET)
        rho = np.fromfile(file=fin, dtype=np.dtype(np.float32)).reshape((nx, ny, nz), order='F')
        fin.close()
    if float64:
        rho = rho.astype(np.float64)
    if returnABC:
        return rho, (a, b, c)
    else:
        return rho, side


def write_xplor(rho, side, filename="map.xplor"):
    """Write an XPLOR formatted electron density map."""
    xs, ys, zs = rho.shape
    title_lines = ['REMARK FILENAME="' + filename + '"', 'REMARK DATE= ' + str(datetime.datetime.today())]
    with open(filename, 'w') as f:
        f.write("\n")
        f.write("%8d !NTITLE\n" % len(title_lines))
        for line in title_lines:
            f.write("%-264s\n" % line)
        # f.write("%8d%8d%8d%8d%8d%8d%8d%8d%8d\n" % (xs,0,xs-1,ys,0,ys-1,zs,0,zs-1))
        f.write("%8d%8d%8d%8d%8d%8d%8d%8d%8d\n" % (
        xs, -xs / 2 + 1, xs / 2, ys, -ys / 2 + 1, ys / 2, zs, -zs / 2 + 1, zs / 2))
        f.write("% -.5E% -.5E% -.5E% -.5E% -.5E% -.5E\n" % (side, side, side, 90, 90, 90))
        f.write("ZYX\n")
        for k in range(zs):
            f.write("%8s\n" % k)
            for j in range(ys):
                for i in range(xs):
                    if (i + j * ys) % 6 == 5:
                        f.write("% -.5E\n" % rho[i, j, k])
                    else:
                        f.write("% -.5E" % rho[i, j, k])
            f.write("\n")
        f.write("    -9999\n")
        f.write("  %.4E  %.4E" % (np.average(rho), np.std(rho)))


def pad_rho(rho, newshape):
    """Pad rho with zeros to achieve new shape"""
    a = rho
    a_nx, a_ny, a_nz = a.shape
    b_nx, b_ny, b_nz = newshape
    padx1 = (b_nx - a_nx) // 2
    padx2 = (b_nx - a_nx) - padx1
    pady1 = (b_ny - a_ny) // 2
    pady2 = (b_ny - a_ny) - pady1
    padz1 = (b_nz - a_nz) // 2
    padz2 = (b_nz - a_nz) - padz1
    # np.pad cannot take negative values, i.e. where the array will be cropped
    # however, can instead just use slicing to do the equivalent
    # but first need to identify which pad values are negative
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
    a = np.pad(a, ((padx1, padx2), (pady1, pady2), (padz1, padz2)), 'constant')[
        slcx1:slcx2, slcy1:slcy2, slcz1:slcz2]
    return a


def zoom_rho(rho, vx, dx):
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
    # zoom factors
    zx, zy, zz = vx / dx, vy / dy, vz / dz
    newrho = ndimage.zoom(rho, (zx, zy, zz), order=1, mode="wrap")

    return newrho


def _fit_by_least_squares(radial, vectors, nmin=None, nmax=None):
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
    return np.convolve(x, np.ones(N) / N, mode='same')


def loadOutFile(filename):
    """Loads a GNOM .out file and returns q, Ireg, sqrt(Ireg), and a
    dictionary of miscellaneous results from GNOM. Taken from the BioXTAS
    RAW software package, used with permission under the GPL license."""

    five_col_fit = re.compile(
        r'\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*$')
    three_col_fit = re.compile(r'\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*$')
    two_col_fit = re.compile(r'\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s*$')

    results_fit = re.compile(
        r'\s*Current\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s+\d*[.]\d*[+eE-]*\d*\s*\d*[.]?\d*[+eE-]*\d*\s*$')

    te_fit = re.compile(r'\s*Total\s+[Ee]stimate\s*:\s+\d*[.]\d+\s*\(?[A-Za-z\s]+\)?\s*$')
    te_num_fit = re.compile(r'\d*[.]\d+')
    te_quality_fit = re.compile(r'[Aa][A-Za-z\s]+\)?\s*$')

    p_rg_fit = re.compile(r'\s*Real\s+space\s*\:?\s*Rg\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*\+-\s*\d*[.]\d+[+eE-]*\d*')
    q_rg_fit = re.compile(r'\s*Reciprocal\s+space\s*\:?\s*Rg\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*')

    p_i0_fit = re.compile(
        r'\s*Real\s+space\s*\:?[A-Za-z0-9\s\.,+-=]*\(0\)\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*\+-\s*\d*[.]\d+[+eE-]*\d*')
    q_i0_fit = re.compile(r'\s*Reciprocal\s+space\s*\:?[A-Za-z0-9\s\.,+-=]*\(0\)\:?\s*\=?\s*\d*[.]\d+[+eE-]*\d*\s*')

    qfull = []
    qshort = []
    Jexp = []
    Jerr = []
    Jreg = []
    Ireg = []

    R = []
    P = []
    Perr = []

    outfile = []

    # In case it returns NaN for either value, and they don't get picked up in the regular expression
    q_rg = None  # Reciprocal space Rg
    q_i0 = None  # Reciprocal space I0

    with open(filename, errors='ignore') as f:
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
                # print line
                found = threecol_match.group().split()

                R.append(float(found[0]))
                P.append(float(found[1]))
                Perr.append(float(found[2]))

            elif fivecol_match:
                # print line
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

    chisq = np.sum(np.square(np.array(Jexp) - np.array(Jreg)) / np.square(Jerr)) / (
            len(Jexp) - 1)  # DOF normalied chi squared

    results = {'dmax': R[-1],  # Dmax
               'TE': TE_out,  # Total estimate
               'rg': rg,  # Real space Rg
               'rger': rger,  # Real space rg error
               'i0': i0,  # Real space I0
               'i0er': i0er,  # Real space I0 error
               'q_rg': q_rg,  # Reciprocal space Rg
               'q_i0': q_i0,  # Reciprocal space I0
               'quality': quality,  # Quality of GNOM out file
               'discrp': Actual_DISCRP,
               # DISCRIP, kind of chi squared (normalized by number of points, with a regularization parameter thing thrown in)
               'oscil': Actual_OSCILL,  # Oscillation of solution
               'stabil': Actual_STABIL,  # Stability of solution
               'sysdev': Actual_SYSDEV,  # Systematic deviation of solution
               'positv': Actual_POSITV,  # Relative norm of the positive part of P(r)
               'valcen': Actual_VALCEN,  # Validity of the chosen interval in real space
               'smooth': Actual_SMOOTH,
               # Smoothness of the chosen interval? -1 indicates no real value, for versions of GNOM < 5.0 (ATSAS <2.8)
               'filename': name,  # GNOM filename
               'algorithm': 'GNOM',  # Lets us know what algorithm was used to find the IFT
               'chisq': chisq  # Actual chi squared value
               }

    # Jreg and Jerr are the raw data on the qfull axis
    Jerr = np.array(Jerr)
    prepend = np.zeros((len(Ireg) - len(Jerr)))
    prepend += np.mean(Jerr[:10])
    Jerr = np.concatenate((prepend, Jerr))
    Jreg = np.array(Jreg)
    Jreg = np.concatenate((prepend * 0, Jreg))
    Jexp = np.array(Jexp)
    Jexp = np.concatenate((prepend * 0, Jexp))

    return np.array(qfull), Jexp, Jerr, np.array(Ireg), results


def loadDatFile(filename):
    ''' Loads a Primus .dat format file. Taken from the BioXTAS RAW software package,
    used with permission under the GPL license.'''

    iq_pattern = re.compile(r'\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*')

    i = []
    q = []
    err = []

    with open(filename, errors='ignore') as f:
        lines = f.readlines()

    comment = ''
    line = lines[0]
    j = 0
    while line.split() and line.split()[0].strip()[0] == '#':
        comment = comment + line
        j = j + 1
        line = lines[j]

    fileHeader = {'comment': comment}
    parameters = {'filename': os.path.split(filename)[1],
                  'counters': fileHeader}

    if comment.find('model_intensity') > -1:
        # FoXS file with a fit! has four data columns
        is_foxs_fit = True
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

    # Check to see if there is any header from RAW, and if so get that.
    header = []
    for j in range(len(lines)):
        if '### HEADER:' in lines[j]:
            header = lines[j + 1:]

    hdict = None
    results = {}

    if len(header) > 0:
        hdr_str = ''
        for each_line in header:
            hdr_str = hdr_str + each_line
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

    iq_pattern = re.compile(r'\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*')

    i = []
    q = []
    err = []

    with open(filename, errors='ignore') as f:
        lines = f.readlines()

    comment = ''
    line = lines[0]
    j = 0
    while line.split() and line.split()[0].strip()[0] == '#':
        comment = comment + line
        j = j + 1
        line = lines[j]

    fileHeader = {'comment': comment}
    parameters = {'filename': os.path.split(filename)[1],
                  'counters': fileHeader}

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

    # grab some header info if available
    header = []
    for j in range(len(lines)):
        # If this is a _fit.dat or .fit file from DENSS, grab the header values beginning with hashtag #.
        if '# Parameter Values:' in lines[j]:
            header = lines[j + 1:j + 9]

    hdict = None
    results = {}

    if len(header) > 0:
        hdr_str = '{'
        for each_line in header:
            line = each_line.split()
            hdr_str = hdr_str + "\"" + line[1] + "\"" + ":" + line[3] + ","
        hdr_str = hdr_str.rstrip(',') + "}"
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

    iq_pattern = re.compile(r'\s*\d*[.]\d*[+eE-]*\d+\s+-?\d*[.]\d*[+eE-]*\d+\s+\d*[.]\d*[+eE-]*\d+\s*')

    i = []
    q = []
    err = []

    with open(filename, errors='ignore') as f:
        lines = f.readlines()

    comment = ''
    line = lines[0]
    j = 0
    while line.split() and line.split()[0].strip()[0] == '#':
        comment = comment + line
        j = j + 1
        line = lines[j]

    fileHeader = {'comment': comment}
    parameters = {'filename': os.path.split(filename)[1],
                  'counters': fileHeader}

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

    # If this is a _fit.dat file from DENSS, grab the header values.
    header = []
    for j in range(len(lines)):
        if '# Parameter Values:' in lines[j]:
            header = lines[j + 1:j + 9]

    hdict = None
    results = {}

    if len(header) > 0:
        hdr_str = '{'
        for each_line in header:
            line = each_line.split()
            hdr_str = hdr_str + "\"" + line[1] + "\"" + ":" + line[3] + ","
        hdr_str = hdr_str.rstrip(',') + "}"
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

    # just return i for ifit to be consistent with other functions' output
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
        # Ifit here is just I, since it's just a data file
        q, I, Ierr, Ifit, results = loadDatFile(fname)
        isfit = False

    # keys = {key.lower().strip().translate(str.maketrans('','', '_ ')): key for key in list(results.keys())}
    keys = {key.lower().strip(): key for key in list(results.keys())}

    if 'dmax' in keys:
        dmax = float(results[keys['dmax']])
    else:
        dmax = -1.

    if units == "nm":
        # DENSS assumes 1/angstrom, so convert from 1/nm to 1/angstrom
        q /= 10
        if dmax != -1:
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
    # first, check if there is a q=0 value.
    if min(Iq[:, 0]) > 1e-8:
        # allow for some floating point error
        return True
    else:
        return False


def clean_up_data(Iq):
    """Do a quick cleanup by removing zero intensities and zero errors.

    Iq - N x 3 numpy array, where N is the number of data  points, and the
    three columns are q, I, error.
    """
    return Iq[(~np.isclose(Iq[:, 1], 0)) & (~np.isclose(Iq[:, 2], 0))]


def calc_rg_I0_by_guinier(Iq, nb=None, ne=None):
    """calculate Rg, I(0) by fitting Guinier equation to data.
    Use only desired q range in input arrays."""
    if nb is None:
        nb = 0
    if ne is None:
        ne = Iq.shape[0]
    while True:
        m, b = stats.linregress(Iq[nb:ne, 0] ** 2, np.log(Iq[nb:ne, 1]))[:2]
        if m < 0.0:
            break
        else:
            # the slope should be negative
            # if the slope is positive, shift the
            # region forward by one point and try again
            nb += 1
            ne += 1
            if nb > 50:
                raise ValueError("Guinier estimation failed. Guinier region slope is positive.")
    rg = (-3 * m) ** (0.5)
    I0 = np.exp(b)
    return rg, I0


def calc_rg_by_guinier_first_2_points(q, I, DENSS_GPU=False):
    """calculate Rg using Guinier law, but only use the
    first two data points. This is meant to be used with a
    calculated scattering profile, such as Imean from denss()."""
    m = (mylog(I[1], DENSS_GPU) - mylog(I[0], DENSS_GPU)) / (q[1] ** 2 - q[0] ** 2)
    rg = (-3 * m) ** (0.5)
    return rg


def calc_rg_by_guinier_peak(Iq, exp=1, nb=0, ne=None):
    """roughly estimate Rg using the Guinier peak method.
    Use only desired q range in input arrays.
    exp - the exponent in q^exp * I(q)"""
    d = exp
    if ne is None:
        ne = Iq.shape[0]
    q = Iq[:, 0]  # [nb:ne,0]
    I = Iq[:, 1]  # [nb:ne,1]
    qdI = q ** d * I
    try:
        # fit a quick quadratic for smoothness, ax^2 + bx + c
        a, b, c = np.polyfit(q, qdI, 2)
        # get the peak position
        qpeak = -b / (2 * a)
    except:
        # if polyfit fails, just grab the maximum position
        qpeaki = np.argmax(qdI)
        qpeak = q[qpeaki]
    # calculate Rg from the peak position
    rg = (3. * d / 2.) ** 0.5 / qpeak
    return rg


def direct_I2P(q, I, D=None):
    # the maximum possible distance is pi/dq
    # use 0.5 just to be sure theres at minimum 2 points per shannon channel
    if D is None:
        rmax = 0.5 * np.pi / (q[2] - q[1])
    else:
        rmax = D
    # r = np.logspace(0.001,np.log10(rmax),500) - 1
    r = np.linspace(0, rmax, 1000)
    P = np.zeros(len(r))
    for ri in range(len(r)):
        qrsinqr = q * r[ri] * np.sin(q * r[ri])
        P[ri] += np.trapz(qrsinqr * I, q)
    return r, 1 / (2 * np.pi ** 2) * P


def P2Rg(r, P):
    num = np.trapz(r ** 2 * P, r)
    denom = 2 * np.trapz(P, r)
    rg2 = num / denom
    return rg2 ** 0.5


def estimate_dmax(Iq, dmax=None, clean_up=True):
    """Attempt to roughly estimate Dmax directly from data."""
    # first, clean up the data
    if clean_up:
        Iq = clean_up_data(Iq)
    q = Iq[:, 0]
    I = Iq[:, 1]
    nq = len(q)
    if dmax is None:
        # first, estimate a very rough rg from the first 20 data points
        nmax = 20
        try:
            rg, I0 = calc_rg_I0_by_guinier(Iq, ne=nmax)
        except:
            rg = calc_rg_by_guinier_peak(Iq, exp=1, ne=100)
        # next, dmax is roughly 3.5*rg for most particles
        # so calculate P(r) using a larger dmax, say twice as large, so 7*rg
        D = 7 * rg
    else:
        # allow user to give an initial estimate of Dmax
        # multiply by 2 to allow for enough large r values
        D = 2 * dmax
    # create a calculated q range for Sasrec for low q out to q=0
    qmin = np.min(q)
    dq = (q.max() - q.min()) / (q.size - 1)
    nqc = int(qmin / dq)
    qc = np.concatenate(([0.0], np.arange(nqc) * dq + (qmin - nqc * dq), q))
    # run Sasrec to perform IFT
    sasrec = Sasrec(Iq[:nq // 2], D, qc=None, alpha=0.0, extrapolate=False)
    # if the rg estimate was way off, it would screw up Dmax estimate
    # but the sasrec rg should be more accurate, even with a screwed up guinier estimate
    # so run it again, but this time with the Dmax = 7*sasrec.rg
    # only do this if rg is significantly different
    if np.abs(sasrec.rg - rg) > 0.2 * sasrec.rg:
        sasrec = Sasrec(Iq[:nq // 2], D=7 * sasrec.rg, qc=None, alpha=0.0, extrapolate=False)
    # lets test a bunch of different dmax's on a logarithmic spacing
    # then see where chi2 is minimal. that at least gives us a good ball park of Dmax
    # the main problem is that we don't know the scale even remotely, or the units,
    # so we need to check many orders of magnitude
    Ds = np.logspace(.1, np.log10(2 * 7 * rg), 10)
    chi2 = np.zeros(len(Ds))
    for i in range(len(Ds)):
        sasrec = Sasrec(Iq[:nq // 2], D=Ds[i], qc=None, alpha=0.0, extrapolate=False)
        chi2[i] = sasrec.calc_chi2()
    order = np.argsort(chi2)
    D = 2 * np.interp(2 * chi2.min(), chi2[order], Ds[order])
    # one final time with new D and full q range
    sasrec = Sasrec(Iq, D=D, qc=None, alpha=0.0, extrapolate=False)
    # now filter the P(r) curve for estimating Dmax better
    qmax = 2 * np.pi / D
    # qmax_fraction = 0.5
    r, Pfilt, sigrfilt = filter_P(sasrec.r, sasrec.P, sasrec.Perr, qmax=qmax)  # qmax_fraction*Iq[:,0].max())
    # import matplotlib.pyplot as plt
    # plt.plot(sasrec.r,sasrec.r*0,'k--')
    # plt.plot(sasrec.r, sasrec.P,'b-')
    # plt.plot(r,Pfilt,'r-')
    # estimate D as the first position where P becomes less than 0.01*P.max(), after P.max()
    Pargmax = Pfilt.argmax()
    # catch cases where the P(r) plot goes largely negative at large r values,
    # as this indicates repulsion. Set the new Pargmax, which is really just an
    # identifier for where to begin searching for Dmax, to be any P value whose
    # absolute value is greater than at least 10% of Pfilt.max. The large 10% is to
    # avoid issues with oscillations in P(r).
    argmax_threshold = 0.05
    above_idx = np.where((np.abs(Pfilt) > argmax_threshold * Pfilt.max()) & (r > r[Pargmax]))
    Pargmax = np.max(above_idx)
    dmax_threshold = (0.01 * Pfilt.max())
    near_zero_idx = np.where((np.abs(Pfilt[Pargmax:]) < dmax_threshold))[0]
    near_zero_idx += Pargmax
    D_idx = near_zero_idx[0]
    D = r[D_idx]
    sasrec.D = np.copy(D)
    # plt.plot(sasrec.r,sasrec.r*0+argmax_threshold*Pfilt.max(),'g--')
    # plt.plot(sasrec.r,sasrec.r*0-argmax_threshold*Pfilt.max(),'g--')
    # plt.plot(sasrec.r,sasrec.r*0+dmax_threshold,'r--')
    # plt.axvline(D,c='r')
    # plt.plot()
    # plt.show()
    # exit()
    sasrec.update()
    return D, sasrec


def filter_P(r, P, sigr=None, qmax=0.5, cutoff=0.75, qmin=0.0, cutoffmin=1.25):
    """Filter P(r) and sigr of oscillations."""
    npts = len(r)
    dr = (r.max() - r.min()) / (r.size - 1)
    fs = 1. / dr
    nyq = fs * 0.5
    fc = (cutoff * qmax / (2 * np.pi)) / nyq
    ntaps = npts // 3
    if ntaps % 2 == 0:
        ntaps -= 1
    b = signal.firwin(ntaps, fc, window='hann')
    if qmin > 0.0:
        fcmin = (cutoffmin * qmin / (2 * np.pi)) / nyq
        b = signal.firwin(ntaps, [fcmin, fc], pass_zero=False, window='hann')
    a = np.array([1])
    import warnings
    with warnings.catch_warnings():
        # theres a warning from filtfilt that is a bug in older scipy versions
        # we are just going to suppress that here.
        warnings.filterwarnings("ignore")
        Pfilt = signal.filtfilt(tuple(b), tuple(a), tuple(P), padlen=len(r) - 1)
        r = np.arange(npts) / fs
        if sigr is not None:
            sigrfilt = signal.filtfilt(b, a, sigr, padlen=len(r) - 1) / (2 * np.pi)
            return r, Pfilt, sigrfilt
        else:
            return r, Pfilt


def grid_center(rho):
    return np.array(rho.shape) // 2


def reconstruct_abinitio_from_scattering_profile(q, I, sigq, dmax, qraw=None, Iraw=None, sigqraw=None,
                                                 ne=None, voxel=5., oversampling=3., recenter=True, recenter_steps=None,
                                                 recenter_mode="com", positivity=True, positivity_steps=None, extrapolate=True, output="map",
                                                 steps=None, seed=None, rho_start=None, support_start=None, add_noise=None,
                                                 shrinkwrap=True, shrinkwrap_old_method=False, shrinkwrap_sigma_start=3,
                                                 shrinkwrap_sigma_end=1.5, shrinkwrap_sigma_decay=0.99, shrinkwrap_threshold_fraction=0.2,
                                                 shrinkwrap_iter=20, shrinkwrap_minstep=100, chi_end_fraction=0.01,
                                                 write_xplor_format=False, write_freq=100, enforce_connectivity=True,
                                                 enforce_connectivity_steps=[500], enforce_connectivity_max_features=1, cutout=True, quiet=False, ncs=0,
                                                 ncs_steps=[500], ncs_axis=1, ncs_type="cyclical", abort_event=None, my_logger=logging.getLogger(),
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

    # Initialize variables

    side = oversampling * D
    halfside = side / 2

    n = int(side / voxel)
    # want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n % 2 == 1:
        n += 1
    # store n for later use if needed
    nbox = n

    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = np.linspace(-(n // 2) * dx, (n // 2 - 1) * dx, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    df = 1 / side
    qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
    qz_ = np.fft.rfftfreq(x_.size) * n * df * 2 * np.pi
    # qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qx, qy, qz = np.meshgrid(qx_, qx_, qz_, indexing='ij')
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0]) - 1e-8  # subtract a tiny bit to deal with floating point error
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)

    # create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    qbl = qbin_labels
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)

    # calculate qbinsc as average of q values in shell
    qbinsc = mybinmean(qr.ravel(), qblravel, xcount)

    # allow for any range of q data
    qdata = qbinsc[np.where((qbinsc >= q.min()) & (qbinsc <= q.max()))]
    Idata = np.interp(qdata, q, I)

    if extrapolate:
        qextend = qbinsc[qbinsc >= qdata.max()]
        Iextend = qextend ** -4
        Iextend = Iextend / Iextend[0] * Idata[-1]
        qdata = np.concatenate((qdata, qextend[1:]))
        Idata = np.concatenate((Idata, Iextend[1:]))

    # create list of qbin indices just in region of data for later F scaling
    qbin_args = np.in1d(qbinsc, qdata, assume_unique=True)
    qba = qbin_args  # just for brevity when using it later
    # set qba bins outside of scaling region to false.
    # start with bins in corners
    # qba[qbinsc>qx_.max()] = False

    sigqdata = np.interp(qdata, q, sigq)

    scale_factor = ne ** 2 / Idata[0]
    Idata *= scale_factor
    sigqdata *= scale_factor
    I *= scale_factor
    sigq *= scale_factor

    if steps == 'None' or steps is None or np.int(steps) < 1:
        stepsarr = np.concatenate((enforce_connectivity_steps, [shrinkwrap_minstep]))
        maxec = np.max(stepsarr)
        steps = int(shrinkwrap_iter * (
                    np.log(shrinkwrap_sigma_end / shrinkwrap_sigma_start) / np.log(shrinkwrap_sigma_decay)) + maxec)
        # add enough steps for convergence after shrinkwrap is finished
        # something like 7000 seems reasonable, likely will finish before that on its own
        # then just make a round number when using defaults
        steps += 7621
    else:
        steps = np.int(steps)

    Imean = np.zeros((len(qbins)))

    if qraw is None:
        qraw = q
    if Iraw is None:
        Iraw = I
    if sigqraw is None:
        sigqraw = sigq
    Iq_exp = np.vstack((qraw, Iraw, sigqraw)).T
    Iq_calc = np.vstack((qbinsc, Imean, Imean)).T
    idx = np.where(Iraw > 0)
    Iq_exp = Iq_exp[idx]
    qmax = np.min([Iq_exp[:, 0].max(), Iq_calc[:, 0].max()])
    Iq_exp = Iq_exp[Iq_exp[:, 0] <= qmax]
    Iq_calc = Iq_calc[Iq_calc[:, 0] <= qmax]

    chi = np.zeros((steps + 1))
    rg = np.zeros((steps + 1))
    supportV = np.zeros((steps + 1))
    if support_start is not None:
        support = np.copy(support_start)
    else:
        support = np.ones(x.shape, dtype=bool)

    if seed is None:
        # Have to reset the random seed to get a random in different from other processes
        prng = np.random.RandomState()
        seed = prng.randint(2 ** 31 - 1)
    else:
        seed = int(seed)

    prng = np.random.RandomState(seed)

    if rho_start is not None:
        rho = rho_start  # *dV
        if add_noise is not None:
            noise_factor = rho.max() * add_noise
            noise = prng.random_sample(size=x.shape) * noise_factor
            rho += noise
    else:
        rho = prng.random_sample(size=x.shape)  # - 0.5
    newrho = np.zeros_like(rho)

    sigma = shrinkwrap_sigma_start

    # calculate the starting shrinkwrap volume as the volume of a sphere
    # of radius Dmax, i.e. much larger than the particle size
    swbyvol = True
    swV = V / 2.0
    Vsphere_Dover2 = 4. / 3 * np.pi * (D / 2.) ** 3
    swVend = Vsphere_Dover2
    swV_decay = 0.9
    first_time_swdensity = True
    threshold = shrinkwrap_threshold_fraction
    # erode will make take five outer edge pixels of the support, like a shell,
    # and will make sure no negative density is in that region
    # this is to counter an artifact that occurs when allowing for negative density
    # as the negative density often appears immediately next to positive density
    # at the edges of the object. This ensures (i.e. biases) only real negative density
    # in the interior of the object (i.e. more than five pixels from the support boundary)
    # thus we only need this on when in membrane mode, i.e. when positivity=False
    if shrinkwrap_old_method or positivity:
        erode = False
    else:
        erode = True
        erosion_width = int(20 / dx)  # this is in pixels
        if erosion_width == 0:
            # make minimum of one pixel
            erosion_width = 1

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
    # my_logger.info('Positivity Steps: %s', positivity_steps)
    my_logger.info('Extrapolate high q: %s', extrapolate)
    my_logger.info('Shrinkwrap: %s', shrinkwrap)
    my_logger.info('Shrinkwrap Old Method: %s', shrinkwrap_old_method)
    my_logger.info('Shrinkwrap sigma start (angstroms): %s', shrinkwrap_sigma_start * dx)
    my_logger.info('Shrinkwrap sigma end (angstroms): %s', shrinkwrap_sigma_end * dx)
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
    my_logger.info('Reciprocal space box width (angstroms^(-1)): %3.3f', qx_.max() - qx_.min())
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
        qbinsc = cp.array(qbinsc)
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

        # F = myfftn(rho, DENSS_GPU=DENSS_GPU)
        F = myrfftn(rho, DENSS_GPU=DENSS_GPU)

        # sometimes, when using denss_refine.py with non-random starting rho,
        # the resulting Fs result in zeros in some locations and the algorithm to break
        # here just make those values to be 1e-16 to be non-zero
        F[np.abs(F) == 0] = 1e-16

        # APPLY RECIPROCAL SPACE RESTRAINTS
        # calculate spherical average of intensities from 3D Fs
        # I3D = myabs(F, DENSS_GPU=DENSS_GPU)**2
        I3D = abs2(F)
        Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=DENSS_GPU)

        # scale Fs to match data
        factors = mysqrt(Idata / Imean, DENSS_GPU=DENSS_GPU)
        # do not scale bins outside of desired range
        # so set those factors to 1.0
        factors[~qba] = 1.0
        F *= factors[qbin_labels]

        try:
            Iq_calc[:, 1] = Imean[qbinsc <= qmax]
            chi[j] = calc_chi2(Iq_exp, Iq_calc, scale=True, offset=False, interpolation=True, return_sf=False,
                               return_fit=False)
        except:
            # in case the interpolation fails for whatever reason, like the GPU status or something
            chi[j] = mysum(((Imean[qba] - Idata[qba]) / sigqdata[qba]) ** 2, DENSS_GPU=DENSS_GPU) / Idata[qba].size

        # APPLY REAL SPACE RESTRAINTS
        # rhoprime = myifftn(F, DENSS_GPU=DENSS_GPU).real
        rhoprime = myirfftn(F, DENSS_GPU=DENSS_GPU).real

        # use Guinier's law to approximate quickly
        rg[j] = calc_rg_by_guinier_first_2_points(qbinsc, Imean, DENSS_GPU=DENSS_GPU)

        # Error Reduction
        newrho *= 0
        newrho[support] = rhoprime[support]

        if not DENSS_GPU and j % write_freq == 0:
            if write_xplor_format:
                write_xplor(rhoprime / dV, side, fprefix + "_current.xplor")
            write_mrc(rhoprime / dV, side, fprefix + "_current.mrc")

        # enforce positivity by making all negative density points zero.
        if positivity:  # and j in positivity_steps:
            newrho[newrho < 0] = 0.0

        # apply non-crystallographic symmetry averaging
        if ncs != 0 and j in ncs_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
            newrho = align2xyz(newrho)
            if DENSS_GPU:
                newrho = cp.array(newrho)

        if ncs != 0 and j in [stepi + 1 for stepi in ncs_steps]:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
            if ncs_axis == 1:
                axes = (1, 2)  # longest
                axes2 = (0, 1)  # shortest
            if ncs_axis == 2:
                axes = (0, 2)  # middle
                axes2 = (0, 1)  # shortest
            if ncs_axis == 3:
                axes = (0, 1)  # shortest
                axes2 = (1, 2)  # longest
            degrees = 360. / ncs
            newrho_total = np.copy(newrho)
            if ncs_type == "dihedral":
                # first, rotate original about perpendicular axis by 180
                # then apply n-fold cyclical rotation
                d2fold = ndimage.rotate(newrho, 180, axes=axes2, reshape=False)
                newrhosym = np.copy(newrho) + d2fold
                newrhosym /= 2.0
                newrho_total = np.copy(newrhosym)
            else:
                newrhosym = np.copy(newrho)
            for nrot in range(1, ncs):
                sym = ndimage.rotate(newrhosym, degrees * nrot, axes=axes, reshape=False)
                newrho_total += np.copy(sym)
            newrho = newrho_total / ncs

            # run shrinkwrap after ncs averaging to get new support
            if shrinkwrap_old_method:
                # run the old method
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho, absv=absv, sigma=sigma, threshold=threshold,
                                                              recenter=recenter, recenter_mode=recenter_mode)
            else:
                swN = int(swV / dV)
                # end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho, absv=True, sigma=sigma, N=swN,
                                                                      recenter=recenter, recenter_mode=recenter_mode)
                    swV *= swV_decay
                else:
                    threshold = shrinkwrap_threshold_fraction
                    if first_time_swdensity:
                        if not quiet:
                            if gui:
                                my_logger.info("switched to shrinkwrap by density threshold = %.4f" % threshold)
                            else:
                                print("\nswitched to shrinkwrap by density threshold = %.4f" % threshold)
                        first_time_swdensity = False
                    newrho, support = shrinkwrap_by_density_value(newrho, absv=True, sigma=sigma, threshold=threshold,
                                                                  recenter=recenter, recenter_mode=recenter_mode)

            if DENSS_GPU:
                newrho = cp.array(newrho)

        if recenter and j in recenter_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            # cannot run center_rho_roll() function since we want to also recenter the support
            # perhaps we should fix this in the future to clean it up
            if recenter_mode == "max":
                rhocom = np.unravel_index(newrho.argmax(), newrho.shape)
            else:
                rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(newrho)))
            gridcenter = grid_center(newrho)  # (np.array(newrho.shape)-1.)/2.
            shift = gridcenter - rhocom
            shift = np.rint(shift).astype(int)
            newrho = np.roll(np.roll(np.roll(newrho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
            support = np.roll(np.roll(np.roll(support, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        # update support using shrinkwrap method
        if shrinkwrap and j >= shrinkwrap_minstep and j % shrinkwrap_iter == 1:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            if shrinkwrap_old_method:
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho, absv=absv, sigma=sigma, threshold=threshold,
                                                              recenter=recenter, recenter_mode=recenter_mode)
            else:
                swN = int(swV / dV)
                # end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho, absv=True, sigma=sigma, N=swN,
                                                                      recenter=recenter, recenter_mode=recenter_mode)
                    swV *= swV_decay
                else:
                    threshold = shrinkwrap_threshold_fraction
                    if first_time_swdensity:
                        if not quiet:
                            if gui:
                                my_logger.info("switched to shrinkwrap by density threshold = %.4f" % threshold)
                            else:
                                print("\nswitched to shrinkwrap by density threshold = %.4f" % threshold)
                        first_time_swdensity = False
                    newrho, support = shrinkwrap_by_density_value(newrho, absv=True, sigma=sigma, threshold=threshold,
                                                                  recenter=recenter, recenter_mode=recenter_mode)

            if sigma > shrinkwrap_sigma_end:
                sigma = shrinkwrap_sigma_decay * sigma

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        # run erode when shrinkwrap is run
        if erode and shrinkwrap and j > shrinkwrap_minstep and j % shrinkwrap_iter == 1:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            # eroded is the region of the support _not_ including the boundary pixels
            # so it is the entire interior. erode_region is _just_ the boundary pixels
            eroded = ndimage.binary_erosion(support, np.ones((erosion_width, erosion_width, erosion_width)))
            # get just boundary voxels, i.e. where support=True and eroded=False
            erode_region = np.logical_and(support, ~eroded)
            # set all negative density in boundary pixels to zero.
            newrho[(newrho < 0) & (erode_region)] = 0

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        if enforce_connectivity and j in enforce_connectivity_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)

            # first run shrinkwrap to define the features
            if shrinkwrap_old_method:
                # run the old method
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho, absv=absv, sigma=sigma, threshold=threshold,
                                                              recenter=recenter, recenter_mode=recenter_mode)
            else:
                # end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                swN = int(swV / dV)
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho, absv=True, sigma=sigma, N=swN,
                                                                      recenter=recenter, recenter_mode=recenter_mode)
                else:
                    newrho, support = shrinkwrap_by_density_value(newrho, absv=True, sigma=sigma, threshold=threshold,
                                                                  recenter=recenter, recenter_mode=recenter_mode)

            # label the support into separate segments based on a 3x3x3 grid
            struct = ndimage.generate_binary_structure(3, 3)
            labeled_support, num_features = ndimage.label(support, structure=struct)
            sums = np.zeros((num_features))
            num_features_to_keep = np.min([num_features, enforce_connectivity_max_features])
            if not quiet:
                if not gui:
                    print("EC: %d -> %d " % (num_features, num_features_to_keep))

            # find the feature with the greatest number of electrons
            for feature in range(num_features + 1):
                sums[feature - 1] = np.sum(newrho[labeled_support == feature])
            big_feature = np.argmax(sums) + 1
            # order the indices of the features in descending order based on their sum/total density
            sums_order = np.argsort(sums)[::-1]
            sums_sorted = sums[sums_order]
            # now grab the actual feature numbers (rather than the indices)
            features_sorted = sums_order + 1

            # remove features from the support that are not the primary feature
            # support[labeled_support != big_feature] = False
            # reset support to zeros everywhere
            # then progressively add in regions of support up to num_features_to_keep
            support *= False
            for feature in range(num_features_to_keep):
                support[labeled_support == features_sorted[feature]] = True

            # clean up density based on new support
            newrho[~support] = 0

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        supportV[j] = mysum(support, DENSS_GPU=DENSS_GPU) * dV

        if not quiet:
            if gui:
                my_logger.info("% 5i % 4.2e % 3.2f       % 5i          ", j, chi[j], rg[j], supportV[j])
            else:
                sys.stdout.write("\r% 5i  % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
                sys.stdout.flush()

        # occasionally report progress in logger
        if j % 500 == 0 and not gui:
            my_logger.info('Step % 5i: % 4.2e % 3.2f       % 5i          ', j, chi[j], rg[j], supportV[j])

        if j > 101 + shrinkwrap_minstep:
            if DENSS_GPU:
                lesser = mystd(chi[j - 100:j], DENSS_GPU=DENSS_GPU).get() < chi_end_fraction * mymean(chi[j - 100:j],
                                                                                                      DENSS_GPU=DENSS_GPU).get()
            else:
                lesser = mystd(chi[j - 100:j], DENSS_GPU=DENSS_GPU) < chi_end_fraction * mymean(chi[j - 100:j],
                                                                                                DENSS_GPU=DENSS_GPU)
            if lesser:
                break

        rho = newrho

    # convert back to numpy outside of for loop
    if DENSS_GPU:
        rho = cp.asnumpy(rho)
        qbin_labels = cp.asnumpy(qbin_labels)
        qbin_args = cp.asnumpy(qbin_args)
        sigqdata = cp.asnumpy(sigqdata)
        Imean = cp.asnumpy(Imean)
        chi = cp.asnumpy(chi)
        qbins = cp.asnumpy(qbins)
        qbinsc = cp.asnumpy(qbinsc)
        Idata = cp.asnumpy(Idata)
        support = cp.asnumpy(support)
        supportV = cp.asnumpy(supportV)
        Idata = cp.asnumpy(Idata)
        newrho = cp.asnumpy(newrho)
        qblravel = cp.asnumpy(qblravel)
        xcount = cp.asnumpy(xcount)

    # F = myfftn(rho)
    F = myrfftn(rho)
    # calculate spherical average intensity from 3D Fs
    I3D = abs2(F)
    # I3D = myabs(F)**2
    Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount)

    # scale Fs to match data
    factors = np.sqrt(Idata / Imean)
    factors[~qba] = 1.0
    F *= factors[qbin_labels]
    # rho = myifftn(F)
    rho = myirfftn(F)
    rho = rho.real

    # negative images yield the same scattering, so flip the image
    # to have more positive than negative values if necessary
    # to make sure averaging is done properly
    # whether theres actually more positive than negative values
    # is ambiguous, but this ensures all maps are at least likely
    # the same designation when averaging
    if np.sum(np.abs(rho[rho < 0])) > np.sum(rho[rho > 0]):
        rho *= -1

    # scale total number of electrons
    if ne is not None:
        rho *= ne / np.sum(rho)

    rg[j + 1] = calc_rg_by_guinier_first_2_points(qbinsc, Imean)
    supportV[j + 1] = supportV[j]

    # change rho to be the electron density in e-/angstroms^3, rather than number of electrons,
    # which is what the FFT assumes
    rho /= dV
    my_logger.info('FINISHED DENSITY REFINEMENT')

    if cutout:
        # here were going to cut rho out of the large real space box
        # to the voxels that contain the particle
        # use D to estimate particle size
        # assume the particle is in the center of the box
        # calculate how many voxels needed to contain particle of size D
        # use bigger than D to make sure we don't crop actual particle in case its larger than expected
        # lets clip it to a maximum of 2*D to be safe
        nD = int(2 * D / dx) + 1
        # make sure final box will still have even samples
        if nD % 2 == 1:
            nD += 1

        nmin = nbox // 2 - nD // 2
        nmax = nbox // 2 + nD // 2 + 2
        # create new rho array containing only the particle
        newrho = rho[nmin:nmax, nmin:nmax, nmin:nmax]
        rho = newrho
        # do the same for the support
        newsupport = support[nmin:nmax, nmin:nmax, nmin:nmax]
        support = newsupport
        # update side to new size of box
        side = dx * (nmax - nmin)

    if write_xplor_format:
        write_xplor(rho, side, fprefix + ".xplor")
        write_xplor(np.ones_like(rho) * support, side, fprefix + "_support.xplor")

    write_mrc(rho, side, fprefix + ".mrc")
    write_mrc(np.ones_like(rho) * support, side, fprefix + "_support.mrc")

    # return original unscaled values of Idata (and therefore Imean) for comparison with real data
    Idata /= scale_factor
    sigqdata /= scale_factor
    Imean /= scale_factor
    I /= scale_factor
    sigq /= scale_factor

    # Write some more output files
    Iq_exp = np.vstack((qraw, Iraw, sigqraw)).T
    Iq_calc = np.vstack((qbinsc, Imean, Imean * 0.01)).T
    idx = np.where(Iraw > 0)
    Iq_exp = Iq_exp[idx]
    qmax = np.min([Iq_exp[:, 0].max(), Iq_calc[:, 0].max()])
    Iq_exp = Iq_exp[Iq_exp[:, 0] <= qmax]
    Iq_calc = Iq_calc[Iq_calc[:, 0] <= qmax]
    final_chi2, exp_scale_factor, offset, fit = calc_chi2(Iq_exp, Iq_calc, scale=True, offset=False, interpolation=True,
                                                          return_sf=True, return_fit=True)

    np.savetxt(fprefix + '_map.fit', fit, delimiter=' ', fmt='%.5e',
               header='q(data),I(data),error(data),I(density); chi2=%.3f' % final_chi2)
    np.savetxt(fprefix + '_stats_by_step.dat', np.vstack((chi, rg, supportV)).T,
               delimiter=" ", fmt="%.5e", header='Chi2 Rg SupportVolume')

    chi[j + 1] = final_chi2

    my_logger.info('Number of steps: %i', j)
    my_logger.info('Final Chi2: %.3e', chi[j + 1])
    my_logger.info('Final Rg: %3.3f', rg[j + 1])
    my_logger.info('Final Support Volume: %3.3f', supportV[j + 1])
    my_logger.info('Mean Density (all voxels): %3.5f', np.mean(rho))
    my_logger.info('Std. Dev. of Density (all voxels): %3.5f', np.std(rho))
    my_logger.info('RMSD of Density (all voxels): %3.5f', np.sqrt(np.mean(np.square(rho))))

    return qdata, Idata, sigqdata, qbinsc, Imean, chi, rg, supportV, rho, side, fit, final_chi2


def shrinkwrap_by_density_value(rho, absv=True, sigma=3.0, threshold=0.2, recenter=True, recenter_mode="com"):
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
    rho_blurred = ndimage.filters.gaussian_filter(tmp, sigma=sigma, mode='wrap')

    support = np.zeros(rho.shape, dtype=bool)
    support[rho_blurred >= threshold * rho_blurred.max()] = True

    return rho, support


def shrinkwrap_by_volume(rho, N, absv=True, sigma=3.0, recenter=True, recenter_mode="com"):
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
    rho_blurred = ndimage.filters.gaussian_filter(tmp, sigma=sigma, mode='wrap')

    # grab the N largest values of the array
    idx = largest_indices(rho_blurred, N)
    support = np.zeros(rho.shape, dtype=bool)
    support[idx] = True
    # now, calculate the threshold that would correspond to the by_density_value method
    threshold = np.min(rho_blurred[idx]) / rho_blurred.max()

    return rho, support, threshold


def ecdf(x):
    """convenience function for computing the empirical CDF"""
    n = x.size
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return np.vstack((vals, ecdf)).T


def find_nearest_i(array, value):
    """Return the index of the array item nearest to specified value"""
    return (np.abs(array - value)).argmin()


def center_rho(rho, centering="com", return_shift=False, maxfirst=True, iterations=1):
    """Move electron density map so its center of mass aligns with the center of the grid

    centering - which part of the density to center on. By default, center on the
                center of mass ("com"). Can also center on maximum density value ("max").
    """
    ne_rho = np.sum((rho))
    gridcenter = grid_center(rho)  # (np.array(rho.shape)-1.)/2.
    total_shift = np.zeros(3)
    if maxfirst:
        # sometimes the density crosses the box boundary, meaning
        # the center of mass calculation becomes an issue
        # first roughly center using the maximum density value (by
        # rolling to avoid interpolation artifacts). Then perform
        # the center of mass translation.
        rho, shift = center_rho_roll(rho, recenter_mode="max", return_shift=True)
        total_shift += shift.astype(float)
    for i in range(iterations):
        if centering == "max":
            rhocom = np.unravel_index(rho.argmax(), rho.shape)
        else:
            rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
        shift = gridcenter - rhocom
        rho = ndimage.interpolation.shift(rho, shift, order=3, mode='wrap')
        rho = rho * ne_rho / np.sum(rho)
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
    total_shift = np.zeros(3, dtype=int)
    gridcenter = grid_center(rho)  # (np.array(rho.shape)-1.)/2.
    if maxfirst:
        # sometimes the density crosses the box boundary, meaning
        # the center of mass calculation becomes an issue
        # first roughly center using the maximum density value (by
        # rolling to avoid interpolation artifacts). Then perform
        # the center of mass translation.
        rhoargmax = np.unravel_index(np.abs(rho).argmax(), rho.shape)
        shift = gridcenter - rhoargmax
        shift = np.rint(shift).astype(int)
        rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
        total_shift += shift
    if recenter_mode == "max":
        rhocom = np.unravel_index(np.abs(rho).argmax(), rho.shape)
    else:
        rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    shift = gridcenter - rhocom
    shift = np.rint(shift).astype(int)
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    total_shift += shift
    if return_shift:
        return rho, total_shift
    else:
        return rho


def spherical_to_euler(phi, theta):
    """
    Convert spherical coordinates (phi, theta) to Euler angles (alpha, beta, gamma).
    """
    alpha = theta
    beta = phi
    gamma = 0  # We set gamma to 0, but it could be set to any value or made random
    return alpha, beta, gamma


def euler_grid_search(refrho, movrho, topn=1, abort_event=None):
    """Simple grid search on uniformly sampled sphere to optimize alignment.
        Return the topn candidate maps (default=1, i.e. the best candidate)."""
    # taken from https://stackoverflow.com/a/44164075/2836338

    # the euler angles search implicitly assumes the object is located
    # at the center of the grid, which may not be the case
    # first translate both refrho and movrho to center of grid, then
    # calculate optimal coarse rotations, the translate back
    refrhocen, refshift = center_rho_roll(refrho, return_shift=True)
    movrhocen, movshift = center_rho_roll(movrho, return_shift=True)

    refrho2 = ndimage.gaussian_filter(refrhocen, sigma=1.0, mode='wrap')
    movrho2 = ndimage.gaussian_filter(movrhocen, sigma=1.0, mode='wrap')
    n = refrho2.shape[0]
    b, e = (int(n / 4), int(3 * n / 4))
    refrho3 = refrho2[b:e, b:e, b:e]
    movrho3 = movrho2[b:e, b:e, b:e]

    # Sample about 100 rotations total
    n_samples = 99  # 33 * 3
    n_gamma = 3  # 3 roll samples
    num_sphere = n_samples // n_gamma  # number of points on the unit sphere

    # Fibonacci sphere sampling
    indices = np.arange(num_sphere, dtype=float)
    phi = np.arccos(1 - 2 * indices / (num_sphere - 1))
    theta = (np.pi * (1 + 5 ** 0.5) * indices) % (2 * np.pi)  # Normalize theta to [0, 2π)

    # Ensure the last phi is pi (or very close to it)
    phi[-1] = np.pi

    # Convert to Euler angles
    alpha, beta, gamma = spherical_to_euler(phi, theta)
    # Generate gamma values
    gamma = np.linspace(0, 2 * np.pi, n_gamma, endpoint=False)
    # Create all combinations
    euler_angles = np.array([(a % (2 * np.pi), b, g) for a, b in zip(alpha, beta) for g in gamma])
    alpha = euler_angles[:, 0]
    beta = euler_angles[:, 1]
    gamma = euler_angles[:, 2]

    scores = np.zeros(n_samples)

    for i in range(n_samples):
        T = np.array([alpha[i], beta[i], gamma[i], 0, 0, 0])
        tmp = transform_rho(T=T, rho=movrho3)
        scores[i] = real_space_correlation_coefficient(refrho3, tmp)
        if abort_event is not None:
            if abort_event.is_set():
                return None, None

    best_pt = largest_indices(scores, topn)
    best_scores = scores[best_pt]
    movrhos = np.zeros((topn, movrho.shape[0], movrho.shape[1], movrho.shape[2]))

    for i in range(topn):
        T = [alpha[best_pt[0][i]], beta[best_pt[0][i]], gamma[best_pt[0][i]], 0, 0, 0]
        movrhos[i] = transform_rho(movrho, T=T)
        # now that the top five rotations are calculated, move each one back
        # to the same center of mass as the original refrho, i.e. by -refrhoshift
        # movrhos[i] = ndimage.interpolation.shift(movrhos[i],-refshift,order=3,mode='wrap')
        movrhos[i] = np.roll(np.roll(np.roll(movrho, -refshift[0], axis=0), -refshift[1], axis=1), -refshift[2], axis=2)

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
        movrhos = movrho[np.newaxis, ...]
        scores = np.zeros(topn)

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


def minimize_rho(refrho, movrho, T=np.zeros(6)):
    """Optimize superposition of electron density maps. Move movrho to refrho."""
    bounds = np.zeros(12).reshape(6, 2)
    bounds[:3, 0] = -20 * np.pi
    bounds[:3, 1] = 20 * np.pi
    bounds[3:, 0] = -5
    bounds[3:, 1] = 5
    save_movrho = np.copy(movrho)
    save_refrho = np.copy(refrho)

    # first translate both to center
    # then afterwards translate back by -refshift
    refrho, refshift = center_rho_roll(refrho, return_shift=True)
    movrho, movshift = center_rho_roll(movrho, return_shift=True)

    # for alignment only, run a low-pass filter to remove noise
    sigma = 1.0
    refrho2 = ndimage.gaussian_filter(refrho, sigma=sigma, mode='wrap')
    movrho2 = ndimage.gaussian_filter(movrho, sigma=sigma, mode='wrap')
    n = refrho2.shape[0]
    # to speed it up crop out the solvent
    b, e = (int(n / 4), int(3 * n / 4))
    refrho3 = refrho2[b:e, b:e, b:e]
    movrho3 = movrho2[b:e, b:e, b:e]
    result = optimize.fmin_l_bfgs_b(minimize_rho_score, T, factr=0.1,
                                    maxiter=100, maxfun=200, epsilon=0.05,
                                    args=(refrho3, movrho3), approx_grad=True)
    Topt = result[0]
    newrho = transform_rho(movrho, Topt)
    rscc = real_space_correlation_coefficient(newrho, refrho)
    # now move newrho back by -refshift
    newrho = np.roll(np.roll(np.roll(newrho, -refshift[0], axis=0), -refshift[1], axis=1), -refshift[2], axis=2)
    finalscore = -1. * rho_overlap_score(save_refrho, newrho)
    return newrho, finalscore


def minimize_rho_score(T, refrho, movrho):
    """Scoring function for superposition of electron density maps.

        refrho - fixed, reference rho
        movrho - moving rho
        T - 6-element list containing alpha, beta, gamma, Tx, Ty, Tz in that order
        to move movrho by.
        """
    newrho = transform_rho(movrho, T)
    score = rho_overlap_score(refrho, newrho)
    return score


def real_space_correlation_coefficient(rho1, rho2, threshold=None):
    """Real space correlation coefficient between two density maps.

    threshold - fraction of max(rho1)"""
    if threshold is None:
        rho1_norm = rho1 - rho1.mean()
        rho2_norm = rho2 - rho1.mean()
        n = np.sum(rho1_norm * rho2_norm)
        d = (np.sum(rho1_norm ** 2) * np.sum(rho2_norm ** 2)) ** 0.5
    else:
        # if there's a threshold, base it on only one map, then use
        # those indices for both maps to ensure the same pixels are compared
        idx = np.where(np.abs(rho1) > threshold * np.abs(rho1).max())
        rho1_norm = rho1 - rho1.mean()
        rho2_norm = rho2 - rho1.mean()
        n = np.sum(rho1_norm[idx] * rho2_norm[idx])
        d = (np.sum(rho1_norm[idx] ** 2) * np.sum(rho2_norm[idx] ** 2)) ** 0.5
    rscc = n / d
    return rscc


def rho_overlap_score(rho1, rho2, threshold=None):
    """Scoring function for superposition of electron density maps."""
    rscc = real_space_correlation_coefficient(rho1, rho2, threshold=threshold)
    # -score for optimization, i.e. want to minimize, not maximize score
    return -rscc


def transform_rho(rho, T, order=1):
    """ Rotate and translate electron density map by T vector.
        T = [alpha, beta, gamma, x, y, z], angles in radians
        order = interpolation order (0-5)
    """
    ne_rho = np.sum(rho)
    R = euler2matrix(T[0], T[1], T[2])

    # Use the geometric center instead of center of mass
    c_out = np.array(rho.shape) / 2.0
    offset = c_out - R.dot(c_out)
    offset += T[3:]

    rho = ndimage.interpolation.affine_transform(rho, R, order=order,
                                                 offset=offset, output=np.float64, mode='wrap')
    rho *= ne_rho / np.sum(rho)
    return rho


def euler2matrix(alpha=0.0, beta=0.0, gamma=0.0):
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
    return reduce(np.dot, R[::-1])


def inertia_tensor(rho, side):
    """Calculate the moment of inertia tensor for the given electron density map."""
    halfside = side / 2.
    n = rho.shape[0]
    x_ = np.linspace(-halfside, halfside, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    Ixx = np.sum((y ** 2 + z ** 2) * rho)
    Iyy = np.sum((x ** 2 + z ** 2) * rho)
    Izz = np.sum((x ** 2 + y ** 2) * rho)
    Ixy = -np.sum(x * y * rho)
    Iyz = -np.sum(y * z * rho)
    Ixz = -np.sum(x * z * rho)
    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    return I


def principal_axes(I):
    """Calculate the principal inertia axes and order them Ia < Ib < Ic."""
    w, v = np.linalg.eigh(I)
    return w, v


def principal_axis_alignment(refrho, movrho):
    """ Align movrho principal axes to refrho."""
    side = 1.0
    ne_movrho = np.sum((movrho))
    # first center refrho and movrho, save refrho shift
    rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(refrho)))
    gridcenter = grid_center(refrho)  # (np.array(refrho.shape)-1.)/2.
    shift = gridcenter - rhocom
    refrho = ndimage.interpolation.shift(refrho, shift, order=3, mode='wrap')
    # calculate, save and perform rotation of refrho to xyz for later
    refI = inertia_tensor(refrho, side)
    refw, refv = principal_axes(refI)
    refR = refv.T
    refrho = align2xyz(refrho)
    # align movrho to xyz too
    # check for best enantiomer, eigh is ambiguous in sign
    movrho = align2xyz(movrho)
    enans = generate_enantiomers(movrho)
    scores = np.zeros(enans.shape[0])
    for i in range(enans.shape[0]):
        scores[i] = -rho_overlap_score(refrho, enans[i])
    movrho = enans[np.argmax(scores)]
    # now rotate movrho by the inverse of the refrho rotation
    R = np.linalg.inv(refR)
    c_in = np.array(ndimage.measurements.center_of_mass(np.abs(movrho)))
    c_out = (np.array(movrho.shape) - 1.) / 2.
    offset = c_in - c_out.dot(R)
    movrho = ndimage.interpolation.affine_transform(movrho, R.T, order=3, offset=offset, mode='wrap')
    # now shift it back to where refrho was originally
    movrho = ndimage.interpolation.shift(movrho, -shift, order=3, mode='wrap')
    movrho *= ne_movrho / np.sum(movrho)
    return movrho


def align2xyz(rho, return_transform=False):
    """ Align rho such that principal axes align with XYZ axes."""
    side = 1.0
    ne_rho = np.sum(rho)
    # shift refrho to the center
    rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
    gridcenter = grid_center(rho)  # (np.array(rho.shape)-1.)/2.
    shift = gridcenter - rhocom
    rho = ndimage.interpolation.shift(rho, shift, order=3, mode='wrap')
    # calculate, save and perform rotation of refrho to xyz for later
    I = inertia_tensor(rho, side)
    w, v = principal_axes(I)
    R = v.T
    refR = np.copy(R)
    refshift = np.copy(shift)
    # apparently need to run this a few times to get good alignment
    # maybe due to interpolation artifacts?
    for i in range(3):
        I = inertia_tensor(rho, side)
        w, v = np.linalg.eigh(I)  # principal axes
        R = v.T  # rotation matrix
        c_in = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
        c_out = (np.array(rho.shape) - 1.) / 2.
        offset = c_in - c_out.dot(R)
        rho = ndimage.interpolation.affine_transform(rho, R.T, order=3,
                                                     offset=offset, mode='wrap')
    # also need to run recentering a few times
    for i in range(3):
        rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(rho)))
        shift = gridcenter - rhocom
        rho = ndimage.interpolation.shift(rho, shift, order=3, mode='wrap')
    rho *= ne_rho / np.sum(rho)
    if return_transform:
        return rho, refR, refshift
    else:
        return rho


def generate_enantiomers(rho):
    """ Generate all enantiomers of given density map.
        Output maps are original, and flipped over z.
        """
    rho_zflip = rho[:, :, ::-1]
    enans = np.array([rho, rho_zflip])
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
            movrho *= ne_rho / np.sum(movrho)

        return movrho, score

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass


def select_best_enantiomer(refrho, rho, abort_event=None):
    """ Generate, align and select the enantiomer that best fits the reference map."""
    # translate refrho to center in case not already centered
    # just use roll to approximate translation to avoid interpolation, since
    # fine adjustments and interpolation will happen during alignment step

    try:
        sleep(1)
        c_refrho = center_rho_roll(refrho)
        # center rho in case it is not centered. use roll to get approximate location
        # and avoid interpolation
        c_rho = center_rho_roll(rho)
        # generate an array of the enantiomers
        enans = generate_enantiomers(c_rho)
        # allow for abort
        if abort_event is not None:
            if abort_event.is_set():
                return None, None

        # align each enantiomer and store the aligned maps and scores in results list
        results = [align(c_refrho, enan, abort_event=abort_event) for enan in enans]

        # now select the best enantiomer
        # rather than return the aligned and therefore interpolated enantiomer,
        # instead just return the original enantiomer, flipped from the original map
        # then no interpolation has taken place. So just dont overwrite enans essentially.
        # enans = np.array([results[k][0] for k in range(len(results))])
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
        rhos = rhos[np.newaxis, ...]
    if refrho is None:
        refrho = rhos[0]

    # in parallel, select the best enantiomer for each rho
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
        rhos = rhos[np.newaxis, ...]
    # first, center all the rhos, then shift them to where refrho is
    cen_refrho, refshift = center_rho_roll(refrho, return_shift=True)
    shift = -refshift
    for i in range(rhos.shape[0]):
        rhos[i] = center_rho_roll(rhos[i])
        ne_rho = np.sum(rhos[i])
        # now shift each rho back to where refrho was originally
        # rhos[i] = ndimage.interpolation.shift(rhos[i],-refshift,order=3,mode='wrap')
        rhos[i] = np.roll(np.roll(np.roll(rhos[i], shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
        rhos[i] *= ne_rho / np.sum(rhos[i])

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
    average_rho = (rho1 + rho2) / 2
    return average_rho


def multi_average_two(niter, **kwargs):
    """ Wrapper script for averaging two maps for multiprocessing."""
    try:
        sleep(1)
        return average_two(kwargs['rho1'][niter], kwargs['rho2'][niter], abort_event=kwargs['abort_event'])
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass


def average_pairs(rhos, cores=1, abort_event=None, single_proc=False):
    """ Average pairs of electron density maps, second half to first half."""
    # create even/odd pairs, odds are the references
    rho_args = {'rho1': rhos[::2], 'rho2': rhos[1::2], 'abort_event': abort_event}

    if not single_proc:
        pool = multiprocessing.Pool(cores)
        try:
            mapfunc = partial(multi_average_two, **rho_args)
            average_rhos = pool.map(mapfunc, list(range(rhos.shape[0] // 2)))
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
            sys.exit(1)
            raise
    else:
        average_rhos = [multi_average_two(niter, **rho_args) for niter in
                        range(rhos.shape[0] // 2)]

    return np.array(average_rhos)


def binary_average(rhos, cores=1, abort_event=None, single_proc=False):
    """ Generate a reference electron density map using binary averaging."""
    twos = 2 ** np.arange(20)
    nmaps = np.max(twos[twos <= rhos.shape[0]])
    # eight maps should be enough for the reference
    nmaps = np.max([nmaps, 8])
    levels = int(np.log2(nmaps)) - 1
    rhos = rhos[:nmaps]
    for level in range(levels):
        rhos = average_pairs(rhos, cores, abort_event=abort_event,
                             single_proc=single_proc)
    refrho = center_rho_roll(rhos[0])
    return refrho


def calc_fsc(rho1, rho2, side):
    """ Calculate the Fourier Shell Correlation between two electron density maps."""
    df = 1.0 / side
    n = rho1.shape[0]
    qx_ = np.fft.fftfreq(n) * n * df
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
    qx_max = qx.max()
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)
    # create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    F1 = np.fft.fftn(rho1)
    F2 = np.fft.fftn(rho2)
    numerator = ndimage.sum(np.real(F1 * np.conj(F2)), labels=qbin_labels,
                            index=np.arange(0, qbin_labels.max() + 1))
    term1 = ndimage.sum(np.abs(F1) ** 2, labels=qbin_labels,
                        index=np.arange(0, qbin_labels.max() + 1))
    term2 = ndimage.sum(np.abs(F2) ** 2, labels=qbin_labels,
                        index=np.arange(0, qbin_labels.max() + 1))
    denominator = (term1 * term2) ** 0.5
    FSC = numerator / denominator
    qidx = np.where(qbins < qx_max)
    return np.vstack((qbins[qidx], FSC[qidx])).T


def fsc2res(fsc, cutoff=0.5, return_plot=False):
    """Calculate resolution from the FSC curve using the cutoff given.

    fsc - an Nx2 array, where the first column is the x axis given as
          as 1/resolution (angstrom).
    cutoff - the fsc value at which to estimate resolution, default=0.5.
    return_plot - return additional arrays for plotting (x, y, resx)
    """
    x = np.linspace(fsc[0, 0], fsc[-1, 0], 1000)
    y = np.interp(x, fsc[:, 0], fsc[:, 1])
    if np.min(fsc[:, 1]) > 0.5:
        # if the fsc curve never falls below zero, then
        # set the resolution to be the maximum resolution
        # value sampled by the fsc curve
        resx = np.max(fsc[:, 0])
        resn = float(1. / resx)
        # print("Resolution: < %.1f A (maximum possible)" % resn)
    else:
        idx = np.where(y >= 0.5)
        # resi = np.argmin(y>=0.5)
        # resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])
        resx = np.max(x[idx])
        resn = float(1. / resx)
        # print("Resolution: %.1f A" % resn)
    if return_plot:
        return resn, x, y, resx
    else:
        return resn


def sigmoid(x, x0, k, b, L):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def sigmoid_find_x_value_given_y(y, x0, k, b, L):
    """find the corresponding x value given a desired y value on a sigmoid curve"""
    x = -1 / k * np.log(L / (y - b) - 1) + x0
    return x


def create_lowq(q):
    """Create a calculated q range for Sasrec for low q out to q=0.
    Just the q values, not any extrapolation of intensities."""
    dq = (q.max() - q.min()) / (q.size - 1)
    nq = int(q.min() / dq)
    qc = np.concatenate(([0.0], np.arange(nq) * dq + (q.min() - nq * dq), q))
    return qc


class Sasrec(object):
    def __init__(self, Iq, D, qc=None, r=None, nr=None, alpha=0.0, ne=2, extrapolate=True):
        self.Iq = Iq
        self.q = Iq[:, 0]
        self.I = Iq[:, 1]
        self.Ierr = Iq[:, 2]
        self.q.clip(1e-10)
        self.I[np.abs(self.I) < 1e-10] = 1e-10
        self.Ierr.clip(1e-10)
        self.q_data = np.copy(self.q)
        self.I_data = np.copy(self.I)
        self.Ierr_data = np.copy(self.Ierr)
        self.nq_data = len(self.q_data)
        if qc is None:
            self.qc = create_lowq(self.q)
        else:
            self.qc = qc
        if extrapolate:
            self.extrapolation = True
            self.extrapolate()
        else:
            self.extrapolation = False
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
            self.r = np.linspace(0, self.D, self.nr)
        else:
            self.nr = self.nq
            self.r = np.linspace(0, self.D, self.nr)
        self.alpha = alpha
        self.ne = ne
        self.update()

    def update(self):
        # self.r = np.linspace(0,self.D,self.nr)
        self.ri = np.arange(self.nr)
        self.n = self.shannon_channels(qmax=self.qmax, D=self.D) + self.ne
        self.Ni = np.arange(self.n)
        self.N = self.Ni + 1
        self.Mi = np.copy(self.Ni)
        self.M = np.copy(self.N)
        self.qn = np.pi / self.D * self.N
        self.In = np.zeros((self.nq))
        self.Inerr = np.zeros((self.nq))
        self.B_data = self.Bt(q=self.q_data)
        self.B = self.Bt()
        # Bc is for the calculated q values in
        # cases where qc is not equal to q.
        self.Bc = self.Bt(q=self.qc)
        self.S = self.St()
        self.Y = self.Yt()
        self.C = self.Ct2()
        self.Cinv = np.linalg.inv(self.C)
        self.In = np.linalg.solve(self.C, self.Y)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.Inerr = np.diagonal(self.Cinv) ** (0.5)
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
        self.chi2 = self.calc_chi2()

    def create_lowq(self):
        """Create a calculated q range for Sasrec for low q out to q=0.
        Just the q values, not any extrapolation of intensities."""
        dq = (self.q.max() - self.q.min()) / (self.q.size - 1)
        nq = int(self.q.min() / dq)
        self.qc = np.concatenate(([0.0], np.arange(nq) * dq + (self.q.min() - nq * dq), self.q))

    def extrapolate(self):
        """Extrapolate to high q values"""
        # create a range of 1001 data points from 1*qmax to 3*qmax
        self.ne = 1001
        qe = np.linspace(1.0 * self.q[-1], 3.0 * self.q[-1], self.ne)
        qe = qe[qe > self.q[-1]]
        qce = np.linspace(1.0 * self.qc[-1], 3.0 * self.q[-1], self.ne)
        qce = qce[qce > self.qc[-1]]
        # extrapolated intensities can be anything, since they will
        # have infinite errors and thus no impact on the calculation
        # of the fit, so just make them a constant
        Ie = np.ones_like(qe)
        # set infinite error bars so that the actual intensities don't matter
        Ierre = Ie * np.inf
        self.q = np.hstack((self.q, qe))
        self.I = np.hstack((self.I, Ie))
        self.Ierr = np.hstack((self.Ierr, Ierre))
        self.qc = np.hstack((self.qc, qce))

    def optimize_alpha(self):
        """Scan alpha values to find optimal alpha"""
        ideal_chi2 = self.calc_chi2()
        al = []
        chi2 = []
        # here, alphas are actually the exponents, since the range can
        # vary from 10^-20 upwards of 10^20. This should cover nearly all likely values
        alphas = np.arange(-30, 30., 2)
        i = 0
        nalphas = len(alphas)
        for alpha in alphas:
            i += 1
            sys.stdout.write("\rScanning alphas... {:.0%} complete".format(i * 1. / nalphas))
            sys.stdout.flush()
            try:
                self.alpha = 10. ** alpha
                # self.update()
                # don't run the full update, just update the Ins with the new alpha for speed
                # then run the full update at the end
                # updating alpha just updates C, so all steps from C to In calculation need to be run
                self.C = self.Ct2()
                self.Cinv = np.linalg.inv(self.C)
                self.In = np.linalg.solve(self.C, self.Y)
            except:
                continue
            chi2value = self.calc_chi2()
            al.append(alpha)
            chi2.append(chi2value)
        al = np.array(al)
        chi2 = np.array(chi2)
        print()
        # find optimal alpha value based on where chi2 begins to rise, to 10% above the ideal chi2
        # interpolate between tested alphas to find more precise value
        x = np.linspace(al[0], al[-1], 1000)
        y = np.interp(x, al, chi2)
        use_sigmoid = True
        if use_sigmoid:
            chif = 1.01
            # try and use a sigmoid fit
            # guess the midpoint of the sigmoid
            chi2_mid = (chi2.max() - chi2.min()) / 2
            idx = find_nearest_i(chi2_mid, y)
            al_mid = x[idx]
            # guess the min, max, etc.
            L = max(chi2)
            b = min(chi2)
            x0_guess = al_mid
            k_guess = np.median(al)
            p0 = [x0_guess, k_guess]
            # constrain b and L, only fit x0 and k
            popt, pcov = optimize.curve_fit(lambda x, x0, k: sigmoid(x, x0, k, b, L), al, chi2, p0, method='dogbox')
            fit = sigmoid(x, popt[0], popt[1], b, L)
            # find the value of the sigmoid closest to 1.1*ideal_chi2
            # minimum of the sigmoid is b parameter, which can be taken from popt
            opt_alpha_exponent = sigmoid_find_x_value_given_y(chif * b, popt[0], popt[1], b, L)
        else:
            chif = 1.1
            # take the maximum alpha value (x) where the chi2 just starts to rise above ideal
            try:
                ali = np.argmax(x[y <= chif * ideal_chi2])
            except:
                # if it fails, it may mean that the lowest alpha value of 10^-20 is still too large, so just take that.
                ali = 0
            # set the optimal alpha to be 10^alpha, since we were actually using exponents
            # also interpolate between the two neighboring alpha values, to get closer to the chif*ideal_chi2
            opt_alpha_exponent = np.interp(chif * ideal_chi2, [y[ali], y[ali - 1]], [x[ali], x[ali - 1]])
        opt_alpha = 10.0 ** (opt_alpha_exponent)
        self.alpha = opt_alpha
        self.update()
        return self.alpha

    def calc_chi2(self):
        Ish = self.In
        Bn = self.B_data
        # calculate Ic at experimental q vales for chi2 calculation
        self.Ic_qe = 2 * np.einsum('n,nq->q', Ish, Bn)
        self.chi2 = 1 / (self.nq_data - 1) * np.sum(1 / (self.Ierr_data ** 2) * (self.I_data - self.Ic_qe) ** 2)
        return self.chi2

    def estimate_Vp_etal(self):
        """Estimate Porod volume using modified method based on oversmoothing.

        Oversmooth the P(r) curve with a high alpha. This helps to remove shape
        scattering that distorts Porod assumptions. """
        # how much to oversmooth by, i.e. multiply alpha times this factor
        oversmoothing = 1.0e1
        # use a different qmax to limit effects of shape scattering.
        # use 8/Rg as the new qmax, but be sure to keep these effects
        # separate from the rest of sasrec, as it is only used for estimating
        # porod volume.
        qmax = 8. / self.rg
        if np.isnan(qmax):
            qmax = 8. / (self.D / 3.5)
        Iq = np.vstack((self.q, self.I, self.Ierr)).T
        sasrec4vp = Sasrec(Iq[self.q < qmax], self.D, alpha=self.alpha * oversmoothing, extrapolate=self.extrapolation)
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
        num_channels = int((qmax - qmin) / width)
        return num_channels

    def Bt(self, q=None):
        N = self.N[:, None]
        if q is None:
            q = self.q
        else:
            q = q
        D = self.D
        # catch cases where qD==nPi, not often, but possible
        x = (N * np.pi) ** 2 - (q * D) ** 2
        y = np.where(x == 0, (N * np.pi) ** 2, x)
        # B = (N*np.pi)**2/((N*np.pi)**2-(q*D)**2) * np.sinc(q*D/np.pi) * (-1)**(N+1)
        B = (N * np.pi) ** 2 / y * np.sinc(q * D / np.pi) * (-1) ** (N + 1)
        return B

    def St(self):
        N = self.N[:, None]
        r = self.r
        D = self.D
        S = r / (2 * D ** 2) * N * np.sin(N * np.pi * r / D)
        return S

    def Yt(self):
        """Return the values of Y, an m-length vector."""
        I = self.I
        Ierr = self.Ierr
        Bm = self.B
        Y = np.einsum('q, nq->n', I / Ierr ** 2, Bm)
        return Y

    def Ct(self):
        """Return the values of C, a m x n variance-covariance matrix"""
        Ierr = self.Ierr
        Bm = self.B
        Bn = self.B
        C = 2 * np.einsum('ij,kj->ik', Bm / Ierr ** 2, Bn)
        return C

    def Gmn(self):
        """Return the mxn matrix of coefficients for the integral of (2nd deriv of P(r))**2 used for smoothing"""
        M = self.M
        N = self.N
        D = self.D
        gmn = np.zeros((self.n, self.n))
        mm, nn = np.meshgrid(M, N, indexing='ij')
        # two cases, one where m!=n, one where m==n. Do both separately.
        idx = np.where(mm != nn)
        gmn[idx] = np.pi ** 2 / (2 * D ** 5) * (mm[idx] * nn[idx]) ** 2 * (mm[idx] ** 4 + nn[idx] ** 4) / (
                    mm[idx] ** 2 - nn[idx] ** 2) ** 2 * (-1) ** (mm[idx] + nn[idx])
        idx = np.where(mm == nn)
        gmn[idx] = nn[idx] ** 4 * np.pi ** 2 / (48 * D ** 5) * (2 * nn[idx] ** 2 * np.pi ** 2 + 33)
        return gmn

    def Ct2(self):
        """Return the values of C, a m x n variance-covariance matrix while smoothing P(r)"""
        n = self.n
        Ierr = self.Ierr
        Bm = self.B
        Bn = self.B
        alpha = self.alpha
        gmn = self.Gmn()
        return alpha * gmn + 2 * np.einsum('ij,kj->ik', Bm / Ierr ** 2, Bn)

    def Ish2Iq(self):
        """Calculate I(q) from intensities at Shannon points."""
        Ish = self.In
        Bn = self.Bc
        I = 2 * np.einsum('n,nq->q', Ish, Bn)
        return I

    def Ish2P(self):
        """Calculate P(r) from intensities at Shannon points."""
        Ish = self.In
        Sn = self.S
        P = np.einsum('n,nr->r', Ish, Sn)
        return P

    def Icerrt(self):
        """Return the standard errors on I_c(q)."""
        Bn = self.Bc
        Bm = self.Bc
        err2 = 2 * np.einsum('nq,mq,nm->q', Bn, Bm, self.Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            err = err2 ** (.5)
        return err

    def Perrt(self):
        """Return the standard errors on P(r)."""
        Sn = self.S
        Sm = self.S
        err2 = np.einsum('nr,mr,nm->r', Sn, Sm, self.Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            err = err2 ** (.5)
        return err

    def Ish2I0(self):
        """Calculate I0 from Shannon intensities"""
        N = self.N
        Ish = self.In
        I0 = 2 * np.sum(Ish * (-1) ** (N + 1))
        return I0

    def I0errf(self):
        """Calculate error on I0 from Shannon intensities from inverse C variance-covariance matrix"""
        N = self.N
        M = self.M
        Cinv = self.Cinv
        s2 = 2 * np.einsum('n,m,nm->', (-1) ** (N), (-1) ** M, Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s = s2 ** (0.5)
        return s

    def Ft(self):
        """Calculate Fn function, for use in Rg calculation"""
        N = self.N
        F = (1 - 6 / (N * np.pi) ** 2) * (-1) ** (N + 1)
        return F

    def Ish2rg(self):
        """Calculate Rg from Shannon intensities"""
        N = self.N
        Ish = self.In
        D = self.D
        I0 = self.I0
        F = self.F
        summation = np.sum(Ish * F)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rg2 = D ** 2 / I0 * summation
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
        s2 = np.einsum('n,m,nm->', Fn, Fm, Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s = D ** 2 / (I0 * rg) * s2 ** (0.5)
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
        s2 = np.einsum('n,m,nm->', Fn, Fm, Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rgerr = D ** 2 / (I0 * rg) * s2 ** (0.5)
        return rgerr

    def Et(self):
        """Calculate En function, for use in ravg calculation"""
        N = self.N
        E = ((-1) ** N - 1) / (N * np.pi) ** 2 - (-1) ** N / 2.
        return E

    def Ish2avgr(self):
        """Calculate average vector length r from Shannon intensities"""
        Ish = self.In
        I0 = self.I0
        D = self.D
        E = self.E
        avgr = 4 * D / I0 * np.sum(Ish * E)
        return avgr

    def avgrerrf(self):
        """Calculate error on Rg from Shannon intensities from inverse C variance-covariance matrix"""
        D = self.D
        Cinv = self.Cinv
        I0 = self.I0
        En = self.E
        Em = self.E
        s2 = np.einsum('n,m,nm->', En, Em, Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            avgrerr = 4 * D / I0 * s2 ** (0.5)
        return avgrerr

    def Ish2Q(self):
        """Calculate Porod Invariant Q from Shannon intensities"""
        D = self.D
        N = self.N
        Ish = self.In
        Q = (np.pi / D) ** 3 * np.sum(Ish * N ** 2)
        return Q

    def Qerrf(self):
        """Calculate error on Q from Shannon intensities from inverse C variance-covariance matrix"""
        D = self.D
        Cinv = self.Cinv
        N = self.N
        M = self.M
        s2 = np.einsum('n,m,nm->', N ** 2, M ** 2, Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s = (np.pi / D) ** 3 * s2 ** (0.5)
        return s

    def gamma0(self):
        """Calculate gamma at r=0. gamma is P(r)/4*pi*r^2"""
        Ish = self.In
        D = self.D
        Q = self.Q
        return 1 / (8 * np.pi ** 3) * Q

    def Ish2Vp(self):
        """Calculate Porod Volume from Shannon intensities"""
        Q = self.Q
        I0 = self.I0
        Vp = 2 * np.pi ** 2 * I0 / Q
        return Vp

    def Vperrf(self):
        """Calculate error on Vp from Shannon intensities from inverse C variance-covariance matrix"""
        I0 = self.I0
        Q = self.Q
        I0s = self.I0err
        Qs = self.Qerr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            Vperr2 = (2 * np.pi / Q) ** 2 * (I0s) ** 2 + (2 * np.pi * I0 / Q ** 2) ** 2 * Qs ** 2
            Vperr = Vperr2 ** (0.5)
        return Vperr

    def Ish2mwVp(self):
        """Calculate molecular weight via Porod Volume from Shannon intensities"""
        Vp = self.Vp
        mw = Vp / 1.66
        return mw

    def mwVperrf(self):
        """Calculate error on mwVp from Shannon intensities from inverse C variance-covariance matrix"""
        Vps = self.Vperr
        return Vps / 1.66

    def Ish2Vc(self):
        """Calculate Volume of Correlation from Shannon intensities"""
        Ish = self.In
        N = self.N
        I0 = self.I0
        D = self.D
        area_qIq = 2 * np.pi / D ** 2 * np.sum(N * Ish * special.sici(N * np.pi)[0])
        Vc = I0 / area_qIq
        return Vc

    def Vcerrf(self):
        """Calculate error on Vc from Shannon intensities from inverse C variance-covariance matrix"""
        I0 = self.I0
        Vc = self.Vc
        N = self.N
        M = self.M
        D = self.D
        Cinv = self.Cinv
        Sin = special.sici(N * np.pi)[0]
        Sim = special.sici(M * np.pi)[0]
        s2 = np.einsum('n,m,nm->', N * Sin, M * Sim, Cinv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            Vcerr = (2 * np.pi * Vc ** 2 / (D ** 2 * I0)) * s2 ** (0.5)
        return Vcerr

    def Ish2Qr(self):
        """Calculate Rambo Invariant Qr (Vc^2/Rg) from Shannon intensities"""
        Vc = self.Vc
        Rg = self.rg
        Qr = Vc ** 2 / Rg
        return Qr

    def Ish2mwVc(self, RNA=False):
        """Calculate molecular weight via the Volume of Correlation from Shannon intensities"""
        Qr = self.Qr
        if RNA:
            mw = (Qr / 0.00934) ** (0.808)
        else:
            mw = (Qr / 0.1231) ** (1.00)
        return mw

    def mwVcerrf(self):
        Vc = self.Vc
        Rg = self.rg
        Vcs = self.Vcerr
        Rgs = self.rgerr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            mwVcs = Vc / (0.1231 * Rg) * (4 * Vcs ** 2 + (Vc / Rg * Rgs) ** 2) ** (0.5)
        return mwVcs

    def Ish2lc(self):
        """Calculate length of correlation from Shannon intensities"""
        Vp = self.Vp
        Vc = self.Vc
        lc = Vp / (2 * np.pi * Vc)
        return lc

    def lcerrf(self):
        """Calculate error on lc from Shannon intensities from inverse C variance-covariance matrix"""
        Vp = self.Vp
        Vc = self.Vc
        Vps = self.Vperr
        Vcs = self.Vcerr
        s2 = Vps ** 2 + (Vp / Vc) ** 2 * Vcs ** 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            lcerr = 1 / (2 * np.pi * Vc) * s2 ** (0.5)
        return lcerr


class PDB(object):
    """Load pdb file."""

    def __init__(self, filename=None, natoms=None, ignore_waters=True, chain=None):
        if isinstance(filename, int):
            # if a user gives no keyword argument, but just an integer,
            # assume the user means the argument is to be interpreted
            # as natoms, rather than filename
            natoms = filename
            filename = None
        if filename is not None:
            ext = os.path.splitext(filename)[1]
            if ext == ".cif":
                self.read_cif(filename, ignore_waters=ignore_waters, chain=chain)
                # print("ERROR: Cannot parse .cif files (yet).")
                # exit()
            else:
                # if it's not a cif file, default to a pdb file. This allows
                # for extensions that are not .pdb, such as .pdb1 which exists.
                try:
                    self.read_pdb(filename, ignore_waters=ignore_waters, chain=chain)
                except:
                    print("ERROR: Cannot parse file: %s" % filename)
                    exit()
        elif natoms is not None:
            self.generate_pdb_from_defaults(natoms)
        self.rij = None
        self.radius = None
        self.unique_radius = None
        self.unique_volume = None

    def read_pdb(self, filename, ignore_waters=True, chain=None):
        self.filename = filename
        self.natoms = 0
        with open(filename) as f:
            for line in f:
                if line[0:6] == "ENDMDL":
                    break
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue  # skip other lines
                if ignore_waters and ((line[17:20] == "HOH") or (line[17:20] == "TIP")):
                    continue
                if chain is not None:
                    # Ignore any atoms not in desired chain
                    if line[21].strip() != chain:
                        continue
                self.natoms += 1
        self.atomnum = np.zeros((self.natoms), dtype=int)
        self.atomname = np.zeros((self.natoms), dtype=np.dtype((str, 3)))
        self.atomalt = np.zeros((self.natoms), dtype=np.dtype((str, 1)))
        self.resname = np.zeros((self.natoms), dtype=np.dtype((str, 3)))
        self.resnum = np.zeros((self.natoms), dtype=int)
        self.chain = np.zeros((self.natoms), dtype=np.dtype((str, 1)))
        self.coords = np.zeros((self.natoms, 3))
        self.occupancy = np.zeros((self.natoms))
        self.b = np.zeros((self.natoms))
        self.atomtype = np.zeros((self.natoms), dtype=np.dtype((str, 2)))
        self.charge = np.zeros((self.natoms), dtype=np.dtype((str, 2)))
        self.nelectrons = np.zeros((self.natoms), dtype=int)
        self.vdW = np.zeros(self.natoms)
        self.numH = np.zeros(self.natoms)
        self.unique_exvolHradius = np.zeros(self.natoms)
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
                    continue  # skip other lines
                if ignore_waters and ((line[17:20] == "HOH") or (line[17:20] == "TIP")):
                    continue
                if chain is not None:
                    # Ignore any atoms not in desired chain
                    if line[21].strip() != chain:
                        continue
                try:
                    self.atomnum[atom] = int(line[6:11])
                except ValueError as e:
                    self.atomnum[atom] = int(line[6:11], 36)
                self.atomname[atom] = line[12:16].split()[0]
                self.atomalt[atom] = line[16]
                self.resname[atom] = line[17:20]
                try:
                    self.resnum[atom] = int(line[22:26])
                except ValueError as e:
                    self.resnum[atom] = int(line[22:26], 36)
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
                    # if atomtype column is not in pdb file, set to first
                    # character of atomname that is in the database of elements
                    # (sometimes a number is the first character)
                    # otherwise default to Carbon
                    atomtype = self.atomname[atom][0]
                    if atomtype not in ffcoeff:
                        atomtype = self.atomname[atom][1]
                    if atomtype not in ffcoeff:
                        print("%s atomtype not recognized for atom number %s" % (atomtype, self.atomnum[atom]))
                        print("Setting atomtype to default Carbon.")
                        atomtype = "C"
                self.atomtype[atom] = atomtype
                self.charge[atom] = line[78:80].strip('\n')
                self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(), 6)
                if len(self.atomtype[atom]) == 1:
                    atomtype = self.atomtype[atom][0].upper()
                else:
                    atomtype = self.atomtype[atom][0].upper() + self.atomtype[atom][1].lower()
                try:
                    dr = vdW[atomtype]
                except:
                    try:
                        dr = vdW[atomtype[0]]
                    except:
                        # default to carbon
                        dr = vdW['C']
                self.vdW[atom] = dr
                atom += 1

    def read_cif(self, filename, ignore_waters=True):
        """
        Read a CIF file and load atom information.

        :param filename: Path to the CIF file
        :param chain_id: Optional chain ID to filter atoms (default: None, read all chains)
        :param ignore_waters: Whether to ignore water molecules (default: True)
        """
        self.filename = filename
        self.natoms = 0
        atoms = []

        with open(filename, 'r') as f:
            reading_atoms = False
            for line in f:
                if line.startswith('_atom_site.'):
                    reading_atoms = True
                    continue
                if reading_atoms and line.startswith('#'):
                    break
                if reading_atoms and not line.strip():
                    continue
                if reading_atoms:
                    fields = line.split()
                    if len(fields) < 20:  # Ensure we have enough fields
                        continue

                    atom_chain = fields[16]

                    resname = fields[5]
                    if ignore_waters and resname in ['HOH', 'WAT']:
                        continue

                    atoms.append({
                        'atomnum': int(fields[1]),
                        'atomname': fields[3],
                        'atomalt': fields[4],
                        'resname': resname,
                        'chain': atom_chain,
                        'resnum': int(fields[8]),
                        'x': float(fields[10]),
                        'y': float(fields[11]),
                        'z': float(fields[12]),
                        'occupancy': float(fields[13]),
                        'b': float(fields[14]),
                        'atomtype': fields[2],
                        'charge': fields[17] if len(fields) > 17 else ''
                    })

        self.natoms = len(atoms)

        # Initialize numpy arrays
        self.atomnum = np.zeros(self.natoms, dtype=int)
        self.atomname = np.zeros(self.natoms, dtype='U4')
        self.atomalt = np.zeros(self.natoms, dtype='U1')
        self.resname = np.zeros(self.natoms, dtype='U3')
        self.resnum = np.zeros(self.natoms, dtype=int)
        self.chain = np.zeros(self.natoms, dtype='U1')
        self.coords = np.zeros((self.natoms, 3))
        self.occupancy = np.zeros(self.natoms)
        self.b = np.zeros(self.natoms)
        self.atomtype = np.zeros(self.natoms, dtype='U2')
        self.charge = np.zeros(self.natoms, dtype='U2')
        self.nelectrons = np.zeros(self.natoms, dtype=int)
        self.vdW = np.zeros(self.natoms)
        self.numH = np.zeros(self.natoms)
        self.unique_exvolHradius = np.zeros(self.natoms)
        self.exvolHradius = np.zeros(self.natoms)

        # Fill numpy arrays with data
        for i, atom in enumerate(atoms):
            self.atomnum[i] = atom['atomnum']
            self.atomname[i] = atom['atomname']
            self.atomalt[i] = atom['atomalt']
            self.resname[i] = atom['resname']
            self.resnum[i] = atom['resnum']
            self.chain[i] = atom['chain']
            self.coords[i] = [atom['x'], atom['y'], atom['z']]
            self.occupancy[i] = atom['occupancy']
            self.b[i] = atom['b']
            self.atomtype[i] = atom['atomtype']
            self.charge[i] = atom['charge']
            self.nelectrons[i] = electrons.get(self.atomtype[i].upper(), 6)

            if len(self.atomtype[i]) == 1:
                atomtype = self.atomtype[i][0].upper()
            else:
                atomtype = self.atomtype[i][0].upper() + self.atomtype[i][1].lower()
            try:
                dr = vdW[i]
            except:
                try:
                    dr = vdW[i[0]]
                except:
                    # default to carbon
                    dr = vdW['C']
            self.vdW[i] = dr

    def generate_pdb_from_defaults(self, natoms):
        self.natoms = natoms
        # simple array of incrementing integers, starting from 1
        self.atomnum = np.arange((self.natoms), dtype=int) + 1
        # all carbon atoms by default
        self.atomname = np.full((self.natoms), "C", dtype=np.dtype((str, 3)))
        # no alternate conformations by default
        self.atomalt = np.zeros((self.natoms), dtype=np.dtype((str, 1)))
        # all Alanines by default
        self.resname = np.full((self.natoms), "ALA", dtype=np.dtype((str, 3)))
        # each atom belongs to a new residue by default
        self.resnum = np.arange((self.natoms), dtype=int)
        # chain A by default
        self.chain = np.full((self.natoms), "A", dtype=np.dtype((str, 1)))
        # all atoms at (0,0,0) by default
        self.coords = np.zeros((self.natoms, 3))
        # all atoms 1.0 occupancy by default
        self.occupancy = np.ones((self.natoms))
        # all atoms 20 A^2 by default
        self.b = np.ones((self.natoms)) * 20.0
        # all atom types carbon by default
        self.atomtype = np.full((self.natoms), "C", dtype=np.dtype((str, 2)))
        # all atoms neutral by default
        self.charge = np.zeros((self.natoms), dtype=np.dtype((str, 2)))
        # all atoms carbon so have six electrons by default
        self.nelectrons = np.ones((self.natoms), dtype=int) * 6
        self.radius = np.zeros(self.natoms)
        self.vdW = np.zeros(self.natoms)
        self.unique_volume = np.zeros(self.natoms)
        self.unique_radius = np.zeros(self.natoms)
        # set a variable with H radius to be used for exvol radii optimization
        # set a variable for number of hydrogens bonded to atoms
        # self.exvolHradius = implicit_H_radius
        self.unique_exvolHradius = np.zeros(self.natoms)
        self.implicitH = False
        self.numH = np.zeros((self.natoms))
        # for CRYST1 card, use default defined by PDB, but 100 A side
        self.cella = 100.0
        self.cellb = 100.0
        self.cellc = 100.0
        self.cellalpha = 90.0
        self.cellbeta = 90.0
        self.cellgamma = 90.0

    def calculate_distance_matrix(self, return_squareform=False):
        if return_squareform:
            self.rij = spatial.distance.squareform(spatial.distance.pdist(self.coords))
        else:
            self.rij = spatial.distance.pdist(self.coords)

    def calculate_unique_volume(self, n=16, use_b=False, atomidx=None):
        """Generate volumes and radii for each atom of a pdb by accounting for overlapping sphere volumes,
        i.e., each radius is set to the value that yields a volume of a sphere equal to the
        corrected volume of the sphere after subtracting spherical caps from bonded atoms."""
        # first, for each atom, find all atoms closer than the sum of the two vdW radii
        ns = np.array([8, 16, 32])
        corrections = np.array([1.53, 1.19, 1.06])  # correction for n=8 voxels (1.19 for n=16, 1.06 for n=32)
        correction = np.interp(n, ns, corrections)  # a rough approximation.
        # print("Calculating unique atomic volumes...")
        if self.unique_volume is None:
            self.unique_volume = np.zeros(self.natoms)
        if atomidx is None:
            atomidx = range(self.natoms)
        for i in atomidx:
            # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,self.natoms))
            # sys.stdout.flush()
            # for each atom, make a box of voxels around it
            ra = self.vdW[i]  # ra is the radius of the main atom
            if use_b:
                ra += B2u(self.b[i])
            side = 2 * ra
            # n = 8 #yields somewhere around 0.2 A voxel spacing depending on atom size
            dx = side / n
            dV = dx ** 3
            x_ = np.linspace(-side / 2, side / 2, n)
            x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
            minigrid = np.zeros(x.shape, dtype=np.bool_)
            shift = np.ones(3) * dx / 2.
            # create a column stack of coordinates for the minigrid
            xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            # for simplicity assume the atom is at the center of the minigrid, (0,0,0),
            # therefore we need to subtract the vector shift (i.e. the coordinates
            # of the atom) from each of the neighboring atoms, so grab those coordinates
            p = np.copy(self.coords[i])
            # calculate all distances from the atom to the minigrid points
            center = np.zeros(3)
            xa, ya, za = center
            dist = spatial.distance.cdist(center[None, :], xyz)[0].reshape(n, n, n)
            # now, any elements of minigrid that have a dist less than ra make true
            minigrid[dist <= ra] = True
            # grab atoms nearby this atom just based on xyz coordinates
            # first, recenter all coordinates in this frame
            coordstmp = self.coords - p
            # next, get all atoms whose x, y, and z coordinates are within the nearby box
            # of length 4 A (more than the sum of two atoms vdW radii, with the limit being about 2.5 A)
            bl = 5.0
            idx_close = np.where(
                (coordstmp[:, 0] >= xa - bl / 2) & (coordstmp[:, 0] <= xa + bl / 2) &
                (coordstmp[:, 1] >= ya - bl / 2) & (coordstmp[:, 1] <= ya + bl / 2) &
                (coordstmp[:, 2] >= za - bl / 2) & (coordstmp[:, 2] <= za + bl / 2)
            )[0]
            idx_close = idx_close[idx_close != i]  # ignore this atom
            nclose = len(idx_close)
            for j in range(nclose):
                # get index of next closest atom
                idx_j = idx_close[j]
                # get the coordinates of the  neighboring atom, and shift using the same vector p as the main atom
                cb = self.coords[idx_j] - p  # center of neighboring atom in new coordinate frame
                xb, yb, zb = cb
                rb = self.vdW[idx_j]
                if use_b:
                    rb += B2u(self.b[idx_j])
                a, b, c, d = equation_of_plane_from_sphere_intersection(xa, ya, za, ra, xb, yb, zb, rb)
                normal = np.array([a, b, c])  # definition of normal to a plane
                # for each grid point, calculate the distance to the plane in the direction of the vector normal
                # if the distance is positive, then that gridpoint is beyond the plane
                # we can calculate the center of the circle which lies on the plane, so thats a good point to use
                circle_center = center_of_circle_from_sphere_intersection(xa, ya, za, ra, xb, yb, zb, rb, a, b, c, d)
                xyz_minus_cc = xyz - circle_center
                # calculate distance matrix to neighbor
                dist2neighbor = spatial.distance.cdist(cb[None, :], xyz)[0].reshape(n, n, n)
                overlapping_voxels = np.zeros(n ** 3, dtype=bool)
                overlapping_voxels[minigrid.ravel() & np.ravel(dist2neighbor <= rb)] = True
                # calculate the distance to the plane for each minigrid voxel
                # there may be a way to vectorize this if its too slow
                noverlap = overlapping_voxels.sum()
                # print(noverlap, overlapping_voxels.size)
                d2plane = np.zeros(x.size)
                for k in range(n ** 3):
                    if overlapping_voxels[k]:
                        d2plane[k] = np.dot(normal, xyz_minus_cc[k, :])
                d2plane = d2plane.reshape(n, n, n)
                # all voxels with a positive d2plane value are _beyond_ the plane
                minigrid[d2plane > 0] = False
            # add up all the remaining voxels in the minigrid to get the volume
            # also correct for limited voxel size
            self.unique_volume[i] = minigrid.sum() * dV * correction

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
                print("%s:%s not found in volumes dictionary. Calculating unique volume." % (
                self.resname[i], self.atomname[i]))
                # print("Setting volume to ALA:CA.")
                # self.unique_volume[i] = atomic_volumes['ALA']['CA']
                self.calculate_unique_volume(atomidx=[i])

    def add_ImplicitH(self):
        if 'H' in self.atomtype:
            self.remove_by_atomtype('H')

        for i in range(len(self.atomname)):
            res = self.resname[i]
            atom = self.atomname[i]

            # For each atom, atom should be a key in "numH", so now just look up value
            # associated with atom
            try:
                H_count = np.rint(numH[res][atom])  # the number of H attached
                # print(res, atom, numH[res][atom])
                # Hbond_count = protein_residues.normal[res]['numH']
                # H_count = Hbond_count[atom]
                H_mean_volume = volH[res][atom]  # the average volume of each H attached
            except:
                # print("atom ", atom, " not in ", res, " list. setting numH to 0.")
                H_count = 0
                H_mean_volume = 0

            # Add number of hydrogens for the atom to a pdb object so it can
            # be carried with pdb class
            self.numH[i] = H_count  # the number of H attached
            self.unique_exvolHradius[i] = sphere_radius_from_volume(H_mean_volume)
            self.nelectrons[i] += H_count

    def remove_waters(self):
        idx = np.where((self.resname == "HOH") | (self.resname == "TIP"))
        self.remove_atoms_from_object(idx)

    def remove_by_atomtype(self, atomtype):
        idx = np.where((self.atomtype == atomtype))
        self.remove_atoms_from_object(idx)

    def remove_by_atomname(self, atomname):
        idx = np.where((self.atomname == atomname))
        self.remove_atoms_from_object(idx)

    def remove_by_atomnum(self, atomnum):
        idx = np.where((self.atomnum == atomnum))
        self.remove_atoms_from_object(idx)

    def remove_by_resname(self, resname):
        idx = np.where((self.resname == resname))
        self.remove_atoms_from_object(idx)

    def remove_by_resnum(self, resnum):
        idx = np.where((self.resnum == resnum))
        self.remove_atoms_from_object(idx)

    def remove_by_chain(self, chain):
        idx = np.where((self.chain == chain))
        self.remove_atoms_from_object(idx)

    def remove_atomalt(self):
        idx = np.where((self.atomalt != ' ') & (self.atomalt != 'A'))
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
        anum, rc = (np.unique(self.atomnum, return_counts=True))
        if np.any(rc > 1):
            # in case default atom numbers are repeated, just renumber them
            self_numbering = True
        else:
            self_numbering = False
        for i in range(self.natoms):
            if self_numbering:
                atomnum = '%5i' % ((i + 1) % 99999)
            else:
                atomnum = '%5i' % (self.atomnum[i] % 99999)
            atomname = '%3s' % self.atomname[i]
            atomalt = '%1s' % self.atomalt[i]
            resnum = '%4i' % (self.resnum[i] % 9999)
            resname = '%3s' % self.resname[i]
            chain = '%1s' % self.chain[i]
            x = '%8.3f' % self.coords[i, 0]
            y = '%8.3f' % self.coords[i, 1]
            z = '%8.3f' % self.coords[i, 2]
            o = '% 6.2f' % self.occupancy[i]
            b = '%6.2f' % self.b[i]
            atomtype = '%2s' % self.atomtype[i]
            charge = '%2s' % self.charge[i]
            records.append([
                               'ATOM  ' + atomnum + '  ' + atomname + ' ' + resname + ' ' + chain + resnum + '    ' + x + y + z + o + b + '          ' + atomtype + charge])
        # np.savetxt(filename, records, fmt='%80s'.encode('ascii'))
        if sys.version_info[0] < 3:
            # Python 2 approach
            np.savetxt(filename, records, fmt='%80s'.encode('ascii'))
        else:
            # Python 3 approach
            np.savetxt(filename, records, fmt='%80s')


def sphere_volume_from_radius(R):
    V_sphere = 4 * np.pi / 3 * R ** 3
    return V_sphere


def sphere_radius_from_volume(V):
    R_sphere = (3 * V / (4 * np.pi)) ** (1. / 3)
    return R_sphere


def cap_heights(r1, r2, d):
    """Calculate the heights h1, h2 of spherical caps from overlapping spheres of radii r1, r2 a distance d apart"""
    h1 = (r2 - r1 + d) * (r2 + r1 - d) / (2 * d)
    h2 = (r1 - r2 + d) * (r1 + r2 - d) / (2 * d)
    return h1, h2


def spherical_cap_volume(R, h):
    # sphere of radius R, cap of height h
    V_cap = 1. / 3 * np.pi * h ** 2 * (3 * R - h)
    return V_cap


def equation_of_plane_from_sphere_intersection(x1, y1, z1, r1, x2, y2, z2, r2):
    """Calculate coefficients a,b,c,d of equation of a plane (ax+by+cz+d=0) formed by the
    intersection of two spheres with centers (x1,y1,z1), (x2,y2,z2) and radii r1,r2.
    from: http://ambrnet.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm"""
    a = 2 * (x2 - x1)
    b = 2 * (y2 - y1)
    c = 2 * (z2 - z1)
    d = x1 ** 2 - x2 ** 2 + y1 ** 2 - y2 ** 2 + z1 ** 2 - z2 ** 2 - r1 ** 2 + r2 ** 2
    return a, b, c, d


def center_of_circle_from_sphere_intersection(x1, y1, z1, r1, x2, y2, z2, r2, a, b, c, d):
    """Calculate the center of the circle formed by the intersection of two spheres"""
    # print(a*(x1-x2), b*(y1-y2), c*(z1-z2))
    # print((a*(x1-x2) + b*(y1-y2) +c*(z1-z2)))
    # print((x1*a + y1*b + z1*c + d))
    t = (x1 * a + y1 * b + z1 * c + d) / (a * (x1 - x2) + b * (y1 - y2) + c * (z1 - z2))
    xc = x1 + t * (x2 - x1)
    yc = y1 + t * (y2 - y1)
    zc = z1 + t * (z2 - z1)
    return (xc, yc, zc)


def calc_rho0(mw, conc):
    """Estimate bulk solvent density, rho0, from list of molecular weights
    and molar concentrations of components.
    mw and conc can be lists (nth element of mw corresponds to nth element of concentration)
    mw in g/mol
    concentration in mol/L.
    """
    mw = np.atleast_1d(mw)
    conc = np.atleast_1d(conc)
    return 0.334 * (1 + np.sum(mw * conc * 0.001))


def rotate_coordinates(coordinates, degrees_x=0, degrees_y=0, degrees_z=0):
    # Convert degrees to radians
    radians_x = np.deg2rad(degrees_x)
    radians_y = np.deg2rad(degrees_y)
    radians_z = np.deg2rad(degrees_z)

    # Create rotation object for each axis
    rotation_x = spatial.transform.Rotation.from_euler('x', radians_x)
    rotation_y = spatial.transform.Rotation.from_euler('y', radians_y)
    rotation_z = spatial.transform.Rotation.from_euler('z', radians_z)

    # Apply rotations sequentially
    rotated_coordinates = rotation_z.apply(rotation_y.apply(rotation_x.apply(coordinates)))

    return rotated_coordinates


def regrid_Iq(Iq, qmin=None, qmax=None, nq=None, qc=None, use_sasrec=False, D=None):
    """ Interpolate Iq_calc to desired qgrid.
    qmax - maximum desired q value (e.g. 0.5, optional)
    nq   - number of q points desired (equispaced from qmin to qmax, e.g., 501, optional)
    qc   - desired q grid (takes precedence over qmax/nq)
    use_sasrec - rather than using simple scipy interpolation, use the more accurate (but slower) sasrec for interpolation
    D    - maximum dimension of particle (useful for sasrec, if not given, estimated automatically from the data)
    """
    if qc is None:
        # set some defaults
        if qmax is None:
            qmax = Iq[:,0].max()
        if qmin is None:
            qmin = Iq[:,0].min()
        if nq is None:
            nq = 501
        if qmax > Iq[:,0].max():
            print(f'WARNING: interpolated qmax ({qmax:0.2f}) outside range of input qmax ({Iq[:,0].max():0.2f})')
        qc = np.linspace(qmin, qmax, nq)
    if use_sasrec and D is not None:
        sasrec = Sasrec(Iq, D=D, qc=qc, alpha=0.0, extrapolate=False)
        Iq_calc_interp = np.vstack((sasrec.qc, sasrec.Ic, sasrec.Icerr)).T
    else:
        I_interpolator = interpolate.interp1d(Iq[:, 0], Iq[:, 1], kind='cubic', fill_value='extrapolate')
        Ic = I_interpolator(qc)
        err_interpolator = interpolate.interp1d(Iq[:, 0], Iq[:, 2], kind='cubic', fill_value='extrapolate')
        Ic_err = err_interpolator(qc)
        Iq_calc_interp = np.vstack((qc, Ic, Ic_err)).T
    return Iq_calc_interp


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
                 global_B=None,
                 resolution=None,
                 voxel=None,
                 side=None,
                 nsamples=None,
                 rho0=None,
                 shell_contrast=None,
                 shell_mrcfile=None,
                 shell_type='water',
                 Icalc_interpolation=True,
                 fit_scale=True,
                 fit_offset=False,
                 data_filename=None,
                 data_units="a",
                 n1=None,
                 n2=None,
                 qmin=None,
                 qmax=None,
                 nq=None,
                 penalty_weight=1.0,
                 penalty_weights=[1.0, 0.01],
                 fit_rho0=True,
                 fit_shell=True,
                 fit_all=True,
                 min_method='Nelder-Mead',
                 min_opts='{"adaptive": True}',
                 fast=False,
                 use_sasrec_during_fitting=False,
                 ignore_warnings=False,
                 quiet=False,
                 run_all_on_init=False,
                 logger=None,
                 ):
        self.quiet = quiet
        self.pdb = pdb
        self.ignore_waters = ignore_waters
        self.explicitH = explicitH
        if self.explicitH is None:
            # only use explicitH if H exists in the pdb file
            # for atoms that are not waters
            if 'H' not in self.pdb.atomtype:  # [pdb.resname!="HOH"]:
                self.explicitH = False
            else:
                self.explicitH = True
        if not self.explicitH:
            self.pdb.add_ImplicitH()
            if not self.quiet: print("Implicit hydrogens used")
        # add a line here that will delete alternate conformations if they exist
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
            if not self.quiet: print("Calculating unique atomic volumes...")
            self.pdb.unique_volume = np.zeros(self.pdb.natoms)
            self.pdb.calculate_unique_volume()
        elif self.pdb.unique_volume is None:
            if not self.quiet: print("Looking up unique atomic volumes...")
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
        # calculate some optimal grid values
        self.optimal_side = estimate_side_from_pdb(self.pdb, use_convex_hull=True)
        self.D = self.optimal_side/3 + 2*(1.7+2.8*2)
        self.optimal_voxel = 1.0
        self.optimal_nsamples = np.ceil(self.optimal_side / self.optimal_voxel).astype(int)
        self.nsamples_limit = 256
        self.global_B = global_B
        if self.global_B is None:
            self.limit_global_B = True
        else:
            # if the user explicitly sets global_B at the command line
            # let them do it.
            self.limit_global_B = False
        self.resolution = resolution
        self.voxel = voxel
        self.side = side
        self.nsamples = nsamples
        if rho0 is None:
            self.rho0 = 0.334
        else:
            self.rho0 = rho0
        if shell_contrast is None:
            self.shell_contrast = 0.011
        else:
            self.shell_contrast = shell_contrast
        self.shell_mrcfile = shell_mrcfile
        if shell_type is None:
            self.shell_type = 'water'
        else:
            self.shell_type = shell_type
        self.Icalc_interpolation = Icalc_interpolation
        self.fit_scale = fit_scale
        self.fit_offset = fit_offset
        self.data_filename = data_filename
        self.data_units = data_units
        self.n1 = n1
        self.n2 = n2
        self.qmin4fitting = qmin
        self.qmax4fitting = qmax
        self.nq4prediction = nq
        self.penalty_weight = penalty_weight
        self.penalty_weights = penalty_weights
        self.fit_rho0 = fit_rho0
        self.fit_shell = fit_shell
        self.fit_all = fit_all
        self.fit_params = False  # start with no fitting
        self.min_method = min_method
        self.min_opts = min_opts
        self.param_names = ['rho0', 'shell_contrast']
        self.params = np.array([self.rho0, self.shell_contrast])
        self.params_target = np.copy(self.params)
        self.penalties_initialized = False
        self.fast = fast
        self.use_sasrec_during_fitting = use_sasrec_during_fitting
        self.ignore_warnings = ignore_warnings
        fname_nopath = os.path.basename(self.pdb.filename)
        self.pdb_basename, ext = os.path.splitext(fname_nopath)
        if logger is not None:
            self.logger = logger
        else:
            logging.basicConfig(filename=self.pdb_basename + '.log', level=logging.INFO, filemode='w',
                                format='%(asctime)s %(message)s')  # , datefmt='%Y-%m-%d %I:%M:%S %p')
            self.logger = logging.getLogger()
        self.logger.info('Current Directory: %s', os.getcwd())
        self.logger.info('PDB Filename: %s', self.pdb.filename)
        self.logger.info('Center PDB: %s', self.center_coords)
        self.logger.info('Ignore waters: %s', self.ignore_waters)
        self.logger.info('Excluded volume type: %s', self.exvol_type)
        self.logger.info('Number of atoms: %s' % (self.pdb.natoms))
        types, n_per_type = np.unique(self.pdb.atomtype, return_counts=True)
        for i in range(len(types)):
            self.logger.info('Number of %s atoms: %s' % (types[i], n_per_type[i]))
        self.logger.info('Use atomic B-factors: %s', self.use_b)
        self.logger.info('Use explicit Hydrogens: %s', self.explicitH)

        if run_all_on_init:
            self.run_all()

    def run_all(self):
        """Run all necessary steps to generate density maps, structure factors, and scattering profile using current settings."""
        self.scale_radii()
        self.make_grids()
        self.calculate_global_B()
        self.calculate_invacuo_density()
        self.calculate_excluded_volume()
        self.calculate_hydration_shell()
        self.calculate_structure_factors()
        self.calc_I_with_modified_params(self.params)
        self.calc_rho_with_modified_params(self.params)

    def scale_radii(self, radii_sf=None):
        """Scale all the modifiable atom type radii in the pdb"""
        if self.pdb.radius is None:
            self.pdb.radius = np.copy(self.pdb.unique_radius)
        if radii_sf is None:
            radii_sf = self.radii_sf
        for i in range(len(self.modifiable_atom_types)):
            if not self.explicitH:
                if self.modifiable_atom_types[i] == 'H':
                    self.pdb.exvolHradius = radii_sf[i] * self.pdb.unique_exvolHradius
                else:
                    self.pdb.radius[self.pdb.atomtype == self.modifiable_atom_types[i]] = radii_sf[i] * \
                                                                                          self.pdb.unique_radius[
                                                                                              self.pdb.atomtype ==
                                                                                              self.modifiable_atom_types[
                                                                                                  i]]
            else:
                self.pdb.exvolHradius = np.zeros(self.pdb.natoms)
                self.pdb.radius[self.pdb.atomtype == self.modifiable_atom_types[i]] = radii_sf[i] * \
                                                                                      self.pdb.unique_radius[
                                                                                          self.pdb.atomtype ==
                                                                                          self.modifiable_atom_types[i]]

    def set_radii(self, atom_types, radii):
        """For each atom type in atom_types, set its value to corresponding radius in radii."""
        if self.pdb.radius is None:
            self.pdb.radius = np.ones(self.pdb.natoms)
        for i in range(len(atom_types)):
            self.pdb.exvolHradius = np.zeros(self.pdb.natoms)
            self.pdb.radius[self.pdb.atomtype == atom_types[i]] = radii[i]

    def calculate_average_radii(self):
        self.mean_radius = np.ones(len(self.modifiable_atom_types))
        for i in range(len(self.modifiable_atom_types)):
            # try using a scale factor for radii instead
            if self.modifiable_atom_types[i] == 'H' and not self.explicitH:
                self.mean_radius[i] = self.pdb.exvolHradius[i]
            else:
                self.mean_radius[i] = self.pdb.radius[self.pdb.atomtype == self.modifiable_atom_types[i]].mean()
        self.logger.info("Calculated average radii:")
        for i in range(len(self.modifiable_atom_types)):
            self.logger.info("%s: %.3f" % (self.modifiable_atom_types[i], self.mean_radius[i]))

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
            # if v, n, s are all given, side and nsamples dominates
            side = side
            nsamples = nsamples
            voxel = side / nsamples
        elif voxel is not None and nsamples is not None and side is None:
            # if v and n given, voxel and nsamples dominates
            voxel = voxel
            nsamples = nsamples
            side = voxel * nsamples
        elif voxel is not None and nsamples is None and side is not None:
            # if v and s are given, adjust voxel to match nearest integer value of n
            voxel = voxel
            side = side
            nsamples = np.ceil(side / voxel).astype(int)
            voxel = side / nsamples
        elif voxel is not None and nsamples is None and side is None:
            # if v is given, voxel thus dominates, so estimate side, calculate nsamples.
            voxel = voxel
            nsamples = np.ceil(optimal_side / voxel).astype(int)
            side = voxel * nsamples
            # if n > 256, adjust side length
            if nsamples > nsamples_limit:
                nsamples = nsamples_limit
                side = voxel * nsamples
        elif voxel is None and nsamples is not None and side is not None:
            # if n and s are given, set voxel size based on those
            nsamples = nsamples
            side = side
            voxel = side / nsamples
        elif voxel is None and nsamples is not None and side is None:
            # if n is given, set side, adjust voxel.
            nsamples = nsamples
            side = optimal_side
            voxel = side / nsamples
        elif voxel is None and nsamples is None and side is not None:
            # if s is given, set voxel, adjust nsamples, reset voxel if necessary
            side = side
            voxel = optimal_voxel
            nsamples = np.ceil(side / voxel).astype(int)
            if nsamples > nsamples_limit:
                nsamples = nsamples_limit
            voxel = side / nsamples
        elif voxel is None and nsamples is None and side is None:
            # if none given, set side and voxel, adjust nsamples, reset voxel if necessary
            side = optimal_side
            voxel = optimal_voxel
            nsamples = np.ceil(side / voxel).astype(int)
            if nsamples > nsamples_limit:
                nsamples = nsamples_limit
            voxel = side / nsamples

        if not self.ignore_warnings:
            # make some warnings for certain cases
            side_small_warning = """
                Side length may be too small and may result in undersampling errors."""
            side_way_too_small_warning = """
                Disabling interpolation of I_calc due to severe undersampling."""
            voxel_big_warning = """
                Voxel size is greater than 1 A. This may lead to less accurate I(q) estimates at high q."""
            nsamples_warning = """
                To avoid long computation times and excessive memory requirements, the number of voxels
                has been limited to {n:d} and the voxel size has been set to {v:.2f},
                which may be too large and lead to less accurate I(q) estimates at high q.""".format(v=voxel,
                                                                                                     n=nsamples)
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
            if side < 2 / 3 * optimal_side:
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

        # make the real space grid
        halfside = side / 2
        n = int(side / voxel)
        # want n to be even for speed/memory optimization with the FFT,
        # ideally a power of 2, but wont enforce that
        if n % 2 == 1: n += 1
        dx = side / n
        dV = dx ** 3
        x_ = np.linspace(-(n // 2) * dx, (n // 2 - 1) * dx, n)
        x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
        xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        # make the reciprocal space grid
        df = 1 / side
        qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
        qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
        qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
        qmax = np.max(qr)
        qstep = np.min(qr[qr > 0]) - 1e-8
        nbins = int(qmax / qstep)
        qbins = np.linspace(0, nbins * qstep, nbins + 1)
        # create an array labeling each voxel according to which qbin it belongs
        qbin_labels = np.searchsorted(qbins, qr, "right")
        qbin_labels -= 1
        qblravel = qbin_labels.ravel()
        xcount = np.bincount(qblravel)  # the number of voxels in each q bin
        # create modified qbins and put qbins in center of bin rather than at left edge of bin.
        qbinsc = mybinmean(qr.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
        q_calc = np.copy(qbinsc)

        # make attributes for all that
        self.halfside = halfside
        self.side = side
        self.n = n
        self.dx = dx
        self.dx = dx
        self.dV = dV
        self.x_ = x_
        self.x, self.y, self.z = x, y, z
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
        # generate a scale factor for each q bin to scale the uncertainties by,
        # based on the relative number of voxels in the bin (sqrt)
        self.xcount_factor = np.sqrt(xcount.max())/np.sqrt(xcount)

        # save an array of indices containing only desired q range for speed
        qmax4calc = self.qx_.max() * 1.1
        self.qidx = np.where((self.qr <= qmax4calc))

        self.logger.info('Optimal Side length: %.2f', self.optimal_side)
        self.logger.info('Optimal N samples:   %d', self.optimal_nsamples)
        self.logger.info('Optimal Voxel size:  %.4f', self.optimal_voxel)
        self.logger.info('Actual  Side length: %.2f', self.side)
        self.logger.info('Actual  N samples:   %d', self.n)
        self.logger.info('Actual  Voxel size:  %.4f', self.dx)

    def calculate_global_B(self):
        if self.dx is None:
            # if make_grids has not been run yet, run it
            self.make_grids()
        if self.global_B is None:
            self.global_B = u2B(0.25 * self.dx)  # this helps with voxel sampling issues
        else:
            self.global_B = self.global_B
        # greater than 2A causes problems
        if self.limit_global_B and (B2u(self.global_B) > 1.5):
            self.global_B = u2B(1.5)

        self.logger.info('Global B-factor:  %.4f', self.global_B)

    def calculate_invacuo_density(self):
        if not self.quiet: print('Calculating in vacuo density...')
        self.rho_invacuo, self.support = pdb2map_multigauss(self.pdb,
                                                            x=self.x, y=self.y, z=self.z,
                                                            global_B=self.global_B,
                                                            use_b=self.use_b,
                                                            ignore_waters=self.ignore_waters)
        if not self.quiet: print('Finished in vacuo density.')

    def calculate_excluded_volume(self, quiet=False):
        if not self.quiet: print('Calculating excluded volume...')
        if self.exvol_type == "gaussian":
            # generate excluded volume assuming gaussian dummy atoms
            # this function outputs in electron count units
            self.rho_exvol, self.supportexvol = pdb2map_simple_gauss_by_radius(self.pdb,
                                                                               self.x, self.y, self.z,
                                                                               rho0=self.rho0,
                                                                               ignore_waters=self.ignore_waters,
                                                                               )  # global_B=self.global_B)
        elif self.exvol_type == "flat":
            # generate excluded volume assuming flat solvent
            if not self.explicitH:
                v = 4 * np.pi / 3 * self.pdb.vdW ** 3 + self.pdb.numH * 4 / 3 * np.pi * vdW['H'] ** 3
                radjusted = sphere_radius_from_volume(v)
            else:
                radjusted = self.pdb.vdW
            self.supportexvol = pdb2support_fast(self.pdb, self.x, self.y, self.z, radius=radjusted,
                                                 probe=B2u(self.pdb.b))
            # estimate excluded volume electron count based on unique volumes of atoms
            v = np.sum(4 / 3 * np.pi * self.pdb.radius ** 3)
            ne = v * self.rho0
            # blur the exvol to have gaussian-like edges
            sigma = 1.0 / self.dx  # best exvol sigma to match water molecule exvol thing is 1 A
            # self.rho_exvol = ndimage.gaussian_filter(self.supportexvol*1.0,sigma=sigma,mode='wrap')
            self.rho_exvol = 1.0 * self.supportexvol
            self.rho_exvol *= ne / self.rho_exvol.sum()  # put in electron count units
        if not self.quiet: print('Finished excluded volume.')

    def calculate_hydration_shell(self):
        if not self.quiet: print('Calculating hydration shell...')
        self.r_water = r_water = 1.4
        uniform_shell = calc_uniform_shell(self.pdb, self.x, self.y, self.z, thickness=self.r_water * 2,
                                           distance=self.r_water)
        self.water_shell_idx = water_shell_idx = uniform_shell.astype(bool)

        if self.dx > 2 * self.r_water and self.shell_type == "water":
            print("Voxel size too large for water form factor hydration shell. Changing shell type to uniform.")
            self.shell_type = "uniform"

        if self.shell_mrcfile is not None:
            # allow user to provide mrc filename to read in a custom shell
            rho_shell, sidex = read_mrc(self.shell_mrcfile)
            rho_shell *= self.dV  # assume mrc file is in units of density, convert to electron count
            if not np.isclose(sidex, self.side, rtol=1e-3, atol=1e-3) or (rho_shell.shape[0] != self.x.shape[0]):
                print("Error: shell_mrcfile does not match grid.")
                print("Use denss_mrcops.py to resample onto the desired grid.")
                exit()
        elif self.shell_type == "water":
            # the default is water type shell
            # make two supports, one for the outer half of the shell, one for the inner half of the shell
            # grab all voxels outside the protein surface plus a water radius
            # subtract a half voxel because we will dilate a inner shell by one voxel, so that we have
            # at least one voxel that is the center of the water shell at distance zero, so that puts
            # the center of the water shell halfway between the inner and outer halves of the shell
            protein_idx = pdb2support_fast(self.pdb, self.x, self.y, self.z, radius=self.pdb.vdW, probe=0)
            protein_rw_idx = pdb2support_fast(self.pdb, self.x, self.y, self.z, radius=self.pdb.vdW,
                                              probe=self.r_water - self.dx / 2)
            if not self.quiet: print('Calculating dist transform...')
            # calculate the distance of each voxel outside the particle to the surface of the protein+water support
            dist1 = np.zeros(self.x.shape)
            dist1 = ndimage.distance_transform_edt(protein_rw_idx)
            # now calculate the distance of each voxel inside the protein+water support by inverting it,
            # but first add one voxel
            protein_rw_idx2 = ndimage.binary_dilation(protein_rw_idx)
            dist2 = np.zeros(self.x.shape)
            dist2 = ndimage.distance_transform_edt(~protein_rw_idx2)
            # now merge the distances of the outer voxels and the inner voxels
            dist = dist1 + dist2
            # convert dist from pixels to angstroms
            dist *= self.dx
            # for form factor calculation, look at only the voxels near the shell for efficiency
            rho_shell = np.zeros(self.x.shape)
            if not self.quiet: print('Calculating shell values...')
            shell_idx = np.where(dist < 2 * r_water)
            shell_idx_bool = np.zeros(self.x.shape, dtype=bool)
            shell_idx_bool[shell_idx] = True
            if shell_idx_bool.sum() == 0:
                shell_idx_bool = np.ones(self.x.shape, dtype=bool)
            rho_shell[shell_idx_bool] = realspace_formfactor(element='HOH', r=dist[shell_idx_bool],
                                                             B=u2B(0.25) + self.global_B)
            # estimate initial shell scale based on contrast using mean density
            shell_mean_density = np.mean(rho_shell[water_shell_idx]) / self.dV
            # scale the mean density of the invacuo shell to match the desired mean density
            rho_shell *= self.shell_contrast / shell_mean_density
            # shell should still be in electron count units
        elif self.shell_type == "uniform":
            rho_shell = water_shell_idx * (self.shell_contrast)
            rho_shell *= self.dV  # convert to electron units
        else:
            print("Error: no valid shell_type given (water or uniform). Disabling hydration shell.")
            rho_shell = self.x * 0.0
        self.rho_shell = rho_shell
        if not self.quiet: print('Finished hydration shell.')

    def calculate_structure_factors(self):
        if not self.quiet: print('Calculating structure factors...')
        # F_invacuo
        self.F_invacuo = myfftn(self.rho_invacuo)

        # perform B-factor sharpening to correct for B-factor sampling workaround
        Bsharp = -self.global_B
        Bsharp3D = np.exp(-(Bsharp) * (self.qr / (4 * np.pi)) ** 2)
        self.F_invacuo *= Bsharp3D

        # exvol F_exvol
        self.F_exvol = myfftn(self.rho_exvol)
        # self.F_exvol *= Bsharp3D

        # shell invacuo F_shell
        self.F_shell = myfftn(self.rho_shell)

        if not self.quiet: print('Finished structure factors.')

    def load_data(self, filename=None, Iq_exp=None, units=None):
        if not self.quiet: print('Loading data...')
        if (filename is None and self.data_filename is None) and (Iq_exp is None):
            print("ERROR: No data filename or array given.")
        elif filename is not None:
            fn = filename
            self.data_filename = filename
        elif self.data_filename is not None:
            fn = self.data_filename
        else:
            fn = None

        if units is not None:
            self.data_units = units

        if fn is not None:
            Iq_exp = np.genfromtxt(fn, invalid_raise=False, usecols=(0, 1, 2))

        if Iq_exp is not None:
            if len(Iq_exp.shape) < 2:
                print("Invalid data format. Data must have 3 columns: q, I, errors.")
                exit()
            if Iq_exp.shape[1] < 3:
                print("Not enough columns (Data must have 3 columns: q, I, errors).")
                exit()
            Iq_exp = Iq_exp[~np.isnan(Iq_exp).any(axis=1)]
            # get rid of any data points equal to zero in the intensities or errors columns
            idx = np.where((Iq_exp[:, 1] != 0) & (Iq_exp[:, 2] != 0))
            Iq_exp = Iq_exp[idx]
            if self.data_units == "nm":
                Iq_exp[:, 0] *= 0.1
            Iq_exp_orig = np.copy(Iq_exp)
            if self.n1 is None:
                self.n1 = 0
            if self.n2 is None:
                self.n2 = len(Iq_exp[:, 0])
            if self.qmin4fitting is not None:
                # determine n1 value associated with qmin4fitting
                self.n1 = find_nearest_i(Iq_exp[:, 0], self.qmin4fitting)
            if self.qmax4fitting is not None:
                # determine n2 value associated with qmax4fitting
                self.n2 = find_nearest_i(Iq_exp[:, 0], self.qmax4fitting)
            self.Iq_exp = Iq_exp[self.n1:self.n2]
            self.q_exp = self.Iq_exp[:, 0]
            self.I_exp = self.Iq_exp[:, 1]
            self.sigq_exp = self.Iq_exp[:, 2]
        else:
            self.Iq_exp = None
            self.q_exp = None
            self.I_exp = None
            self.sigq_exp = None
            self.fit_params = False

        # save an array of indices containing only desired q range for speed
        if self.data_filename:
            self.qmax4calc = self.q_exp.max() * 1.1
        else:
            self.qmax4calc = self.qx_.max() * 1.1
        self.qidx = np.where((self.qr <= self.qmax4calc))
        # if the voxel size of the map is too large for the data to sample, limit the data
        if self.q_exp.max() > self.qx_.max():
            idx = np.where(self.q_exp < self.qx_.max())
            self.q_exp = self.q_exp[idx]
            self.I_exp = self.I_exp[idx]
            self.sigq_exp = self.sigq_exp[idx]
            self.Iq_exp = self.Iq_exp[idx]
        if not self.quiet: print('Data loaded.')

        self.logger.info('Data filename: %s', self.data_filename)
        self.logger.info('First data point: %s', self.n1)
        self.logger.info('Last data point: %s', self.n2)

    def initialize_penalties(self, penalty_weight=None):
        """Initialize penalty weights more efficiently by avoiding redundant calculations"""
        # Set target parameters and temporary zero penalty
        self.params_target = self.params
        self.penalty_weight = 0.0

        # We only need to calculate intensities once
        # Call calc_score directly, which will call calc_I internally
        self.calc_score_with_modified_params(self.params)
        chi2_nofit = self.chi2

        # Set the actual penalty weight
        if penalty_weight is None:
            self.penalty_weight = chi2_nofit * 100.0
        else:
            self.penalty_weight = penalty_weight

        self.penalties_initialized = True

    def minimize_parameters(self, fit_radii=False):
        if not self.penalties_initialized:
            self.initialize_penalties()

        if fit_radii:
            self.param_names += self.modifiable_atom_types

        # generate a set of bounds
        self.bounds = np.zeros((len(self.param_names), 2))

        # don't bother fitting if none is requested (default)
        if self.fit_rho0:
            self.bounds[0, 0] = 0
            self.bounds[0, 1] = np.inf
            self.fit_params = True
        else:
            self.bounds[0, 0] = self.rho0
            self.bounds[0, 1] = self.rho0
        if self.fit_shell:
            self.bounds[1, 0] = -np.inf
            self.bounds[1, 1] = np.inf
            self.fit_params = True
        else:
            self.bounds[1, 0] = self.shell_contrast
            self.bounds[1, 1] = self.shell_contrast

        if fit_radii:
            # radii_sf, i.e. radii scale factors
            self.bounds[2:, 0] = 0
            self.bounds[2:, 1] = np.inf
            self.params = np.append(self.params, self.radii_sf)
            self.penalty_weights = np.append(self.penalty_weights, np.ones(len(self.param_names[2:])))

        if not self.fit_all:
            # disable all fitting if requested
            self.fit_params = False

        self.params_guess = self.params
        self.params_target = self.params_guess

        if self.fit_params:
            if not self.quiet: print('Optimizing parameters...')
            # self.logger.info('Optimizing parameters...')
            if self.penalty_weight != 0:
                if not self.quiet:
                    print(["scale_factor"], self.param_names, ["penalty"], ["chi2"])
                    print("-" * 100)
            else:
                if not self.quiet:
                    print(["scale_factor"], self.param_names, ["chi2"])
                    print("-" * 100)
            results = optimize.minimize(self.calc_score_with_modified_params, self.params_guess,
                                        bounds=self.bounds,
                                        method=self.min_method,
                                        options=eval(self.min_opts),
                                        # method='L-BFGS-B', options={'eps':0.001},
                                        )
            self.optimized_params = results.x
            self.optimized_chi2 = results.fun
            if not self.quiet: print('Finished minimizing parameters.')
        else:
            self.calc_score_with_modified_params(self.params)
            self.optimized_params = self.params_guess
            self.optimized_chi2 = self.chi2
        self.params = np.copy(self.optimized_params)

        self.logger.info('Final Parameter Values:')
        for i in range(len(self.params)):
            self.logger.info("%s : %.5e" % (self.param_names[i], self.params[i]))

    def calc_chi2(self):
        self.optimized_chi2, self.exp_scale_factor, self.offset, self.fit = calc_chi2(self.Iq_exp, self.Iq_calc,
                                                                                      interpolation=self.Icalc_interpolation,
                                                                                      scale=self.fit_scale,
                                                                                      offset=self.fit_offset,
                                                                                      return_sf=True, return_fit=True,
                                                                                      use_sasrec=True, D=self.D)
        self.logger.info("Scale factor: %.5e " % self.exp_scale_factor)
        self.logger.info("Offset: %.5e " % self.offset)
        self.logger.info("chi2 of fit:  %.5e " % self.optimized_chi2)

    def calc_score_with_modified_params(self, params):
        self.calc_I_with_modified_params(params)
        # sasrec is slow, so don't use it for fitting by default, but allow it by the user
        # but we will still use it for final output profile calculation
        if self.use_sasrec_during_fitting:
            self.chi2, self.exp_scale_factor = calc_chi2(self.Iq_exp, self.Iq_calc,
                                                         scale=self.fit_scale,
                                                         offset=self.fit_offset,
                                                         interpolation=self.Icalc_interpolation,
                                                         return_sf=True,
                                                         use_sasrec=True, D=self.D)
        else:
            self.chi2, self.exp_scale_factor = calc_chi2(self.Iq_exp, self.Iq_calc,
                                                         scale=self.fit_scale,
                                                         offset=self.fit_offset,
                                                         interpolation=self.Icalc_interpolation,
                                                         return_sf=True)
        self.calc_penalty(params)
        self.score = self.chi2 + self.penalty
        if self.fit_params:
            if self.penalty_weight != 0:
                # include printing of penalties if that option is give
                if not self.quiet:
                    print("%.5e" % self.exp_scale_factor, ' '.join("%.5e" % param for param in params),
                          "%.3f" % self.penalty, "%.3f" % self.chi2)
            else:
                if not self.quiet:
                    print("%.5e" % self.exp_scale_factor, ' '.join("%.5e" % param for param in params),
                          "%.3f" % self.chi2)
        return self.score

    def calc_penalty(self, params):
        """Calculates a penalty using quadratic loss function
        for parameters dependent on a target value for each parameter.
        """
        nparams = len(params)
        params_weights = np.ones(nparams)  # note, different than penalty_weights
        params_target = self.params_target
        penalty_weight = self.penalty_weight
        penalty_weights = self.penalty_weights
        # set the individual parameter penalty weights
        # to be 1/params_target, so that each penalty
        # is weighted as a fraction of the target rather than an
        # absolute number.
        for i in range(nparams):
            if params_target[i] != 0:
                params_weights[i] = 1 / params_target[i]
        # multiply each weight by the desired individual penalty weight
        if penalty_weights is not None:
            params_weights *= penalty_weights
        # use quadratic loss function
        penalty = 1 / nparams * np.sum((params_weights * (params - params_target)) ** 2)
        penalty *= penalty_weight
        self.penalty = penalty

    def calc_F_with_modified_params(self, params, full_qr=False):
        """Calculates structure factor sum from set of parameters with optimized performance"""
        # Calculate scaling factors
        sf_ex = params[0] / self.rho0 if self.rho0 != 0 else 1.0
        sf_sh = params[1] / self.shell_contrast if self.shell_contrast != 0 else 1.0

        # Ensure F array exists
        if not hasattr(self, 'F') or self.F is None:
            self.F = np.zeros_like(self.F_invacuo)

        if full_qr:
            # For full q-range, use direct assignment
            self.F = self.F_invacuo - sf_ex * self.F_exvol + sf_sh * self.F_shell
        else:
            # Fall back to standard calculation
            self.F[self.qidx] = self.F_invacuo[self.qidx] - sf_ex * self.F_exvol[self.qidx] + sf_sh * self.F_shell[
                self.qidx]

    def calc_I_with_modified_params(self, params):
        """Calculates intensity profile for optimization of parameters"""
        if len(params) > 2:
            # Handle radii scaling
            self.scale_radii(radii_sf=params[2:])
            self.calculate_excluded_volume(quiet=True)
            self.F_exvol = myfftn(self.rho_exvol)

        # Calculate structure factors
        self.calc_F_with_modified_params(params)

        # Calculate intensities
        if not hasattr(self, 'I3D') or self.I3D is None:
            self.I3D = np.empty_like(self.F, dtype=np.float64)
        abs2_fast(self.F, out=self.I3D)

        # Bin intensities
        if not hasattr(self, 'I_calc') or self.I_calc is None or len(self.I_calc) != len(self.xcount):
            self.I_calc = np.empty(len(self.xcount), dtype=np.float64)
        mybinmean_optimized(self.I3D.ravel(), self.qblravel, xcount=self.xcount, out=self.I_calc)

        # Calculate errors and create output array
        errors = self.I_calc[0] * 0.002 + self.I_calc ** 0.5 * 0.01
        errors *= self.xcount_factor
        self.Iq_calc = np.vstack((self.qbinsc, self.I_calc, errors)).T

        # Calculate Rg and I0
        self.Rg = calc_rg_by_guinier_first_2_points(self.q_calc, self.I_calc)
        self.I0 = self.I_calc[0]

    def calc_rho_with_modified_params(self, params):
        """Calculates electron density map for protein in solution. Includes the excluded volume and
        hydration shell calculations."""
        if self.rho0 != 0:
            sf_ex = params[0] / self.rho0
        else:
            sf_ex = 1.0
        # sf_sh is ratio of params[1] to initial shell_contrast
        if self.shell_contrast != 0:
            sf_sh = params[1] / self.shell_contrast
        else:
            sf_sh = 1.0
        self.sf_ex = sf_ex
        self.sf_sh = sf_sh
        self.rho_insolvent = self.rho_invacuo - sf_ex * self.rho_exvol + sf_sh * self.rho_shell

    def calculate_excluded_volume_in_A3(self):
        """Calculate the excluded volume of the particle in angstroms cubed."""
        self.exvol_in_A3 = np.sum(sphere_volume_from_radius(self.pdb.radius)) + np.sum(
            sphere_volume_from_radius(self.pdb.exvolHradius) * self.pdb.numH)
        self.logger.info("Calculated excluded volume: %.2f" % (self.exvol_in_A3))

    def save_Iq_calc(self, prefix=None, qmax=None, nq=None, qc=None, use_sasrec=True):
        """Save the calculated Iq curve to a .dat file."""
        header = ' '.join('%s: %.5e ; ' % (self.param_names[i], self.params[i]) for i in range(len(self.params)))
        # store Dmax in the header, which is helpful for other tasks
        header += f"\nDmax  = {self.D:.5e}"
        header_dat = header + "\nq_calc I_calc err_calc"
        if prefix is None:
            prefix = self.pdb_basename
        if qmax is not None or nq is not None or qc is not None or use_sasrec:
            self.Iq_calc = regrid_Iq(Iq=self.Iq_calc, qmax=qmax, nq=nq, qc=qc, use_sasrec=use_sasrec, D=self.D)
        if qmax is None:
            qmax = self.qx_.max() - 1e-8
            # only write out values less than the edge of the box
        Iq_calc = self.Iq_calc[self.Iq_calc[:,0] <= qmax]
        np.savetxt(prefix + '.dat', Iq_calc, delimiter=' ', fmt='%.8e', header=header_dat)

    def save_fit(self, prefix=None):
        """Save the combined experimental and calculated Iq curve to a .fit file."""
        if self.fit is not None:
            header = ' '.join('%s: %.5e ; ' % (self.param_names[i], self.params[i]) for i in range(len(self.params)))
            # store Dmax in the header, which is helpful for other tasks
            header += f"\nDmax  = {self.D:.5e}"
            header_fit = header + '\n q, I, error, fit ; chi2= %.3e' % (self.chi2 if self.chi2 is not None else 0)
            if prefix is None:
                prefix = self.pdb_basename
            np.savetxt(prefix + '.fit', self.fit, delimiter=' ', fmt='%.8e', header=header_fit)
        else:
            print("No fit exists. First calculate a fit with pdb2mrc.calc_chi2()")


class PDB2SAS(object):
    """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.

    pdb - a saxstats PDB file object.
    q - q values to use for calculations (optional).
    """

    def __init__(self, pdb, q=None, numba=True):
        self.pdb = pdb
        if q is None:
            q = np.linspace(0, 0.5, 101)
        self.q = q
        # self.calc_I()

    def calc_form_factors(self, B=0.0):
        """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.

        B - B-factors (i.e. Debye-Waller/temperature factors) of atoms (default=0.0)
        """
        B = np.atleast_1d(B)
        if B.shape[0] == 1:
            B = np.ones(self.pdb.natoms) * B
        self.ff = np.zeros((self.pdb.natoms, len(self.q)))
        for i in range(self.pdb.natoms):
            try:
                self.ff[i, :] = formfactor(self.pdb.atomtype[i], q=self.q, B=B[i])
            except Exception as e:
                print("pdb.atomtype unknown for atom %d" % i)
                print("attempting to use pdb.atomname instead")
                print(e)
                try:
                    self.ff[i, :] = formfactor(self.pdb.atomname[i][0], q=self.q, B=B[i])
                except Exception as e:
                    print("pdb.atomname unknown for atom %d" % i)
                    print("Defaulting to Carbon form factor.")
                    print(e)
                    self.ff[i, :] = formfactor("C", q=self.q, B=B[i])

    def calc_debye(self, natoms_limit=1000):
        """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
        """
        if self.pdb.natoms > natoms_limit:
            print(
                "Error: Too many atoms. This function is not suitable for large macromolecules over %i atoms" % natoms_limit)
            # if natoms is too large, sinc lookup table has huge memory requirements
        else:
            if self.pdb.rij is None:
                self.pdb.calculate_distance_matrix(return_squareform=True)
            s = np.sinc(self.q * self.pdb.rij[..., None] / np.pi)
            self.I = np.einsum('iq,jq,ijq->q', self.ff, self.ff, s)

    def calc_I(self, numba=True):
        self.calc_form_factors()
        if numba:
            try:
                # try numba function first
                self.I = calc_debye_numba(self.pdb.coords, self.q, self.ff)
            except:
                print("numba failed. Calculating debye slowly.")
                self.calc_debye()
        else:
            self.calc_debye()


if HAS_NUMBA:
    @nb.njit(fastmath=True, parallel=True, error_model="numpy", cache=True)
    def numba_cdist(A, B):
        assert A.shape[1] == B.shape[1]
        C = np.empty((A.shape[0], B.shape[0]), A.dtype)

        # workaround to get the right datatype for acc
        init_val_arr = np.zeros(1, A.dtype)
        init_val = init_val_arr[0]

        for i in nb.prange(A.shape[0]):
            for j in range(B.shape[0]):
                acc = init_val
                for k in range(A.shape[1]):
                    acc += (A[i, k] - B[j, k]) ** 2
                C[i, j] = np.sqrt(acc)
        return C


    @nb.njit(fastmath=True, parallel=True, error_model="numpy", cache=True)
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
            acc = 0
            for ri in nb.prange(nr):
                for rj in nb.prange(nr):
                    # loop through all atoms, and add the debye formulism for the pair
                    qri_j = q[qi] * np.linalg.norm(coords[ri] - coords[rj])
                    if qri_j != 0:
                        s = np.sin(qri_j) / (qri_j)
                    else:
                        s = 1.0
                    acc += ff_T[qi, ri] * ff_T[qi, rj] * s
            I[qi] = acc
        return I


def pdb2map_simple_gauss_by_radius(pdb, x, y, z, cutoff=3.0, global_B=None, rho0=0.334, ignore_waters=True):
    """Simple isotropic single gaussian sum at coordinate locations.

    This function only calculates the values at
    grid points near the atom for speed.

    pdb - instance of PDB class (required, must have pdb.radius attribute)
    x,y,z - meshgrids for x, y, and z (required)
    cutoff - maximum distance from atom to calculate density
    rho0 - average bulk solvent density used for excluded volume estimation (0.334 for water)
    """
    side = x[-1, 0, 0] - x[0, 0, 0]
    halfside = side / 2
    n = x.shape[0]
    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = x[:, 0, 0]
    shift = 0  # np.ones(3)*dx/2.
    # print("\n Calculate density map from PDB... ")
    values = np.zeros(x.shape)
    support = np.zeros(x.shape, dtype=bool)
    if global_B is None:
        global_B = 0.0
    B = global_B * np.ones(pdb.natoms)
    cutoffs = 2 * pdb.vdW
    gxmin = x.min()
    gxmax = x.max()
    gymin = y.min()
    gymax = y.max()
    gzmin = z.min()
    gzmax = z.max()
    for i in range(pdb.coords.shape[0]):
        if ignore_waters and pdb.resname[i] == "HOH":
            continue
        if rho0 == 0:
            continue
        # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        # sys.stdout.flush()
        # this will cut out the grid points that are near the atom
        # first, get the min and max distances for each dimension
        # also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i]  # for convenience, store up x,y,z coordinates of atom
        # ignore atoms whose coordinates are outside the box limits
        if (
                (xa < gxmin) or
                (xa > gxmax) or
                (ya < gymin) or
                (ya > gymax) or
                (za < gzmin) or
                (za > gzmax)
        ):
            # print()
            print("Atom %d outside boundary of cell ignored." % i)
            continue
        cutoff = cutoffs[i]
        xmin = int(np.floor((xa - cutoff) / dx)) + n // 2
        xmax = int(np.ceil((xa + cutoff) / dx)) + n // 2
        ymin = int(np.floor((ya - cutoff) / dx)) + n // 2
        ymax = int(np.ceil((ya + cutoff) / dx)) + n // 2
        zmin = int(np.floor((za - cutoff) / dx)) + n // 2
        zmax = int(np.ceil((za + cutoff) / dx)) + n // 2
        # handle edges
        xmin = max([xmin, 0])
        xmax = min([xmax, n])
        ymin = max([ymin, 0])
        ymax = min([ymax, n])
        zmin = max([zmin, 0])
        zmax = min([zmax, n])
        # now lets create a slice object for convenience
        slc = np.s_[xmin:xmax, ymin:ymax, zmin:zmax]
        nx = xmax - xmin
        ny = ymax - ymin
        nz = zmax - zmin
        # now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(), y[slc].ravel(), z[slc].ravel()))
        dist = spatial.distance.cdist(pdb.coords[None, i] - shift, xyz)

        V = sphere_volume_from_radius(pdb.radius[i])
        VoneH = sphere_volume_from_radius(pdb.exvolHradius[i])
        VallH = pdb.numH[i] * VoneH
        Vtot = V + VallH
        V = Vtot
        tmpvalues = realspace_gaussian_formfactor(r=dist, rho0=rho0, V=V, radius=None, B=B[i])

        # rescale total number of electrons by expected number of electrons
        if np.sum(tmpvalues) > 1e-8:
            ne_total = rho0 * V
            tmpvalues *= ne_total / np.sum(tmpvalues)

        values[slc] += tmpvalues.reshape(nx, ny, nz)
        support[slc] = True
    return values, support


def pdb2map_multigauss(pdb, x, y, z, cutoff=3.0, global_B=None, use_b=False, ignore_waters=True):
    """5-term gaussian sum at coordinate locations using Cromer-Mann coefficients.

    This function only calculates the values at
    grid points near the atom for speed.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    cutoff - maximum distance from atom to calculate density
    global_B - desired resolution of density map, calculated as a B-factor
    corresonding to atomic displacement equal to resolution.
    """
    side = x[-1, 0, 0] - x[0, 0, 0]
    halfside = side / 2
    n = x.shape[0]
    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = x[:, 0, 0]
    shift = 0  # np.ones(3)*dx/2.
    # print("\n Calculate density map from PDB... ")
    values = np.zeros(x.shape)
    support = np.zeros(x.shape, dtype=bool)
    if global_B is None:
        global_B = 0.0
    cutoff = max(cutoff, 2 * B2u(global_B))
    if use_b:
        B = global_B + pdb.b
    else:
        B = global_B + pdb.b * 0
    gxmin = x.min()
    gxmax = x.max()
    gymin = y.min()
    gymax = y.max()
    gzmin = z.min()
    gzmax = z.max()
    for i in range(pdb.coords.shape[0]):
        if ignore_waters and pdb.resname[i] == "HOH":
            continue
        # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        # sys.stdout.flush()
        # this will cut out the grid points that are near the atom
        # first, get the min and max distances for each dimension
        # also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i]  # for convenience, store up x,y,z coordinates of atom
        # ignore atoms whose coordinates are outside the box limits
        if (
                (xa < gxmin) or
                (xa > gxmax) or
                (ya < gymin) or
                (ya > gymax) or
                (za < gzmin) or
                (za > gzmax)
        ):
            # print()
            print("Atom %d outside boundary of cell ignored." % i)
            continue
        xmin = int(np.floor((xa - cutoff) / dx)) + n // 2
        xmax = int(np.ceil((xa + cutoff) / dx)) + n // 2
        ymin = int(np.floor((ya - cutoff) / dx)) + n // 2
        ymax = int(np.ceil((ya + cutoff) / dx)) + n // 2
        zmin = int(np.floor((za - cutoff) / dx)) + n // 2
        zmax = int(np.ceil((za + cutoff) / dx)) + n // 2
        # handle edges
        xmin = max([xmin, 0])
        xmax = min([xmax, n])
        ymin = max([ymin, 0])
        ymax = min([ymax, n])
        zmin = max([zmin, 0])
        zmax = min([zmax, n])
        # now lets create a slice object for convenience
        slc = np.s_[xmin:xmax, ymin:ymax, zmin:zmax]
        nx = xmax - xmin
        ny = ymax - ymin
        nz = zmax - zmin
        # now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(), y[slc].ravel(), z[slc].ravel()))
        dist = spatial.distance.cdist(pdb.coords[None, i] - shift, xyz)[0]
        try:
            element = pdb.atomtype[i]
            ffcoeff[element]
        except:
            try:
                element = pdb.atomname[i][0].upper() + pdb.atomname[i][1].lower()
                ffcoeff[element]
            except:
                try:
                    element = pdb.atomname[i][0]
                    ffcoeff[element]
                except:
                    print("Atom type %s or name not recognized for atom # %s"
                          % (pdb.atomtype[i],
                             pdb.atomname[i][0].upper() + pdb.atomname[i][1].lower(),
                             i))
                    print("Using default form factor for Carbon")
                    element = 'C'
                    ffcoeff[element]

        if pdb.numH[i] > 0:
            Va = V_without_impH = sphere_volume_from_radius(pdb.radius[i])
            Vb = V_with_impH = V_without_impH + pdb.numH[i] * sphere_volume_from_radius(pdb.exvolHradius[i])
            ra = sphere_radius_from_volume(Va)
            rb = sphere_radius_from_volume(Vb)
            Ba = u2B(ra) / 8
            Bb = u2B(rb) / 8
            Bdiff = Bb - Ba
        else:
            Bdiff = 0.0
        tmpvalues = realspace_formfactor(element=element, r=dist, B=B[i] + Bdiff)
        # rescale total number of electrons by expected number of electrons
        # pdb.nelectrons is already corrected with the number of electrons including hydrogens
        if np.sum(tmpvalues) > 1e-8:
            ne_total = pdb.nelectrons[i]
            tmpvalues *= ne_total / tmpvalues.sum()

        values[slc] += tmpvalues.reshape(nx, ny, nz)
        support[slc] = True
    # values *= pdb.nelectrons.sum()/values.sum()
    return values, support


def pdb2F_multigauss(pdb, qx, qy, qz, qr=None, radii=None, B=None):
    """Calculate structure factors F from pdb coordinates.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    radii - float or list of radii of atoms in pdb (optional, uses spherical form factor rather than Kromer-Mann)
    """
    radii = np.atleast_1d(radii)
    if qr is None:
        qr = (qx ** 2 + qy ** 2 + qz ** 2) ** 0.5
    n = qr.shape[0]
    F = np.zeros(qr.shape, dtype=complex)
    if B is None:
        B = np.zeros(pdb.natoms)
    if radii[0] is None:
        useradii = False
    else:
        useradii = True
        radii = np.ones(radii.size) * radii
    for i in range(pdb.natoms):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i + 1, pdb.natoms))
        sys.stdout.flush()
        Fatom = formfactor(element=pdb.atomtype[i], q=qr, B=B[i]) * np.exp(
            -1j * (qx * pdb.coords[i, 0] + qy * pdb.coords[i, 1] + qz * pdb.coords[i, 2]))
        ne_total = electrons[pdb.atomtype[i]] + pdb.numH[i]
        Fatom *= ne_total / Fatom[0, 0, 0].real
        F += Fatom
    return F


def pdb2F_simple_gauss_by_radius(pdb, qx, qy, qz, qr=None, rho0=0.334, radii=None, B=None):
    """Calculate structure factors F from pdb coordinates.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    radii - float or list of radii of atoms in pdb (optional, uses spherical form factor rather than Kromer-Mann)
    """
    radii = np.atleast_1d(radii)
    if qr is None:
        qr = (qx ** 2 + qy ** 2 + qz ** 2) ** 0.5
    n = qr.shape[0]
    F = np.zeros(qr.shape, dtype=complex)
    if B is None:
        B = np.zeros(pdb.natoms)
    if radii[0] is None:
        useradii = False
    else:
        useradii = True
        radii = np.ones(radii.size) * radii
    for i in range(pdb.natoms):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i + 1, pdb.natoms))
        sys.stdout.flush()
        V = (4 * np.pi / 3) * pdb.radius[i] ** 3 + pdb.numH[i] * (4 * np.pi / 3) * pdb.exvolHradius ** 3
        Fatom = reciprocalspace_gaussian_formfactor(q=qr, rho0=rho0, V=V) * np.exp(
            -1j * (qx * pdb.coords[i, 0] + qy * pdb.coords[i, 1] + qz * pdb.coords[i, 2]))
        ne_total = rho0 * V
        Fatom *= ne_total / Fatom[0, 0, 0].real
        F += Fatom
    return F


def pdb2map_FFT(pdb, x, y, z, radii=None, restrict=True):
    """Calculate electron density from pdb coordinates by FFT of Fs.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    radii - float or list of radii of atoms in pdb (optional, uses spherical form factor rather than Kromer-Mann)
    """
    radii = np.atleast_1d(radii)
    side = x[-1, 0, 0] - x[0, 0, 0]
    halfside = side / 2
    n = x.shape[0]
    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = x[:, 0, 0]
    df = 1 / side
    qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)
    # create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)
    # create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = mybinmean(qr.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
    F = np.zeros(qr.shape, dtype=complex)
    natoms = pdb.coords.shape[0]
    if radii[0] is None:
        useradii = False
    else:
        useradii = True
        radii = np.ones(radii.size) * radii
    for i in range(pdb.natoms):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i + 1, pdb.natoms))
        sys.stdout.flush()
        if useradii:
            F += sphere(q=qr, R=radii[i], I0=pdb.nelectrons[i], amp=True) * np.exp(
                -1j * (qx * pdb.coords[i, 0] + qy * pdb.coords[i, 1] + qz * pdb.coords[i, 2]))
        else:
            F += formfactor(element=pdb.atomtype[i], q=qr) * np.exp(
                -1j * (qx * pdb.coords[i, 0] + qy * pdb.coords[i, 1] + qz * pdb.coords[i, 2]))
    I3D = abs2(F)
    Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
    rho = myifftn(F).real
    # rho[rho<0] = 0
    # need to shift rho to center of grid, since FFT is offset by half a grid length
    shift = [n // 2 - 1, n // 2 - 1, n // 2 - 1]
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    if restrict:
        xyz = np.column_stack([x.flat, y.flat, z.flat])
        pdb.coords -= np.ones(3) * dx / 2.
        pdbidx = pdb2support(pdb, xyz=xyz, probe=0.0)
        rho[~pdbidx] = 0.0
    return rho, pdbidx


def pdb2support_fast(pdb, x, y, z, radius=None, probe=0.0):
    """Return a boolean 3D density map with support from PDB coordinates"""

    support = np.zeros(x.shape, dtype=np.bool_)
    n = x.shape[0]
    side = x.max() - x.min()
    dx = side / n
    shift = np.ones(3) * dx / 2.

    if radius is None:
        radius = pdb.vdW

    radius = np.atleast_1d(radius)
    if len(radius) != pdb.natoms:
        print("Error: radius argument does not have same length as pdb.")
        exit()

    dr = radius + probe

    natoms = pdb.natoms
    for i in range(natoms):
        # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        # sys.stdout.flush()
        # if a grid point of env is within the desired distance, dr, of
        # the atom coordinate, add it to env
        # to save memory, only run the distance matrix one atom at a time
        # and will only look at grid points within a box of size dr near the atom
        # this will cut out the grid points that are near the atom
        # first, get the min and max distances for each dimension
        # also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i]  # for convenience, store up x,y,z coordinates of atom
        xmin = int(np.floor((xa - dr[i]) / dx)) + n // 2
        xmax = int(np.ceil((xa + dr[i]) / dx)) + n // 2
        ymin = int(np.floor((ya - dr[i]) / dx)) + n // 2
        ymax = int(np.ceil((ya + dr[i]) / dx)) + n // 2
        zmin = int(np.floor((za - dr[i]) / dx)) + n // 2
        zmax = int(np.ceil((za + dr[i]) / dx)) + n // 2
        # handle edges
        xmin = max([xmin, 0])
        xmax = min([xmax, n])
        ymin = max([ymin, 0])
        ymax = min([ymax, n])
        zmin = max([zmin, 0])
        zmax = min([zmax, n])
        # now lets create a slice object for convenience
        slc = np.s_[xmin:xmax, ymin:ymax, zmin:zmax]
        nx = xmax - xmin
        ny = ymax - ymin
        nz = zmax - zmin
        # now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(), y[slc].ravel(), z[slc].ravel()))
        # now calculate all distances from the atom to the minigrid points
        dist = spatial.distance.cdist(pdb.coords[None, i] - shift, xyz)
        # now, add any grid points within dr of atom to the env grid
        # first, create a dummy array to hold booleans of size dist.size
        tmpenv = np.zeros(dist.shape, dtype=np.bool_)
        # now, any elements that have a dist less than dr make true
        tmpenv[dist <= dr[i]] = True
        # now reshape for inserting into env
        tmpenv = tmpenv.reshape(nx, ny, nz)
        support[slc] += tmpenv
    return support


def pdb2support_vdW(pdb, x, y, z, radius=None, probe=0.0):
    """Return a boolean 3D density map with support from PDB coordinates"""

    support = np.zeros(x.shape, dtype=np.bool_)
    n = x.shape[0]
    side = x.max() - x.min()
    dx = side / (n - 1)

    if radius is None:
        radius = pdb.vdW

    radius = np.atleast_1d(radius)
    if len(radius) != pdb.natoms:
        print("Error: radius argument does not have same length as pdb.")
        exit()

    dr = radius + probe

    natoms = pdb.natoms
    for i in range(natoms):
        # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        # sys.stdout.flush()
        # if a grid point of env is within the desired distance, dr, of
        # the atom coordinate, add it to env
        # to save memory, only run the distance matrix one atom at a time
        # and will only look at grid points within a box of size dr near the atom
        # this will cut out the grid points that are near the atom
        # first, get the min and max distances for each dimension
        # also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i]  # for convenience, store up x,y,z coordinates of atom
        xmin = int(np.floor((xa - dr[i]) / dx)) + n // 2  # - 1
        xmax = int(np.ceil((xa + dr[i]) / dx)) + n // 2  # - 1
        ymin = int(np.floor((ya - dr[i]) / dx)) + n // 2  # - 1
        ymax = int(np.ceil((ya + dr[i]) / dx)) + n // 2  # - 1
        zmin = int(np.floor((za - dr[i]) / dx)) + n // 2  # - 1
        zmax = int(np.ceil((za + dr[i]) / dx)) + n // 2  # - 1
        # handle edges
        xmin = max([xmin, 0])
        xmax = min([xmax, n])
        ymin = max([ymin, 0])
        ymax = min([ymax, n])
        zmin = max([zmin, 0])
        zmax = min([zmax, n])
        # now lets create a slice object for convenience
        slc = np.s_[xmin:xmax, ymin:ymax, zmin:zmax]
        nx = xmax - xmin
        ny = ymax - ymin
        nz = zmax - zmin
        # now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(), y[slc].ravel(), z[slc].ravel()))
        # now calculate all distances from the atom to the minigrid points
        dist = spatial.distance.cdist(pdb.coords[None, i], xyz)
        # now, add any grid points within dr of atom to the env grid
        # first, create a dummy array to hold booleans of size dist.size
        tmpenv = np.zeros(dist.shape, dtype=np.bool_)
        # now, any elements that have a dist less than dr make true
        tmpenv[dist <= dr[i]] = True
        # now reshape for inserting into env
        tmpenv = tmpenv.reshape(nx, ny, nz)
        support[slc] += tmpenv
    return support


# Solvent excluded surface function
def pdb2support_SES(pdb, x, y, z, radius=None, probe=1.4):
    """Return a 3D Boolean array of the Solvent Excluded Surface from a set of coordinates.

    pdb - instance of PDB() class
    x,y,z - meshgrid arrays containing x, y, and z coordinates for all voxels of 3D support space
    probe - probe radius for ses determination (defaults to 1.4 for water)
    """

    n = x.shape[0]
    side = x.max() - x.min()
    dx = side / (n - 1)

    if radius is None:
        radius = pdb.vdW

    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    support = np.zeros(x.shape, dtype=bool)

    vdw_support = pdb2support_vdW(pdb, x, y, z, probe=0.0, radius=radius)
    vdw_probe_support = pdb2support_vdW(pdb, x, y, z, probe=probe, radius=radius)
    big_support = pdb2support_vdW(pdb, x, y, z, probe=probe + dx * 1.01, radius=radius)

    # all voxels inside the vdw_support will be true. But first
    # we need to find voxels not inside the vdw support
    # but that _are_ inside the ses. So first make a search region
    # which contains voxels inside big_support that are not inside vdw_support,
    # kind of like a shell (but with gaps in between atoms included)
    # this will contain the voxels where a water molecule can be
    # centered while not overlapping the protein. Then remove the voxels
    # far from that surface to be left with the voxels at the edge where
    # a water molecule can be placed.
    # this is called the solvent accessible surface
    sas = np.copy(big_support)
    sas[vdw_probe_support] = False
    sas_xyz = xyz[sas.ravel()]

    # Take the surface voxels (sas_xyz), and run pdb2support_vdW, but set the radius
    # to zero and the probe to the water. That will create a support quickly around each
    # surface voxel that excludes the reentrant surface. Then, grab all the reentrant
    # voxels that fall within the vdw_probe_shell.
    # first make a pdb file from the surface voxels
    sas_xyz_pdb = PDB(len(sas_xyz))
    sas_xyz_pdb.coords = sas_xyz
    reentrant = pdb2support_vdW(sas_xyz_pdb, x, y, z, radius=np.zeros(sas_xyz_pdb.natoms), probe=probe)
    # this is inverted, so flip it
    reentrant = ~reentrant
    # now set everything outside of the vdw_probe support to be false, to remove outside voxels
    reentrant[~vdw_probe_support] = False

    ses = reentrant
    ses[vdw_support] = True

    return ses


def u2B(u):
    """Calculate B-factor from atomic displacement, u"""
    return np.sign(u) * 8 * np.pi ** 2 * u ** 2


def B2u(B):
    """Calculate atomic displacement, u, from B-factor"""
    return np.sign(B) * (np.abs(B) / (8 * np.pi ** 2)) ** 0.5


def v2B(v):
    """Calculate B-factor from atomic volume displacement, v"""
    u = sphere_radius_from_volume(v)
    return np.sign(u) * 8 * np.pi ** 2 * u ** 2


def sphere(R, q=np.linspace(0, 0.5, 501), I0=1., amp=False):
    """Calculate the scattering of a uniform sphere."""
    q = np.atleast_1d(q)
    a = np.where(q == 0.0, 1.0, (3 * (np.sin(q * R) - q * R * np.cos(q * R)) / (q * R) ** 3))
    if np.isnan(a).any():
        a[np.where(np.isnan(a))] = 1.
    if amp:
        return I0 * a
    else:
        return I0 * a ** 2


def formfactor(element, q=(np.arange(500) + 1) / 1000., B=None):
    """Calculate atomic form factors"""
    if B is None:
        B = 0.0
    q = np.atleast_1d(q)
    ff = np.zeros(q.shape)
    for i in range(4):
        ff += ffcoeff[element]['a'][i] * np.exp(-ffcoeff[element]['b'][i] * (q / (4 * np.pi)) ** 2)
    # ff += ffcoeff[element]['c']
    ff *= np.exp(-B * (q / (4 * np.pi)) ** 2)
    return ff


def realspace_formfactor(element, r=(np.arange(501)) / 1000., B=None):
    """Calculate real space atomic form factors"""
    if B is None:
        B = 0.0
    r = np.atleast_1d(r)
    ff = np.zeros(r.shape)
    for i in range(4):
        ai = ffcoeff[element]['a'][i]
        bi = ffcoeff[element]['b'][i]
        ff += (4 * np.pi / (bi + B)) ** (3 / 2.) * ai * np.exp(-4 * np.pi ** 2 * r ** 2 / (bi + B))
    # i = np.where((r==0))
    # ff += signal.unit_impulse(r.shape, i) * ffcoeff[element]['c']
    return ff


def reciprocalspace_gaussian_formfactor(q=np.linspace(0, 0.5, 501), rho0=0.334, V=None, radius=None, B=None):
    """Calculate reciprocal space atomic form factors assuming an isotropic gaussian sphere (for excluded volume)."""
    if B is None:
        B = 0.0
    if (V is None) and (radius is None):
        print("Error: either radius or volume of atom must be given.")
        exit()
    elif V is None:
        # calculate volume from radius assuming sphere
        V = (4 * np.pi / 3) * radius ** 3
    ff = rho0 * V * np.exp(-q ** 2 * V ** (2. / 3) / (4 * np.pi))
    ff *= np.exp(-B * (q / (4 * np.pi)) ** 2)
    return ff


def realspace_gaussian_formfactor(r=np.linspace(-3, 3, 101), rho0=0.334, V=None, radius=None, B=None):
    """Calculate real space atomic form factors assuming an isotropic gaussian sphere (for excluded volume)."""
    if B is None:
        B = 0.0
    if (V is None) and (radius is None):
        print("Error: either radius or volume of atom must be given.")
        exit()
    elif V is None:
        # calculate volume from radius assuming sphere
        V = (4 * np.pi / 3) * radius ** 3
    if V <= 0:
        ff = r * 0
    else:
        # ff = rho0 * np.exp(-np.pi*r**2/V**(2./3))
        ff = 8 * rho0 * np.pi ** (3. / 2) * V / (B + 4 * np.pi * V ** (2. / 3)) ** (3. / 2) * np.exp(
            -4 * np.pi ** 2 * r ** 2 / (B + 4 * np.pi * V ** (2. / 3)))
    return ff


def estimate_side_from_pdb(pdb, use_convex_hull=False):
    # roughly estimate maximum dimension
    # calculate max distance along x, y, z
    # take the maximum of the three
    # triple that value to set the default side
    # i.e. set oversampling to 3, like in denss
    # use_convex_hull is a bit more accurate
    if pdb.rij is not None:
        # if pdb.rij has already been calculated
        # then just take the Dmax from that
        D = np.max(pdb.rij)
    elif use_convex_hull:
        # print("using convex hull")
        hull = spatial.ConvexHull(pdb.coords)
        hull_points = pdb.coords[hull.vertices]
        # Compute the pairwise distances between points on the convex hull
        distances = spatial.distance.pdist(hull_points, 'euclidean')
        D = np.max(distances)
    else:
        # if pdb.rij has not been calculated,
        # rather than calculating the whole distance
        # matrix, which can be slow and memory intensive
        # for large models, just approximate the maximum
        # length as the max of the range of x, y, or z
        # values of the coordinates.
        xmin = np.min(pdb.coords[:, 0]) - 1.7
        xmax = np.max(pdb.coords[:, 0]) + 1.7
        ymin = np.min(pdb.coords[:, 1]) - 1.7
        ymax = np.max(pdb.coords[:, 1]) + 1.7
        zmin = np.min(pdb.coords[:, 2]) - 1.7
        zmax = np.max(pdb.coords[:, 2]) + 1.7
        wx = xmax - xmin
        wy = ymax - ymin
        wz = zmax - zmin
        D = np.max([wx, wy, wz])
    side = 3 * D
    return side


def calc_chi2(Iq_exp, Iq_calc, scale=True, offset=False, interpolation=True, return_sf=False, return_fit=False, use_sasrec=True, D=None):
    """Calculates a chi2 comparing experimental vs calculated intensity profiles using interpolation.

    Iq_exp (ndarray) - Experimental data, q, I, sigq (required)
    Iq_calc (ndarray) - calculated scattering profile's q, I, and sigq (required)
    scale (bool) - whether to allow scaling of Iq_exp to Iq_calc
    offset (bool) - whether to allow offset of Iq_exp to Iq_calc
    interpolation (bool) - whether to allow fine interpolation of Icalc to qexp using cubic spline
    return_sf (bool) - return scale factor of Iq_exp
    return_fit (bool) - return fit formatted as Nx4 array of [qexp, c*Iexp+b, c*err, Icalc], where c,b are scale and offset
    """
    q_exp = np.copy(Iq_exp[:, 0])
    I_exp = np.copy(Iq_exp[:, 1])
    sigq_exp = np.copy(Iq_exp[:, 2])
    q_calc = np.copy(Iq_calc[:, 0])
    I_calc = np.copy(Iq_calc[:, 1])
    if interpolation:
        Iq_calc_interp = regrid_Iq(Iq_calc, qc=q_exp, use_sasrec=use_sasrec, D=D)
        I_calc = Iq_calc_interp[:,1]
    else:
        # if interpolation of (coarse) calculated profile is disabled, we still need to at least
        # put the experimental data on the correct grid for comparison, so regrid the exp arrays
        # with simple 1D linear interpolation. Note, this interpolates the exp to calc, not calc to exp,
        # because exp is finely sampled usually, whereas for denss in particular calc is coarse
        I_exp = np.interp(q_calc, q_exp, I_exp)
        sigq_exp = np.interp(q_calc, q_exp, sigq_exp)
        q_exp = np.copy(q_calc)

    if scale and offset:
        # the offset is effectively a constant that is the same for each
        # data point (i.e., a bunch of ones times a number),
        # so we fit two vectors to Icalc (one is Iexp and the other is the array of ones),
        # such that Icalc = c*Iexp + b*ones
        # and then the coefficients output by least squares are c (scale factor) and b (offset)
        # then, the chi2 equation is chi2 = 1/N * sum((Icalc-(cIexp+b))/sigq)^2
        # which can be refactored as chi2 = 1/N * sum( Icalc/sigq - cIexp/sigq - b/sigq )^2
        # so for np.linalg.lstsq, give the curves divided by sigq to get back c and b coefficients
        exp = np.vstack((I_exp / sigq_exp, np.ones(len(sigq_exp)) * 1 / sigq_exp))
        calc = I_calc / sigq_exp
        exp_scale_factor, offset = _fit_by_least_squares(calc, exp)
    elif scale:
        exp = I_exp / sigq_exp
        calc = I_calc / sigq_exp
        exp_scale_factor = _fit_by_least_squares(calc, exp)
        offset = 0.0
    else:
        exp_scale_factor = 1.0
        offset = 0.0
    I_exp *= exp_scale_factor
    I_exp += offset
    sigq_exp *= exp_scale_factor
    chi2 = 1 / len(q_exp) * np.sum(((I_exp - I_calc) / sigq_exp) ** 2)
    fit = np.vstack((q_exp, I_exp, sigq_exp, I_calc)).T
    if return_sf and return_fit:
        return chi2, exp_scale_factor, offset, fit
    elif return_sf and not return_fit:
        return chi2, exp_scale_factor
    elif not return_sf and return_fit:
        return chi2, fit
    else:
        return chi2


def calc_uniform_shell(pdb, x, y, z, thickness=2.8, distance=1.4):
    """Create a uniform density hydration shell around the particle.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    thickness - thickness of the shell (e.g. water diameter)
    distance - distance from the protein surface defining the center of the shell (e.g., water radius)
    """
    inner_support = pdb2support_fast(pdb, x, y, z, radius=pdb.vdW, probe=distance - thickness / 2)
    outer_support = pdb2support_fast(pdb, x, y, z, radius=pdb.vdW, probe=distance + thickness / 2)
    shell_idx = outer_support
    shell_idx[inner_support] = False
    shell = shell_idx * 1.0
    return shell


def denss_3DFs(rho_start, dmax, ne=None, voxel=5., oversampling=3., positivity=True,
               output="map", steps=2001, seed=None, shrinkwrap=True, shrinkwrap_sigma_start=3,
               shrinkwrap_sigma_end=1.5, shrinkwrap_sigma_decay=0.99, shrinkwrap_threshold_fraction=0.2,
               shrinkwrap_iter=20, shrinkwrap_minstep=50, write_freq=100, support=None,
               enforce_connectivity=True, enforce_connectivity_steps=[6000], quiet=False):
    """Calculate electron density from starting map by refining phases only."""
    D = dmax
    side = oversampling * D
    halfside = side / 2
    n = int(side / voxel)
    # want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n % 2 == 1: n += 1
    # store n for later use if needed
    nbox = n
    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = np.linspace(-halfside, halfside, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    df = 1 / side
    qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
    qz_ = np.fft.rfftfreq(x_.size) * n * df * 2 * np.pi
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)
    # create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    qbinsc[1:] += qstep / 2.
    # create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    if steps == 'None' or steps is None or steps < 1:
        steps = int(shrinkwrap_iter * (np.log(shrinkwrap_sigma_end / shrinkwrap_sigma_start) / np.log(
            shrinkwrap_sigma_decay)) + shrinkwrap_minstep)
        steps += 3000
    else:
        steps = np.int(steps)
    Imean = np.zeros((steps + 1, len(qbins)))
    chi = np.zeros((steps + 1))
    rg = np.zeros((steps + 1))
    supportV = np.zeros((steps + 1))
    chibest = np.inf
    usesupport = True
    if support is None:
        support = np.ones(x.shape, dtype=bool)
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
        # APPLY RECIPROCAL SPACE RESTRAINTS
        # calculate spherical average of intensities from 3D Fs
        I3D = np.abs(F) ** 2
        Imean[j] = ndimage.mean(I3D, labels=qbin_labels, index=np.arange(0, qbin_labels.max() + 1))
        # scale Fs to match data
        F *= Amp / np.abs(F)
        chi[j] = 1.0
        # APPLY REAL SPACE RESTRAINTS
        rhoprime = np.fft.ifftn(F, rho.shape)
        rhoprime = rhoprime.real
        if j % write_freq == 0:
            write_mrc(rhoprime / dV, side, output + "_current.mrc")
        rg[j] = rho2rg(rhoprime, r=r, support=support, dx=dx)
        newrho = np.zeros_like(rho)
        # Error Reduction
        newrho[support] = rhoprime[support]
        newrho[~support] = 0.0
        # enforce positivity by making all negative density points zero.
        if positivity:
            netmp = np.sum(newrho)
            newrho[newrho < 0] = 0.0
            if np.sum(newrho) != 0:
                newrho *= netmp / np.sum(newrho)
        supportV[j] = np.sum(support) * dV

        if not quiet:
            sys.stdout.write("\r% 5i % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
            sys.stdout.flush()

        rho = newrho

    if not quiet:
        print()

    F = np.fft.fftn(rho)
    # calculate spherical average intensity from 3D Fs
    Imean[j + 1] = ndimage.mean(np.abs(F) ** 2, labels=qbin_labels, index=np.arange(0, qbin_labels.max() + 1))
    # scale Fs to match data
    F *= Amp / np.abs(F)
    rho = np.fft.ifftn(F, rho.shape)
    rho = rho.real

    # scale total number of electrons
    if ne is not None:
        rho *= ne / np.sum(rho)

    rg[j + 1] = rho2rg(rho=rho, r=r, support=support, dx=dx)
    supportV[j + 1] = supportV[j]

    # change rho to be the electron density in e-/angstroms^3, rather than number of electrons,
    # which is what the FFT assumes
    rho /= dV

    return rho




