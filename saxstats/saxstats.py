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
#    Copyright 2017, 2018, 2019, 2020 The Research Foundation for SUNY
#
#    Additional authors:
#    Nhan D. Nguyen
#    Jesse Hopkins
#    Andrew Bruno
#    Esther Gallmeier
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

# import numba

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

if CUPY_LOADED:
    gpu_abs2 = cp.ElementwiseKernel(
             'complex128 z', 'float64 d', 'double re=real(z), im=imag(z); d=re*re+im*im;', "gpu_abs2")

def abs2(x, DENSS_GPU=False):
    #a faster way to calculate abs(x)**2, for calculating intensities
    if DENSS_GPU:
       return gpu_abs2(x)
    else:
        return x.real**2 + x.imag**2

# @numba.jit(nopython=True)
# def mybincount(x, weights):
#     result = np.zeros(x.max() + 1, int)
#     for i in x:
#         result[i] += weights[i]

def mybinmean(xravel,binsravel,xcount=None,DENSS_GPU=False):
    if DENSS_GPU:
        if isinstance(binsravel, np.ndarray):
            binsravel = cp.asarray(binsravel)
        xsum = cp.bincount(binsravel, xravel)
        if xcount is None:
            xcount = cp.bincount(binsravel)
        elif isinstance(xcount, np.ndarray):
            xcount = cp.asarray(xcount)
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

def read_mrc(filename="map.mrc",returnABC=False):
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
        print("Error. Argument 'side' must be float or 3-tuple")
    #zoom factors
    zx, zy, zz = vx/dx, vy/dx, vz/dx
    newrho = ndimage.zoom(rho,(zx, zy, zz),order=1,mode="wrap")

    return newrho

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
    recenter_mode="com", positivity=True, extrapolate=True, output="map",
    steps=None, seed=None, flatten_low_density=True, rho_start=None, add_noise=None,
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

    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    qbinsc[1:] += qstep/2.

    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)

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
    qba = np.copy(qbin_args) #just for brevity when using it later
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
    support = np.ones(x.shape,dtype=bool)

    if seed is None:
        #Have to reset the random seed to get a random in different from other processes
        prng = np.random.RandomState()
        seed = prng.randint(2**31-1)
    else:
        seed = int(seed)

    prng = np.random.RandomState(seed)

    if rho_start is not None:
        rho = rho_start*dV
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
        # F[np.abs(F)==0] = 1e-16

        #APPLY RECIPROCAL SPACE RESTRAINTS
        #calculate spherical average of intensities from 3D Fs
        # I3D = myabs(F, out=I3D, DENSS_GPU=DENSS_GPU)**2
        I3D = abs2(F, DENSS_GPU=DENSS_GPU)
        Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=DENSS_GPU)

        #scale Fs to match data
        factors = mysqrt(Idata/Imean, DENSS_GPU=DENSS_GPU)
        F *= factors[qbin_labels]

        chi[j] = mysum(((Imean[qba]-Idata[qba])/sigqdata[qba])**2, DENSS_GPU=DENSS_GPU)/Idata[qba].size

        #APPLY REAL SPACE RESTRAINTS
        rhoprime = myifftn(F, DENSS_GPU=DENSS_GPU).real

        if not DENSS_GPU and j%write_freq == 0:
            if write_xplor_format:
                write_xplor(rhoprime/dV, side, fprefix+"_current.xplor")
            write_mrc(rhoprime/dV, side, fprefix+"_current.mrc")

        # use Guinier's law to approximate quickly
        rg[j] = calc_rg_by_guinier_first_2_points(qbinsc, Imean, DENSS_GPU=DENSS_GPU)

        #Error Reduction
        newrho *= 0
        if DENSS_GPU:
            newrho = cp.asarray(newrho)
        newrho[support] = rhoprime[support]

        # enforce positivity by making all negative density points zero.
        if positivity:
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
                if j>500:
                    absv = True
                else:
                    absv = False
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
                #run the old method
                if j>500:
                    absv = True
                else:
                    absv = False
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
        if erode and j > shrinkwrap_minstep and j%shrinkwrap_iter==1:
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

    F = np.fft.fftn(rho)
    #calculate spherical average intensity from 3D Fs
    Imean = ndimage.mean(np.abs(F)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    #chi[j+1] = np.sum(((Imean[j+1,qbin_args]-Idata)/sigqdata)**2)/qbin_args.size

    #scale Fs to match data
    #factors = np.ones((len(qbins)))
    factors = np.sqrt(Idata/Imean)
    F *= factors[qbin_labels]
    rho = np.fft.ifftn(F,rho.shape)
    rho = rho.real

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

    # rg[j+1] = rho2rg(rho=rho,r=r,support=support,dx=dx)
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
    #support[rho_blurred >= threshold] = True
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
                #sasrec = saxs.Sasrec(Iq[n1:n2], D, qc=qc, r=r, nr=args.nr, alpha=10.**alpha, ne=nes, extrapolate=args.extrapolate)
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

    def read_pdb(self, filename, ignore_waters=False):
        self.natoms = 0
        with open(filename) as f:
            for line in f:
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
        with open(filename) as f:
            atom = 0
            for line in f:
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
                try:
                    self.atomnum[atom] = int(line[6:11])
                except ValueError as e:
                    self.atomnum[atom] = int(line[6:11],36)
                if ignore_waters and ((line[17:20]=="HOH") or (line[17:20]=="TIP")):
                    continue
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
                self.atomtype[atom] = atomtype
                self.charge[atom] = line[78:80].strip('\n')
                self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(),6)
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
        #for CRYST1 card, use default defined by PDB, but 100 A side
        self.cella = 100.0
        self.cellb = 100.0
        self.cellc = 100.0
        self.cellalpha = 90.0
        self.cellbeta = 90.0
        self.cellgamma = 90.0

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

def pdb2map_gauss(pdb,xyz,sigma,mode="slow",eps=1e-6):
    """Simple isotropic gaussian sum at coordinate locations.

    Fast mode uses KDTree to only calculate density at grid points with
    a density above a threshold.
    see https://stackoverflow.com/questions/52208434"""
    n = int(round(xyz.shape[0]**(1/3.)))
    sigma /= 4.
    dx = xyz[1,2] - xyz[0,2]
    shift = np.ones(3)*dx/2.
    #dist = spatial.distance.cdist(pdb.coords, xyz)
    #rho = np.sum(values,axis=0).reshape(n,n,n)
    #run cdist in a loop over atoms to avoid overloading memory
    print("\n Calculate density map from PDB... ")
    if mode == "fast":
        if eps is None:
            eps = np.finfo('float64').eps
        thr = -np.log(eps) * 2 * sigma**2
        data_tree = spatial.cKDTree(pdb.coords-shift)
        discr = 1000 # you can tweak this to get best results on your system
        values = np.empty(n**3)
        for i in range(n**3//discr + 1):
            sys.stdout.write("\r%i / %i chunks" % (i+1, n**3//discr + 1))
            sys.stdout.flush()
            slc = slice(i * discr, i * discr + discr)
            grid_tree = spatial.cKDTree(xyz[slc])
            dists = grid_tree.sparse_distance_matrix(data_tree, thr, output_type = 'coo_matrix')
            dists.data = 1./np.sqrt(2*np.pi*sigma**2) * np.exp(-dists.data/(2*sigma**2))
            values[slc] = dists.sum(1).squeeze()
    else:
        values = np.zeros((xyz.shape[0]))
        for i in range(pdb.coords.shape[0]):
            sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
            sys.stdout.flush()
            dist = spatial.distance.cdist(pdb.coords[None,i]-shift, xyz)
            dist *= dist
            values += pdb.nelectrons[i]*1./np.sqrt(2*np.pi*sigma**2) * np.exp(-dist[0]/(2*sigma**2))
    print()
    return values.reshape(n,n,n)

def pdb2map_fastgauss(pdb,x,y,z,sigma,r=20.0,ignore_waters=True):
    """Simple isotropic gaussian sum at coordinate locations.

    This fastgauss function only calculates the values at
    grid points near the atom for speed.

    pdb - instance of PDB class (required)
    x,y,z - meshgrids for x, y, and z (required)
    sigma - width of Gaussian, i.e. effectively resolution
    r - maximum distance from atom to calculate density
    """
    side = x[-1,0,0] - x[0,0,0]
    halfside = side/2
    n = x.shape[0]
    dx = side/n
    dV = dx**3
    V = side**3
    x_ = x[:,0,0]
    sigma /= 4. #to make compatible with e2pdb2mrc/chimera sigma
    shift = np.ones(3)*dx/2.
    print("\n Calculate density map from PDB... ")
    values = np.zeros(x.shape)
    for i in range(pdb.coords.shape[0]):
        if ignore_waters and pdb.resname[i]=="HOH":
            continue
        sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        sys.stdout.flush()
        #this will cut out the grid points that are near the atom
        #first, get the min and max distances for each dimension
        #also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i] # for convenience, store up x,y,z coordinates of atom
        xmin = int(np.floor((xa-r)/dx)) + n//2
        xmax = int(np.ceil((xa+r)/dx)) + n//2
        ymin = int(np.floor((ya-r)/dx)) + n//2
        ymax = int(np.ceil((ya+r)/dx)) + n//2
        zmin = int(np.floor((za-r)/dx)) + n//2
        zmax = int(np.ceil((za+r)/dx)) + n//2
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
        dist *= dist
        tmpvalues = pdb.nelectrons[i]*1./np.sqrt(2*np.pi*sigma**2) * np.exp(-dist[0]/(2*sigma**2))
        values[slc] += tmpvalues.reshape(nx,ny,nz)
    print()
    return values

def pdb2map_multigauss(pdb,x,y,z,cutoff=3.0,resolution=0.0,ignore_waters=True):
    """5-term gaussian sum at coordinate locations using Cromer-Mann coefficients.

    This fastgauss function only calculates the values at
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
    print("\n Calculate density map from PDB... ")
    values = np.zeros(x.shape)
    support = np.zeros(x.shape,dtype=bool)
    cutoff = max(cutoff,2*resolution)
    #convert resolution to B-factor for form factor calculation
    #set resolution equal to atomic displacement
    B = u2B(resolution)
    for i in range(pdb.coords.shape[0]):
        if ignore_waters and pdb.resname[i]=="HOH":
            continue
        sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        sys.stdout.flush()
        #this will cut out the grid points that are near the atom
        #first, get the min and max distances for each dimension
        #also, convert those distances to indices by dividing by dx
        xa, ya, za = pdb.coords[i] # for convenience, store up x,y,z coordinates of atom
        #ignore atoms whose coordinates are outside the box limits
        if (
            (xa < x.min()) or
            (xa > x.max()) or
            (ya < y.min()) or
            (ya > y.max()) or
            (za < z.min()) or
            (za > z.max())
           ):
           print()
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
        dist = spatial.distance.cdist(pdb.coords[None,i]-shift, xyz)
        dist *= dist
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

        tmpvalues = realspace_formfactor(element=element,r=dist,B=B)
        #rescale total number of electrons by expected number of electrons
        if np.sum(tmpvalues)>1e-8:
            tmpvalues *= electrons[element] / np.sum(tmpvalues)
        else:
            print()
            print("Voxel spacing too coarse. No density for atom %d."%i)

        values[slc] += tmpvalues.reshape(nx,ny,nz)
        support[slc] = True
    print()
    return values, support

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
    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    #only move the non-zero terms, since the zeroth term should be at q=0.
    qbinsc[1:] += qstep/2.
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    F = np.zeros(qr.shape,dtype=complex)
    Fmean = np.zeros((len(qbins)))
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
    print()
    print("Total number of electrons = %f " % np.abs(F[0,0,0]))
    qbin_labels = np.zeros(F.shape, dtype=int)
    qbin_labels = np.digitize(qr, qbins)
    qbin_labels -= 1
    Imean = ndimage.mean(np.abs(F)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    rho = np.fft.ifftn(F,x.shape).real
    rho[rho<0] = 0
    #need to shift rho to center of grid, since FFT is offset by half a grid length
    shift = np.array(rho.shape)/2
    shift -= 1
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    if restrict:
        xyz = np.column_stack([x.flat,y.flat,z.flat])
        pdb.coords -= np.ones(3)*dx/2.
        pdbidx = pdb2support(pdb, xyz=xyz,probe=0.0)
        rho[~pdbidx] = 0.0
    return rho, pdbidx

def pdb2support(pdb,xyz,probe=0.0):
    """Calculate support from pdb coordinates."""
    n = int(round(xyz.shape[0]**(1/3.)))
    support = np.zeros((n,n,n),dtype=bool)
    xyz_nearby = []
    xyz_nearby_i = []
    for i in range(pdb.natoms):
        sys.stdout.write("\r% 5i / % 5i" % (i+1,pdb.natoms))
        sys.stdout.flush()
        xyz_nearby_i.append(np.where(spatial.distance.cdist(pdb.coords[i,None],xyz) < 1.7+probe)[1])
    xyz_nearby_i = np.unique(np.concatenate(xyz_nearby_i))
    support[np.unravel_index(xyz_nearby_i,support.shape)] = True
    return support

def pdb2support_fast(pdb,x,y,z,dr=2.0):
    """Return a boolean 3D density map with support from PDB coordinates"""

    support = np.zeros(x.shape,dtype=np.bool_)
    n = x.shape[0]
    side = x.max()-x.min()
    dx = side/n
    shift = np.ones(3)*dx/2.

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
        xmin = int(np.floor((xa-dr)/dx)) + n//2
        xmax = int(np.ceil((xa+dr)/dx)) + n//2
        ymin = int(np.floor((ya-dr)/dx)) + n//2
        ymax = int(np.ceil((ya+dr)/dx)) + n//2
        zmin = int(np.floor((za-dr)/dx)) + n//2
        zmax = int(np.ceil((za+dr)/dx)) + n//2
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
        tmpenv[dist<=dr] = True
        #now reshape for inserting into env
        tmpenv = tmpenv.reshape(nx,ny,nz)
        support[slc] += tmpenv
    #print()
    return support

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

def formfactor(element, q=(np.arange(500)+1)/1000.,B=0):
    """Calculate atomic form factors"""
    q = np.atleast_1d(q)
    ff = np.zeros(q.shape)
    for i in range(4):
        ff += ffcoeff[element]['a'][i] * np.exp(-ffcoeff[element]['b'][i]*(q/(4*np.pi))**2)
    ff += ffcoeff[element]['c']
    ff *= np.exp(-B*q**2)
    return ff

def u2B(u):
    """Calculate B-factor from atomic displacement, u"""
    if u<0:
        return -8 * np.pi**2 * np.abs(u)**2
    else:
        return 8 * np.pi**2 * u**2

def B2u(B):
    """Calculate atomic displacement, u, from B-factor"""
    if B<0:
        return -(np.abs(B)/(8*np.pi**2))**0.5
    else:
        return (B/(8*np.pi**2))**0.5

def realspace_formfactor(element, r=(np.arange(501))/1000., B=0.0):
    """Calculate real space atomic form factors"""
    r = np.atleast_1d(r)
    ff = np.zeros(r.shape)
    for i in range(4):
        ai = ffcoeff[element]['a'][i]
        bi = ffcoeff[element]['b'][i]
        #the form factor in the next line is based on an analytical Fourier
        #tranform of the form factor in reciprocal space
        #ff += 2*(2/bi)**0.5*np.pi * ai * np.exp(-(2*np.pi*r)**2/bi)
        #the form factor in the next line additionally accounts for a B-factor
        ff += 2*(6)**0.5 * np.pi * ai * np.exp(-3*(2*np.pi*r)**2/(3*bi+B)) / (3*bi+B)**0.5
    i = np.where((r==0))
    ff += signal.unit_impulse(r.shape, i) * ffcoeff[element]['c']
    return ff

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



electrons = {'H': 1,'HE': 2,'He': 2,'LI': 3,'Li': 3,'BE': 4,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'NE': 10,'Ne': 10,'NA': 11,'Na': 11,'MG': 12,'Mg': 12,'AL': 13,'Al': 13,'SI': 14,'Si': 14,'P': 15,'S': 16,'CL': 17,'Cl': 17,'AR': 18,'Ar': 18,'K': 19,'CA': 20,'Ca': 20,'SC': 21,'Sc': 21,'TI': 22,'Ti': 22,'V': 23,'CR': 24,'Cr': 24,'MN': 25,'Mn': 25,'FE': 26,'Fe': 26,'CO': 27,'Co': 27,'NI': 28,'Ni': 28,'CU': 29,'Cu': 29,'ZN': 30,'Zn': 30,'GA': 31,'Ga': 31,'GE': 32,'Ge': 32,'AS': 33,'As': 33,'SE': 34,'Se': 34,'Se': 34,'Se': 34,'BR': 35,'Br': 35,'KR': 36,'Kr': 36,'RB': 37,'Rb': 37,'SR': 38,'Sr': 38,'Y': 39,'ZR': 40,'Zr': 40,'NB': 41,'Nb': 41,'MO': 42,'Mo': 42,'TC': 43,'Tc': 43,'RU': 44,'Ru': 44,'RH': 45,'Rh': 45,'PD': 46,'Pd': 46,'AG': 47,'Ag': 47,'CD': 48,'Cd': 48,'IN': 49,'In': 49,'SN': 50,'Sn': 50,'SB': 51,'Sb': 51,'TE': 52,'Te': 52,'I': 53,'XE': 54,'Xe': 54,'CS': 55,'Cs': 55,'BA': 56,'Ba': 56,'LA': 57,'La': 57,'CE': 58,'Ce': 58,'PR': 59,'Pr': 59,'ND': 60,'Nd': 60,'PM': 61,'Pm': 61,'SM': 62,'Sm': 62,'EU': 63,'Eu': 63,'GD': 64,'Gd': 64,'TB': 65,'Tb': 65,'DY': 66,'Dy': 66,'HO': 67,'Ho': 67,'ER': 68,'Er': 68,'TM': 69,'Tm': 69,'YB': 70,'Yb': 70,'LU': 71,'Lu': 71,'HF': 72,'Hf': 72,'TA': 73,'Ta': 73,'W': 74,'RE': 75,'Re': 75,'OS': 76,'Os': 76,'IR': 77,'Ir': 77,'PT': 78,'Pt': 78,'AU': 79,'Au': 79,'HG': 80,'Hg': 80,'TL': 81,'Tl': 81,'PB': 82,'Pb': 82,'BI': 83,'Bi': 83,'PO': 84,'Po': 84,'AT': 85,'At': 85,'RN': 86,'Rn': 86,'FR': 87,'Fr': 87,'RA': 88,'Ra': 88,'AC': 89,'Ac': 89,'TH': 90,'Th': 90,'PA': 91,'Pa': 91,'U': 92,'NP': 93,'Np': 93,'PU': 94,'Pu': 94,'AM': 95,'Am': 95,'CM': 96,'Cm': 96,'BK': 97,'Bk': 97,'CF': 98,'Cf': 98,'ES': 99,'Es': 99,'FM': 100,'Fm': 100,'MD': 101,'Md': 101,'NO': 102,'No': 102,'LR': 103,'Lr': 103,'RF': 104,'Rf': 104,'DB': 105,'Db': 105,'SG': 106,'Sg': 106,'BH': 107,'Bh': 107,'HS': 108,'Hs': 108,'MT': 109,'Mt': 109}

#form factors taken from http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
ffcoeff = {
    "H": {"a": [0.489918, 0.262003, 0.196767, 0.049879], "b": [20.6593, 7.74039, 49.5519, 2.20159], "c": 0.001305},
    "H1-": {"a": [0.897661, 0.565616, 0.415815, 0.116973], "b": [53.1368, 15.187, 186.576, 3.56709], "c": 0.002389},
    "He": {"a": [0.8734, 0.6309, 0.3112, 0.178], "b": [9.1037, 3.3568, 22.9276, 0.9821], "c": 0.0064},
    "Li": {"a": [1.1282, 0.7508, 0.6175, 0.4653], "b": [3.9546, 1.0524, 85.3905, 168.261], "c": 0.0377},
    "Li1+": {"a": [0.6968, 0.7888, 0.3414, 0.1563], "b": [4.6237, 1.9557, 0.6316, 10.0953], "c": 0.0167},
    "Be": {"a": [1.5919, 1.1278, 0.5391, 0.7029], "b": [43.6427, 1.8623, 103.483, 0.542], "c": 0.0385},
    "Be2+": {"a": [6.2603, 0.8849, 0.7993, 0.1647], "b": [0.0027, 0.8313, 2.2758, 5.1146], "c": -6.1092},
    "B": {"a": [2.0545, 1.3326, 1.0979, 0.7068], "b": [23.2185, 1.021, 60.3498, 0.1403], "c": -0.1932},
    "C": {"a": [2.31, 1.02, 1.5886, 0.865], "b": [20.8439, 10.2075, 0.5687, 51.6512], "c": 0.2156},
    "Cval": {"a": [2.26069, 1.56165, 1.05075, 0.839259], "b": [22.6907, 0.656665, 9.75618, 55.5949], "c": 0.286977},
    "N": {"a": [12.2126, 3.1322, 2.0125, 1.1663], "b": [0.0057, 9.8933, 28.9975, 0.5826], "c": -11.529},
    "O": {"a": [3.0485, 2.2868, 1.5463, 0.867], "b": [13.2771, 5.7011, 0.3239, 32.9089], "c": 0.2508},
    "O1-": {"a": [4.1916, 1.63969, 1.52673, -20.307], "b": [12.8573, 4.17236, 47.0179, -0.01404], "c": 21.9412},
    "F": {"a": [3.5392, 2.6412, 1.517, 1.0243], "b": [10.2825, 4.2944, 0.2615, 26.1476], "c": 0.2776},
    "F1-": {"a": [3.6322, 3.51057, 1.26064, 0.940706], "b": [5.27756, 14.7353, 0.442258, 47.3437], "c": 0.653396},
    "Ne": {"a": [3.9553, 3.1125, 1.4546, 1.1251], "b": [8.4042, 3.4262, 0.2306, 21.7184], "c": 0.3515},
    "Na": {"a": [4.7626, 3.1736, 1.2674, 1.1128], "b": [3.285, 8.8422, 0.3136, 129.424], "c": 0.676},
    "Na1+": {"a": [3.2565, 3.9362, 1.3998, 1.0032], "b": [2.6671, 6.1153, 0.2001, 14.039], "c": 0.404},
    "Mg": {"a": [5.4204, 2.1735, 1.2269, 2.3073], "b": [2.8275, 79.2611, 0.3808, 7.1937], "c": 0.8584},
    "Mg2+": {"a": [3.4988, 3.8378, 1.3284, 0.8497], "b": [2.1676, 4.7542, 0.185, 10.1411], "c": 0.4853},
    "Al": {"a": [6.4202, 1.9002, 1.5936, 1.9646], "b": [3.0387, 0.7426, 31.5472, 85.0886], "c": 1.1151},
    "Al3+": {"a": [4.17448, 3.3876, 1.20296, 0.528137], "b": [1.93816, 4.14553, 0.228753, 8.28524], "c": 0.706786},
    "Siv": {"a": [6.2915, 3.0353, 1.9891, 1.541], "b": [2.4386, 32.3337, 0.6785, 81.6937], "c": 1.1407},
    "Sival": {"a": [5.66269, 3.07164, 2.62446, 1.3932], "b": [2.6652, 38.6634, 0.916946, 93.5458], "c": 1.24707},
    "Si4+": {"a": [4.43918, 3.20345, 1.19453, 0.41653], "b": [1.64167, 3.43757, 0.2149, 6.65365], "c": 0.746297},
    "P": {"a": [6.4345, 4.1791, 1.78, 1.4908], "b": [1.9067, 27.157, 0.526, 68.1645], "c": 1.1149},
    "S": {"a": [6.9053, 5.2034, 1.4379, 1.5863], "b": [1.4679, 22.2151, 0.2536, 56.172], "c": 0.8669},
    "Cl": {"a": [11.4604, 7.1964, 6.2556, 1.6455], "b": [0.0104, 1.1662, 18.5194, 47.7784], "c": -9.5574},
    "Cl1-": {"a": [18.2915, 7.2084, 6.5337, 2.3386], "b": [0.0066, 1.1717, 19.5424, 60.4486], "c": -16.378},
    "Ar": {"a": [7.4845, 6.7723, 0.6539, 1.6442], "b": [0.9072, 14.8407, 43.8983, 33.3929], "c": 1.4445},
    "K": {"a": [8.2186, 7.4398, 1.0519, 0.8659], "b": [12.7949, 0.7748, 213.187, 41.6841], "c": 1.4228},
    "K1+": {"a": [7.9578, 7.4917, 6.359, 1.1915], "b": [12.6331, 0.7674, -0.002, 31.9128], "c": -4.9978},
    "Ca": {"a": [8.6266, 7.3873, 1.5899, 1.0211], "b": [10.4421, 0.6599, 85.7484, 178.437], "c": 1.3751},
    "Ca2+": {"a": [15.6348, 7.9518, 8.4372, 0.8537], "b": [-0.0074, 0.6089, 10.3116, 25.9905], "c": -14.875},
    "Sc": {"a": [9.189, 7.3679, 1.6409, 1.468], "b": [9.0213, 0.5729, 136.108, 51.3531], "c": 1.3329},
    "Sc3+": {"a": [13.4008, 8.0273, 1.65943, 1.57936], "b": [0.29854, 7.9629, -0.28604, 16.0662], "c": -6.6667},
    "Ti": {"a": [9.7595, 7.3558, 1.6991, 1.9021], "b": [7.8508, 0.5, 35.6338, 116.105], "c": 1.2807},
    "Ti2+": {"a": [9.11423, 7.62174, 2.2793, 0.087899], "b": [7.5243, 0.457585, 19.5361, 61.6558], "c": 0.897155},
    "Ti3+": {"a": [17.7344, 8.73816, 5.25691, 1.92134], "b": [0.22061, 7.04716, -0.15762, 15.9768], "c": -14.652},
    "Ti4+": {"a": [19.5114, 8.23473, 2.01341, 1.5208], "b": [0.178847, 6.67018, -0.29263, 12.9464], "c": -13.28},
    "V": {"a": [10.2971, 7.3511, 2.0703, 2.0571], "b": [6.8657, 0.4385, 26.8938, 102.478], "c": 1.2199},
    "V2+": {"a": [10.106, 7.3541, 2.2884, 0.0223], "b": [6.8818, 0.4409, 20.3004, 115.122], "c": 1.2298},
    "V3+": {"a": [9.43141, 7.7419, 2.15343, 0.016865], "b": [6.39535, 0.383349, 15.1908, 63.969], "c": 0.656565},
    "V5+": {"a": [15.6887, 8.14208, 2.03081, -9.576], "b": [0.679003, 5.40135, 9.97278, 0.940464], "c": 1.7143},
    "Cr": {"a": [10.6406, 7.3537, 3.324, 1.4922], "b": [6.1038, 0.392, 20.2626, 98.7399], "c": 1.1832},
    "Cr2+": {"a": [9.54034, 7.7509, 3.58274, 0.509107], "b": [5.66078, 0.344261, 13.3075, 32.4224], "c": 0.616898},
    "Cr3+": {"a": [9.6809, 7.81136, 2.87603, 0.113575], "b": [5.59463, 0.334393, 12.8288, 32.8761], "c": 0.518275},
    "Mn": {"a": [11.2819, 7.3573, 3.0193, 2.2441], "b": [5.3409, 0.3432, 17.8674, 83.7543], "c": 1.0896},
    "Mn2+": {"a": [10.8061, 7.362, 3.5268, 0.2184], "b": [5.2796, 0.3435, 14.343, 41.3235], "c": 1.0874},
    "Mn3+": {"a": [9.84521, 7.87194, 3.56531, 0.323613], "b": [4.91797, 0.294393, 10.8171, 24.1281], "c": 0.393974},
    "Mn4+": {"a": [9.96253, 7.97057, 2.76067, 0.054447], "b": [4.8485, 0.283303, 10.4852, 27.573], "c": 0.251877},
    "Fe": {"a": [11.7695, 7.3573, 3.5222, 2.3045], "b": [4.7611, 0.3072, 15.3535, 76.8805], "c": 1.0369},
    "Fe2+": {"a": [11.0424, 7.374, 4.1346, 0.4399], "b": [4.6538, 0.3053, 12.0546, 31.2809], "c": 1.0097},
    "Fe3+": {"a": [11.1764, 7.3863, 3.3948, 0.0724], "b": [4.6147, 0.3005, 11.6729, 38.5566], "c": 0.9707},
    "Co": {"a": [12.2841, 7.3409, 4.0034, 2.3488], "b": [4.2791, 0.2784, 13.5359, 71.1692], "c": 1.0118},
    "Co2+": {"a": [11.2296, 7.3883, 4.7393, 0.7108], "b": [4.1231, 0.2726, 10.2443, 25.6466], "c": 0.9324},
    "Co3+": {"a": [10.338, 7.88173, 4.76795, 0.725591], "b": [3.90969, 0.238668, 8.35583, 18.3491], "c": 0.286667},
    "Ni": {"a": [12.8376, 7.292, 4.4438, 2.38], "b": [3.8785, 0.2565, 12.1763, 66.3421], "c": 1.0341},
    "Ni2+": {"a": [11.4166, 7.4005, 5.3442, 0.9773], "b": [3.6766, 0.2449, 8.873, 22.1626], "c": 0.8614},
    "Ni3+": {"a": [10.7806, 7.75868, 5.22746, 0.847114], "b": [3.5477, 0.22314, 7.64468, 16.9673], "c": 0.386044},
    "Cu": {"a": [13.338, 7.1676, 5.6158, 1.6735], "b": [3.5828, 0.247, 11.3966, 64.8126], "c": 1.191},
    "Cu1+": {"a": [11.9475, 7.3573, 6.2455, 1.5578], "b": [3.3669, 0.2274, 8.6625, 25.8487], "c": 0.89},
    "Cu2+": {"a": [11.8168, 7.11181, 5.78135, 1.14523], "b": [3.37484, 0.244078, 7.9876, 19.897], "c": 1.14431},
    "Zn": {"a": [14.0743, 7.0318, 5.1652, 2.41], "b": [3.2655, 0.2333, 10.3163, 58.7097], "c": 1.3041},
    "Zn2+": {"a": [11.9719, 7.3862, 6.4668, 1.394], "b": [2.9946, 0.2031, 7.0826, 18.0995], "c": 0.7807},
    "Ga": {"a": [15.2354, 6.7006, 4.3591, 2.9623], "b": [3.0669, 0.2412, 10.7805, 61.4135], "c": 1.7189},
    "Ga3+": {"a": [12.692, 6.69883, 6.06692, 1.0066], "b": [2.81262, 0.22789, 6.36441, 14.4122], "c": 1.53545},
    "Ge": {"a": [16.0816, 6.3747, 3.7068, 3.683], "b": [2.8509, 0.2516, 11.4468, 54.7625], "c": 2.1313},
    "Ge4+": {"a": [12.9172, 6.70003, 6.06791, 0.859041], "b": [2.53718, 0.205855, 5.47913, 11.603], "c": 1.45572},
    "As": {"a": [16.6723, 6.0701, 3.4313, 4.2779], "b": [2.6345, 0.2647, 12.9479, 47.7972], "c": 2.531},
    "Se": {"a": [17.0006, 5.8196, 3.9731, 4.3543], "b": [2.4098, 0.2726, 15.2372, 43.8163], "c": 2.8409},
    "Br": {"a": [17.1789, 5.2358, 5.6377, 3.9851], "b": [2.1723, 16.5796, 0.2609, 41.4328], "c": 2.9557},
    "Br1-": {"a": [17.1718, 6.3338, 5.5754, 3.7272], "b": [2.2059, 19.3345, 0.2871, 58.1535], "c": 3.1776},
    "Kr": {"a": [17.3555, 6.7286, 5.5493, 3.5375], "b": [1.9384, 16.5623, 0.2261, 39.3972], "c": 2.825},
    "Rb": {"a": [17.1784, 9.6435, 5.1399, 1.5292], "b": [1.7888, 17.3151, 0.2748, 164.934], "c": 3.4873},
    "Rb1+": {"a": [17.5816, 7.6598, 5.8981, 2.7817], "b": [1.7139, 14.7957, 0.1603, 31.2087], "c": 2.0782},
    "Sr": {"a": [17.5663, 9.8184, 5.422, 2.6694], "b": [1.5564, 14.0988, 0.1664, 132.376], "c": 2.5064},
    "Sr2+": {"a": [18.0874, 8.1373, 2.5654, -34.193], "b": [1.4907, 12.6963, 24.5651, -0.0138], "c": 41.4025},
    "Y": {"a": [17.776, 10.2946, 5.72629, 3.26588], "b": [1.4029, 12.8006, 0.125599, 104.354], "c": 1.91213},
    "Y3+": {"a": [17.9268, 9.1531, 1.76795, -33.108], "b": [1.35417, 11.2145, 22.6599, -0.01319], "c": 40.2602},
    "Zr": {"a": [17.8765, 10.948, 5.41732, 3.65721], "b": [1.27618, 11.916, 0.117622, 87.6627], "c": 2.06929},
    "Zr4+": {"a": [18.1668, 10.0562, 1.01118, -2.6479], "b": [1.2148, 10.1483, 21.6054, -0.10276], "c": 9.41454},
    "Nb": {"a": [17.6142, 12.0144, 4.04183, 3.53346], "b": [1.18865, 11.766, 0.204785, 69.7957], "c": 3.75591},
    "Nb3+": {"a": [19.8812, 18.0653, 11.0177, 1.94715], "b": [0.019175, 1.13305, 10.1621, 28.3389], "c": -12.912},
    "Nb5+": {"a": [17.9163, 13.3417, 10.799, 0.337905], "b": [1.12446, 0.028781, 9.28206, 25.7228], "c": -6.3934},
    "Mo": {"a": [3.7025, 17.2356, 12.8876, 3.7429], "b": [0.2772, 1.0958, 11.004, 61.6584], "c": 4.3875},
    "Mo3+": {"a": [21.1664, 18.2017, 11.7423, 2.30951], "b": [0.014734, 1.03031, 9.53659, 26.6307], "c": -14.421},
    "Mo5+": {"a": [21.0149, 18.0992, 11.4632, 0.740625], "b": [0.014345, 1.02238, 8.78809, 23.3452], "c": -14.316},
    "Mo6+": {"a": [17.8871, 11.175, 6.57891, 0], "b": [1.03649, 8.48061, 0.058881, 0], "c": 0.344941},
    "Tc": {"a": [19.1301, 11.0948, 4.64901, 2.71263], "b": [0.864132, 8.14487, 21.5707, 86.8472], "c": 5.40428},
    "Ru": {"a": [19.2674, 12.9182, 4.86337, 1.56756], "b": [0.80852, 8.43467, 24.7997, 94.2928], "c": 5.37874},
    "Ru3+": {"a": [18.5638, 13.2885, 9.32602, 3.00964], "b": [0.847329, 8.37164, 0.017662, 22.887], "c": -3.1892},
    "Ru4+": {"a": [18.5003, 13.1787, 4.71304, 2.18535], "b": [0.844582, 8.12534, 0.36495, 20.8504], "c": 1.42357},
    "Rh": {"a": [19.2957, 14.3501, 4.73425, 1.28918], "b": [0.751536, 8.21758, 25.8749, 98.6062], "c": 5.328},
    "Rh3+": {"a": [18.8785, 14.1259, 3.32515, -6.1989], "b": [0.764252, 7.84438, 21.2487, -0.01036], "c": 11.8678},
    "Rh4+": {"a": [18.8545, 13.9806, 2.53464, -5.6526], "b": [0.760825, 7.62436, 19.3317, -0.0102], "c": 11.2835},
    "Pd": {"a": [19.3319, 15.5017, 5.29537, 0.605844], "b": [0.698655, 7.98929, 25.2052, 76.8986], "c": 5.26593},
    "Pd2+": {"a": [19.1701, 15.2096, 4.32234, 0], "b": [0.696219, 7.55573, 22.5057, 0], "c": 5.2916},
    "Pd4+": {"a": [19.2493, 14.79, 2.89289, -7.9492], "b": [0.683839, 7.14833, 17.9144, 0.005127], "c": 13.0174},
    "Ag": {"a": [19.2808, 16.6885, 4.8045, 1.0463], "b": [0.6446, 7.4726, 24.6605, 99.8156], "c": 5.179},
    "Ag1+": {"a": [19.1812, 15.9719, 5.27475, 0.357534], "b": [0.646179, 7.19123, 21.7326, 66.1147], "c": 5.21572},
    "Ag2+": {"a": [19.1643, 16.2456, 4.3709, 0], "b": [0.645643, 7.18544, 21.4072, 0], "c": 5.21404},
    "Cd": {"a": [19.2214, 17.6444, 4.461, 1.6029], "b": [0.5946, 6.9089, 24.7008, 87.4825], "c": 5.0694},
    "Cd2+": {"a": [19.1514, 17.2535, 4.47128, 0], "b": [0.597922, 6.80639, 20.2521, 0], "c": 5.11937},
    "In": {"a": [19.1624, 18.5596, 4.2948, 2.0396], "b": [0.5476, 6.3776, 25.8499, 92.8029], "c": 4.9391},
    "In3+": {"a": [19.1045, 18.1108, 3.78897, 0], "b": [0.551522, 6.3247, 17.3595, 0], "c": 4.99635},
    "Sn": {"a": [19.1889, 19.1005, 4.4585, 2.4663], "b": [5.8303, 0.5031, 26.8909, 83.9571], "c": 4.7821},
    "Sn2+": {"a": [19.1094, 19.0548, 4.5648, 0.487], "b": [0.5036, 5.8378, 23.3752, 62.2061], "c": 4.7861},
    "Sn4+": {"a": [18.9333, 19.7131, 3.4182, 0.0193], "b": [5.764, 0.4655, 14.0049, -0.7583], "c": 3.9182},
    "Sb": {"a": [19.6418, 19.0455, 5.0371, 2.6827], "b": [5.3034, 0.4607, 27.9074, 75.2825], "c": 4.5909},
    "Sb3+": {"a": [18.9755, 18.933, 5.10789, 0.288753], "b": [0.467196, 5.22126, 19.5902, 55.5113], "c": 4.69626},
    "Sb5+": {"a": [19.8685, 19.0302, 2.41253, 0], "b": [5.44853, 0.467973, 14.1259, 0], "c": 4.69263},
    "Te": {"a": [19.9644, 19.0138, 6.14487, 2.5239], "b": [4.81742, 0.420885, 28.5284, 70.8403], "c": 4.352},
    "I": {"a": [20.1472, 18.9949, 7.5138, 2.2735], "b": [4.347, 0.3814, 27.766, 66.8776], "c": 4.0712},
    "I1-": {"a": [20.2332, 18.997, 7.8069, 2.8868], "b": [4.3579, 0.3815, 29.5259, 84.9304], "c": 4.0714},
    "Xe": {"a": [20.2933, 19.0298, 8.9767, 1.99], "b": [3.9282, 0.344, 26.4659, 64.2658], "c": 3.7118},
    "Cs": {"a": [20.3892, 19.1062, 10.662, 1.4953], "b": [3.569, 0.3107, 24.3879, 213.904], "c": 3.3352},
    "Cs1+": {"a": [20.3524, 19.1278, 10.2821, 0.9615], "b": [3.552, 0.3086, 23.7128, 59.4565], "c": 3.2791},
    "Ba": {"a": [20.3361, 19.297, 10.888, 2.6959], "b": [3.216, 0.2756, 20.2073, 167.202], "c": 2.7731},
    "Ba2+": {"a": [20.1807, 19.1136, 10.9054, 0.77634], "b": [3.21367, 0.28331, 20.0558, 51.746], "c": 3.02902},
    "La": {"a": [20.578, 19.599, 11.3727, 3.28719], "b": [2.94817, 0.244475, 18.7726, 133.124], "c": 2.14678},
    "La3+": {"a": [20.2489, 19.3763, 11.6323, 0.336048], "b": [2.9207, 0.250698, 17.8211, 54.9453], "c": 2.4086},
    "Ce": {"a": [21.1671, 19.7695, 11.8513, 3.33049], "b": [2.81219, 0.226836, 17.6083, 127.113], "c": 1.86264},
    "Ce3+": {"a": [20.8036, 19.559, 11.9369, 0.612376], "b": [2.77691, 0.23154, 16.5408, 43.1692], "c": 2.09013},
    "Ce4+": {"a": [20.3235, 19.8186, 12.1233, 0.144583], "b": [2.65941, 0.21885, 15.7992, 62.2355], "c": 1.5918},
    "Pr": {"a": [22.044, 19.6697, 12.3856, 2.82428], "b": [2.77393, 0.222087, 16.7669, 143.644], "c": 2.0583},
    "Pr3+": {"a": [21.3727, 19.7491, 12.1329, 0.97518], "b": [2.6452, 0.214299, 15.323, 36.4065], "c": 1.77132},
    "Pr4+": {"a": [20.9413, 20.0539, 12.4668, 0.296689], "b": [2.54467, 0.202481, 14.8137, 45.4643], "c": 1.24285},
    "Nd": {"a": [22.6845, 19.6847, 12.774, 2.85137], "b": [2.66248, 0.210628, 15.885, 137.903], "c": 1.98486},
    "Nd3+": {"a": [21.961, 19.9339, 12.12, 1.51031], "b": [2.52722, 0.199237, 14.1783, 30.8717], "c": 1.47588},
    "Pm": {"a": [23.3405, 19.6095, 13.1235, 2.87516], "b": [2.5627, 0.202088, 15.1009, 132.721], "c": 2.02876},
    "Pm3+": {"a": [22.5527, 20.1108, 12.0671, 2.07492], "b": [2.4174, 0.185769, 13.1275, 27.4491], "c": 1.19499},
    "Sm": {"a": [24.0042, 19.4258, 13.4396, 2.89604], "b": [2.47274, 0.196451, 14.3996, 128.007], "c": 2.20963},
    "Sm3+": {"a": [23.1504, 20.2599, 11.9202, 2.71488], "b": [2.31641, 0.174081, 12.1571, 24.8242], "c": 0.954586},
    "Eu": {"a": [24.6274, 19.0886, 13.7603, 2.9227], "b": [2.3879, 0.1942, 13.7546, 123.174], "c": 2.5745},
    "Eu2+": {"a": [24.0063, 19.9504, 11.8034, 3.87243], "b": [2.27783, 0.17353, 11.6096, 26.5156], "c": 1.36389},
    "Eu3+": {"a": [23.7497, 20.3745, 11.8509, 3.26503], "b": [2.22258, 0.16394, 11.311, 22.9966], "c": 0.759344},
    "Gd": {"a": [25.0709, 19.0798, 13.8518, 3.54545], "b": [2.25341, 0.181951, 12.9331, 101.398], "c": 2.4196},
    "Gd3+": {"a": [24.3466, 20.4208, 11.8708, 3.7149], "b": [2.13553, 0.155525, 10.5782, 21.7029], "c": 0.645089},
    "Tb": {"a": [25.8976, 18.2185, 14.3167, 2.95354], "b": [2.24256, 0.196143, 12.6648, 115.362], "c": 3.58324},
    "Tb3+": {"a": [24.9559, 20.3271, 12.2471, 3.773], "b": [2.05601, 0.149525, 10.0499, 21.2773], "c": 0.691967},
    "Dy": {"a": [26.507, 17.6383, 14.5596, 2.96577], "b": [2.1802, 0.202172, 12.1899, 111.874], "c": 4.29728},
    "Dy3+": {"a": [25.5395, 20.2861, 11.9812, 4.50073], "b": [1.9804, 0.143384, 9.34972, 19.581], "c": 0.68969},
    "Ho": {"a": [26.9049, 17.294, 14.5583, 3.63837], "b": [2.07051, 0.19794, 11.4407, 92.6566], "c": 4.56796},
    "Ho3+": {"a": [26.1296, 20.0994, 11.9788, 4.93676], "b": [1.91072, 0.139358, 8.80018, 18.5908], "c": 0.852795},
    "Er": {"a": [27.6563, 16.4285, 14.9779, 2.98233], "b": [2.07356, 0.223545, 11.3604, 105.703], "c": 5.92046},
    "Er3+": {"a": [26.722, 19.7748, 12.1506, 5.17379], "b": [1.84659, 0.13729, 8.36225, 17.8974], "c": 1.17613},
    "Tm": {"a": [28.1819, 15.8851, 15.1542, 2.98706], "b": [2.02859, 0.238849, 10.9975, 102.961], "c": 6.75621},
    "Tm3+": {"a": [27.3083, 19.332, 12.3339, 5.38348], "b": [1.78711, 0.136974, 7.96778, 17.2922], "c": 1.63929},
    "Yb": {"a": [28.6641, 15.4345, 15.3087, 2.98963], "b": [1.9889, 0.257119, 10.6647, 100.417], "c": 7.56672},
    "Yb2+": {"a": [28.1209, 17.6817, 13.3335, 5.14657], "b": [1.78503, 0.15997, 8.18304, 20.39], "c": 3.70983},
    "Yb3+": {"a": [27.8917, 18.7614, 12.6072, 5.47647], "b": [1.73272, 0.13879, 7.64412, 16.8153], "c": 2.26001},
    "Lu": {"a": [28.9476, 15.2208, 15.1, 3.71601], "b": [1.90182, 9.98519, 0.261033, 84.3298], "c": 7.97628},
    "Lu3+": {"a": [28.4628, 18.121, 12.8429, 5.59415], "b": [1.68216, 0.142292, 7.33727, 16.3535], "c": 2.97573},
    "Hf": {"a": [29.144, 15.1726, 14.7586, 4.30013], "b": [1.83262, 9.5999, 0.275116, 72.029], "c": 8.58154},
    "Hf4+": {"a": [28.8131, 18.4601, 12.7285, 5.59927], "b": [1.59136, 0.128903, 6.76232, 14.0366], "c": 2.39699},
    "Ta": {"a": [29.2024, 15.2293, 14.5135, 4.76492], "b": [1.77333, 9.37046, 0.295977, 63.3644], "c": 9.24354},
    "Ta5+": {"a": [29.1587, 18.8407, 12.8268, 5.38695], "b": [1.50711, 0.116741, 6.31524, 12.4244], "c": 1.78555},
    "W": {"a": [29.0818, 15.43, 14.4327, 5.11982], "b": [1.72029, 9.2259, 0.321703, 57.056], "c": 9.8875},
    "W6+": {"a": [29.4936, 19.3763, 13.0544, 5.06412], "b": [1.42755, 0.104621, 5.93667, 11.1972], "c": 1.01074},
    "Re": {"a": [28.7621, 15.7189, 14.5564, 5.44174], "b": [1.67191, 9.09227, 0.3505, 52.0861], "c": 10.472},
    "Os": {"a": [28.1894, 16.155, 14.9305, 5.67589], "b": [1.62903, 8.97948, 0.382661, 48.1647], "c": 11.0005},
    "Os4+": {"a": [30.419, 15.2637, 14.7458, 5.06795], "b": [1.37113, 6.84706, 0.165191, 18.003], "c": 6.49804},
    "Ir": {"a": [27.3049, 16.7296, 15.6115, 5.83377], "b": [1.59279, 8.86553, 0.417916, 45.0011], "c": 11.4722},
    "Ir3+": {"a": [30.4156, 15.862, 13.6145, 5.82008], "b": [1.34323, 7.10909, 0.204633, 20.3254], "c": 8.27903},
    "Ir4+": {"a": [30.7058, 15.5512, 14.2326, 5.53672], "b": [1.30923, 6.71983, 0.167252, 17.4911], "c": 6.96824},
    "Pt": {"a": [27.0059, 17.7639, 15.7131, 5.7837], "b": [1.51293, 8.81174, 0.424593, 38.6103], "c": 11.6883},
    "Pt2+": {"a": [29.8429, 16.7224, 13.2153, 6.35234], "b": [1.32927, 7.38979, 0.263297, 22.9426], "c": 9.85329},
    "Pt4+": {"a": [30.9612, 15.9829, 13.7348, 5.92034], "b": [1.24813, 6.60834, 0.16864, 16.9392], "c": 7.39534},
    "Au": {"a": [16.8819, 18.5913, 25.5582, 5.86], "b": [0.4611, 8.6216, 1.4826, 36.3956], "c": 12.0658},
    "Au1+": {"a": [28.0109, 17.8204, 14.3359, 6.58077], "b": [1.35321, 7.7395, 0.356752, 26.4043], "c": 11.2299},
    "Au3+": {"a": [30.6886, 16.9029, 12.7801, 6.52354], "b": [1.2199, 6.82872, 0.212867, 18.659], "c": 9.0968},
    "Hg": {"a": [20.6809, 19.0417, 21.6575, 5.9676], "b": [0.545, 8.4484, 1.5729, 38.3246], "c": 12.6089},
    "Hg1+": {"a": [25.0853, 18.4973, 16.8883, 6.48216], "b": [1.39507, 7.65105, 0.443378, 28.2262], "c": 12.0205},
    "Hg2+": {"a": [29.5641, 18.06, 12.8374, 6.89912], "b": [1.21152, 7.05639, 0.284738, 20.7482], "c": 10.6268},
    "Tl": {"a": [27.5446, 19.1584, 15.538, 5.52593], "b": [0.65515, 8.70751, 1.96347, 45.8149], "c": 13.1746},
    "Tl1+": {"a": [21.3985, 20.4723, 18.7478, 6.82847], "b": [1.4711, 0.517394, 7.43463, 28.8482], "c": 12.5258},
    "Tl3+": {"a": [30.8695, 18.3481, 11.9328, 7.00574], "b": [1.1008, 6.53852, 0.219074, 17.2114], "c": 9.8027},
    "Pb": {"a": [31.0617, 13.0637, 18.442, 5.9696], "b": [0.6902, 2.3576, 8.618, 47.2579], "c": 13.4118},
    "Pb2+": {"a": [21.7886, 19.5682, 19.1406, 7.01107], "b": [1.3366, 0.488383, 6.7727, 23.8132], "c": 12.4734},
    "Pb4+": {"a": [32.1244, 18.8003, 12.0175, 6.96886], "b": [1.00566, 6.10926, 0.147041, 14.714], "c": 8.08428},
    "Bi": {"a": [33.3689, 12.951, 16.5877, 6.4692], "b": [0.704, 2.9238, 8.7937, 48.0093], "c": 13.5782},
    "Bi3+": {"a": [21.8053, 19.5026, 19.1053, 7.10295], "b": [1.2356, 6.24149, 0.469999, 20.3185], "c": 12.4711},
    "Bi5+": {"a": [33.5364, 25.0946, 19.2497, 6.91555], "b": [0.91654, 0.39042, 5.71414, 12.8285], "c": -6.7994},
    "Po": {"a": [34.6726, 15.4733, 13.1138, 7.02588], "b": [0.700999, 3.55078, 9.55642, 47.0045], "c": 13.677},
    "At": {"a": [35.3163, 19.0211, 9.49887, 7.42518], "b": [0.68587, 3.97458, 11.3824, 45.4715], "c": 13.7108},
    "Rn": {"a": [35.5631, 21.2816, 8.0037, 7.4433], "b": [0.6631, 4.0691, 14.0422, 44.2473], "c": 13.6905},
    "Fr": {"a": [35.9299, 23.0547, 12.1439, 2.11253], "b": [0.646453, 4.17619, 23.1052, 150.645], "c": 13.7247},
    "Ra": {"a": [35.763, 22.9064, 12.4739, 3.21097], "b": [0.616341, 3.87135, 19.9887, 142.325], "c": 13.6211},
    "Ra2+": {"a": [35.215, 21.67, 7.91342, 7.65078], "b": [0.604909, 3.5767, 12.601, 29.8436], "c": 13.5431},
    "Ac": {"a": [35.6597, 23.1032, 12.5977, 4.08655], "b": [0.589092, 3.65155, 18.599, 117.02], "c": 13.5266},
    "Ac3+": {"a": [35.1736, 22.1112, 8.19216, 7.05545], "b": [0.579689, 3.41437, 12.9187, 25.9443], "c": 13.4637},
    "Th": {"a": [35.5645, 23.4219, 12.7473, 4.80703], "b": [0.563359, 3.46204, 17.8309, 99.1722], "c": 13.4314},
    "Th4+": {"a": [35.1007, 22.4418, 9.78554, 5.29444], "b": [0.555054, 3.24498, 13.4661, 23.9533], "c": 13.376},
    "Pa": {"a": [35.8847, 23.2948, 14.1891, 4.17287], "b": [0.547751, 3.41519, 16.9235, 105.251], "c": 13.4287},
    "U": {"a": [36.0228, 23.4128, 14.9491, 4.188], "b": [0.5293, 3.3253, 16.0927, 100.613], "c": 13.3966},
    "U3+": {"a": [35.5747, 22.5259, 12.2165, 5.37073], "b": [0.52048, 3.12293, 12.7148, 26.3394], "c": 13.3092},
    "U4+": {"a": [35.3715, 22.5326, 12.0291, 4.7984], "b": [0.516598, 3.05053, 12.5723, 23.4582], "c": 13.2671},
    "U6+": {"a": [34.8509, 22.7584, 14.0099, 1.21457], "b": [0.507079, 2.8903, 13.1767, 25.2017], "c": 13.1665},
    "Np": {"a": [36.1874, 23.5964, 15.6402, 4.1855], "b": [0.511929, 3.25396, 15.3622, 97.4908], "c": 13.3573},
    "Np3+": {"a": [35.7074, 22.613, 12.9898, 5.43227], "b": [0.502322, 3.03807, 12.1449, 25.4928], "c": 13.2544},
    "Np4+": {"a": [35.5103, 22.5787, 12.7766, 4.92159], "b": [0.498626, 2.96627, 11.9484, 22.7502], "c": 13.2116},
    "Np6+": {"a": [35.0136, 22.7286, 14.3884, 1.75669], "b": [0.48981, 2.81099, 12.33, 22.6581], "c": 13.113},
    "Pu": {"a": [36.5254, 23.8083, 16.7707, 3.47947], "b": [0.499384, 3.26371, 14.9455, 105.98], "c": 13.3812},
    "Pu3+": {"a": [35.84, 22.7169, 13.5807, 5.66016], "b": [0.484938, 2.96118, 11.5331, 24.3992], "c": 13.1991},
    "Pu4+": {"a": [35.6493, 22.646, 13.3595, 5.18831], "b": [0.481422, 2.8902, 11.316, 21.8301], "c": 13.1555},
    "Pu6+": {"a": [35.1736, 22.7181, 14.7635, 2.28678], "b": [0.473204, 2.73848, 11.553, 20.9303], "c": 13.0582},
    "Am": {"a": [36.6706, 24.0992, 17.3415, 3.49331], "b": [0.483629, 3.20647, 14.3136, 102.273], "c": 13.3592},
    "Cm": {"a": [36.6488, 24.4096, 17.399, 4.21665], "b": [0.465154, 3.08997, 13.4346, 88.4834], "c": 13.2887},
    "Bk": {"a": [36.7881, 24.7736, 17.8919, 4.23284], "b": [0.451018, 3.04619, 12.8946, 86.003], "c": 13.2754},
    "Cf": {"a": [36.9185, 25.1995, 18.3317, 4.24391], "b": [0.437533, 3.00775, 12.4044, 83.7881], "c": 13.2674},
}







