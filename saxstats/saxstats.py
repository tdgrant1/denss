#!/usr/bin/env python
#
#    saxstats.py
#    SAXStats
#    A collection of python functions useful for solution scattering
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tgrant@hwi.buffalo.edu>
#    Copyright 2017, 2018 The Research Foundation for SUNY
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

import sys
import re
import os
import json
import struct
import logging
import argparse
import matplotlib
import numpy as np
from multiprocessing import Pool
import datetime, time
from scipy import ndimage
import matplotlib.pyplot as plt
from functools import partial
from _version import __version__
from scipy import stats, signal, interpolate, spatial, special, optimize, ndimage, linalg

def chi2(exp, calc, sig):
    """Return the chi2 discrepancy between experimental and calculated data"""
    return np.sum(np.square(exp - calc) / np.square(sig))

def center_rho(rho, centering="com"):
    """Move electron density map so its center of mass aligns with the center of the grid

    centering - which part of the density to center on. By default, center on the
                center of mass ("com"). Can also center on maximum density value ("max").
    """
    ne_rho= np.sum((rho))
    if centering == "max":
        rhocom = np.unravel_index(rho.argmax(), rho.shape)
    else:
        rhocom = np.array(ndimage.measurements.center_of_mass(rho))
    gridcenter = np.array(rho.shape)/2.
    shift = gridcenter-rhocom
    rho = ndimage.interpolation.shift(rho,shift,order=3,mode='wrap')
    rho = rho*ne_rho/np.sum(rho)
    return rho

def rho2rg(rho,side=None,r=None,support=None,dx=None):
    """Calculate radius of gyration from an electron density map."""
    if side is None and r is None:
        print "Error: To calculate Rg, must provide either side or r parameters."
        sys.exit()
    if side is not None and r is None:
        n = rho.shape[0]
        x_ = np.linspace(-side/2.,side/2.,n)
        x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
        r = np.sqrt(x**2 + y**2 + z**2)
    if support is None:
        support = np.ones_like(rho,dtype=bool)
    if dx is None:
        print "Error: To calculate Rg, must provide dx"
        sys.exit()
    rhocom = (np.array(ndimage.measurements.center_of_mass(rho))-np.array(rho.shape)/2.)*dx
    rg2 = np.sum(r[support]**2*rho[support])/np.sum(rho[support])
    rg2 = rg2 - np.linalg.norm(rhocom)**2
    rg = np.sign(rg2)*np.abs(rg2)**0.5
    return rg

def write_mrc(rho,side,filename="map.mrc"):
    """Write an MRC formatted electron density map.
       See here: http://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/#image
    """
    xs, ys, zs = rho.shape
    nxstart = -xs/2+1
    nystart = -ys/2+1
    nzstart = -zs/2+1
    with open(filename, "wb") as fout:
        # NC, NR, NS, MODE = 2 (image : 32-bit reals)
        fout.write(struct.pack('<iiii', xs, ys, zs, 2))
        # NCSTART, NRSTART, NSSTART
        fout.write(struct.pack('<iii', nxstart, nystart, nzstart))
        # MX, MY, MZ
        fout.write(struct.pack('<iii', xs, ys, zs))
        # X length, Y, length, Z length
        fout.write(struct.pack('<fff', side, side, side))
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
        fout.write(struct.pack('<fff', nxstart*(side/xs), nystart*(side/ys), nzstart*(side/zs)))
        # MAP
        fout.write('MAP ')
        # MACHST (little endian)
        fout.write(struct.pack('<BBBB', 0x44, 0x41, 0x00, 0x00))
        # RMS (std)
        fout.write(struct.pack('<f', np.std(rho)))
        # NLABL
        fout.write(struct.pack('<i', 0))
        # LABEL(20,10) 10 80-character text labels
        for i in xrange(0, 800):
            fout.write(struct.pack('<B', 0x00))

        # Write out data
        for k in range(zs):
            for j in range(ys):
                for i in range(xs):
                    s = struct.pack('<f', rho[i,j,k])
                    fout.write(s)

def read_mrc(filename="map.mrc"):
    """
        See MRC format at http://bio3d.colorado.edu/imod/doc/mrc_format.txt for offsets
    """
    with open(filename, 'rb') as fin:
        MRCdata=fin.read()
        nx = struct.unpack_from('<i',MRCdata, 0)
        ny = struct.unpack_from('<i',MRCdata, 4)
        nz = struct.unpack_from('<i',MRCdata, 8)
        nx,ny,nz = [np.array(i) for i in [nx,ny,nz]]
        rho_shape = (nx[0],ny[0],nz[0])
        
        
        side = struct.unpack_from('<f',MRCdata,40)[0]
        
        fin.seek(1024, os.SEEK_SET)
        rho = np.fromfile(file=fin, dtype=np.dtype(np.float32)).reshape(rho_shape)
        rho = rho.T
        fin.close()
        #print rho.shape
    return rho, side

#def average
#def enantiomer

def write_xplor(rho,side,filename="map.xplor"):
    """Write an XPLOR formatted electron density map."""
    xs, ys, zs = rho.shape
    title_lines = ['REMARK FILENAME="'+filename+'"','REMARK DATE= '+str(datetime.datetime.today())]
    with open(filename,'wb') as f:
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

    Jerr = np.array(Jerr)
    prepend = np.zeros((len(Ireg)-len(Jerr)))
    prepend += np.mean(Jerr[:10])
    Jerr = np.concatenate((prepend,Jerr))

    return np.array(qfull), np.array(Ireg), Jerr, results

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
        for each in hdict.iterkeys():
            if each != 'filename':
                results[each] = hdict[each]

    if 'analysis' in results:
        if 'GNOM' in results['analysis']:
            results = results['analysis']['GNOM']

    return q, i, err, results

def loadProfile(fname):
    """Determines which loading function to run, and then runs it."""

    if os.path.splitext(fname)[1] == '.out':
        q, I, Ierr, results = loadOutFile(fname)
        isout = True
    else:
        q, I, Ierr, results = loadDatFile(fname)
        isout = False

    keys = {key.lower().strip().translate(None, '_ '): key for key in results.keys()}

    if 'dmax' in keys:
        dmax = float(results[keys['dmax']])
    else:
        dmax = -1.

    return q, I, Ierr, dmax, isout

def denss(q, I, sigq, dmax, ne=None, voxel=5., oversampling=3., limit_dmax=False, limit_dmax_steps=[500],
        recenter=True, recenter_steps=None, recenter_mode="com", positivity=True, extrapolate=True,
        output="map", steps=None, seed=None,  minimum_density=None,  maximum_density=None, shrinkwrap=True, shrinkwrap_sigma_start=3,
        shrinkwrap_sigma_end=1.5, shrinkwrap_sigma_decay=0.99, shrinkwrap_threshold_fraction=0.2,
        shrinkwrap_iter=20, shrinkwrap_minstep=100, chi_end_fraction=0.01, write_xplor_format=False, write_freq=100,
        enforce_connectivity=True, enforce_connectivity_steps=[500],cutout=True,quiet=False):
    """Calculate electron density from scattering data."""
    
    D = dmax
    
    logging.info('q range of input data: %3.3f < q < %3.3f', q.min(), q.max())
    logging.info('Maximum dimension: %3.3f', D)
    logging.info('Sampling ratio: %3.3f', oversampling)
    logging.info('Requested real space voxel size: %3.3f', voxel)
    logging.info('Number of electrons: %3.3f', ne)
    logging.info('Limit Dmax: %s', limit_dmax)
    logging.info('Limit Dmax Steps: %s', limit_dmax_steps)
    logging.info('Recenter: %s', recenter)
    logging.info('Recenter Steps: %s', recenter_steps)
    logging.info('Recenter Mode: %s', recenter_mode)
    logging.info('Positivity: %s', positivity)
    logging.info('Minimum Density: %s', minimum_density)
    logging.info('Maximum Density: %s', maximum_density)
    logging.info('Extrapolate high q: %s', extrapolate)
    logging.info('Shrinkwrap: %s', shrinkwrap)
    logging.info('Shrinkwrap sigma start: %s', shrinkwrap_sigma_start)
    logging.info('Shrinkwrap sigma end: %s', shrinkwrap_sigma_end)
    logging.info('Shrinkwrap sigma decay: %s', shrinkwrap_sigma_decay)
    logging.info('Shrinkwrap threshold fraction: %s', shrinkwrap_threshold_fraction)
    logging.info('Shrinkwrap iterations: %s', shrinkwrap_iter)
    logging.info('Shrinkwrap starting step: %s', shrinkwrap_minstep)
    logging.info('Enforce connectivity: %s', enforce_connectivity)
    logging.info('Enforce connectivity steps: %s', enforce_connectivity_steps)
    logging.info('Chi2 end fraction: %3.3e', chi_end_fraction)

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
    realne = np.copy(ne)
    sigqdata = np.interp(qdata,q,sigq)
    scale_factor = realne**2 / Idata[0]
    Idata *= scale_factor
    sigqdata *= scale_factor
    if steps is None:
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
    support = np.ones(x.shape,dtype=bool)
    if seed is None:
        seed = np.random.randint(2**31-1)
    else:
        seed = int(seed)
    prng = np.random.RandomState(seed)
    rho = prng.random_sample(size=x.shape)
    update_support = True
    sigma = shrinkwrap_sigma_start
    #convert density values to absolute number of electrons
    #since FFT and rho given in electrons, not density, until converted at the end
    rho_min = minimum_density
    rho_max = maximum_density
    if rho_min is not None:
        rho_min *= dV
        #print rho_min
    if rho_max is not None:
        rho_max *= dV
        #print rho_max

    logging.info('Maximum number of steps: %i', steps)
    logging.info('Grid size (voxels): %i x %i x %i', n, n, n)
    logging.info('Real space box width (angstroms): %3.3f', side)
    logging.info('Real space box range (angstroms): %3.3f < x < %3.3f', x_.min(), x_.max())
    logging.info('Real space box volume (angstroms^3): %3.3f', V)
    logging.info('Real space voxel size (angstroms): %3.3f', dx)
    logging.info('Real space voxel volume (angstroms^3): %3.3f', dV)
    logging.info('Reciprocal space box width (angstroms^(-1)): %3.3f', qx_.max()-qx_.min())
    logging.info('Reciprocal space box range (angstroms^(-1)): %3.3f < qx < %3.3f', qx_.min(), qx_.max())
    logging.info('Maximum q vector (diagonal) (angstroms^(-1)): %3.3f', qr.max())
    logging.info('Number of q shells: %i', nbins)
    logging.info('Width of q shells (angstroms^(-1)): %3.3f', qstep)
    logging.info('Random seed: %i', seed)
    if not quiet:
        print "\n Step     Chi2     Rg    Support Volume"
        print " ----- --------- ------- --------------"

    for j in range(steps):
        F = np.fft.fftn(rho)
        #APPLY RECIPROCAL SPACE RESTRAINTS
        #calculate spherical average of intensities from 3D Fs
        I3D = np.abs(F)**2
        Imean[j] = ndimage.mean(I3D, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
        """
        if j==0:
            np.savetxt(output+'_step0_saxs.dat',np.vstack((qbinsc,Imean[j],Imean[j]*.05)).T,delimiter=" ",fmt="%.5e")
            #write_xplor(rho,side,output+"_original.xplor")
        """
        #scale Fs to match data
        factors = np.ones((len(qbins)))
        factors[qbin_args] = np.sqrt(Idata/Imean[j,qbin_args])
        F *= factors[qbin_labels]
        chi[j] = np.sum(((Imean[j,qbin_args]-Idata)/sigqdata)**2)/qbin_args.size
        #APPLY REAL SPACE RESTRAINTS
        rhoprime = np.fft.ifftn(F,rho.shape)
        rhoprime = rhoprime.real
        if j%write_freq == 0:
            if write_xplor_format:
                write_xplor(rhoprime,side,output+"_current.xplor")
            write_mrc(rhoprime,side,output+"_current.mrc")
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
        #allow further bounds on density, rather than just positivity
        if rho_min is not None:
            netmp = np.sum(newrho)
            newrho[newrho<rho_min] = rho_min
            if np.sum(newrho) != 0:
                newrho *= netmp / np.sum(newrho)
        if rho_max is not None:
            netmp = np.sum(newrho)
            newrho[newrho>rho_max] = rho_max
            if np.sum(newrho) != 0:
                newrho *= netmp / np.sum(newrho)
        #update support using shrinkwrap method
        if recenter and j in recenter_steps:
            if recenter_mode == "max":
                rhocom = np.unravel_index(newrho.argmax(), newrho.shape)
            else:
                rhocom = np.array(ndimage.measurements.center_of_mass(newrho))
            gridcenter = np.array(rho.shape)/2.
            shift = gridcenter-rhocom
            shift = shift.astype(int)
            newrho = np.roll(np.roll(np.roll(newrho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
            support = np.roll(np.roll(np.roll(support, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
        if shrinkwrap and j >= shrinkwrap_minstep and j%shrinkwrap_iter==0:
            rho_blurred = ndimage.filters.gaussian_filter(newrho,sigma=sigma,mode='wrap')
            support = np.zeros(rho.shape,dtype=bool)
            support[rho_blurred >= shrinkwrap_threshold_fraction*rho_blurred.max()] = True
            if sigma > shrinkwrap_sigma_end:
                sigma = shrinkwrap_sigma_decay*sigma
            if enforce_connectivity and j in enforce_connectivity_steps:
                #label the support into separate segments based on a 3x3x3 grid
                struct = ndimage.generate_binary_structure(3, 3)
                labeled_support, num_features = ndimage.label(support, structure=struct)
                sums = np.zeros((num_features))
                
                #find the feature with the greatest number of electrons
                for feature in range(num_features):
                    sums[feature-1] = np.sum(newrho[labeled_support==feature])
                big_feature = np.argmax(sums)+1
                #remove features from the support that are not the primary feature
                support[labeled_support != big_feature] = False
        if limit_dmax and j in limit_dmax_steps:
            support[r>0.6*D] = False
            if np.sum(support) <= 0:
                support = np.ones(rho.shape,dtype=bool)
        supportV[j] = np.sum(support)*dV

        if not quiet:
            #sys.stdout.write("\r ----- --------- ------- --------------")
            sys.stdout.write("\r% 5i % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
            sys.stdout.flush()

        if j > 101 + shrinkwrap_minstep and np.std(chi[j-100:j]) < chi_end_fraction * np.median(chi[j-100:j]):
            break

        rho = newrho
    if not quiet:
        print

    F = np.fft.fftn(rho)
    #calculate spherical average intensity from 3D Fs
    Imean[j+1] = ndimage.mean(np.abs(F)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    #chi[j+1] = np.sum(((Imean[j+1,qbin_args]-Idata)/sigqdata)**2)/qbin_args.size
    #scale Fs to match data
    factors = np.ones((len(qbins)))
    factors[qbin_args] = np.sqrt(Idata/Imean[j+1,qbin_args])
    F *= factors[qbin_labels]
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
        if nD%2==1: nD += 1
        min = nbox/2 - nD/2
        max = nbox/2 + nD/2 + 2
        #create new rho array containing only the particle
        newrho = rho[min:max,min:max,min:max]
        rho = newrho
        #do the same for the support
        newsupport = support[min:max,min:max,min:max]
        support = newsupport
        #update side to new size of box
        side = dx * (max-min)

    if write_xplor_format:
        write_xplor(rho,side,output+".xplor")
        write_xplor(np.ones_like(rho)*support,side,output+"_support.xplor")
    write_mrc(rho,side,output+".mrc")
    write_mrc(np.ones_like(rho)*support,side,output+"_support.mrc")

    logging.info('Number of steps: %i', j)
    logging.info('Final Chi2: %.3e', chi[j])
    logging.info('Final Rg: %3.3f', rg[j+1])
    logging.info('Final Support Volume: %3.3f', supportV[j+1])

    #return original unscaled values of Idata (and therefore Imean) for comparison with real data
    Idata /= scale_factor
    sigqdata /= scale_factor
    Imean /= scale_factor

    return qdata, Idata, sigqdata, qbinsc, Imean[j], chi, rg, supportV, rho, side

def center_rho_roll(rho):
    """Move electron density map so its center of mass aligns with the center of the grid"""
    rhocom = np.array(ndimage.measurements.center_of_mass(rho))
    #rhocom = np.unravel_index(rho.argmax(), rho.shape)
    gridcenter = np.array(rho.shape)/2.
    shift = gridcenter-rhocom
    shift = shift.astype(int)
    print rhocom
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    print np.array(ndimage.measurements.center_of_mass(rho))
    return rho

def minimize_rho(refrho, movrho, T = np.zeros(6)):
    """Optimize superposition of electron density maps. Move movrho to refrho."""
    #print "initial score: ", 1/rho_overlap_score(refrho,movrho)
    bounds = np.zeros(12).reshape(6,2)
    bounds[:3,0] = -20*np.pi
    bounds[:3,1] = 20*np.pi
    bounds[3:,0] = -5
    bounds[3:,1] = 5
    save_movrho = np.copy(movrho)
    save_refrho = np.copy(refrho)
    result = optimize.fmin_l_bfgs_b(minimize_rho_score, T, factr= 0.1, maxiter=100, maxfun=200, epsilon=0.05, args=(refrho,movrho),bounds=bounds, approx_grad=True)
    Topt = result[0]
    newrho = transform_rho(save_movrho, Topt, order=2)
    finalscore = 1/rho_overlap_score(save_refrho,newrho)
    return newrho, finalscore

def minimize_rho_score(T, refrho, movrho):
    """Scoring function for superposition of electron density maps.
        
        refrho - fixed, reference rho
        movrho - moving rho
        T - 6-element list containing alpha, beta, gamma, Tx, Ty, Tz in that order
        to move movrho by.
        """
    newrho=transform_rho(movrho, T)
    score = rho_overlap_score(refrho,newrho)
    return score

def rho_overlap_score(rho1,rho2):
    """Scoring function for superposition of electron density maps."""
    n=2*np.sum(np.abs(rho1*rho2))
    d=(2*np.sum(rho1**2)**0.5*np.sum(rho2**2)**0.5)
    score = n/d
    #1/score for least squares minimization, i.e. want to minimize, not maximize score
    return 1/score

def transform_rho(rho, T, order=1):
    """ Rotate and translate electron density map by T vector.
    
        T = [alpha, beta, gamma, x, y, z], angles in radians
        order = interpolation order (0-5)
    """
    ne_rho= np.sum((rho))
    R = euler2matrix(T[0],T[1],T[2])
    coordinates=np.array(ndimage.measurements.center_of_mass(rho))
    offset=coordinates-np.dot(coordinates,R)
    newrho = ndimage.interpolation.affine_transform(rho,R.T,order=order,offset=offset,output=np.float64,mode='wrap')
    newrho = ndimage.interpolation.shift(newrho,T[3:],order=order,mode='wrap',output=np.float64)
    newrho = newrho*ne_rho/np.sum(newrho)
    return newrho

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
    w,v = np.linalg.eigh(I.T)
    return w,v

def principal_axis_alignment(refrho,movrho):
    """ Align movrho principal axes to refrho."""
    side = 1.0
    ne_movrho= np.sum((movrho))
    I1=inertia_tensor(refrho, side)
    w1,v1=principal_axes(I1)
    I2=inertia_tensor(movrho, side)
    w2,v2=principal_axes(I2)
    R1=v1
    R2=v2
    R=np.dot(R1,R2.T)
    coordinates=np.array(ndimage.measurements.center_of_mass(movrho))
    offset = coordinates - np.dot(coordinates,R)
    newrho = ndimage.interpolation.affine_transform(movrho,R.T,order=1, output=np.float64, offset=offset, mode='wrap')
    newrho = newrho*ne_movrho/np.sum(newrho)
    return newrho

def align2xyz(rho):
    """ Align rho such that principal axes align with XYZ axes."""
    side = 1.0
    ne_rho= np.sum((rho))
    rho = center_rho(rho)
    I = inertia_tensor(rho, side)
    w,v = principal_axes(I)
    R = v.T
    coordinates=np.array(ndimage.measurements.center_of_mass(rho))
    offset=coordinates-np.dot(coordinates,R)
    newrho = ndimage.interpolation.affine_transform(rho,R.T,order=1, offset=offset, mode='wrap')
    newrho = newrho*ne_rho/np.sum(newrho)
    return newrho

def generate_enantiomers(rho):
    """ Generate all enantiomers of given density map.
        Output maps are flipped over x,y,z,xy,yz,zx, and xyz, respectively
        """
    rho = align2xyz(rho)
    rho_xflip = rho[::-1,:,:]
    rho_yflip = rho[:,::-1,:]
    rho_zflip = rho[:,:,::-1]
    rho_xyflip = rho_xflip[:,::-1,:]
    rho_yzflip = rho_yflip[:,:,::-1]
    rho_zxflip = rho_zflip[::-1,:,:]
    rho_xyzflip = rho_xyflip[:,:,::-1]
    enans = np.array([rho,rho_xflip,rho_yflip,rho_zflip,
                      rho_xyflip,rho_yzflip,rho_zxflip,
                      rho_xyzflip])
    return enans

def align(refrho,movrho):
    """ Align second electron density map to the first."""
    ne_rho= np.sum((movrho))
    rhocom = np.array(ndimage.measurements.center_of_mass(refrho))
    gridcenter = np.array(refrho.shape)/2.
    shift = gridcenter-rhocom
    newrefrho = ndimage.interpolation.shift(refrho,shift,order=3,mode='wrap')
    movrho = principal_axis_alignment(newrefrho, movrho)
    movrho, score = minimize_rho(newrefrho, movrho)
    movrho = ndimage.interpolation.shift(movrho,-shift,order=3,mode='wrap')
    movrho = movrho*ne_rho/np.sum(movrho)
    return movrho, score

def multi_align(niter, **kwargs):
    """ Wrapper script for alignment for multiprocessing."""
    try:
        kwargs['refrho']=kwargs['refrho'][niter]
        if niter >= kwargs['movrho'].shape[0]-1:
            print "\r Finishing alignment: %i / %i" % (niter+1, kwargs['movrho'].shape[0])
        else:
            sys.stdout.write( "\r Running alignment: %i / %i" % (niter+1, kwargs['movrho'].shape[0]))
            sys.stdout.flush()
        kwargs['movrho']=kwargs['movrho'][niter]
        time.sleep(1)
        return align(**kwargs)
    except KeyboardInterrupt:
        pass

def average_two(rho1, rho2):
    """ Align and average two electron density maps and return the average."""
    rho2, score = align(rho1, rho2)
    average_rho = (rho1+rho2)/2
    return average_rho

def multi_average_two(niter, **kwargs):
    """ Wrapper script for averaging two maps for multiprocessing."""
    try:
        kwargs['rho1']=kwargs['rho1'][niter]
        kwargs['rho2']=kwargs['rho2'][niter]
        time.sleep(1)
        return average_two(**kwargs)
    except KeyboardInterrupt:
        pass

def average_pairs(rhos, cores = 1):
    """ Average pairs of electron density maps, second half to first half."""
    #create even/odd pairs, odds are the references
    rho_args = {'rho1':rhos[::2], 'rho2':rhos[1::2]}
    pool = Pool(cores)
    try:
        mapfunc = partial(multi_average_two, **rho_args)
        average_rhos = pool.map(mapfunc, range(rhos.shape[0]/2))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.close()
        sys.exit(1)
    return np.array(average_rhos)

def binary_average(rhos, cores=1):
    """ Generate a reference electron density map using binary averaging."""
    twos = 2**np.arange(20)
    nmaps = np.max(twos[twos<=rhos.shape[0]])
    levels = int(np.log2(nmaps))-1
    rhos = rhos[:nmaps]
    for level in range(levels):
         rhos = average_pairs(rhos,cores)
    refrho = center_rho(rhos[0])
    print "\n Generating reference is complete."
    return refrho

def align_multiple(refrho, rhos, cores = 1):
    """ Align each map in the set to the reference."""
    #need to make a duplicate of refrho nmaps times to feed as separate
    #arguments to pool.map
    rho_args = {'refrho':np.array([refrho for i in range(rhos.shape[0])]), 'movrho':rhos}
    pool = Pool(cores)
    try:
        mapfunc = partial(multi_align, **rho_args)
        results = pool.map(mapfunc, range(rhos.shape[0]))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.close()
        sys.exit(1)
    aligned_rhos = np.array([results[i][0] for i in range(len(results))])
    scores = np.array([results[i][1] for i in range(len(results))])
    return aligned_rhos, scores

def select_best_enantiomer(refrho, rho, cores = 1):
    """ Generate, align and select the best enantiomer. """
    enans = generate_enantiomers(rho)
    aligned, scores = align_multiple(refrho, enans, cores)
    best_i = np.argmax(scores)
    best_enan, score = aligned[best_i], scores[best_i]
    return best_enan, score

def select_best_enantiomers(refrho, rhos, cores=1):
    """ Generate, align and select the best enantiomer for each map in the set."""
    best_enans = []
    scores = []
    for i in range(rhos.shape[0]):
        best_enan, score = select_best_enantiomer(refrho, rhos[i], cores)
        best_enans.append(best_enan)
        scores.append(score)
    best_enans = np.array(best_enans)
    scores = np.array(scores)
    return best_enans, scores

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
    numerator = ndimage.sum(np.real(F1*np.conj(F2)), labels=qbin_labels, index = np.arange(0,qbin_labels.max()+1))
    term1 = ndimage.sum(np.abs(F1)**2, labels=qbin_labels, index = np.arange(0,qbin_labels.max()+1))
    term2 = ndimage.sum(np.abs(F2)**2, labels=qbin_labels, index = np.arange(0,qbin_labels.max()+1))
    denominator = (term1*term2)**0.5
    #print "numerator ", numerator
    #print "denominator ", denominator
    FSC = numerator/denominator
    qidx = np.where(qbins<qx_max)
    return  np.vstack((qbins[qidx],FSC[qidx])).T

class PDB(object):
    """Load pdb file."""
    def __init__(self, filename):
        self.natoms = 0
        with open(filename) as f:
            for line in f:
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                self.natoms += 1
        self.atomnum = np.zeros((self.natoms),dtype=int)
        self.atomname = np.zeros((self.natoms),dtype=np.dtype((str,3)))
        self.atomalt = np.zeros((self.natoms),dtype=np.dtype((str,1)))
        self.resname = np.zeros((self.natoms),dtype=np.dtype((str,3)))
        self.resnum = np.zeros((self.natoms),dtype=int)
        self.chain = np.zeros((self.natoms),dtype=np.dtype((str,1)))
        self.coords = np.zeros((self.natoms, 3))
        self.occupancy = np.zeros((self.natoms))
        self.b = np.zeros((self.natoms))
        self.atomtype = np.zeros((self.natoms),dtype=np.dtype((str,2)))
        self.charge = np.zeros((self.natoms),dtype=np.dtype((str,2)))
        self.nelectrons = np.zeros((self.natoms),dtype=int)
        with open(filename) as f:
            atom = 0
            for line in f:
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                self.atomnum[atom] = int(line[6:11])
                self.atomname[atom] = line[13:16]
                self.atomalt[atom] = line[16]
                self.resname[atom] = line[17:20]
                self.resnum[atom] = int(line[22:26])
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
                self.charge[atom] = line[78:80]
                self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(),6)
                atom += 1

    def write(self, filename):
        """Write PDB file format using pdb object as input."""
        records = []
        for i in range(self.natoms):
            atomnum = '%5i' % self.atomnum[i]
            atomname = '%3s' % self.atomname[i]
            atomalt = '%1s' % self.atomalt[i]
            resnum = '%4i' % self.resnum[i]
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
        np.savetxt(filename, records, fmt = '%80s')

def pdb2map_gauss(pdb,xyz,sigma):
    """Simple isotropic gaussian sum at coordinate locations."""
    n = int(round(xyz.shape[0]**(1/3.)))
    sigma /= 4.
    #dist = spatial.distance.cdist(coords, xyz)
    #rho = np.sum(values,axis=0).reshape(n,n,n)
    #run cdist in a loop over atoms to avoid overloading memory
    values = np.zeros((xyz.shape[0]))
    print "\n Read density map from PDB... "
    for i in range(pdb.coords.shape[0]):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        sys.stdout.flush()
        dist = spatial.distance.cdist(pdb.coords[None,i], xyz)
        dist *= dist
        values += pdb.nelectrons[i]*1./np.sqrt(2*np.pi*sigma**2) * np.exp(-dist[0]/(2*sigma**2))
    print
    rho = values.reshape(n,n,n)
    return rho


electrons = {'H': 1,'HE': 2,'He': 2,'LI': 3,'Li': 3,'BE': 4,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'NE': 10,'Ne': 10,'NA': 11,'Na': 11,'MG': 12,'Mg': 12,'AL': 13,'Al': 13,'SI': 14,'Si': 14,'P': 15,'S': 16,'CL': 17,'Cl': 17,'AR': 18,'Ar': 18,'K': 19,'CA': 20,'Ca': 20,'SC': 21,'Sc': 21,'TI': 22,'Ti': 22,'V': 23,'CR': 24,'Cr': 24,'MN': 25,'Mn': 25,'FE': 26,'Fe': 26,'CO': 27,'Co': 27,'NI': 28,'Ni': 28,'CU': 29,'Cu': 29,'ZN': 30,'Zn': 30,'GA': 31,'Ga': 31,'GE': 32,'Ge': 32,'AS': 33,'As': 33,'SE': 34,'Se': 34,'Se': 34,'Se': 34,'BR': 35,'Br': 35,'KR': 36,'Kr': 36,'RB': 37,'Rb': 37,'SR': 38,'Sr': 38,'Y': 39,'ZR': 40,'Zr': 40,'NB': 41,'Nb': 41,'MO': 42,'Mo': 42,'TC': 43,'Tc': 43,'RU': 44,'Ru': 44,'RH': 45,'Rh': 45,'PD': 46,'Pd': 46,'AG': 47,'Ag': 47,'CD': 48,'Cd': 48,'IN': 49,'In': 49,'SN': 50,'Sn': 50,'SB': 51,'Sb': 51,'TE': 52,'Te': 52,'I': 53,'XE': 54,'Xe': 54,'CS': 55,'Cs': 55,'BA': 56,'Ba': 56,'LA': 57,'La': 57,'CE': 58,'Ce': 58,'PR': 59,'Pr': 59,'ND': 60,'Nd': 60,'PM': 61,'Pm': 61,'SM': 62,'Sm': 62,'EU': 63,'Eu': 63,'GD': 64,'Gd': 64,'TB': 65,'Tb': 65,'DY': 66,'Dy': 66,'HO': 67,'Ho': 67,'ER': 68,'Er': 68,'TM': 69,'Tm': 69,'YB': 70,'Yb': 70,'LU': 71,'Lu': 71,'HF': 72,'Hf': 72,'TA': 73,'Ta': 73,'W': 74,'RE': 75,'Re': 75,'OS': 76,'Os': 76,'IR': 77,'Ir': 77,'PT': 78,'Pt': 78,'AU': 79,'Au': 79,'HG': 80,'Hg': 80,'TL': 81,'Tl': 81,'PB': 82,'Pb': 82,'BI': 83,'Bi': 83,'PO': 84,'Po': 84,'AT': 85,'At': 85,'RN': 86,'Rn': 86,'FR': 87,'Fr': 87,'RA': 88,'Ra': 88,'AC': 89,'Ac': 89,'TH': 90,'Th': 90,'PA': 91,'Pa': 91,'U': 92,'NP': 93,'Np': 93,'PU': 94,'Pu': 94,'AM': 95,'Am': 95,'CM': 96,'Cm': 96,'BK': 97,'Bk': 97,'CF': 98,'Cf': 98,'ES': 99,'Es': 99,'FM': 100,'Fm': 100,'MD': 101,'Md': 101,'NO': 102,'No': 102,'LR': 103,'Lr': 103,'RF': 104,'Rf': 104,'DB': 105,'Db': 105,'SG': 106,'Sg': 106,'BH': 107,'Bh': 107,'HS': 108,'Hs': 108,'MT': 109,'Mt': 109}









