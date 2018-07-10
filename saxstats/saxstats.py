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
#    Copyright 2017 The Research Foundation for SUNY
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
import logging
import datetime
import re
import os
import json
import struct
import numpy as np
from scipy import ndimage, special
from collections import OrderedDict

def chi2(exp, calc, sig):
    """Return the chi2 discrepancy between experimental and calculated data"""
    return np.sum(np.square(exp - calc) / np.square(sig))

def center_rho(rho,centering="com"):
    """Move electron density map so its center of mass aligns with the center of the grid

    centering - which part of the density to center on. By default, center on the
                center of mass ("com"). Can also center on maximum density value ("max").
    """
    if centering == "max":
        rhocom = np.unravel_index(rho.argmax(), rho.shape)
    else:
        rhocom = np.array(ndimage.measurements.center_of_mass(rho))
    gridcenter = np.array(rho.shape)/2.
    shift = gridcenter-rhocom
    rho = ndimage.interpolation.shift(rho,shift,order=3,mode='wrap')
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
        # ISPG, NSYMBT, LSKFLG
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

def read_mrc(filename):
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
    
    return rho, side

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

def denss(q, I, sigq, D, ne=None, voxel=5., oversampling=3., limit_dmax=False, limit_dmax_steps=[500],
        recenter=True, recenter_steps=None, recenter_mode="com", positivity=True, extrapolate=True,write=True,
        filename="map", steps=None, seed=None, rho_min=None, rho_max=None, shrinkwrap=True, shrinkwrap_sigma_start=3,
        shrinkwrap_sigma_end=1.5, shrinkwrap_sigma_decay=0.99, shrinkwrap_threshold_fraction=0.2,
        shrinkwrap_iter=20, shrinkwrap_minstep=100, chi_end_fraction=0.01, write_xplor_format=False, write_freq=100,
        enforce_connectivity=True, enforce_connectivity_steps=[500],cutout=True):
    """Calculate electron density from scattering data."""
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

    print "Step  Chi2      Rg      Support Volume"
    print "----- --------- ------- --------------"
    for j in range(steps):
        F = np.fft.fftn(rho)
        #APPLY RECIPROCAL SPACE RESTRAINTS
        #calculate spherical average of intensities from 3D Fs
        I3D = np.abs(F)**2
        Imean[j] = ndimage.mean(I3D, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
        """
        if j==0:
            np.savetxt(filename+'_step0_saxs.dat',np.vstack((qbinsc,Imean[j],Imean[j]*.05)).T,delimiter=" ",fmt="%.5e")
            #write_xplor(rho,side,filename+"_original.xplor")
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
                write_xplor(rhoprime,side,filename+"_current.xplor")
            write_mrc(rhoprime,side,filename+"_current.mrc")
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
                print num_features
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
        sys.stdout.write("\r% 5i % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
        sys.stdout.flush()

        if j > 101 + shrinkwrap_minstep and np.std(chi[j-100:j]) < chi_end_fraction * np.median(chi[j-100:j]):
            break

        rho = newrho
        
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
        write_xplor(rho,side,filename+".xplor")
        write_xplor(np.ones_like(rho)*support,side,filename+"_support.xplor")
    write_mrc(rho,side,filename+".mrc")
    write_mrc(np.ones_like(rho)*support,side,filename+"_support.mrc")

    logging.info('Number of steps: %i', j)
    logging.info('Final Chi2: %.3e', chi[j])
    logging.info('Final Rg: %3.3f', rg[j+1])
    logging.info('Final Support Volume: %3.3f', supportV[j+1])

    #return original unscaled values of Idata (and therefore Imean) for comparison with real data
    Idata /= scale_factor
    sigqdata /= scale_factor
    Imean /= scale_factor

    return qdata, Idata, sigqdata, qbinsc, Imean[j], chi, rg, supportV

class Sasrec(object):
    def __init__(self, Iq, D, qc=None, r=None, alpha=0.0, ne=2):
        self.q = Iq[:,0]
        self.I = Iq[:,1]
        self.Ierr = Iq[:,2]
        self.q.clip(1e-10)
        self.I[np.abs(self.I)<1e-10] = 1e-10
        self.Ierr.clip(1e-10)
        self.D = D
        self.qmin = np.min(self.q)
        self.qmax = np.max(self.q)
        self.nq = len(self.q)
        self.qi = np.arange(self.nq)
        if qc is None:
            self.qc = self.q
        else:
            self.qc = qc
        if r is None:
            self.nr = self.nq
            self.r = np.linspace(0,self.D,self.nr)
        else:
            self.r = r
            self.nr = len(self.r)
        self.alpha = alpha
        self.ne = ne
        self.update()

    def update(self):
        self.ri = np.arange(self.nr)
        self.n = self.shannon_channels(self.qmax,self.D) + self.ne
        self.Ni = np.arange(self.n)
        self.N = self.Ni + 1
        self.Mi = np.copy(self.Ni)
        self.M = np.copy(self.N)
        self.qn = np.pi/self.D * self.N
        self.In = np.zeros((self.nq))
        self.Inerr = np.zeros((self.nq))
        self.B = np.zeros((self.n, self.nq))
        self.S = np.zeros((self.n, self.nr))
        self.Y = np.zeros((self.n))
        self.C = np.zeros((self.n, self.n))
        self.B[self.Ni] = self.Bt(self.N[:, None], self.q, self.D)
        self.S[self.Ni] = self.St(self.N[:, None], self.r, self.D)
        self.Y[self.Mi] = self.Yt(I=self.I, Ierr=self.Ierr, Bm=self.B[self.Mi])
        self.C[self.Mi[:, None], self.Ni] = self.Ct2(Ierr=self.Ierr, Bm=self.B[self.Mi], Bn=self.B[self.Ni], alpha=self.alpha, D=self.D)
        self.Cinv = np.linalg.inv(self.C)
        self.In = 0.5*np.linalg.solve(self.C,self.Y)
        #self.In = 0.5*optimize.nnls(self.C, self.Y)[0]
        self.Inerr = 0.5*(np.diagonal(self.Cinv))**(0.5)
        self.Ic = self.Ish2Iq(Ish=self.In,D=self.D,q=self.qc)[:,1]
        self.P = self.Ish2P(self.In,self.D,self.r)
        self.Perr = self.Perrf(r=self.r, D=self.D, Sn=self.S[self.Ni[:, None], self.ri], Sm=self.S[self.Mi[:, None], self.ri], Cinv=self.Cinv)
        self.I0 = self.Ish2I0(self.In,self.D)
        self.I0err = self.I0errf(self.Cinv)
        self.rg2 = self.Ish2rg2(self.In,self.D)
        self.rg = self.Ish2rg(self.In,self.D)
        self.rgerr = self.rgerrf(self.In,self.D,self.Cinv)
        self.ravg = self.Ish2ravg(self.In,self.D)
        self.ravgerr = self.ravgerrf(self.In,self.D,self.Cinv)
        self.Q = self.Ish2Q(self.In,self.D)
        self.Qerr = self.Qerrf(self.D,self.Cinv)
        self.Vp = self.Ish2Vp(self.In,self.D)
        self.Vperr = self.Vperrf(self.In,self.D,self.Cinv)
        self.mwVp = self.Ish2mwVp(self.In,self.D)
        self.mwVperr = self.mwVperrf(self.In,self.D,self.Cinv)
        self.Vc = self.Ish2Vc(self.In,self.D)
        self.Vcerr = self.Vcerrf(self.In,self.D,self.Cinv)
        self.mwVc = self.Ish2mwVc(self.In,self.D)
        self.mwVcerr = self.mwVcerrf(self.In,self.D,self.Cinv)
        self.lc = self.Ish2lc(self.In,self.D)
        self.lcerr = self.lcerrf(self.In,self.D,self.Cinv)


    def shannon_channels(self, D, qmax=0.5, qmin=0.0):
        """Return the number of Shannon channels given a q range and maximum particle dimension"""
        width = np.pi / D
        num_channels = int((qmax-qmin) / width)
        return num_channels

    def Bt(self,n,q,D):
        return (n*np.pi)**2/((n*np.pi)**2-(q*D)**2) * np.sinc(q*D/np.pi) * (-1)**(n+1)

    def St(self,n,r,D):
        return r/D**2 * n * np.sin(n*np.pi*r/D)

    def Yt(self,I, Ierr, Bm):
        """Return the values of Y, an m-length vector."""
        return np.einsum('i, ki->k', I/Ierr**2, Bm)

    def Ct(self,Ierr, Bm, Bn):
        """Return the values of C, a m x n variance-covariance matrix"""
        return np.einsum('ij,kj->ik', Bm/Ierr**2, Bn)

    def Amn(self,m,n,D):
        """Return the mxn matrix of coefficients for the integral of (2nd deriv of P(r))**2 used for smoothing"""
        m = np.atleast_1d(m).astype(float)
        n = np.atleast_1d(n).astype(float)
        nm = len(m)
        nn = len(n)
        amn = np.zeros((nm,nn))
        for i in range(nm):
            for j in range(nn):
                if m[i] != n[j]:
                    amn[i,j] = np.pi**2/(2*D**5) * (m[i]*n[j])**2 * (m[i]**4+n[j]**4)/(m[i]**2+n[j]**2)**2 * (-1)**(m[i]+n[j])
                if m[i] == n[j]:
                    amn[i,j] = np.pi**4/(2*D**5) * n[j]**6
        return amn

    def Ct2(self,Ierr, Bm, Bn, alpha, D):
        """Return the values of C, a m x n variance-covariance matrix while smoothing P(r)"""
        m = Bm.shape[0]
        n = Bn.shape[0]
        ni = Bm.shape[1]
        M = np.arange(m)+1
        N = np.arange(n)+1
        amn = self.Amn(M,N,D)
        return ni/8. * alpha * amn + np.einsum('ij,kj->ik', Bm/Ierr**2, Bn)

    def Perrt(self,r, Sn, Sm, Cinv):
        """Return the standard errors on P(r).
            
            To work, Sn will generally equal something like S[N[:, None], pts] where N is
            the array of n shannon channels, same for Sm, and pts is the array of
            indices for the number of data points, e.g. np.arange(npts).
            This is required for correct broadcasting.
            Example: Perr(r=r, Sn=S[N[:, None], pts]), Sm=S[M[:, None], pts], Cinv=Cinv)
            """
        return r*np.einsum('ni,mi,nm->i', Sn, Sm, Cinv)**(.5)

    def Iq2Ish(self,q,I,sigq,D,ne=2):
        """Calculate Shannon intensities from experimental I(q) profile."""
        npts = len(q)
        qmax = np.max(q)
        D = float(D)
        nmax = shannon_channels(qmax, D)+ne
        
        sigq.clip(1e-10)
        
        B = np.zeros((nmax, npts))
        C = np.zeros((nmax, nmax))
        Y = np.zeros((nmax))
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        pts = np.arange(npts)
        
        qsh = N*shannon_width(D)
        Ish = np.zeros((nmax,3))
        
        B[Ni] = Bt(N[:, None], q, D)
        Y[Mi] = Yt(I=I,Ierr=sigq,Bm=B[Mi])
        C[Mi[:, None], Ni] = Ct(Ierr=sigq, Bm=B[Mi], Bn=B[Ni])
        
        Cinv = np.linalg.inv(C)
        
        Ish[:,0] = qsh
        Ish[:,1] = 0.5*np.linalg.solve(C,Y)
        Ish[:,2] = 0.5*(np.diagonal(Cinv))**(0.5)
        
        return Ish

    def Iq2Cinv(self,q,I,sigq,D,ne=2):
        """Calculate Shannon intensities from experimental I(q) profile."""
        npts = len(q)
        qmax = np.max(q)
        D = float(D)
        nmax = shannon_channels(qmax, D)+ne
        
        sigq.clip(1e-10)
        
        B = np.zeros((nmax, npts))
        C = np.zeros((nmax, nmax))
        Y = np.zeros((nmax))
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        pts = np.arange(npts)
        
        qsh = N*shannon_width(D)
        Ish = np.zeros((nmax,3))
        
        B[Ni] = Bt(N[:, None], q, D)
        Y[Mi] = Yt(I=I,sigq=sigq,Bm=B[Mi])
        C[Mi[:, None], Ni] = Ct(sigq=sigq, Bm=B[Mi], Bn=B[Ni])
        
        Cinv = np.linalg.inv(C)
        
        Ish[:,0] = qsh
        Ish[:,1] = 0.5*np.linalg.solve(C,Y)
        Ish[:,2] = 0.5*(np.diagonal(Cinv))**(0.5)
        
        return Cinv

    def Ish2Iq(self,Ish,D,q=(np.arange(500)+1.)/1000):
        """Calculate I(q) from intensities at Shannon points."""
        q = np.atleast_1d(q)
        Ish = np.atleast_1d(Ish)
        Iq = np.zeros((len(q),2))
        Iq[:,0] = q
        n = len(Ish)
        N = np.arange(n)+1
        for i in range(len(q)):
            denominator = ((N*np.pi)**2-(q[i]*D)**2)
            Iq[i,1] = 2 * np.sum(Ish*(N*np.pi)**2 / denominator * np.sinc(q[i]*D/np.pi) * (-1)**(N+1))
            if np.isinf(Iq[i,1]):
                j = find_nearest_i(D2qsh(D,qmax=q[-1]),q[i])
                Iq[i,1] = Ish[j]
        return Iq

    def Ish2P(self,Ish,D,r=None,dr=None):
        """Calculate P(r) from intensities at Shannon points."""
        if r is None:
            if dr is None:
                dr = 0.10
            r = np.linspace(0,D,D/dr)
        r = np.atleast_1d(r)
        Ish = np.atleast_1d(Ish)
        N = np.arange(len(Ish))+1
        P = 1./(2*D**2) * np.einsum('k,kj->k',r,N*Ish*np.sin(N*np.pi*r[:,None]/D))
        return P

    def _Ish2P(self,Ish,D,r=None,dr=None,extend=True,nstart=0):
        """Calculate P(r) from intensities at Shannon points."""
        if r is None:
            if dr is None:
                dr = 0.10
            r = np.linspace(0,D,D/dr)
        r = np.atleast_1d(r)
        Ish = np.atleast_1d(Ish)
        Ish = Ish[nstart:]
        P = np.zeros((len(r),2))
        P[:,0] = r
        N = np.arange(len(Ish))+1+nstart
        for i in range(len(r)):
            if r[i]>D:
                P[i,1] = 0.0
            else:
                P[i,1] = r[i]/(2*D**2) * np.sum(N*Ish*np.sin(N*np.pi*r[i]/D))
                if extend:
                    Np = np.arange(N[-1]+1,100)
                    k = Ish[-1]/(N[-1]*np.pi/D)**(-4)
                    P[i,1] += r[i]/(2*D**2) * k * np.sum(Np*(Np*np.pi/D)**(-4)*np.sin(Np*np.pi*r[i]/D))
        return P

    def Perrf(self,r, D, Sn, Sm, Cinv):
        """Return the standard errors on P(r).
            
            To work, Sn will generally equal something like S[N[:, None], pts] where N is
            the array of n shannon channels, same for Sm, and pts is the array of
            indices for the number of data points, e.g. np.arange(npts).
            This is required for correct broadcasting.
            Example: Perr(r=r, Sn=S[N[:, None], pts]), Sm=S[M[:, None], pts], Cinv=Cinv)
            """
        nmax = len(Sn)
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        #THIS IS WRONG, FIX IT WITH CORRECT EQUATION
        return r/(4*D**2)*np.einsum('ni,mi,nm->i', N[:,None]*Sn, M[:,None]*Sm, Cinv)**(.5)

    def Ish2I0(self,Ish,D):
        """Calculate I0 from Shannon intensities"""
        n = len(Ish)
        N = np.arange(n)+1
        I0 = 2 * np.sum(Ish*(-1)**(N+1))
        return I0

    def I0errf(self,Cinv):
        """Calculate error on I0 from Shannon intensities from inverse C variance-covariance matrix"""
        nmax = len(Cinv)
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        s2 = np.einsum('n,m,nm->',(-1)**(N),(-1)**M,Cinv)
        return s2**(0.5)

    def Ish2rg(self,Ish,D):
        """Calculate Rg from Shannon intensities"""
        n = len(Ish)
        N = np.arange(n)+1
        I0 = self.Ish2I0(Ish,D)
        summation = np.sum(Ish*(1-6/(N*np.pi)**2)*(-1)**(N+1))
        rg = np.sqrt(D**2/I0 * summation)
        return rg

    def Ish2rg2(self,Ish,D):
        """Calculate Rg^2 from Shannon intensities"""
        n = len(Ish)
        N = np.arange(n)+1
        I0 = self.Ish2I0(Ish,D)
        summation = np.sum(Ish*(1-6/(N*np.pi)**2)*(-1)**(N+1))
        rg2 = D**2/I0 * summation
        #rg = np.sqrt(rg2)
        return rg2

    def rgerrf(self,Ish,D,Cinv):
        """Calculate error on Rg from Shannon intensities from inverse C variance-covariance matrix"""
        rg = self.Ish2rg(Ish,D)
        I0 = self.Ish2I0(Ish,D)
        nmax = len(Cinv)
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        s2 = np.einsum('n,m,nm->',(1-6/(N*np.pi)**2)*(-1)**(N),(1-6/(M*np.pi)**2)*(-1)**M,Cinv)
        return D**2/(I0*rg)*s2**(0.5)

    def Ish2ravg(self,Ish,D):
        """Calculate average vector length r from Shannon intensities"""
        n = len(Ish)
        N = np.arange(n)+1
        I0 = self.Ish2I0(Ish,D)
        summation = np.sum(Ish * ( ((-1)**N-1)/(N*np.pi)**2 - (-1)**N/2. ))
        avgr = 4*D/I0 * summation
        return avgr

    def ravgerrf(self,Ish,D,Cinv):
        """Calculate error on Rg from Shannon intensities from inverse C variance-covariance matrix"""
        avgr = self.Ish2ravg(Ish,D)
        I0 = self.Ish2I0(Ish,D)
        nmax = len(Cinv)
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        s2 = np.einsum('n,m,nm->',((-1)**N-1)/(N*np.pi)**2 - (-1)**N/2.,((-1)**M-1)/(M*np.pi)**2 - (-1)**M/2.,Cinv)
        return 2*D/I0 * s2**(0.5)

    def Ish2Q(self,Ish,D):
        """Calculate Porod Invariant Q from Shannon intensities"""
        n = len(Ish)
        N = np.arange(n)+1
        Q = (np.pi/D)**3 * np.sum(Ish*N**2)
        return Q

    def Qerrf(self,D,Cinv):
        """Calculate error on Q from Shannon intensities from inverse C variance-covariance matrix"""
        nmax = len(Cinv)
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        s2 = np.einsum('n,m,nm->', N**2, M**2,Cinv)
        return (np.pi/D)**3 * s2**(0.5)

    def gamma0(self,Ish, D):
        """Calculate gamma at r=0. gamma is P(r)/4*pi*r^2"""
        Q = self.Ish2Q(Ish,D)
        return 1/(8*np.pi**3) * Q

    def Ish2Vp(self,Ish,D):
        """Calculate Porod Volume from Shannon intensities"""
        Q = self.Ish2Q(Ish,D)
        I0 = self.Ish2I0(Ish,D)
        Vp = 2*np.pi**2 * I0/Q
        return Vp

    def Vperrf(self,Ish, D, Cinv):
        """Calculate error on Vp from Shannon intensities from inverse C variance-covariance matrix"""
        I0 = self.Ish2I0(Ish,D)
        Q = self.Ish2Q(Ish,D)
        I0s = self.I0errf(Cinv)
        Qs = self.Qerrf(D,Cinv)
        s2 = (2*np.pi/Q)**2*(I0s)**2 + (2*np.pi*I0/Q**2)**2*Qs**2
        return s2**(0.5)

    def Ish2mwVp(self,Ish,D):
        """Calculate molecular weight via Porod Volume from Shannon intensities"""
        Vp = self.Ish2Vp(Ish,D)
        mw = Vp/1.66
        return mw

    def mwVperrf(self,Ish,D,Cinv):
        """Calculate error on mwVp from Shannon intensities from inverse C variance-covariance matrix"""
        Vps = self.Vperrf(Ish,D,Cinv)
        return Vps/1.66

    def Ish2Vc(self,Ish,D):
        """Calculate Volume of Correlation from Shannon intensities"""
        n = len(Ish)
        N = np.arange(n)+1
        I0 = self.Ish2I0(Ish,D)
        area_qIq = 2*np.pi/D**2 * np.sum(N * Ish * special.sici(N*np.pi)[0])
        Vc = I0/area_qIq
        return Vc

    def Vcerrf(self,Ish, D, Cinv):
        """Calculate error on Vc from Shannon intensities from inverse C variance-covariance matrix"""
        I0 = self.Ish2I0(Ish,D)
        Vc = self.Ish2Vc(Ish,D)
        nmax = len(Cinv)
        Ni = np.arange(nmax) #indices
        Mi = np.arange(nmax) #indices
        N = Ni+1 #values
        M = Mi+1 #values
        s2 = np.einsum('n,m,nm->', N*special.sici(N*np.pi)[0], M*special.sici(M*np.pi)[0],Cinv)
        return (2*np.pi*Vc**2/(D**2*I0)) * s2**(0.5)

    def Ish2Qr(self,Ish,D):
        """Calculate Rambo Invariant Qr (Vc^2/Rg) from Shannon intensities"""
        Vc = self.Ish2Vc(Ish,D)
        Rg = self.Ish2rg(Ish,D)
        Qr = Vc**2/Rg
        return Qr

    def Ish2mwVc(self,Ish,D,RNA=False):
        """Calculate molecular weight via the Volume of Correlation from Shannon intensities"""
        Qr = self.Ish2Qr(Ish,D)
        if RNA:
            mw = (Qr/0.00934)**(0.808)
        else:
            mw = (Qr/0.1231)**(1.00)
        return mw

    def mwVcerrf(self,Ish,D,Cinv):
        Vc = self.Ish2Vc(Ish,D)
        Rg = self.Ish2rg(Ish,D)
        Vcs = self.Vcerrf(Ish,D,Cinv)
        Rgs = self.rgerrf(Ish,D,Cinv)
        mwVcs = Vc/(0.1231*Rg) * (4*Vcs**2 + (Vc/Rg*Rgs)**2)**(0.5)
        return mwVcs

    def Ish2lc(self,Ish,D):
        """Calculate length of correlation from Shannon intensities"""
        Vp = self.Ish2Vp(Ish,D)
        Vc = self.Ish2Vc(Ish,D)
        lc = Vp/(2*np.pi*Vc)
        return lc

    def lcerrf(self,Ish, D, Cinv):
        """Calculate error on lc from Shannon intensities from inverse C variance-covariance matrix"""
        Vp = self.Ish2Vp(Ish,D)
        Vc = self.Ish2Vc(Ish,D)
        Vps = self.Vperrf(Ish,D,Cinv)
        Vcs = self.Vcerrf(Ish,D,Cinv)
        s2 = Vps**2 + (Vp/Vc)**2*Vcs**2
        return 1/(2*np.pi*Vc) * s2**(0.5)

    def Ish2areaIq(self,Ish,D):
        """Calculate area under I(q) from Shannon intensities"""
        I0 = self.Ish2I0(Ish,D)
        area_Iq = np.pi/D * (I0/2 + np.sum(Ish))
        #the following also works
        #n = len(Ish)
        #N = np.arange(n)+1
        #area_Iq = np.pi/D * np.sum(Ish*(1+(-1)**(N+1)))
        return area_Iq














