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

import sys, logging, datetime
import numpy as np
from scipy import ndimage

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

def write_xplor(rho,side,filename="map.xplor"):
    """Write an XPLOR formatted electron density map."""
    xs, ys, zs = rho.shape
    title_lines = ['REMARK FILENAME="'+filename+'"','REMARK DATE= '+str(datetime.datetime.today())]
    f = open(filename,'wb')
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
    f.close()

def denss(q,I,sigq,D,ne=None,voxel=5.,oversampling=3.,limit_dmax=False,dmax_start_step=500,recenter=True,recenter_maxstep=None,positivity=True,extrapolate=True,write=True,filename="map",steps=3000,seed=None,shrinkwrap=True,shrinkwrap_sigma_start=3,shrinkwrap_sigma_end=1.5,shrinkwrap_sigma_decay=0.99,shrinkwrap_threshold_fraction=0.2,shrinkwrap_iter=20,shrinkwrap_minstep=100,chi_end_fraction=0.01,write_freq=100,enforce_connectivity=True,enforce_connectivity_steps=[500]):
    """Calculate electron density from scattering data."""
    side = oversampling*D
    halfside = side/2
    n = int(side/voxel)
    #want odd n so that there exists an F[0,0,0] in the center that equals the number of electrons for easy scaling
    if n%2==0: n += 1
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
    #only move the non-zero terms, since the zeroth term should be at q=0.
    qbinsc[1:] += qstep/2.
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.digitize(qr, qbins)
    qbin_labels -= 1
    #allow for any range of q data
    qdata = qbinsc[np.where( (qbinsc>=q.min()) & (qbinsc<=q.max()) )]
    Idata = np.interp(qdata,q,I)
    if extrapolate:
        qextend = qbinsc[np.where(qbinsc>=qdata.max())]
        Iextend = qextend**-4
        Iextend = Iextend/Iextend[0] * Idata[-1]
        qdata = np.concatenate((qdata,qextend[1:]))
        Idata = np.concatenate((Idata,Iextend[1:]))
    #create list of qbin indices just in region of data for later F scaling
    qbin_args = qbin_args = np.in1d(qbinsc,qdata,assume_unique=True)
    realne = np.copy(ne)
    sigqdata = np.interp(qdata,q,sigq)
    Imean = np.zeros((steps+1,len(qbins)))
    chi = np.zeros((steps+1))
    rg = np.zeros((steps+1))
    supportV = np.zeros((steps+1))
    chibest = np.inf
    usesupport = True
    support = np.ones(x.shape,dtype=bool)
    if seed is None:
        seed = np.random.randint(2**32-1)
    else:
        seed = int(seed)
    prng = np.random.RandomState(seed)
    rho = prng.random_sample(size=x.shape)
    update_support = True
    sigma = shrinkwrap_sigma_start

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
        if j==0:
            np.savetxt(filename+'_step0_saxs.dat',np.vstack((qbinsc,Imean[j],Imean[j]*.05)).T,delimiter=" ",fmt="%.5e")
            write_xplor(rho,side,filename+"_original.xplor")
        #scale Fs to match data
        factors = np.ones((len(qbins)))
        factors[qbin_args] = np.sqrt(Idata/Imean[j,qbin_args])
        F *= factors[qbin_labels]
        chi[j] = np.sum(((Imean[j,qbin_args]-Idata)/sigqdata)**2)/qbin_args.size
        #APPLY REAL SPACE RESTRAINTS
        rhoprime = np.fft.ifftn(F,rho.shape)
        rhoprime = rhoprime.real
        if j%write_freq == 0:
            write_xplor(rhoprime,side,filename+"_current.xplor")
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
        #update support using shrinkwrap method
        if j%shrinkwrap_iter==0:
            if recenter:
                if recenter_maxstep is None:
                    newrho = center_rho(newrho)
                elif j <= recenter_maxstep:
                    newrho = center_rho(newrho,centering="max")
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
        if limit_dmax and j > dmax_start_step:
            support[r>0.6*D] = False
            if np.sum(support) <= 0:
                support = np.ones(rho.shape,dtype=bool)
        supportV[j] = np.sum(support)*dV
        sys.stdout.write("\r% 5i % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
        sys.stdout.flush()
        
        if j > 101 + shrinkwrap_minstep and np.std(chi[j-100:j]) < chi_end_fraction * np.median(chi[j-100:j]):
            rho = newrho
            F = np.fft.fftn(rho)
            break
        else:
            rho = newrho
            F = np.fft.fftn(rho)
    print
    
    #calculate spherical average intensity from 3D Fs
    Imean[j+1] = ndimage.mean(np.abs(F)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    chi[j+1] = np.sum(((Imean[j+1,qbin_args]-Idata)/sigqdata)**2)/qbin_args.size
    #scale Fs to match data
    factors = np.ones((len(qbins)))
    factors[qbin_args] = np.sqrt(Idata/Imean[j+1,qbin_args])
    F *= factors[qbin_labels]
    rho = np.fft.ifftn(F,rho.shape)
    rho = rho.real
    #recenter rho
    write_xplor(rho,side,filename+"_precentered.xplor")
    rho = center_rho(rho)
    if ne is not None:
        rho *= ne / np.sum(rho)

    rg[j+1] = rho2rg(rho=rho,r=r,support=support,dx=dx)
    supportV[j+1] = supportV[j]

    if write:
        write_xplor(rho,side,filename+".xplor")
        write_xplor(np.ones_like(rho)*support,side,filename+"_support.xplor")

    logging.info('Number of steps: %i', j)
    logging.info('Final Chi2: %3.3f', chi[j+1])
    logging.info('Final Rg: %3.3f', rg[j+1])
    logging.info('Final Support Volume: %3.3f', supportV[j+1])

    return qdata, Idata, sigqdata, qbinsc, Imean[j+1], chi, rg, supportV


