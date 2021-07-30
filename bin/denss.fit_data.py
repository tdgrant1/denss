#!/usr/bin/env python
#
#    denss.fit_data.py
#    A tool for fitting solution scattering data with smooth function
#    based on Moore's algorithm for fitting a trigonometric series.
#
#    Part of the DENSS package
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tgrant@hwi.buffalo.edu>
#    Copyright 2018 The Research Foundation for SUNY
#
#    Additional authors:
#    Christopher Handelmann
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
import datetime, time
import os, argparse, sys
import logging
import numpy as np
from saxstats._version import __version__
import saxstats.saxstats as saxs

parser = argparse.ArgumentParser(description="A tool for fitting solution scattering data with smooth function based on Moore's algorithm for fitting a trigonometric series.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="SAXS data file for input (either .dat or .out)")
parser.add_argument("-d", "--dmax", default=None, type=float, help="Estimated maximum dimension")
parser.add_argument("-a", "--alpha", default=None, type=float, help="Set alpha smoothing factor")
parser.add_argument("-n1", "--n1", default=None, type=int, help="First data point to use")
parser.add_argument("-n2", "--n2", default=None, type=int, help="Last data point to use")
parser.add_argument("-q", "--qfile", default=None, type=str, help="ASCII text filename to use for setting the calculated q values (like a SAXS .dat file, but just uses first column, optional)")
parser.add_argument("--nes", default=2, type=int, help=argparse.SUPPRESS)
parser.add_argument("--max_dmax", default=None, type=float, help="Maximum limit for allowed Dmax values (for plotting slider)")
parser.add_argument("--max_alpha", default=None, type=float, help="Maximum limit for allowed alpha values (for plotting slider)")
parser.add_argument("--max_nes", default=10, type=int, help=argparse.SUPPRESS)
parser.add_argument("--no_gui", dest="plot", action="store_false", help="Do not run the interactive GUI mode.")
parser.add_argument("--no_log", dest="log", action="store_false", help="Do not plot on log y axis.")
parser.add_argument("--no_extrapolation", dest="extrapolate", action="store_false", help="Do not extrapolate high q data.")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
parser.set_defaults(plot=True)
args = parser.parse_args()

if args.plot:
    #if plotting is enabled, try to import matplotlib
    #if import fails, set plotting to false
    try:
        import matplotlib
        #matplotlib.use('TkAgg')
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
    except ImportError:
        args.plot = False

if __name__ == "__main__":

    alpha = args.alpha

    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        output = basename
    else:
        output = args.output

    Iq = np.genfromtxt(args.file, invalid_raise = False, usecols=(0,1,2))
    Iq = Iq[~np.isnan(Iq).any(axis = 1)]
    #get rid of any data points equal to zero in the intensities or errors columns
    idx = np.where((Iq[:,1]!=0)&(Iq[:,2]!=0))
    Iq = Iq[idx]
    nes = args.nes

    if args.n1 is None:
        n1 = 0
    else:
        n1 = args.n1
    if args.n2 is None:
        n2 = len(Iq[:,0])
    else:
        n2 = args.n2

    Iq = Iq[n1:n2]

    if args.dmax is None:
        #estimate dmax directly from data
        D, sasrec = saxs.estimate_dmax(Iq,clean_up=True)
    else:
        D = args.dmax
        sasrec = saxs.Sasrec(Iq, D, alpha=0.0)

    if args.max_dmax is None:
        args.max_dmax = 2.*D

    if args.extrapolate:
        qe = np.linspace(1.0*Iq[-1,0],3.0*Iq[-1,0],1001)
        qsph = qe[qe>Iq[-1,0]]
        #simulate scattering of sphere of Rg of particle
        Isph = saxs.sphere(q=qsph,R=3./5.*sasrec.rg**0.5)
        Isph += np.random.normal(Isph*1e-2,size=len(Isph))
        idx = np.where(Isph>0)
        qsph = qsph[idx]
        Isph = Isph[idx]
        #scale extrapolated intensity to match data
        Isph *= Iq[-50:,1].mean()/Isph[:20].mean()
        ssph = Isph*1.e6 #try huge error bars
        Iqe = np.concatenate((Iq,np.vstack((qsph,Isph,ssph)).T))
        #save original Iq for plotting
        Iq_orig = np.copy(Iq)
        #now reset Iq to be the extrapolated Iq
        Iq = np.copy(Iqe)
    else:
        Iq_orig = np.copy(Iq)

    n1 = 0
    n2 = len(Iq)

    q = Iq[:,0]
    #create a calculated q range for Sasrec
    qmax = q.max()
    qmin = q.min()
    dq = (qmax-qmin)/(q.size-1)
    nq = int(qmin/dq)
    qc = np.concatenate(([0.0],np.arange(nq)*dq+(qmin-nq*dq),q))
    Icerr = np.concatenate((np.ones(nq+1)*Iq[0,2],Iq[:,2]))

    if args.qfile is not None:
        qc = np.loadtxt(args.qfile,usecols=(0,))

    sasrec = saxs.Sasrec(Iq, D, alpha=0.0)
    ideal_chi2 = sasrec.calc_chi2()

    if args.alpha is None:
        al = []
        chi2 = []
        #here, alphas are actually the exponents, since the range can
        #vary from 10^-10 upwards of 10^20. This should cover nearly all likely values
        alphas = np.arange(-10,20.)
        for alpha in alphas:
            #print("***** ALPHA ****** %.5e"%alpha)
            sasrec = saxs.Sasrec(Iq[n1:n2], D, qc=qc, r=None, alpha=10.**alpha, ne=nes)
            r = sasrec.r
            pi = np.pi
            N = sasrec.N[:,None]
            In = sasrec.In[:,None]
            chi2value = sasrec.calc_chi2()
            al.append(alpha)
            chi2.append(chi2value)
        chi2 = np.array(chi2)

        #find optimal alpha value based on where chi2 begins to rise, to 10% above the ideal chi2 (where alpha=0)
        x = np.linspace(alphas[0],alphas[-1],1000)
        y = np.interp(x, alphas, chi2)
        chif = 2.0
        ali = np.argmin(y<=chif*ideal_chi2)
        opt_alpha = 10.0**(np.interp(chif*ideal_chi2,[y[ali+1],y[ali]],[x[ali+1],x[ali]])-1)
        alpha = opt_alpha
    else:
        alpha = args.alpha
    sasrec = saxs.Sasrec(Iq[n1:n2], D, qc=qc, r=None, alpha=alpha, ne=nes)
    #get a rough estimate for a reasonable alpha based on I(0)
    #to set a maximum alpha range, so when users click in the slider
    #it at least does something reasonable, rather than either nothing
    #significant, or so huge it becomes difficult to find the right value
    if args.max_alpha is None:
        if alpha == 0.0:
            max_alpha = 2.0
        else:
            max_alpha = 2*alpha

    def store_parameters_as_string(event=None):
        param_str = ("Parameter Values:\n"
        "Dmax  = {dmax:.5e}\n"
        "alpha = {alpha:.5e}\n"
        "I(0)  = {I0:.5e} +- {I0err:.5e}\n"
        "Rg    = {rg:.5e} +- {rgerr:.5e}\n"
        "r_avg = {r:.5e} +- {rerr:.5e}\n"
        "Vp    = {Vp:.5e} +- {Vperr:.5e}\n"
        "MW_Vp = {mwVp:.5e} +- {mwVperr:.5e}\n"
        "Vc    = {Vc:.5e} +- {Vcerr:.5e}\n"
        "MW_Vc = {mwVc:.5e} +- {mwVcerr:.5e}\n"
        "Lc    = {lc:.5e} +- {lcerr:.5e}\n"
        ).format(dmax=sasrec.D,alpha=sasrec.alpha,
            I0=sasrec.I0,I0err=sasrec.I0err,
            rg=sasrec.rg,rgerr=sasrec.rgerr,
            r=sasrec.avgr,rerr=sasrec.avgrerr,
            Vp=sasrec.Vp,Vperr=sasrec.Vperr,
            mwVp=sasrec.mwVp,mwVperr=sasrec.mwVperr,
            Vc=sasrec.Vc,Vcerr=sasrec.Vcerr,
            mwVc=sasrec.mwVc,mwVcerr=sasrec.mwVcerr,
            lc=sasrec.lc,lcerr=sasrec.lcerr)
        return param_str

    def print_values(event=None):
        print("---------------------------------")
        param_str = store_parameters_as_string()
        print(param_str)

    def save_file(event=None):
        #sascif = saxs.Sascif(sasrec)
        #sascif.write(output+".sascif")
        #print "%s file saved" % (output+".sascif")
        param_str = store_parameters_as_string()
        #add column headers to param_str for output
        param_str += 'q, I, error, fit'
        #quick, interpolate the raw data, sasrec.I, to the new qc values, but be sure to 
        #put zeros in for the q values not measured behind the beamstop
        Iinterp = np.interp(sasrec.qc, sasrec.q, sasrec.I, left=0.0, right=0.0)
        np.savetxt(output+'.fit', np.vstack((sasrec.qc, Iinterp, sasrec.Icerr, sasrec.Ic)).T,delimiter=' ',fmt='%.5e',header=param_str)
        np.savetxt(output+'_pr.dat', np.vstack((sasrec.r, sasrec.P, sasrec.Perr)).T,delimiter=' ',fmt='%.5e')
        print("%s and %s files saved" % (output+".fit",output+"_pr.dat"))

    if args.plot:

        #fig, (axI, axP) = plt.subplots(1, 2, figsize=(12,6))
        fig = plt.figure(0, figsize=(12,6))
        fig.canvas.set_window_title(output)
        fig.suptitle(output)
        axI = plt.subplot2grid((3,2), (0,0),rowspan=2)
        axR = plt.subplot2grid((3,2), (2,0),sharex=axI)
        axP = plt.subplot2grid((3,2), (0,1),rowspan=3)
        plt.subplots_adjust(left=0.068, bottom=0.25, right=0.98, top=0.95)

        I_l1, = axI.plot(sasrec.q, sasrec.I, 'k.')
        I_l2, = axI.plot(sasrec.qc, sasrec.Ic, 'r-', lw=2)
        I_l3, = axI.plot(sasrec.qn, sasrec.In, 'bo', mec='b', mfc='none', mew=2)
        if args.log: axI.semilogy()
        axI.set_ylabel('I(q)')
        axI.set_xlabel('q')

        #residuals
        res = np.log10(np.abs(sasrec.I)) - np.log10(np.interp(sasrec.q, sasrec.qc, np.abs(sasrec.Ic)))
        Ires_l0, = axR.plot(sasrec.q, sasrec.q*0, 'k--')
        Ires_l1, = axR.plot(sasrec.q, res, 'r-')
        axR.set_ylabel('Residuals')
        axR.set_xlabel('q')

        P_l1, = axP.plot(sasrec.r*100, sasrec.r*0, 'k--')
        P_l2, = axP.plot(sasrec.r, sasrec.P, 'b-', lw=2)
        axP.set_ylabel('P(r)')
        axP.set_xlabel('r')

        #axI.set_xlim([0,1.1*np.max(sasrec.q)])
        #axR.set_xlim([0,1.1*np.max(sasrec.q)])
        axI.set_xlim([0,Iq_orig[-1,0]])
        axR.set_xlim([0,Iq_orig[-1,0]])
        axP.set_xlim([0,1.1*np.max(sasrec.r)])
        axI.set_ylim([0.25*np.min(sasrec.Ic[sasrec.qc<Iq_orig[-1,0]]),2*np.max(sasrec.Ic[sasrec.qc<Iq_orig[-1,0]])])
        #axR.set_ylim([0,Iq_orig[-1,0]])
        #axP.set_ylim([0,1.1*np.max(sasrec.r)])

        axcolor = 'lightgoldenrodyellow'
        axdmax = plt.axes([0.05, 0.125, 0.4, 0.03], facecolor=axcolor)
        axalpha = plt.axes([0.05, 0.075, 0.4, 0.03], facecolor=axcolor)
        #axnes = plt.axes([0.05, 0.025, 0.4, 0.03], facecolor=axcolor)

        axI0 = plt.figtext(.57, .125,   "I(0) = %.2e $\pm$ %.2e"%(sasrec.I0,sasrec.I0err),family='monospace')
        axrg = plt.figtext(.57, .075,   "Rg   = %.2e $\pm$ %.2e"%(sasrec.rg,sasrec.rgerr),family='monospace')
        axrav = plt.figtext(.57, .025,  "$\overline{r}$    = %.2e $\pm$ %.2e"%(sasrec.avgr,sasrec.avgrerr),family='monospace')
        axVp = plt.figtext(.77, .125,   "Vp = %.2e $\pm$ %.2e"%(sasrec.Vp,sasrec.Vperr),family='monospace')
        axVc = plt.figtext(.77, .075,   "Vc = %.2e $\pm$ %.2e"%(sasrec.Vc,sasrec.Vcerr),family='monospace')
        axlc = plt.figtext(.77, .025,   "Lc = %.2e $\pm$ %.2e"%(sasrec.lc,sasrec.lcerr),family='monospace')
        #axVpmw = plt.figtext(.55, .075, "Vp MW = %.2e $\pm$ %.2e"%(sasrec.mwVp,sasrec.mwVperr),family='monospace')
        #axVcmw = plt.figtext(.55, .025, "Vc MW = %.2e $\pm$ %.2e"%(sasrec.mwVc,sasrec.mwVcerr),family='monospace')

        sdmax = Slider(axdmax, 'Dmax', 0.0, args.max_dmax, valinit=D)
        sdmax.valtext.set_visible(False)
        # set up ticks marks on the slider to denote the change in interaction
        axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax]) 
        #axdmax.xaxis.tick_top()
        axdmax.tick_params(labelbottom=False)

        salpha = Slider(axalpha, 'Alpha', 0.0, max_alpha, valinit=alpha)
        salpha.valtext.set_visible(False)

        #snes = Slider(axnes, 'NES', 0, args.max_nes, valinit=args.nes, valstep=1)

        dmax = D
        n1 = str(n1)
        n2 = str(n2)

        def analyze(dmax,alpha,n1,n2):
            global sasrec
            sasrec = saxs.Sasrec(Iq[n1:n2], dmax, qc=qc, r=None, alpha=alpha, ne=nes)
            Icinterp = np.interp(sasrec.q, sasrec.qc, np.abs(sasrec.Ic))
            res = np.log10(np.abs(sasrec.I)) - np.log10(Icinterp)
            I_l2.set_data(sasrec.qc[:n2], sasrec.Ic[:n2])
            I_l3.set_data(sasrec.qn, sasrec.In)
            Ires_l1.set_data(sasrec.q, res)
            P_l2.set_data(sasrec.r, sasrec.P)
            axI0.set_text("I(0) = %.2e $\pm$ %.2e"%(sasrec.I0,sasrec.I0err))
            axrg.set_text("Rg   = %.2e $\pm$ %.2e"%(sasrec.rg,sasrec.rgerr))
            axrav.set_text("$\overline{r}$    = %.2e $\pm$ %.2e"%(sasrec.avgr,sasrec.avgrerr))
            axVp.set_text("Vp = %.2e $\pm$ %.2e"%(sasrec.Vp,sasrec.Vperr))
            axVc.set_text("Vc = %.2e $\pm$ %.2e"%(sasrec.Vc,sasrec.Vcerr))
            axlc.set_text("Lc = %.2e $\pm$ %.2e"%(sasrec.lc,sasrec.lcerr))
            #axVpmw.set_text("Vp MW = %.2e $\pm$ %.2e"%(sasrec.mwVp,sasrec.mwVperr))
            #axVcmw.set_text("Vc MW = %.2e $\pm$ %.2e"%(sasrec.mwVc,sasrec.mwVcerr))

        def n1_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(text)
            n2 = int(n2_box.text)
            analyze(dmax,alpha,n1,n2)
            fig.canvas.draw_idle()

        def n2_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(text)
            analyze(dmax,alpha,n1,n2)
            fig.canvas.draw_idle()

        def D_submit(text):
            dmax = float(text)
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            analyze(dmax,alpha,n1,n2)
            # this updates the slider value based on text box value
            sdmax.set_val(dmax)
            axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax])
            fig.canvas.draw_idle()

        def A_submit(text):
            dmax = sdmax.val
            alpha = float(text)
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            analyze(dmax,alpha,n1,n2)
            # this updates the slider value based on text box value
            salpha.set_val(alpha)
            fig.canvas.draw_idle()

        def update(val):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            analyze(dmax,alpha,n1,n2)
            # partitions the slider, so clicking in the upper and lower range scale valmax
            if (dmax > 0.9 * sdmax.valmax) or (dmax < 0.1 * sdmax.valmax):
                sdmax.valmax = 2 * dmax
                sdmax.ax.set_xlim(sdmax.valmin, sdmax.valmax)
                axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax])
            # partions slider as well
            if (alpha > 0.9 * salpha.valmax) or (alpha < 0.1 * salpha.valmax):
                salpha.valmax = 2 * alpha
                # alpha starting at zero makes initial adjustment additive not multiplicative
                if alpha != 0:
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)
                elif alpha == 0:
                    salpha.valmax = alpha + 10
                    salpha.valmin = 0.0
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)

            Dmax_box.set_val("%.4e"%dmax)
            Alpha_box.set_val("%.4e"%alpha)

            fig.canvas.draw_idle()

        # making a text entry for dmax that allows for user input
        Dvalue = "{dmax:.4e}".format(dmax=dmax)
        axIntDmax = plt.axes([0.45, 0.125, 0.08, 0.03])
        Dmax_box = TextBox(axIntDmax, '', initial=Dvalue)
        Dmax_box.on_submit(D_submit)

        # making a text entry for alpha that allows for user input
        Avalue = "{alpha:.4e}".format(alpha=alpha)
        axIntAlpha = plt.axes([0.45, 0.075, 0.08, 0.03])
        Alpha_box = TextBox(axIntAlpha, '', initial=Avalue)
        Alpha_box.on_submit(A_submit)

        # making a text entry for n1 that allows for user input
        n1value = "{}".format(n1)
        plt.figtext(0.0085, 0.178, "First point")
        axIntn1 = plt.axes([0.075, 0.170, 0.08, 0.03])
        n1_box = TextBox(axIntn1, '', initial=n1)
        n1_box.on_submit(n1_submit)

        # making a text entry for n2 that allows for user input
        n2value = "{}".format(n2)
        plt.figtext(0.17, 0.178, "Last point")
        axIntn2 = plt.axes([0.235, 0.170, 0.08, 0.03])
        n2_box = TextBox(axIntn2, '', initial=n2)
        n2_box.on_submit(n2_submit)

        #here is the slider updating
        sdmax.on_changed(update)
        salpha.on_changed(update)
        #snes.on_changed(update)

        axreset = plt.axes([0.05, 0.02, 0.1, 0.04])
        reset_button = Button(axreset, 'Reset Sliders', color=axcolor, hovercolor='0.975')

        def reset_values(event):
            sdmax.reset()
            salpha.reset()
        reset_button.on_clicked(reset_values)

        axprint = plt.axes([0.2, 0.02, 0.1, 0.04])
        print_button = Button(axprint, 'Print Values', color=axcolor, hovercolor='0.975')

        print_button.on_clicked(print_values)

        axsave = plt.axes([0.35, 0.02, 0.1, 0.04])
        save_button = Button(axsave, 'Save File', color=axcolor, hovercolor='0.975')

        save_button.on_clicked(save_file)

        plt.show()

    print_values()
    save_file()
