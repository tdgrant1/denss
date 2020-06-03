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
import os, argparse, sys, imp
import logging
import numpy as np
from saxstats._version import __version__
import saxstats.saxstats as saxs
try:
    imp.find_module('matplotlib')
    matplotlib_found = True
    from matplotlib.gridspec import GridSpec
except ImportError:
    matplotlib_found = False

parser = argparse.ArgumentParser(description="A tool for fitting solution scattering data with smooth function based on Moore's algorithm for fitting a trigonometric series.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="SAXS data file for input (either .dat or .out)")
parser.add_argument("-d", "--dmax", default=100., type=float, help="Estimated maximum dimension")
parser.add_argument("-a", "--alpha", default=0., type=float, help="Set alpha smoothing factor")
parser.add_argument("-n1", "--n1", default=0, type=int, help="First data point to use")
parser.add_argument("-n2", "--n2", default=-1, type=int, help="Last data point to use")
parser.add_argument("-q", "--qfile", default=None, type=str, help="ASCII text filename to use for setting the calculated q values (like a SAXS .dat file, but just uses first column, optional)")
parser.add_argument("--nes", default=2, type=int, help=argparse.SUPPRESS)
parser.add_argument("--max_dmax", default=None, type=float, help="Maximum limit for allowed Dmax values (for plotting slider)")
parser.add_argument("--max_alpha", default=None, type=float, help="Maximum limit for allowed alpha values (for plotting slider)")
parser.add_argument("--max_nes", default=10, type=int, help=argparse.SUPPRESS)
parser.add_argument("--no_gui", dest="plot", action="store_false", help="Do not run the interactive GUI mode.")
parser.add_argument("--no_log", dest="log", action="store_false", help="Do not plot on log y axis.")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
if matplotlib_found:
    parser.set_defaults(plot=True)
else:
    parser.set_defaults(plot=False)
parser.set_defaults(log=True)
args = parser.parse_args()

if __name__ == "__main__":

    alpha = args.alpha

    if args.output is None:
        basename, ext = os.path.splitext(args.file)
        output = basename + '_fit'
    else:
        output = args.output

    Iq = np.genfromtxt(args.file, invalid_raise = False)
    Iq = Iq[~np.isnan(Iq).any(axis = 1)]
    D = args.dmax
    nes = args.nes

    if args.max_dmax is None:
        args.max_dmax = 2.*D
    if args.max_alpha is None:
        args.max_alpha = 10.

    q = Iq[:,0]
    qmin = np.min(q[0])
    dq = q[1] - q[0]
    nq = int(qmin/dq)
    qc = np.concatenate(([0.0],np.arange(nq)*dq+(qmin-nq*dq),q))
    #Icerr = np.concatenate((np.ones(nq+1)*Iq[0,2],Iq[:,2]))

    if args.qfile is not None:
        qc = np.loadtxt(args.qfile,usecols=(0,))

    Icerr = np.interp(qc,q,Iq[:,2])

    sasrec = saxs.Sasrec(Iq[args.n1:args.n2], D, qc=qc, r=None, alpha=alpha, ne=nes)


    if args.plot:
        import matplotlib
        #matplotlib.use('TkAgg')
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

        #fig, (axI, axP) = plt.subplots(1, 2, figsize=(12,6))
        fig = plt.figure(0, figsize=(12,6))
        axI = plt.subplot2grid((3,2), (0,0),rowspan=2)
        axR = plt.subplot2grid((3,2), (2,0),sharex=axI)
        axP = plt.subplot2grid((3,2), (0,1),rowspan=3)
        plt.subplots_adjust(left=0.068, bottom=0.25, right=0.98, top=0.95)

        I_l1, = axI.plot(sasrec.q, sasrec.I, 'k.')
        I_l2, = axI.plot(sasrec.qc, sasrec.Ic, 'r-', lw=2)
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

        axI.set_xlim([0,1.1*np.max(sasrec.q)])
        axR.set_xlim([0,1.1*np.max(sasrec.q)])
        axP.set_xlim([0,1.1*np.max(sasrec.r)])

        axcolor = 'lightgoldenrodyellow'
        axdmax = plt.axes([0.05, 0.125, 0.4, 0.03], facecolor=axcolor)
        axalpha = plt.axes([0.05, 0.075, 0.4, 0.03], facecolor=axcolor)
        #axnes = plt.axes([0.05, 0.025, 0.4, 0.03], facecolor=axcolor)

        axrg = plt.figtext(.55, .125, "Rg = " + str(round(sasrec.rg,2)) + " +- " + str(round(sasrec.rgerr,2)))
        axI0 = plt.figtext(.75, .125, "I(0) = " + str(round(sasrec.I0,2)) + " +- " + str(round(sasrec.I0err,2)))
        axVp = plt.figtext(.55, .075, "Vp = " + str(round(sasrec.Vp,2)) + " +- " + str(round(sasrec.Vperr,2)))
        axVpmw = plt.figtext(.75, .075, "Vp MW = " + str(round(sasrec.mwVp,2)) + " +- " + str(round(sasrec.mwVperr,2)))
        axVcmw = plt.figtext(.55, .025, "Vc MW = " + str(round(sasrec.mwVc,2)) + " +- " + str(round(sasrec.mwVcerr,2)))
        axlc = plt.figtext(.75, .025, "Lc = " + str(round(sasrec.lc,2)) + " +- " + str(round(sasrec.lcerr,2)))

        sdmax = Slider(axdmax, 'Dmax', 0.0, args.max_dmax, valinit=D)
        sdmax.valtext.set_visible(False)
        # set up ticks marks on the slider to denote the change in interaction
        axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax]) 
        #axdmax.xaxis.tick_top()
        axdmax.tick_params(labelbottom=False)

        salpha = Slider(axalpha, 'Alpha', 0.0, args.max_alpha, valinit=alpha)
        salpha.valtext.set_visible(False)

        #snes = Slider(axnes, 'NES', 0, args.max_nes, valinit=args.nes, valstep=1)

        dmax = D
        n1 = str(args.n1)
        n2 = str(args.n2)

        def analyze(dmax,alpha,n1,n2):
            global sasrec
            sasrec = saxs.Sasrec(Iq[n1:n2], dmax, qc=qc[:n2], r=None, alpha=alpha, ne=nes)
            Icinterp = np.interp(sasrec.q, sasrec.qc, np.abs(sasrec.Ic))
            res = np.log10(np.abs(sasrec.I)) - np.log10(Icinterp)
            I_l2.set_data(sasrec.qc, sasrec.Ic)
            Ires_l1.set_data(sasrec.q, res)
            P_l2.set_data(sasrec.r, sasrec.P)
            axrg.set_text("Rg = " + str(round(sasrec.rg,2)) + " +- " + str(round(sasrec.rgerr,2)))
            axI0.set_text("I(0) = " + str(round(sasrec.I0,2)) + " +- " + str(round(sasrec.I0err,2)))
            axVp.set_text("Vp = " + str(round(sasrec.Vp,2)) + " +- " + str(round(sasrec.Vperr,2)))
            axVpmw.set_text("Vp MW = " + str(round(sasrec.mwVp,2)) + " +- " + str(round(sasrec.mwVperr,2)))
            axVcmw.set_text("Vc MW = " + str(round(sasrec.mwVc,2)) + " +- " + str(round(sasrec.mwVcerr,2)))
            axlc.set_text("Lc = " + str(round(sasrec.lc,2)) + " +- " + str(round(sasrec.lcerr,2)))

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
        Dvalue = "{}".format(dmax)
        axIntDmax = plt.axes([0.45, 0.125, 0.08, 0.03])
        Dmax_box = TextBox(axIntDmax, '', initial=Dvalue)
        Dmax_box.on_submit(D_submit)

        # making a text entry for alpha that allows for user input
        Avalue = "{}".format(alpha)
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

        def print_values(event):
            print("---------------------------------")
            print("Dmax = " + str(round(sasrec.D,2)))
            print("alpha = %.5e" % sasrec.alpha)
            print("Rg = " + str(round(sasrec.rg,2)) + " +- " + str(round(sasrec.rgerr,2)))
            print("I(0) = " + str(round(sasrec.I0,2)) + " +- " + str(round(sasrec.I0err,2)))
            print("Vp = " + str(round(sasrec.Vp,2)) + " +- " + str(round(sasrec.Vperr,2)))
            print("Vp MW = " + str(round(sasrec.mwVp,2)) + " +- " + str(round(sasrec.mwVperr,2)))
            print("Vc MW = " + str(round(sasrec.mwVc,2)) + " +- " + str(round(sasrec.mwVcerr,2)))
            print("Lc = " + str(round(sasrec.lc,2)) + " +- " + str(round(sasrec.lcerr,2)))
        print_button.on_clicked(print_values)

        axsave = plt.axes([0.35, 0.02, 0.1, 0.04])
        save_button = Button(axsave, 'Save File', color=axcolor, hovercolor='0.975')

        def save_file(event):
            #sascif = saxs.Sascif(sasrec)
            #sascif.write(output+".sascif")
            #print "%s file saved" % (output+".sascif")
            n2 = int(n2_box.text)
            np.savetxt(output+'.dat', np.vstack((sasrec.qc, sasrec.Ic, Icerr[:n2])).T,delimiter=' ',fmt='%.5e')
            print("%s file saved" % (output+".dat"))
        save_button.on_clicked(save_file)

        plt.show()


    print("---------------------------------")
    print("Dmax = " + str(round(sasrec.D,2)))
    print("alpha = %.5e" % sasrec.alpha)
    print("Rg = " + str(round(sasrec.rg,2)) + " +- " + str(round(sasrec.rgerr,2)))
    print("I(0) = " + str(round(sasrec.I0,2)) + " +- " + str(round(sasrec.I0err,2)))
    print("Vp = " + str(round(sasrec.Vp,2)) + " +- " + str(round(sasrec.Vperr,2)))
    print("Vp MW = " + str(round(sasrec.mwVp,2)) + " +- " + str(round(sasrec.mwVperr,2)))
    print("Vc MW = " + str(round(sasrec.mwVc,2)) + " +- " + str(round(sasrec.mwVcerr,2)))
    print("Lc = " + str(round(sasrec.lc,2)) + " +- " + str(round(sasrec.lcerr,2)))


    #sascif = saxs.Sascif(sasrec)
    #sascif.write(output+".sascif")
    #print "%s file saved" % (output+".sascif")
    n2 = int(n2_box.text)
    np.savetxt(output+'.dat', np.vstack((sasrec.qc, sasrec.Ic, Icerr[:n2])).T,delimiter=' ',fmt='%.5e')
    print("%s file saved" % (output+".dat"))
