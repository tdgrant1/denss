#!/usr/bin/env python
#
#    denss_fit_data.py
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
#    Email:  <tdgrant@buffalo.edu>
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
import time
import os, argparse, sys
import numpy as np

import denss

sasrec = None

def main():
    global sasrec
    parser = argparse.ArgumentParser(description="A tool for fitting solution scattering data with smooth function based on Moore's algorithm for fitting a trigonometric series.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="SAXS data file for input (either .dat or .out)")
    parser.add_argument("-d", "--dmax", default=None, type=float, help="Estimated maximum dimension in angstroms (if q values are in 1/nm, use the -u option to convert).")
    parser.add_argument("-a", "--alpha", default=None, type=float, help="Set alpha smoothing factor")
    parser.add_argument("-u", "--units", default="a", type=str, help="Angular units (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
    parser.add_argument("-n1", "--n1", default=None, type=int, help="First data point to use")
    parser.add_argument("-n2", "--n2", default=None, type=int, help="Last data point to use")
    parser.add_argument("--ignore_errors", dest="ignore_errors", action="store_true", help="Ignore error bars (i.e., set all error bars to 1.0).")
    parser.add_argument("-q", "--qfile", default=None, type=str, help="ASCII text filename to use for setting the calculated q values (like a SAXS .dat file, but just uses first column, optional)")
    parser.add_argument("-qmax", "--qmax", default=None, type=float, help="Maximum q value for calculated intensities (optional)")
    parser.add_argument("-nq", "--nq", default=None, type=int, help="Number of data points in calculated intensity profile (optional)")
    parser.add_argument("-r", "--rfile", default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument("-nr", "--nr", default=None, type=int, help="Number of points in P(r) curve (default = number of points in I(q) profile).")
    parser.add_argument("--nes", default=2, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max_dmax", default=None, type=float, help="Maximum limit for allowed Dmax values (for plotting slider)")
    parser.add_argument("--max_alpha", default=None, type=float, help="Maximum limit for allowed alpha values (for plotting slider)")
    parser.add_argument("--max_nes", default=10, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--no_gui", dest="plot", action="store_false", help="Do not run the interactive GUI mode.")
    parser.add_argument("--no_log", dest="log", action="store_false", help="Do not plot on log y axis.")
    parser.add_argument("--no_extrapolation", dest="extrapolate", action="store_false", help="Do not extrapolate high q data.")
    parser.add_argument("--write_shannon", dest="write_shannon", action="store_true", help="Write a file containing only the Shannon intensities.")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
    parser.set_defaults(plot=True)
    parser.set_defaults(write_shannon=False)
    parser.set_defaults(ignore_errors=False)
    args = parser.parse_args()

    alpha = args.alpha

    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        output = basename
    else:
        output = args.output

    if args.ignore_errors:
        Iq = np.genfromtxt(args.file, invalid_raise=False, usecols=(0, 1))
    else:
        Iq = np.genfromtxt(args.file, invalid_raise=False, usecols=(0, 1, 2))
    if len(Iq.shape) < 2:
        print("Invalid data format. Data file must have 3 columns: q, I, errors. Alternatively, disable errors with --ignore_errors option (sets errors to 1.0).")
        exit()
    if Iq.shape[1] < 3 or args.ignore_errors:
        print("WARNING: Only 2 columns given. Data should have 3 columns: q, I, errors.")
        print("WARNING: Setting error bars to 1.0 (i.e., ignoring error bars)")
        Iq2 = np.zeros((Iq.shape[0], 3))
        Iq2[:,:2] = Iq[:,:2]
        Iq2[:,2] += 1.0 #set error bars to 1.0
        Iq = Iq2
    Iq = Iq[~np.isnan(Iq).any(axis = 1)]
    #get rid of any data points equal to zero in the intensities or errors columns
    idx = np.where((Iq[:,1]!=0)&(Iq[:,2]!=0))
    Iq = Iq[idx]

    nes = args.nes

    if args.units == "nm":
        Iq[:,0] *= 0.1

    if args.n1 is None:
        n1 = 0
    else:
        n1 = args.n1
    if args.n2 is None:
        n2 = len(Iq[:,0])
    else:
        n2 = args.n2

    Iq_orig = np.copy(Iq)

    if args.dmax is None:
        #estimate dmax directly from data
        #note that denss.estimate_dmax does NOT extrapolate
        #the high q data, even though by default
        #denss.Sasrec does extrapolate.
        D, sasrec = denss.estimate_dmax(Iq, clean_up=True)
    else:
        D = args.dmax

    if args.max_dmax is None:
        args.max_dmax = 2.*D

    if args.rfile is not None:
        r = np.loadtxt(args.rfile,usecols=(0,))
    else:
        r = None

    if args.qfile is not None:
        qc = np.loadtxt(args.qfile,usecols=(0,))
    else:
        qc = None

    print("Dmax = %.2f"%D)
    qmax = Iq[n1:n2,0].max()
    if (qc is None) and (args.extrapolate):
        qmaxc = qmax*3.0
    elif (qc is None):
        qmaxc = qmax
    else:
        qmaxc = qc.max()

    #let a user set a desired set of q values to be calculated
    #based on a given qmax and nq
    if (args.qmax is not None) or (args.nq is not None):
        if args.qmax is not None:
            qmaxc = args.qmax
        else:
            qmaxc = qmaxc
        if args.nq is not None:
            nqc = args.nq
        else:
            nqc = 501
        qc = np.linspace(0,qmaxc,nqc)
        #assume if they are giving args.qmax or args.nq, that they want to
        #disable extrapolation
        args.extrapolate = False

    nsh = qmax/(np.pi/D)
    nshc = qmaxc/(np.pi/D)
    print("Number of experimental Shannon channels: %d"%(nsh))
    print("Number of calculated Shannon channels: %d"%(nshc))
    if (nsh > 500) or (nshc>500):
        print("WARNING: Nsh > 500. Calculation may take a while. Please double check Dmax is accurate.")
        #give the user a few seconds to cancel with CTRL-C
        waittime = 10
        try:
            for i in range(waittime+1):
                sys.stdout.write("\rTo cancel, press CTRL-C in the next %d seconds. "%(waittime-i))
                sys.stdout.flush()
                time.sleep(1)
            print()
        except KeyboardInterrupt:
            print("Canceling...")
            exit()


    #calculate chi2 when alpha=0, to get the best possible chi2 for reference
    sasrec = denss.Sasrec(Iq[n1:n2], D, qc=qc, r=r, nr=args.nr, ne=nes, alpha=0.0, extrapolate=args.extrapolate)
    ideal_chi2 = sasrec.calc_chi2()

    if args.alpha is None:
        alpha = sasrec.optimize_alpha()
    else:
        alpha = args.alpha
    sasrec = denss.Sasrec(Iq[n1:n2], D, qc=qc, r=r, nr=args.nr, alpha=alpha, ne=nes, extrapolate=args.extrapolate)

    #implement method of estimating Vp, Vc, etc using oversmoothing
    sasrec.estimate_Vp_etal()

    #set a maximum alpha range, so when users click in the slider
    #it at least does something reasonable, rather than either nothing
    #significant, or so huge it becomes difficult to find the right value
    if args.max_alpha is None:
        if alpha == 0.0:
            max_alpha = 2.0
        else:
            max_alpha = 2*alpha

    def store_parameters_as_string(event=None):
        param_str = ("Parameter Values:\n"
        "Chi2  = {chi2:.5e}\n"
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
        ).format(chi2=sasrec.chi2,
            dmax=sasrec.D,alpha=sasrec.alpha,
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
        param_str = store_parameters_as_string()
        #add column headers to param_str for output
        param_str += 'q, I, error, fit ; chi2 = %.3f'%sasrec.chi2
        #quick, interpolate the raw data, sasrec.I, to the new qc values, but be sure to
        #put zeros in for the q values not measured
        Iinterp = np.interp(sasrec.qc, sasrec.q_data, sasrec.I_data, left=0.0, right=0.0)
        Ierrinterp = np.interp(sasrec.qc, sasrec.q_data, sasrec.Ierr_data)
        np.savetxt(output+'.fit', np.vstack((sasrec.qc, Iinterp, Ierrinterp, sasrec.Ic)).T,delimiter=' ',fmt='%.5e',header=param_str)
        np.savetxt(output+'_pr.dat', np.vstack((sasrec.r, sasrec.P, sasrec.Perr)).T,delimiter=' ',fmt='%.5e')
        print("%s and %s files saved" % (output+".fit",output+"_pr.dat"))
        if args.write_shannon:
            np.savetxt(output+'_Shannon.dat', np.vstack((sasrec.qn, sasrec.In, sasrec.Inerr)).T,delimiter=' ',fmt='%.5e',header=param_str)

    if args.plot:
        #if plotting is enabled, try to import matplotlib
        #if import fails, set plotting to false
        #first try using Qt5Agg backend, then TkAgg
        try:
            import matplotlib
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons #, RangeSlider
            fig = plt.figure(0, figsize=(12,6))
            qtsuccess = True
        except ImportError:
            qtsuccess = False
        if not qtsuccess:
            print("Using TkAgg")
            try:
                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons #, RangeSlider
                fig = plt.figure(0, figsize=(12,6))
            except Exception as e:
                print("Matplotlib failed to create figure. GUI mode disabled.")
                print("Error: %s"%e)
                args.plot = False
        else:
            print("Using QtAgg")

    if args.plot:
        try:
            #this command changed in matplotlib v3.6
            fig.canvas.manager.set_window_title(output)
        except:
            try:
                fig.canvas.set_window_title(output)
            except:
                pass
        fig.suptitle(output)
        axI = plt.subplot2grid((3,2), (0,0),rowspan=2)
        axR = plt.subplot2grid((3,2), (2,0),sharex=axI)
        axP = plt.subplot2grid((3,2), (0,1),rowspan=3)
        plt.subplots_adjust(left=0.068, bottom=0.25, right=0.98, top=0.95)

        #add a plot of untouched light gray data for reference for the user
        I_l0, = axI.plot(Iq_orig[:,0], Iq_orig[:,1], '.', c='0.8', ms=3)
        I_l1, = axI.plot(sasrec.q_data, sasrec.I_data, 'k.', ms=3, label='test')
        I_l2, = axI.plot(sasrec.qc, sasrec.Ic, 'r-', lw=2)
        I_l3, = axI.plot(sasrec.qn, sasrec.In, 'bo', mec='b', mfc='none', mew=2)
        chi2_text = axI.text(0.7,0.9,r"$\chi^2$ = %.3e"%sasrec.chi2,transform=axI.transAxes,fontsize='large')
        if args.log: axI.semilogy()
        axI.set_ylabel('I(q)')
        axI.set_xlabel('q')

        #make plots for each Bn (should we also do each Sn for P(r) plot?)
        # I_Bn = []
        # nn = range(5,20) #range(20) #idx Bns to show
        # for i in nn:
        #     I_Bn.append(axI.plot(sasrec.q, 2*sasrec.In[i]*sasrec.B[i], label='B_%d'%i))


        #residuals
        #first, just ensure that we're comparing similar q ranges, so
        #interpolate from qc to q_data to enable subtraction, since there's
        #q values in qc that are not in q_data, and possibly vice versa
        Icinterp = np.interp(sasrec.q_data, sasrec.qc, sasrec.Ic)
        res = (sasrec.I_data - Icinterp)/sasrec.Ierr_data
        #in case qc were fewer points than the data, for whatever reason,
        #only grab the points up to qc.max
        ridx = np.where((sasrec.q_data<sasrec.qc.max()))
        absolute_maximum_q = np.max([sasrec.qc.max(),sasrec.q.max(),Iq_orig[:,0].max()])
        Ires_l0, = axR.plot([0,absolute_maximum_q], [0,0], 'k--')
        Ires_l1, = axR.plot(sasrec.q_data[ridx], res[ridx], 'r.', ms=3)
        axR.set_ylabel('Residuals')
        axR.set_xlabel('q')

        P_l1, = axP.plot(sasrec.r*100, sasrec.r*0, 'k--')
        P_l2, = axP.plot(sasrec.r, sasrec.P, 'b-', lw=2)
        axP.set_ylabel('P(r)')
        axP.set_xlabel('r')

        #plot Sns on P(r)
        # P_Sn = []
        # for i in nn:
        #     P_Sn.append(axP.plot(sasrec.r, sasrec.In[i]*sasrec.S[i], label='S_%d'%i))
        # mm = nn #range(0,20)
        # P_Sn[-1] = axP.plot(sasrec.r, np.sum(sasrec.In[mm,None]*sasrec.S[mm],axis=0),'k-',lw=4)

        #axI.set_xlim([0,1.1*np.max(sasrec.q)])
        #axR.set_xlim([0,1.1*np.max(sasrec.q)])
        axI.set_xlim([0,Iq_orig[-1,0]])
        axR.set_xlim([0,Iq_orig[-1,0]])
        axP.set_xlim([0,1.1*np.max(sasrec.r)])
        axI.set_ylim([0.25*np.min(sasrec.Ic[sasrec.qc<Iq_orig[-1,0]]),2*np.max(sasrec.Ic[sasrec.qc<Iq_orig[-1,0]])])
        #axR.set_ylim([0,Iq_orig[-1,0]])
        #axP.set_ylim([0,1.1*np.max(sasrec.r)])
        #the "q" axis label is a little low, so let's raise it up a bit
        axR.xaxis.labelpad = -10

        axcolor = 'lightgoldenrodyellow'
        #axn1n2 = plt.axes([0.05, 0.175, 0.4, 0.03], facecolor=axcolor)
        axdmax = plt.axes([0.05, 0.125, 0.4, 0.03], facecolor=axcolor)
        axalpha = plt.axes([0.05, 0.075, 0.4, 0.03], facecolor=axcolor)
        #axnes = plt.axes([0.05, 0.025, 0.4, 0.03], facecolor=axcolor)

        axI0 = plt.figtext(.57, .125,   r"$I(0)$  = %.2e $\pm$ %.2e"%(sasrec.I0,sasrec.I0err),family='monospace')
        axrg = plt.figtext(.57, .075,   r"$R_g$   = %.2e $\pm$ %.2e"%(sasrec.rg,sasrec.rgerr),family='monospace')
        axrav = plt.figtext(.57, .025,  r"$\overline{r}$    = %.2e $\pm$ %.2e"%(sasrec.avgr,sasrec.avgrerr),family='monospace')
        axVp = plt.figtext(.77, .125,   r"$V_p$ = %.2e $\pm$ %.2e"%(sasrec.Vp,sasrec.Vperr),family='monospace')
        axVc = plt.figtext(.77, .075,   r"$V_c$ = %.2e $\pm$ %.2e"%(sasrec.Vc,sasrec.Vcerr),family='monospace')
        axlc = plt.figtext(.77, .025,   r"$\ell_c$ = %.2e $\pm$ %.2e"%(sasrec.lc,sasrec.lcerr),family='monospace')
        #axVpmw = plt.figtext(.55, .075, "Vp MW = %.2e $\pm$ %.2e"%(sasrec.mwVp,sasrec.mwVperr),family='monospace')
        #axVcmw = plt.figtext(.55, .025, "Vc MW = %.2e $\pm$ %.2e"%(sasrec.mwVc,sasrec.mwVcerr),family='monospace')

        #RangeSlider is for very new versions of matplotlib, so for now ignore it
        #sn1n2 = RangeSlider(axn1n2, 'n1n2', 0, Iq_orig.shape[0], valinit=(n1, n2))
        #sn1n2.valtext.set_visible(False)

        sdmax = Slider(axdmax, 'Dmax', 0.0, args.max_dmax, valinit=D)
        sdmax.valtext.set_visible(False)
        # set up ticks marks on the slider to denote the change in interaction
        axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax])
        #axdmax.xaxis.tick_top()
        axdmax.tick_params(labelbottom=False)

        salpha = Slider(axalpha, r'$\alpha$', 0.0, max_alpha, valinit=alpha)
        salpha.valtext.set_visible(False)

        dmax = D
        n1 = str(n1)
        n2 = str(n2)
        nRemove = ""

        def analyze(dmax,alpha,n1,n2,nRemove,extrapolate):
            global sasrec
            points = np.arange(Iq_orig.shape[0])
            mask = np.ones(len(points),dtype=bool)
            mask[points<=n1] = False
            mask[points>=n2] = False
            #parse nRemove string, allow n point or n-m ranges
            ranges = nRemove.split(",")
            for rangei in ranges:
                if "-" in rangei:
                    rangei_a, rangei_b = [int(j) for j in rangei.split("-")]
                    mask[(points>=rangei_a)&(points<=rangei_b)] = False
                elif rangei == "":
                    pass
                else:
                    try:
                        mask[int(rangei)] = False
                    except:
                        print("Invalid Remove Range")
            qc = denss.create_lowq(q=Iq_orig[:,0])
            sasrec = denss.Sasrec(Iq_orig[mask], dmax, qc=qc, r=r, nr=args.nr, alpha=alpha, ne=nes, extrapolate=extrapolate)
            sasrec.estimate_Vp_etal()
            res = (sasrec.I_data - sasrec.Ic_qe)/sasrec.Ierr_data
            ridx = np.where((sasrec.q_data<sasrec.qc.max()))
            I_l1.set_data(sasrec.q_data, sasrec.I_data)
            I_l2.set_data(sasrec.qc, sasrec.Ic)
            I_l3.set_data(sasrec.qn, sasrec.In)
            chi2_text.set_text(r"$\chi^2$ = %.3e"%sasrec.chi2)
            Ires_l1.set_data(sasrec.q_data, res)
            P_l2.set_data(sasrec.r, sasrec.P)
            axI0.set_text(r"$I(0)$  = %.2e $\pm$ %.2e"%(sasrec.I0,sasrec.I0err))
            axrg.set_text(r"$R_g$   = %.2e $\pm$ %.2e"%(sasrec.rg,sasrec.rgerr))
            axrav.set_text(r"$\overline{r}$    = %.2e $\pm$ %.2e"%(sasrec.avgr,sasrec.avgrerr))
            axVp.set_text(r"$V_p$ = %.2e $\pm$ %.2e"%(sasrec.Vp,sasrec.Vperr))
            axVc.set_text(r"$V_c$ = %.2e $\pm$ %.2e"%(sasrec.Vc,sasrec.Vcerr))
            axlc.set_text(r"$\ell_c$ = %.2e $\pm$ %.2e"%(sasrec.lc,sasrec.lcerr))
            #axVpmw.set_text("Vp MW = %.2e $\pm$ %.2e"%(sasrec.mwVp,sasrec.mwVperr))
            #axVcmw.set_text("Vc MW = %.2e $\pm$ %.2e"%(sasrec.mwVc,sasrec.mwVcerr))
            # j = 0
            # for i in nn:
            #     I_Bn[j][0].set_data(sasrec.q, 2*sasrec.In[i]*sasrec.B[i])
            #     j+=1
            # j = 0
            # for i in nn:
            #     P_Sn[j][0].set_data(sasrec.r, sasrec.In[i]*sasrec.S[i])
            #     j+=1
            # P_Sn[-1][0].set_data(sasrec.r, np.sum(sasrec.In[mm,None]*sasrec.S[mm],axis=0))


        def n1_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(text)
            n2 = int(n2_box.text)
            nRemove = str(nRemove_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,nRemove,extrapolate)
            fig.canvas.draw_idle()

        def n2_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(text)
            nRemove = str(nRemove_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,nRemove,extrapolate)
            fig.canvas.draw_idle()

        def nRemove_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            nRemove = str(text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,nRemove,extrapolate)
            fig.canvas.draw_idle()

        def extrapolate_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            nRemove = str(nRemove_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,nRemove,extrapolate)
            fig.canvas.draw_idle()

        def D_submit(text):
            dmax = float(text)
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            nRemove = str(nRemove_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,nRemove,extrapolate)
            # this updates the slider value based on text box value
            sdmax.set_val(dmax)
            if (dmax > 0.9 * sdmax.valmax) or (dmax < 0.1 * sdmax.valmax):
                sdmax.valmax = 2 * dmax
                sdmax.ax.set_xlim(sdmax.valmin, sdmax.valmax)
                axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax])
            fig.canvas.draw_idle()

        def A_submit(text):
            dmax = sdmax.val
            alpha = float(text)
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            nRemove = str(nRemove_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,nRemove,extrapolate)
            # this updates the slider value based on text box value
            salpha.set_val(alpha)
            # partions alpha slider
            if (alpha > 0.9 * salpha.valmax) or (alpha < 0.1 * salpha.valmax):
                salpha.valmax = 2 * alpha
                # alpha starting at zero makes initial adjustment additive not multiplicative
                if alpha != 0:
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)
                elif alpha == 0:
                    salpha.valmax = alpha + 10
                    salpha.valmin = 0.0
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)
            fig.canvas.draw_idle()

        def update(val):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            nRemove = str(nRemove_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            #print(extrapolate)
            analyze(dmax,alpha,n1,n2,nRemove,extrapolate)
            # partitions the slider, so clicking in the upper and lower range scale valmax
            if (dmax > 0.9 * sdmax.valmax) or (dmax < 0.1 * sdmax.valmax):
                sdmax.valmax = 2 * dmax
                sdmax.ax.set_xlim(sdmax.valmin, sdmax.valmax)
                axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax])
            # partitions slider as well
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
        # plt.figtext(0.0085, 0.178, "First:")
        # axIntn1 = plt.axes([0.075, 0.170, 0.08, 0.03])
        plt.figtext(0.02, 0.178, "First:")
        axIntn1 = plt.axes([0.05, 0.170, 0.05, 0.03])
        n1_box = TextBox(axIntn1, '', initial=n1)
        n1_box.on_submit(n1_submit)

        # making a text entry for n2 that allows for user input
        n2value = "{}".format(n2)
        # plt.figtext(0.17, 0.178, "Last:")
        # axIntn2 = plt.axes([0.235, 0.170, 0.08, 0.03])
        plt.figtext(0.12, 0.178, "Last:")
        axIntn2 = plt.axes([0.15, 0.170, 0.05, 0.03])
        n2_box = TextBox(axIntn2, '', initial=n2)
        n2_box.on_submit(n2_submit)

        # making a text entry for removing points that allows for user input
        nRemovevalue = "{}".format(nRemove)
        plt.figtext(0.21, 0.178, "Remove:")
        axnRemove = plt.axes([0.27, 0.170, 0.08, 0.03])
        nRemove_box = TextBox(axnRemove, '', initial=nRemove)
        nRemove_box.on_submit(nRemove_submit)

        # create a checkbox for extrapolation
        axExtrap = plt.axes([0.37, 0.170, 0.015, 0.03], frameon=True)
        axExtrap.margins(0.0)

        extrapolate_check = CheckButtons(axExtrap, ["Extrapolate"], [args.extrapolate])
        check = extrapolate_check
        try:
            #the axes object for the checkbutton is crazy large, and actually
            #blocks the sliders underneath even when frameon=False
            #so we have to manually set the size and location of each of the
            #elements of the checkbox after setting the axes margins to zero above
            #including the rectangle checkbox, the lines for the X, and the label
            size =  1.0 #size relative to axes axExtrap
            for rect in extrapolate_check.rectangles:
                rect.set_x(0.)
                rect.set_y(0.)
                rect.set_width(size)
                rect.set_height(size)
            first = True
            for l in check.lines:
                for ll in l:
                    llx = ll.get_xdata()
                    lly = ll.get_ydata()
                    #print(llx)
                    #print(lly)
                    ll.set_xdata([0.0,size])
                    if first:
                        #there's two lines making
                        #up the checkbox, so need
                        #to set the y values separately
                        #one going from bottom left to
                        #upper right, the other opposite
                        ll.set_ydata([size,0.0])
                        first = False
                    else:
                        ll.set_ydata([0.0, size])
        except:
            #newer versions of matplotlib don't have the check.rectangles attribute. so for now they get a small box.
            pass

        check.labels[0].set_position((1.5,.5))

        #here is the slider updating
        sdmax.on_changed(update)
        salpha.on_changed(update)
        extrapolate_check.on_clicked(update)
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


if __name__ == "__main__":
    main()