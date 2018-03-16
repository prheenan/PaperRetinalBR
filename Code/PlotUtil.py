# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse, enum, copy

from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util,  FEC_Plot
from Lib.UtilForce.UtilIgor.TimeSepForceObj import TimeSepForceObj
from Lib.UtilForce.UtilGeneral import PlotUtilities


from Lib.AppIWT.Code.InverseWeierstrass import FEC_Pulling_Object
from Lib.AppWHAM.Code.UtilLandscape import Conversions

import RetinalUtil

def plot_landscapes(data,energy_obj,ax1=None,
                    ax2=None,ax3=None):
    if ax1 is None:
        ax1 = plt.subplot(3,1,1)
    if ax2 is None:
        ax2 = plt.subplot(3,1,2)
    if ax3 is None:
        ax3 = plt.subplot(3,1,3)
    q = energy_obj.q
    q_nm = q * 1e9
    xlim_nm = [min(q_nm), max(q_nm)]
    G0_plot = energy_obj.G0_kcal_per_mol
    spline_G0 = RetinalUtil.spline_fit(q=q, G0=energy_obj.G0)
    to_kcal_per_mol = Conversions.kcal_per_mol_per_J()
    plt.sca(ax1)
    for d in data:
        plt.plot(d.Separation * 1e9, d.Force * 1e12, markevery=50)
    plt.xlim(xlim_nm)
    PlotUtilities.lazyLabel("", "$F$ (pN)", "")
    PlotUtilities.no_x_label(ax=ax1)
    plt.sca(ax2)
    plt.plot(q_nm, G0_plot)
    plt.plot(q_nm, spline_G0(q) * to_kcal_per_mol, 'r--')
    PlotUtilities.lazyLabel("", "$\Delta G_\mathrm{0}$\n(kcal/mol)", "")
    PlotUtilities.no_x_label(ax=ax2)
    plt.xlim(xlim_nm)
    plt.sca(ax3)
    k_N_per_m = spline_G0.derivative(2)(q)
    k_pN_per_nm = k_N_per_m * 1e3
    plt.plot(q_nm, k_pN_per_nm,linewidth=0.5)
    plt.plot(q_nm, k_pN_per_nm,linewidth=0.5)
    PlotUtilities.lazyLabel("q (nm)", "k (pN/nm)", "")
    lim = 75
    plt.ylim(-lim, lim)