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
    energy_obj = RetinalUtil.valid_landscape(energy_obj)
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
    plt.xlim(xlim_nm)
    PlotUtilities.lazyLabel("q (nm)", "k (pN/nm)", "")
    lim = 75
    plt.ylim(-lim, lim)

def _mean_and_stdev_landcapes(splines,q_interp):
    values = [s(q_interp) for s in splines]
    mean_energy = np.mean(values, axis=0)
    std_energy = np.std(values, axis=0)
    return mean_energy,std_energy

def plot_mean_landscape(q_interp, splines, ax=None,color='c',label=None,
                        fill_between=True):
    """
    :param q_interp: where to interpolate the splines
    :param splines: LstSqUnivariateSpline objects
    :param ax:  which axis to add to
    :return:
    """
    mean_energy, std_energy = _mean_and_stdev_landcapes(splines,q_interp)
    ax = plt.subplot(1, 1, 1) if (ax is None) else ax
    plt.subplot(ax)
    plt.plot(q_interp, mean_energy, color=color,label=label)
    if fill_between:
        plt.fill_between(q_interp, mean_energy - std_energy,
                         mean_energy + std_energy,
                         color=color, alpha=0.2)
    PlotUtilities.lazyLabel("q (nm)", "$\Delta G_0$ (kcal/mol)", "")
    return mean_energy, std_energy



def plot_delta_GF(q_interp,mean_energy,std_energy,max_q_nm=30,linestyle='None',
                  markersize=3,capsize=3,round_energy=-1,round_std=-1,
                  label_offset=0,**kw):
    """
    :param q_interp: extensions
    :param mean_energy:
    :param std_energy:
    :param max_q_nm:
    :return:
    """
    # only look at the first X nm
    max_q_idx = -1
    # determine where the max is, and label it
    max_idx = max_q_idx
    max_energy_mean = mean_energy[max_idx]
    max_energy_std = std_energy[max_idx]
    q_at_max_energy = q_interp[max_idx]
    # subtract the offset (i.e., to show the data with the PEG correction..)
    label_mean = np.round(max_energy_mean-label_offset,round_energy)
    label_std = np.round(max_energy_std,round_std)
    label = (r"$\mathbf{\Delta G}_{GF}$")  + \
            (" = {:.0f} $\pm$ {:.0f} kcal/mol").format(label_mean,label_std)
    plt.errorbar(q_at_max_energy,max_energy_mean,max_energy_std,
                 label=label,markersize=markersize,linestyle=linestyle,
                 capsize=capsize,**kw)
    return q_at_max_energy,max_energy_mean,max_energy_std

