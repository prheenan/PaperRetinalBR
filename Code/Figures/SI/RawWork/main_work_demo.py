# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import re

sys.path.append("../../../")
from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Processing import ProcessingUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec
import re

from Figures import FigureUtil
from Lib.AppIWT.Code.UtilLandscape import Conversions
from scipy import integrate
from scipy.interpolate import interp1d

def kcal_per_mol_per_J():
    return Conversions.kcal_per_mol_per_J()

def _single_work(x,f):
    return integrate.cumtrapz(x=x, y=f, initial=0)

def _calculate_work(x_arr,f_arr):
    works = [ _single_work(x,f)
             for x,f in zip(x_arr,f_arr)]
    return works

def _f_david(kbT,L0,Lp,x):
    """
    See 'Is this an estimate of ddG_retinal?'

    :param kbT:
    :param L0:
    :param Lp:
    :param x:
    :return:
    """
    return kbT * (L0**2) / (4 * Lp * (x -L0)**2)

def _make_work_plot(fec_list,x_arr,works):
    works_kcal = np.array(works)*kcal_per_mol_per_J()
    # get the interpolated work
    x_min = max([min(x) for x in x_arr])
    x_max = min([max(x) for x in x_arr])
    x_interp = np.linspace(x_min,x_max,endpoint=True,num=200)
    interp_f_W = [interp1d(x=x,y=w) for x,w in zip(x_arr, works_kcal)]
    interp_W = [f(x_interp) for f in interp_f_W]
    mean_W = np.mean(interp_W,axis=0)
    std_W = np.std(interp_W,axis=0)
    # use Davids function
    shift = min(x_interp) - 2e-9
    max_david = 32e-9
    x_david = np.linspace(0,max_david,num=100)
    style_david = dict(color='b',linestyle='--',label="David's model")
    legend_kw = dict(handlelength=2)
    david_F = _f_david(kbT=4.1e-21, L0=11e-9+shift, Lp=0.4e-9, x=x_david)
    david_W = _single_work(x=x_david,f=david_F)
    x_david_plot = x_david * 1e9
    W_david_plot = david_W * kcal_per_mol_per_J()
    f_david_plot = david_F * 1e12
    # interpolate each work onto a grid
    _,ylim = FigureUtil._limits(fec_list)
    xlim = [20,45]
    x_interp_plot = x_interp * 1e9
    fudge_work = max(std_W)
    max_W = np.max([max(w) for w in works_kcal])+fudge_work
    ylim_work = [-20,400]
    ax1 = plt.subplot(3,1,1)
    FigureUtil._plot_fec_list(fec_list,xlim,ylim)
    plt.plot(x_david_plot,f_david_plot,**style_david)
    PlotUtilities.legend(**legend_kw)
    FigureUtil._plot_fmt(ax1, xlim, ylim)
    ax2 = plt.subplot(3,1,2)
    for x,w in zip(x_arr,works_kcal):
        plt.plot(x * 1e9,w,linewidth=0.75)
    FigureUtil._plot_fmt(ax2, xlim, ylim_work)
    PlotUtilities.lazyLabel("","W (kcal/mol)","")
    ax3 = plt.subplot(3,1,3)
    plt.errorbar(x_interp_plot,mean_W,fmt='r-',markersize=2,capsize=2)
    plt.fill_between(x_interp_plot,mean_W-std_W,mean_W+std_W,color='r',
                     alpha=0.3,label="$<W> \pm \sigma$",linewidth=0)
    plt.plot(x_david_plot,W_david_plot,'b--',zorder=5,**style_david)
    FigureUtil._plot_fmt(ax3, xlim, ylim_work,is_bottom=True)
    PlotUtilities.legend(**legend_kw)
    PlotUtilities.lazyLabel("Extension (nm)","W (kcal/mol)","")

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir = "../../../../Data/FECs180307/"
    read_f = FigureUtil.read_sample_landscapes
    gallery = CheckpointUtilities.getCheckpoint("./caches.pkl",
                                                read_f,False,base_dir)
    lab_plot = [ ["BR-PEG3400",gallery.PEG3400],
                 ["BO-PEG3400",gallery.BO_PEG3400] ]
    for label,to_use in lab_plot:
        pipeline = FigureUtil._alignment_pipeline(to_use)
        fecs = pipeline.blacklisted.fec_list
        # calculate all the works
        x_arr = [f.Separation for f in fecs]
        f_arr = [f.Force for f in fecs]
        works = _calculate_work(x_arr,f_arr)
        fig = PlotUtilities.figure()
        _make_work_plot(fecs, x_arr, works)
        PlotUtilities.savefig(fig, "FigureS_Work_{:s}.png".format(label))


if __name__ == "__main__":
    run()
