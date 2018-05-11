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

def _mean_work(x_arr,works_kcal):
    x_min = max([min(x) for x in x_arr])
    x_max = min([max(x) for x in x_arr])
    x_interp = np.linspace(x_min,x_max,endpoint=True,num=200)
    interp_f_W = [interp1d(x=x,y=w) for x,w in zip(x_arr, works_kcal)]
    interp_W = [f(x_interp) for f in interp_f_W]
    mean_W = np.mean(interp_W,axis=0)
    std_W = np.std(interp_W,axis=0)
    return x_interp, mean_W, std_W

def xlim_ylim():
    ylim_work = [-20,750]
    xlim = [20,60]
    ylim_force = [-20,300]
    return xlim, ylim_force,ylim_work

def _plot_mean_works(x_interp,mean_W,std_W,color,title):
    x_interp_plot = x_interp * 1e9
    plt.errorbar(x_interp_plot,mean_W,fmt='-',color=color,markersize=2,
                 capsize=2)
    label_W = title.split("-PEG")[0]
    label_W_mean = r"$(\mu  \pm \sigma)" + \
                   r"_{W,\mathbf{" + label_W + r"}}$"
    plt.fill_between(x_interp_plot,mean_W-std_W,mean_W+std_W,color=color,
                     alpha=0.3,label=label_W_mean,linewidth=0)

def _make_work_plot(fec_list,x_arr,works_kcal,gs,col,color,title):
    # get the interpolated work
    x_interp, mean_W, std_W = _mean_work(x_arr, works_kcal)
    # use Davids function
    shift = 23e-9
    max_david = 31e-9
    x_david = np.linspace(0,max_david,num=100)
    style_david = dict(color='b',linestyle='--',label="David")
    legend_kw = dict(handlelength=1.5,handletextpad=0.3,fontsize=6)
    david_F = _f_david(kbT=4.1e-21, L0=11e-9+shift, Lp=0.4e-9, x=x_david)
    david_W = _single_work(x=x_david,f=david_F)
    xlim, ylim_force, ylim_work = xlim_ylim()
    x_david_plot = x_david * 1e9
    W_david_plot = david_W * kcal_per_mol_per_J()
    f_david_plot = david_F * 1e12
    is_left = (col == 0)
    fmt_kw = dict(is_left=is_left)
    label_work = "$W$ (kcal/mol)"
    # interpolate each work onto a grid
    _,ylim = FigureUtil._limits(fec_list)
    xlim = [20,60]
    fudge_work = max(std_W)
    ax1 = plt.subplot(gs[0,col])
    FigureUtil._plot_fec_list(fec_list,xlim,ylim)
    plt.plot(x_david_plot,f_david_plot,**style_david)
    if is_left:
        PlotUtilities.legend(**legend_kw)
    FigureUtil._plot_fmt(ax1, xlim, ylim,**fmt_kw)
    PlotUtilities.title(title,color=color)
    ax2 = plt.subplot(gs[1,col])
    for x,w in zip(x_arr,works_kcal):
        plt.plot(x * 1e9,w,linewidth=0.75)
    FigureUtil._plot_fmt(ax2, xlim, ylim_work,ylabel=label_work,**fmt_kw)
    ax3 = plt.subplot(gs[2,col])
    _plot_mean_works(x_interp, mean_W, std_W, color, title)
    style_lower_david = dict(**style_david)
    if (not is_left):
        style_lower_david['label'] = None
    plt.plot(x_david_plot,W_david_plot,'b--',zorder=5,**style_lower_david)
    PlotUtilities.legend(**legend_kw)
    FigureUtil._plot_fmt(ax3, xlim, ylim_work,is_bottom=True,
                         ylabel=label_work,**fmt_kw)

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
    fig = PlotUtilities.figure(figsize=(4,6))
    gs = gridspec.GridSpec(ncols=1,nrows=2,height_ratios=[3,1])
    gs_pipeline= gridspec.GridSpecFromSubplotSpec(ncols=2,nrows=3,hspace=0.03,
                                                  subplot_spec=gs[0,:])
    colors = ['r','rebeccapurple']
    xlim,_,ylm_W = xlim_ylim()

    mean_works_info = []
    for i,(label,to_use) in enumerate(lab_plot):
        pipeline = FigureUtil._alignment_pipeline(to_use)
        fecs = pipeline.blacklisted.fec_list
        # calculate all the works
        x_arr = [f.Separation for f in fecs]
        f_arr = [f.Force for f in fecs]
        works = _calculate_work(x_arr,f_arr)
        works_kcal = np.array(works) * kcal_per_mol_per_J()
        c = colors[i]
        _make_work_plot(fecs, x_arr, works_kcal,gs_pipeline,col=i,color=c,
                         title=label)
        x_interp, mean_W, std_W= _mean_work(x_arr,works_kcal)
        plt.subplot(gs[-1, :])
        _plot_mean_works(x_interp, mean_W, std_W, color=c, title=label)
        plt.xlim(xlim)
        plt.ylim(ylm_W)
        PlotUtilities.lazyLabel("Extension (nm)" , "$W$ (kcal/mol)","",
                                useLegend=False)
    PlotUtilities.savefig(fig, "FigureS_Work.png".format(label))


if __name__ == "__main__":
    run()
