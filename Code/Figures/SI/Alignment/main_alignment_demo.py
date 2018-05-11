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
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec
import re

from Figures import FigureUtil



def _ensemble_alignment(gs,alignment,col_idx):
    xlim, ylim = FigureUtil._limits(alignment._all_fecs)
    common_kw = dict(xlim=xlim,ylim=ylim)
    kw_fmt = dict(color=False,is_left=(col_idx ==0),**common_kw)
    ax1 = plt.subplot(gs[0,col_idx])
    FigureUtil._plot_fec_list(alignment.zeroed.fec_list,**common_kw)
    FigureUtil._plot_fmt(ax1,**kw_fmt)
    ax2 = plt.subplot(gs[1,col_idx])
    FigureUtil._plot_fec_list(alignment.polished.fec_list,**common_kw)
    FigureUtil._plot_fmt(ax2,**kw_fmt)
    PlotUtilities.no_x_label(ax=ax2)
    ax3 = plt.subplot(gs[2,col_idx])
    FigureUtil._plot_fec_list(alignment.blacklisted.fec_list,**common_kw)
    FigureUtil._plot_fmt(ax3,is_bottom=True,**kw_fmt)


def _heatmap_alignment(gs,alignment,col_idx):
    xlim, ylim = FigureUtil._limits(alignment._all_fecs)
    max_x = xlim[1]
    bin_step_nm = 1
    bin_step_pN = 5
    bins_x = np.arange(xlim[0],xlim[1] + bin_step_nm,step=bin_step_nm)
    bins_y = np.arange(ylim[0],ylim[1] + bin_step_pN,step=bin_step_pN)
    common_kw = dict(separation_max=max_x,use_colorbar=False,title="",
                     bins=(bins_x,bins_y))
    ax1 = plt.subplot(gs[0,col_idx])
    FEC_Plot.heat_map_fec(alignment.zeroed.fec_list,**common_kw)
    FigureUtil._plot_fmt(ax1,xlim,ylim,color=True)
    ax2 = plt.subplot(gs[1,col_idx])
    FEC_Plot.heat_map_fec(alignment.polished.fec_list,**common_kw)
    FigureUtil._plot_fmt(ax2,xlim,ylim,color=True)
    title_kw = dict(color='b',y=0.95,loc='left',fontsize=6)
    downarrow = "$\Downarrow$"
    title_sub = downarrow + " Subtract $X_{\mathbf{PEG3400}}(F)$ + " + \
                "$L_{\mathbf{0,C-term}}$"
    PlotUtilities.title(title_sub,**title_kw)
    PlotUtilities.no_x_label(ax=ax2)
    ax3 = plt.subplot(gs[2,col_idx])
    FEC_Plot.heat_map_fec(alignment.blacklisted.fec_list,**common_kw)
    FigureUtil._plot_fmt(ax3,xlim,ylim,is_bottom=True,color=True)
    PlotUtilities.title(downarrow + " Remove poorly-fit FECs",**title_kw)
    return [ax1,ax2,ax3]

def _make_algned_plot(alignment,label):
    fig = PlotUtilities.figure((3, 4))
    gs = gridspec.GridSpec(3, 2)
    # make the 'standard' alignment plots
    axes = _heatmap_alignment(gs, alignment, 0)
    _ensemble_alignment(gs, alignment, 1)
    out_name = "./FigureS_Alignment_{:s}.png".format(label)
    PlotUtilities.title(label, ax=axes[0])
    PlotUtilities.savefig(fig, out_name,
                          subplots_adjust=dict(hspace=0.12, wspace=0.02))



def _make_plots(galleries_labels):
    alignments =  [FigureUtil._alignment_pipeline(gallery_tmp[0])
                   for gallery_tmp in galleries_labels]
    for i,(_,label) in enumerate(galleries_labels):
        # make the standard aligning plot
        alignment = alignments[i]
        _make_algned_plot(alignment, label)
    # plot the final curves on the same plot
    xlim,ylim = FigureUtil._limits(alignment._all_fecs)
    colors = ['rebeccapurple','g']
    fig = PlotUtilities.figure((5, 3))
    gs = gridspec.GridSpec(2, 2)
    # reverse everything, so PEG600 is on top
    galleries_labels = galleries_labels[::-1]
    alignments = alignments[::-1]
    for i,(_,l) in enumerate(galleries_labels):
        ax = plt.subplot(gs[i,0])
        a = alignments[i]
        FigureUtil._plot_fec_list(a.blacklisted.fec_list, xlim, ylim,label=l,
                                  color=colors[i])
        if (i == 0):
            PlotUtilities.no_x_label(ax)
            PlotUtilities.xlabel("",ax=ax)
    # plot them both on the last column
    plt.subplot(gs[:,1])
    kw = [dict(),dict(linewidth=0.6)]
    for i,(_,l) in enumerate(galleries_labels):
        a = alignments[i]
        FigureUtil._plot_fec_list(a.blacklisted.fec_list, xlim, ylim,label=l,
                                  color=colors[i],**kw[i])
    PlotUtilities.savefig(fig,"FigureS_3400vs600.png")


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
                                                read_f,True,base_dir)
    galleries_labels = [ [gallery.PEG600,"PEG600"],
                         [gallery.PEG3400, "PEG3400"]]
    _make_plots(galleries_labels)


if __name__ == "__main__":
    run()
