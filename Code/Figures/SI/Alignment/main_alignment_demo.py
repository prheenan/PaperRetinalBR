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

class SnapshotFEC(object):
    def __init__(self,step,fec_list):
        self.step = step
        self.fec_list = fec_list

class AlignmentInfo(object):
    def __init__(self,e,zeroed,polished,blacklisted):
        self.landscape = e
        self.zeroed = zeroed
        self.polished = polished
        self.blacklisted = blacklisted
    @property
    def _all_fecs(self):
        all_lists = [self.zeroed,self.polished,self.blacklisted]
        to_ret = [f for list_v in all_lists for f in list_v.fec_list]
        return to_ret

def _snapsnot(base_dir,step):
    corrected_dir = Pipeline._cache_dir(base=base_dir,
                                        enum=step)
    data = CheckpointUtilities.lazy_multi_load(corrected_dir)
    return SnapshotFEC(step,data)


def _alignment_pipeline(e):
    base_dir_landscapes = e.base_dir
    base_dir = base_dir_landscapes.split("landscape_")[0]
    # get the corrected directory (this is *zeroed*)
    zeroed = _snapsnot(base_dir,step=Pipeline.Step.CORRECTED)
    # get the polished / aligned dir
    polished = _snapsnot(base_dir,step=Pipeline.Step.POLISH)
    # get the directory after blacklisting bad curves
    base_landscape = RetinalUtil._landscape_dir(base_dir)
    blacklist = _snapsnot(base_landscape, step=Pipeline.Step.MANUAL)
    to_ret = AlignmentInfo(e,zeroed,polished,blacklist)
    return to_ret

def _plot_fmt(ax,xlim,ylim,is_bottom=False,color=True,is_left=True):
    plt.xlim(xlim)
    plt.ylim(ylim)
    PlotUtilities.title("")
    PlotUtilities.ylabel("$F$ (pN)")
    if (not is_bottom):
        PlotUtilities.no_x_label(ax=ax)
        PlotUtilities.xlabel("")
    if (not is_left):
        PlotUtilities.no_y_label(ax=ax)
        PlotUtilities.ylabel("")
    if color:
        color_kw = dict(ax=ax,color='w',label_color='k')
        PlotUtilities.color_x(**color_kw)
        PlotUtilities.color_y(**color_kw)

def _limits(alignment):
    xlim = [-20,120]
    max_y = np.max([max(f.Force) for f in alignment._all_fecs]) * 1e12
    ylim = [-50,max_y]
    return xlim,ylim

def _plot_fec_list(list_v,xlim,ylim):
    f_x = lambda x_tmp : x_tmp.Separation
    for d in list_v:
        ProcessingUtil.plot_single_fec(d, f_x, xlim, ylim,
                                       style_data=dict(color=None, alpha=0.3,
                                                       linewidth=0.3))

def _ensemble_alignment(gs,alignment,col_idx):
    xlim, ylim = _limits(alignment)
    common_kw = dict(xlim=xlim,ylim=ylim)
    kw_fmt = dict(color=False,is_left=(col_idx ==0),**common_kw)
    ax1 = plt.subplot(gs[0,col_idx])
    _plot_fec_list(alignment.zeroed.fec_list,**common_kw)
    _plot_fmt(ax1,**kw_fmt)
    ax2 = plt.subplot(gs[1,col_idx])
    _plot_fec_list(alignment.polished.fec_list,**common_kw)
    _plot_fmt(ax2,**kw_fmt)
    PlotUtilities.no_x_label(ax=ax2)
    ax3 = plt.subplot(gs[2,col_idx])
    _plot_fec_list(alignment.blacklisted.fec_list,**common_kw)
    _plot_fmt(ax3,is_bottom=True,**kw_fmt)


def _heatmap_alignment(gs,alignment,col_idx):
    xlim, ylim = _limits(alignment)
    max_x = xlim[1]
    bin_step_nm = 1
    bin_step_pN = 5
    bins_x = np.arange(xlim[0],xlim[1] + bin_step_nm,step=bin_step_nm)
    bins_y = np.arange(ylim[0],ylim[1] + bin_step_pN,step=bin_step_pN)
    common_kw = dict(separation_max=max_x,use_colorbar=False,title="",
                     bins=(bins_x,bins_y))
    ax1 = plt.subplot(gs[0,col_idx])
    FEC_Plot.heat_map_fec(alignment.zeroed.fec_list,**common_kw)
    _plot_fmt(ax1,xlim,ylim)
    ax2 = plt.subplot(gs[1,col_idx])
    FEC_Plot.heat_map_fec(alignment.polished.fec_list,**common_kw)
    _plot_fmt(ax2,xlim,ylim)
    title_kw = dict(color='b',y=0.95,loc='left',fontsize=6)
    downarrow = "$\Downarrow$"
    title_sub = downarrow + " Subtract $X_{\mathbf{PEG3400}}(F)$ + " + \
                "$L_{\mathbf{0,C-term}}$"
    PlotUtilities.title(title_sub,**title_kw)
    PlotUtilities.no_x_label(ax=ax2)
    ax3 = plt.subplot(gs[2,col_idx])
    FEC_Plot.heat_map_fec(alignment.blacklisted.fec_list,**common_kw)
    _plot_fmt(ax3,xlim,ylim,True)
    PlotUtilities.title(downarrow + " Remove poorly-fit FECs",**title_kw)


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir = "../../../../Data/FECs180307/"
    base_dir_input = base_dir + "BR+Retinal/"
    in_dir = Pipeline._cache_dir(base=base_dir_input,
                                 enum=Pipeline.Step.CORRECTED)
    energies = CheckpointUtilities.lazy_load(in_dir + "energies.pkl")
    alignment = _alignment_pipeline(energies[0])
    fig = PlotUtilities.figure((3,4))
    gs = gridspec.GridSpec(3, 2)
    _heatmap_alignment(gs,alignment,0)
    _ensemble_alignment(gs, alignment,1)
    PlotUtilities.savefig(fig, "./FigureSX_Alignment.png",
                          subplots_adjust=dict(hspace=0.12,wspace=0.02))


if __name__ == "__main__":
    run()
