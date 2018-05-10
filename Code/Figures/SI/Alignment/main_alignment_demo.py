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

class LandscapeGallery(object):
    def __init__(self,PEG600,PEG3400):
        self.PEG600 = PEG600
        self.PEG3400 = PEG3400

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
    ylim = [-50,max(max_y,300)]
    return xlim,ylim

def _plot_fec_list(list_v,xlim,ylim,label=None,color=None,**kw):
    f_x = lambda x_tmp : x_tmp.Separation
    for i,d in enumerate(list_v):
        label_tmp = label if i == 0 else None
        ProcessingUtil.plot_single_fec(d, f_x, xlim, ylim,label=label_tmp,
                                       style_data=dict(color=color, alpha=0.3,
                                                       linewidth=0.3),**kw)

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
    return [ax1,ax2,ax3]

def read_landscapes(base_dir):
    """
    :param base_dir: input to RetinalUtil._read_all_energies
    :return:
    """
    energies = RetinalUtil._read_all_energies(base_dir)
    names = [e.base_dir.split("FECs180307")[1] for e in energies]
    str_PEG600_example = "/BR+Retinal/300nms/170511FEC/landscape_"
    str_PEG3400_example = "/BR+Retinal/3000nms/170503FEC/landscape_"
    idx_PEG600 = names.index(str_PEG600_example)
    idx_PEG3400 = names.index(str_PEG3400_example)
    to_ret = LandscapeGallery(PEG600=energies[idx_PEG600],
                              PEG3400=energies[idx_PEG3400])
    return to_ret


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
    gallery = CheckpointUtilities.getCheckpoint("./caches.pkl",read_landscapes,
                                                True,base_dir_input)
    galleries_labels = [ [gallery.PEG600,"PEG600"],
                         [gallery.PEG3400, "PEG3400"]]
    alignments =  [_alignment_pipeline(gallery_tmp[0])
                   for gallery_tmp in galleries_labels]
    for i,(_,label) in enumerate(galleries_labels):
        alignment = alignments[i]
        fig = PlotUtilities.figure((3,4))
        gs = gridspec.GridSpec(3, 2)
        # make the 'standard' alignment plots
        axes = _heatmap_alignment(gs,alignment,0)
        _ensemble_alignment(gs, alignment,1)
        out_name = "./FigureS_Alignment_{:s}.png".format(label)
        PlotUtilities.title(label,ax=axes[0])
        PlotUtilities.savefig(fig,out_name,
                              subplots_adjust=dict(hspace=0.12,wspace=0.02))
    # plot the final curves on the same plot
    xlim,ylim = _limits(alignment)
    colors = ['r','g']
    fig = PlotUtilities.figure()
    for i,(_,l) in enumerate(galleries_labels[::-1]):
        a = alignments[i]
        _plot_fec_list(a.blacklisted.fec_list, xlim, ylim,label=l,color=colors[i])
    PlotUtilities.savefig(fig,"FigureS_3400vs600.png")



if __name__ == "__main__":
    run()
