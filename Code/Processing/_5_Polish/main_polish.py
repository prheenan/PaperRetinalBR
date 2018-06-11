# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
import re

sys.path.append("../../")
from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Processing import ProcessingUtil
import RetinalUtil


def _debug_plot(to_ret,d,ext_total,f_grid):
    plt.close()
    xlim = [min(to_ret.Separation) - 10e-9,max(d.Separation)]
    ylim = [-30e-12,max(d.Force)]
    plt.subplot(2,1,1)
    plt.plot(d.Separation,d.Force)
    plt.plot(ext_total,f_grid,'r--')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplot(2,1,2)
    plt.plot(to_ret.Separation,to_ret.Force)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def _polish_single(d):
    # filter the data, making a copy
    to_ret = RetinalUtil._polish_helper(d)
    return to_ret

def polish_data(base_dir):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    for d in all_data:
        to_ret = _polish_single(d)
        yield to_ret

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    default_base = "../../../Data/170321FEC/"
    base_dir = Pipeline._base_dir_from_cmd(default=default_base)
    step = Pipeline.Step.POLISH
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.ALIGNED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    plot_dir = Pipeline._plot_subdir(base=base_dir, enum=step)
    force = True
    limit = None
    functor = lambda : polish_data(in_dir)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    from Lib.AppWHAM.Code import UtilWHAM, WeightedHistogram
    sizes = [d.Force.size for d in data]
    min_s = min(sizes)
    sliced_data = [d._slice(slice(0,min_s,1)) for d in data]
    for d in sliced_data:
        d.Offset = d.ZSnsr[0]
        d.Extension = d.Separation
        d.kT = 4.1e-21
    data_wham = UtilWHAM.to_wham_input(objs=sliced_data, n_ext_bins=200)
    obj_wham = WeightedHistogram.wham(data_wham)
    data_unpolished = CheckpointUtilities.lazy_multi_load(in_dir)
    f_x_zsnsr = lambda x: x.ZSnsr
    ProcessingUtil.heatmap_ensemble_plot(data,out_name=plot_dir + "heatmap.png")
    ProcessingUtil.heatmap_ensemble_plot(data,f_x=f_x_zsnsr,
                                         out_name=plot_dir + "heatmap_Z.png",
                                         kw_map=dict(x_func=f_x_zsnsr))
    # plot each individual
    f_x = lambda x_tmp : x_tmp.Separation
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    name_func = FEC_Util.fec_name_func
    xlim, ylim = ProcessingUtil.nm_and_pN_limits(data,f_x)
    xlim = [-20,100]
    for d_unpolish,d_polish in zip(data_unpolished,data):
        fig = PlotUtilities.figure((6,6))
        # make the Separation column
        ax1,ax2 = plt.subplot(2,2,1), plt.subplot(2,2,3)
        polish_plot(ax1, ax2, d_unpolish, d_polish, xlim, ylim,
                    f_x = lambda x: x.Separation,plot_components_1=True)
        # make the ZSnsr column
        ax3,ax4 = plt.subplot(2,2,2), plt.subplot(2,2,4)
        polish_plot(ax3, ax4, d_unpolish, d_polish, xlim, ylim,
                    f_x = lambda x: x.ZSnsr,plot_components_1=False)
        PlotUtilities.xlabel("Stage Position (nm)", ax=ax4)
        for a in [ax3,ax4]:
            PlotUtilities.no_y_label(ax=a)
            PlotUtilities.ylabel("",ax=a)
        name = plot_subdir + name_func(0, d_polish) + ".png"
        PlotUtilities.savefig(fig,name)

def polish_plot(ax1,ax2,d_unpolish,d_polish,xlim,ylim,f_x,plot_components_1):
    plt.sca(ax1)
    ProcessingUtil._aligned_plot(d_unpolish, f_x, xlim, ylim, use_shift=True,
                                 plot_components=plot_components_1)
    PlotUtilities.xlabel("")
    PlotUtilities.no_x_label(ax1)
    plt.sca(ax2)
    ProcessingUtil._aligned_plot(d_polish, f_x, xlim, ylim, use_shift=True,
                                 plot_components=False)


if __name__ == "__main__":
    run()
