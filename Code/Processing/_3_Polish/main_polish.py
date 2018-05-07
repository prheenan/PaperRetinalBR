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
    data_unpolished = CheckpointUtilities.lazy_multi_load(in_dir)
    ProcessingUtil.heatmap_ensemble_plot(data,out_name=plot_dir + "heatmap.png")
    # plot each individual
    f_x = lambda x_tmp : x_tmp.Separation
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    name_func = FEC_Util.fec_name_func
    xlim, ylim = ProcessingUtil.nm_and_pN_limits(data,f_x)
    xlim = [-10,150]
    for d_unpolish,d_polish in zip(data_unpolished,data):
        fig = PlotUtilities.figure()
        ax1 = plt.subplot(2,1,1)
        ProcessingUtil._aligned_plot(d_unpolish,f_x,xlim,ylim)
        PlotUtilities.xlabel("")
        PlotUtilities.no_x_label(ax1)
        plt.subplot(2,1,2)
        ProcessingUtil._aligned_plot(d_polish,f_x,xlim,ylim)
        name = plot_subdir + name_func(0, d_polish) + ".png"
        PlotUtilities.savefig(fig,name)


if __name__ == "__main__":
    run()
