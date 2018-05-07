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
from Lib.AppWLC.Code import WLC
from Processing.Util import WLC as WLCHao
from Lib.AppWLC.UtilFit import fit_base

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

def offset_L(info):
    # align by the contour length of the protein
    offset_m = 20e-9
    L0 = info.L0_c_terminal - offset_m
    return L0

def _ext_grid(f_grid,x0):
    # get the extension components
    ext_total, ext_components = WLCHao._hao_ext_grid(f_grid, *x0)
    ext_FJC = ext_components[0]
    # make the extension at <= force be zero
    where_f_le = np.where(f_grid <= 0)
    ext_FJC[where_f_le] = 0
    ext_total[where_f_le] = 0
    return ext_total, ext_FJC

def _polish_single(d):
    # filter the data, making a copy
    to_ret = d._slice(slice(0, None, 1))
    # get the slice we are fitting
    inf = to_ret.L0_info
    fit_slice = inf.fit_slice
    x, f = to_ret.Separation.copy(), to_ret.Force.copy()
    # get a grid over all possible forces
    f_grid = np.linspace(min(f), max(f), num=f.size, endpoint=True)
    ext_total, ext_FJC = _ext_grid(f_grid, inf.x0)
    # we now have X_FJC as a function of force. Therefore, we can subtract
    # off the extension of the PEG3400 to determining the force-extension
    # associated with only the protein (*including* its C-term)
    # note we are getting ext_FJC(f), where f is each point in the original
    # data.
    ext_FJC_all_forces = fit_base._grid_to_data(x=f, x_grid=f_grid,
                                                y_grid=ext_FJC,
                                                bounds_error=False)
    # remove the extension associated with the PEG
    to_ret.Separation -= ext_FJC_all_forces
    L0 = offset_L(to_ret.L0_info)
    to_ret.Separation -= L0
    to_ret.ZSnsr -= L0
    # make sure the fitting object knows about the change in extensions...
    ext_total_info, ext_FJC_correct_info = _ext_grid(inf.f_grid, inf.x0)
    to_ret.L0_info.set_x_offset(L0 + ext_FJC_correct_info)
    _debug_plot(to_ret, d, ext_total, f_grid)
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
