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
import RetinalUtil

import warnings
from Lib.AppFEATHER.Code import Detector, Analysis

from multiprocessing import Pool
import multiprocessing

def _debug_plot(to_ret):
    plt.close()
    inf = to_ret.L0_info
    fit_slice = inf.fit_slice
    x, f = to_ret.Separation, to_ret.Force
    # get a grid over all possible forces
    f_grid = np.linspace(min(f), max(f), num=f.size, endpoint=True)
    # get the extension components
    ext_total, ext_components = WLCHao._hao_shift(f_grid, *inf.x0,**inf.kw_fit)
    # determine the
    plt.plot(x, f, color='k', alpha=0.3)
    plt.plot(x[fit_slice], f[fit_slice], 'r')
    plt.plot(ext_total, f_grid, 'b--')
    for c in ext_components:
        plt.plot(c,f_grid)
    plt.xlim([min(x),150e-9])
    plt.axvline(inf.x0[-1])
    plt.show()



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
    step = Pipeline.Step.ALIGNED
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.SANITIZED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    max_n_pool = multiprocessing.cpu_count() - 1
    n_pool = max_n_pool
    N_fit_pts = 20
    min_F_N = 175e-12 if "+Retinal" in base_dir else 90e-12
    data = RetinalUtil.align_data(in_dir,out_dir,force=force,n_pool=n_pool,
                                  min_F_N=min_F_N,N_fit_pts=N_fit_pts)
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    xlim_heatmap_nm = [-20,100]
    ProcessingUtil.heatmap_ensemble_plot(data,xlim=xlim_heatmap_nm,
                                         out_name=plot_subdir + "heatmap.png")
    # get the post-blacklist heapmap, too..
    data_filtered = ProcessingUtil._filter_by_bl(data, in_dir)
    # align the data...
    data_aligned = [RetinalUtil._polish_helper(d) for d in data_filtered]
    out_name = plot_subdir + "heatmap_bl.png"
    ProcessingUtil.heatmap_ensemble_plot(data_aligned, xlim=xlim_heatmap_nm,
                                         out_name=out_name)
    # make individual plots
    ProcessingUtil.make_aligned_plot(base_dir,step,data,
                                     xlim=[-30,150],use_shift=True)



if __name__ == "__main__":
    run()
