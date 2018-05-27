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

def GF2_event_idx(d,min_F_N):
    pred_info = d.info_feather
    tau_n = pred_info.tau_n
    # POST: FEATHER found something; we need to screen for lower-force events..
    event_idx = [i for i in pred_info.event_idx]
    event_slices = [slice(i - tau_n * 2, i, 1) for i in event_idx]
    # determine the coefficients of the fit
    t, f = d.Time, d.Force
    # loading rate helper has return like:
    # fit_x, fit_y, pred, _, _, _
    list_v = [Detector._loading_rate_helper(t, f, e)
              for e in event_slices]
    # get the predicted force (rupture force), which is the last element of the
    # predicted force.
    pred = [e[2] if len(e[0]) > 0 else [0] for e in list_v]
    f_at_idx = [p[-1] for p in pred]
    valid_events = [i for i, f in zip(event_idx, f_at_idx) if f > min_F_N]
    if (len(valid_events) == 0):
        warnings.warn("Couldn't find high-force events for {:s}". \
                      format(d.Meta.Name))
        # just take the maximum
        valid_events = [event_idx[np.argmax(f_at_idx)]]
    # make sure the event makes sense
    max_fit_idx = valid_events[0]
    return max_fit_idx

def align_single(d,min_F_N):
    """
    :param d: FEC to get FJC+WLC fit of
    :param min_F_N: minimum force, in Newtons, for fitting event. helps avoid
     occasional small force events
    :param kw: keywords to use for fitting...
    :return:
    """
    force_N = d.Force
    pred_info = d.info_feather
    max_fit_idx = GF2_event_idx(d,min_F_N)
    where_above_surface = np.where(force_N >= 0)[0]
    first_time_above_surface = where_above_surface[0]
    assert first_time_above_surface < max_fit_idx , \
        "Couldn't find fitting region"
    # start the fit after any potential adhesions
    fit_start = max(first_time_above_surface,pred_info.slice_fit.start)
    fit_slice = slice(fit_start,max_fit_idx,1)
    # slice the object to just the region we want
    obj_slice = d._slice(fit_slice)
    # fit wlc to the f vs x of that slice
    info_fit = WLCHao.hao_fit(obj_slice.Separation,obj_slice.Force)
    info_fit.fit_slice = fit_slice
    offset = info_fit._L_shift - info_fit._Ns * WLCHao._L_planar()
    d.Separation += offset
    d.ZSnsr += offset
    to_ret = ProcessingUtil.AlignedFEC(d,info_fit,feather_info=pred_info)
    return to_ret

def _align_and_cache(d,out_dir,force=False,**kw):
    return ProcessingUtil._cache_individual(d, out_dir, align_single,
                                            force,d, **kw)

def func(args):
    x, out_dir, kw = args
    to_ret = _align_and_cache(x,out_dir,**kw)
    return to_ret


def align_data(base_dir,out_dir,n_pool,**kw):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    input_v = [ [d,out_dir,kw] for d in all_data]
    to_ret = ProcessingUtil._multiproc(func, input_v, n_pool)
    to_ret = [r for r in to_ret if r is not None]
    return to_ret

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
    min_F_N = 175e-12 if "+Retinal" in base_dir else 90e-12
    data = align_data(in_dir,out_dir,force=force,n_pool=n_pool,min_F_N=min_F_N)
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    xlim_heatmap_nm = [0,100]
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
