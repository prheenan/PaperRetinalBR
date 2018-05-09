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


def align_single(d,**kw):
    """
    :param d: FEC to get FJC+WLC fit of
    :param min_F_N: minimum force, in Newtons, for fitting event. helps avoid
     occasional small force events
    :param kw: keywords to use for fitting...
    :return:
    """
    force_N = d.Force
    where_above_surface = np.where(force_N >= 0)[0]
    assert where_above_surface.size > 0, "Force never above surface "
    # use FEATHER; fit to the first event, don't look for adhesion
    d_pred_only = d._slice(slice(0,None,1))
    # first, try removing surface adhesions
    feather_kw =  dict(d=d_pred_only,**kw)
    pred_info,tau_n = RetinalUtil._detect_retract_FEATHER(**feather_kw)
    # if we removed more than 20nm or we didnt find any events, then
    # FEATHER got confused by a near-surface BR. Tell it not to look for
    # surface adhesions
    expected_surface_m = d.Separation[pred_info.slice_fit.start]
    expected_gf_m = 20e-9
    if ((len(pred_info.event_idx) == 0) or (expected_surface_m > expected_gf_m)):
        f_refs = [Detector.delta_mask_function]
        pred_info,tau_n = RetinalUtil._detect_retract_FEATHER(f_refs=f_refs,
                                                              **feather_kw)
    pred_info.tau_n = tau_n
    assert len(pred_info.event_idx) > 0 , "FEATHER can't find an event..."
    to_ret = ProcessingUtil.AlignedFEC(d,info_fit=None,feather_info=pred_info)
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
    step = Pipeline.Step.SANITIZED
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.CORRECTED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    max_n_pool = multiprocessing.cpu_count() - 1
    n_pool = max_n_pool
    kw_feather = dict(pct_approach=0.3, tau_f=0.01, threshold=1e-3)
    data = align_data(in_dir,out_dir,force=force,n_pool=n_pool,
                      **kw_feather)
    # plot all of the FEATHER information
    f_x = lambda x: x.Separation
    _, ylim = ProcessingUtil.nm_and_pN_limits(data, f_x=f_x)
    xlim = [-20,150]
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    for d in data:
        fig = PlotUtilities.figure()
        ProcessingUtil.plot_single_fec(d, f_x, xlim, ylim, markevery=1)
        x = f_x(d) * 1e9
        for i in d.info_feather.event_idx:
            plt.axvline(x[i])
        name = FEC_Util.fec_name_func(0,d)
        PlotUtilities.savefig(fig,plot_subdir + name + ".png")



if __name__ == "__main__":
    run()
