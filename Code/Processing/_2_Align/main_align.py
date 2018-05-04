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
    ext_total, ext_components = WLCHao._hao_ext_grid(f_grid, *inf.x0)
    ext_FJC = ext_components[0]
    # determine the
    plt.plot(x, f, color='k', alpha=0.3)
    plt.plot(x[fit_slice], f[fit_slice], 'r')
    plt.plot(ext_total, f_grid, 'b--')
    plt.xlim([min(x),max(x)])
    plt.show()

def _detect_retract_FEATHER(d,pct_approach,tau_f,threshold,f_refs=None):
    """
    :param d:  TimeSepForce
    :param pct_approach: how much of the retract, starting from the end,
    to use as an effective approach curve
    :param tau_f: fraction for tau
    :param threshold: FEATHERs probability threshold
    :return:
    """
    force_N = d.Force
    # use the last x% as a fake 'approach' (just for noise)
    n = force_N.size
    n_approach = int(np.ceil(n * pct_approach))
    tau_n_points = int(np.ceil(n * tau_f))
    # slice the data for the approach, as described above
    n_approach_start = n - (n_approach + 1)
    fake_approach = d._slice(slice(n_approach_start, n, 1))
    fake_dwell = d._slice(slice(n_approach_start - 1, n_approach_start, 1))
    # make a 'custom' split fec (this is what FEATHER needs for its noise stuff)
    split_fec = Analysis.split_force_extension(fake_approach, fake_dwell, d, tau_n_points)
    # set the 'approach' number of points for filtering to the retract.
    split_fec.set_tau_num_points_approach(split_fec.tau_num_points)
    # set the predicted retract surface index to a few tau. This avoids looking at adhesion
    split_fec.get_predicted_retract_surface_index = lambda: 5 * tau_num_points
    pred_info = Detector._predict_split_fec(split_fec, threshold=threshold,
                                            f_refs=f_refs)
    return pred_info

def align_single(d,min_F_N,**kw):
    """
    :param d: FEC to get FJC+WLC fit of
    :param min_F_N: minimum force, in Newtons, for fitting event. helps avoid occasional small force events
    :param kw: keywords to use for fitting...
    :return:
    """
    force_N = d.Force
    where_above_surface = np.where(force_N >= 0)[0]
    assert where_above_surface.size > 0, "Force never above surface "
    first_time_above_surface = where_above_surface[0]
    # use FEATHER; fit to the first event, don't look for adhesion
    d_pred_only = d._slice(slice(0,None,1))
    pred_info = _detect_retract_FEATHER(d_pred_only,f_refs=[Detector.delta_mask_function],**kw)
    assert len(pred_info.event_idx) > 0 , "FEATHER can't find an event..."
    force_spline_N = pred_info.interp(d.Time)
    valid_events = [i for i in pred_info.event_idx
                    if force_spline_N[i] > min_F_N]
    assert len(valid_events) > 0 , "Couldn't find any valid events"
    # make sure the event makes sense
    max_fit_idx = valid_events[0]
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
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.FILTERED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    max_n_pool = multiprocessing.cpu_count() - 1
    n_pool = max_n_pool
    kw_feather = dict(pct_approach=0.1, tau_f=0.01, threshold=1e-3)
    data =align_data(in_dir,out_dir,force=force,n_pool=n_pool,min_F_N=90e-12,**kw_feather)
    ProcessingUtil.make_aligned_plot(base_dir,step,data)



if __name__ == "__main__":
    run()
