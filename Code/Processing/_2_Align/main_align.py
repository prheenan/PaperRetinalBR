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
    plt.show()

def align_single(d,min_wlc_force_fit_N,max_sep_m):
    force_N = d.Force
    where_GF = np.where((force_N >= min_wlc_force_fit_N) &
                        (d.Separation <= max_sep_m))[0]
    where_above_surface = np.where(force_N >= 0)[0]
    assert where_above_surface.size > 0, "Force never above surface "
    if (where_GF.size > 0):
        last_time_GF = where_GF[-1]
    else:

        msg = "For alignment, {:} never above limit of {:}N; using max".\
            format(d.Meta.Name,min_wlc_force_fit_N)
        warnings.warn(msg,RuntimeWarning)
        last_time_GF = np.argmax(force_N)
    max_fit_idx = np.argmax(force_N[:last_time_GF])
    first_time_above_surface = where_above_surface[0]
    assert first_time_above_surface < max_fit_idx , \
        "Couldn't find fitting region"
    fit_slice = slice(first_time_above_surface,max_fit_idx,1)
    # slice the object to just the region we want
    obj_slice = d._slice(fit_slice)
    # fit wlc to the f vs x of that slice
    fit_info = WLCHao.hao_fit(obj_slice.Separation,obj_slice.Force)
    fit_info.fit_slice = fit_slice
    to_ret = ProcessingUtil.AlignedFEC(d,fit_info)
    return to_ret

def _align_and_cache(d,out_dir,force=False,**kw):
    name = out_dir + FEC_Util.fec_name_func(0,d) + ".pkl"
    data = CheckpointUtilities.getCheckpoint(name,align_single,force,
                                             d,**kw)
    return data


def func(args):
    x, out_dir, kw = args
    to_ret = _align_and_cache(x,out_dir,**kw)
    return to_ret

def align_data(base_dir,out_dir,n_pool,**kw):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    input_v = [ [d,out_dir,kw] for d in all_data]
    p = Pool(n_pool)
    if (n_pool > 1):
        to_ret = p.map(func,input_v)
    else:
        to_ret = [func(d) for d in input_v]
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
    min_wlc_force_fit_N = 200e-12
    max_sep_m = 105e-9
    data =align_data(in_dir,out_dir,max_sep_m=max_sep_m,
                     min_wlc_force_fit_N=min_wlc_force_fit_N,force=force,
                     n_pool=n_pool)
    ProcessingUtil.make_aligned_plot(base_dir,step,data)



if __name__ == "__main__":
    run()
