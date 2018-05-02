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

def align_single(d,min_wlc_force_fit_N,max_sep_m,kw_wlc,brute_dict):
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


def align_data(base_dir,**kw):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    for d in all_data:
        # filter the data, making a copy
        to_ret = d._slice(slice(0, None, 1))
        to_ret = align_single(to_ret,**kw)
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
    step = Pipeline.Step.ALIGNED
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.FILTERED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    limit = None
    min_wlc_force_fit_N = 200e-12
    max_sep_m = 105e-9
    brute_dict = dict(Ns=100,ranges=((10e-9,100e-9),))
    kw_wlc = dict(kbT=4.1e-21, Lp=0.3e-9, K0=1000e-12)
    functor = lambda : align_data(in_dir,
                                  max_sep_m=max_sep_m,
                                  min_wlc_force_fit_N=min_wlc_force_fit_N,
                                  kw_wlc=kw_wlc,brute_dict=brute_dict)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    ProcessingUtil.make_aligned_plot(base_dir,step,data)



if __name__ == "__main__":
    run()
