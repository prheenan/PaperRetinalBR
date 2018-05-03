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

def _filter_single(d,t_filter,f_decimate):
    # filter the data
    delta_t = d.Time[1] - d.Time[0]
    n_filt = int(np.ceil(t_filter / delta_t))
    n_decimate = int(np.ceil(f_decimate * n_filt))
    d_filt = FEC_Util.GetFilteredForce(d, NFilterPoints=n_filt)
    # slice it
    to_ret = d_filt._slice(slice(0, None, n_decimate))
    return to_ret

def func(args):
    d, out_dir, force, t_filter, f_decimate = args
    to_ret = ProcessingUtil._cache_individual(d,out_dir,_filter_single,force,
                                              d,t_filter,f_decimate)
    return to_ret


def filter_data(base_dir,out_dir,force,t_filter,f_decimate):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    input_v = [[d, out_dir,force,t_filter, f_decimate] for d in all_data]
    to_ret = ProcessingUtil._multiproc(func, input_v,n_pool=3)
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
    step = Pipeline.Step.FILTERED
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.READ)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=Pipeline.Step.FILTERED)
    force = True
    limit = None
    f_filter_Hz = 5e3
    # filter to X s
    t_filter_s = 1/f_filter_Hz
    # t_filter -> n_filter_points
    # after filtering, take every N points, where
    # N = f_decimate * n_filter_points
    # in other words, we oversample by 1/f_decimate
    f_decimate = 0.33
    assert f_decimate < 1 and f_decimate > 0
    data = filter_data(in_dir,out_dir,force,t_filter_s,f_decimate)
    ProcessingUtil.plot_data(base_dir,step,data,markevery=1)


if __name__ == "__main__":
    run()
