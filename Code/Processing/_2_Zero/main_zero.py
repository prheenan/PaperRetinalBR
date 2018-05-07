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

def _debug_zero(d,x_spline_t,f_spline_t,zero_x):
    plt.close()
    plt.plot(d.Separation, d.Force)
    plt.plot(x_spline_t, f_spline_t)
    plt.axvline(zero_x)
    plt.show()

def _filter_single(d,n_filter):
    # filter the data
    t, f, x = d.Time, d.Force, d.Separation
    knots = t[1:-1:n_filter]
    common_spline = dict(q=t, k=3, knots=knots)
    f_spline = RetinalUtil.spline_fit(G0=f, **common_spline)
    x_spline = RetinalUtil.spline_fit(G0=x, **common_spline)
    f_spline_t = f_spline(t)
    x_spline_t = x_spline(t)
    f_zero = 0.2
    n_zero = int(np.ceil(f.size * f_zero))
    zero_f = np.median(f_spline_t[-n_zero:])
    idx_zero = np.where(f_spline_t >= 0)[0]
    zero_x = x_spline_t[idx_zero[0]]
    d.Separation -= zero_x
    d.ZSnsr -= zero_x
    d.Force -= zero_f
    return d

def func(args):
    d, out_dir, force, n_filter = args
    to_ret = ProcessingUtil._cache_individual(d,out_dir,_filter_single,force,
                                              d,n_filter)
    return to_ret


def filter_data(base_dir,out_dir,force,n_filter):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    input_v = [[d, out_dir,force, n_filter] for d in all_data]
    to_ret = ProcessingUtil._multiproc(func, input_v,n_pool=1)
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
    step = Pipeline.Step.CORRECTED
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.FILTERED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    n_filter = 10
    data = filter_data(in_dir,out_dir,force,n_filter)
    ProcessingUtil.plot_data(base_dir,step,data,markevery=1)

if __name__ == "__main__":
    run()
