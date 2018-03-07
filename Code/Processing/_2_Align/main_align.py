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

def align_single(d,min_wlc_force_fit_N):
    force_N = d.Force * 1e-12
    where_GF = np.where(force_N >= min_wlc_force_fit_N)[0]
    where_above_surface = np.where(force_N >= 0)[0]
    assert where_GF.size * where_above_surface.size > 0, "Force never above limit "
    last_time_GF = where_GF[-1]
    last_time_zero = where_above_surface[0]
    plt.close()
    plt.plot(d.Time, d.Force)
    plt.axvline(d.Time[last_time_GF])
    plt.axvline(d.Time[last_time_zero])
    plt.show()

def align_data(base_dir,**kw):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    for d in all_data:
        # filter the data
        to_ret = align_single(d,**kw)
        pass

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
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.FILTERED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=Pipeline.Step.ALIGNED)
    force = False
    limit = None
    min_wlc_force_fit_N = 200e-12
    functor = lambda : align_data(in_dir,
                                  min_wlc_force_fit_N=min_wlc_force_fit_N)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)


if __name__ == "__main__":
    run()
