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

def filter_data(base_dir,n_filter_points,n_decimate):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    for d in all_data:
        # filter the data
        d_filt = FEC_Util.GetFilteredForce(d,NFilterPoints=n_filter_points)
        # slice it
        to_ret = d_filt._slice(slice(0,None,n_decimate))
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
    step = Pipeline.Step.FILTERED
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.READ)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=Pipeline.Step.FILTERED)
    force = False
    limit = None
    n_filter_points = 100
    n_decimate = 30
    functor = lambda : filter_data(in_dir,n_filter_points,n_decimate)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    ProcessingUtil.plot_data(base_dir,step,data,markevery=1)


if __name__ == "__main__":
    run()
