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
import RetinalUtil

def slice_data(in_dir,min_sep=40e-9,max_sep=140e-9):
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    for d in data:
        # find where we should start
        sep = d.Separation
        # filter the separation to get a better estimate
        n_pts_filter = int(sep.size//100)
        sep_filter = FEC_Util.SavitskyFilter(sep,n_pts_filter)
        where_ge_0 = np.where(sep_filter > min_sep)[0]
        assert where_ge_0.size > 0 , "Never above zero"
        first_above_surface = where_ge_0[0]
        # find where we should end
        where_le_max =np.where(sep <= max_sep)[0]
        assert where_le_max.size > 0 , "Never in size"
        last_time_slice = where_le_max[-1]
        to_ret = d._slice(slice(first_above_surface,last_time_slice,1))
        yield to_ret


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir = RetinalUtil._landscape_base()
    step = Pipeline.Step.SANITIZED
    in_dir = Pipeline._cache_dir(base=base_dir,
                                 enum=Pipeline.Step.MANUAL)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    limit = None
    min_sep = 15e-9
    max_sep = 80e-9
    functor = lambda : slice_data(in_dir,min_sep=min_sep,max_sep=max_sep)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    plot_dir = Pipeline._plot_subdir(base=base_dir, enum=step)
    ProcessingUtil.heatmap_ensemble_plot(data,out_name=plot_dir + "heatmap.png",
                                         xlim=[-5,max_sep*1e9])
    # plot each individual
    ProcessingUtil.plot_data(base_dir,step,data)



if __name__ == "__main__":
    run()
