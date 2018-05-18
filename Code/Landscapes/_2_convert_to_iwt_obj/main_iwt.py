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
from Lib.AppIWT.Code import WeierstrassUtil
import RetinalUtil
import warnings
from collections import Counter


def to_iwt(in_dir):
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    # make sure they all have the same velocity
    velocities = [d.Velocity for d in data]
    # make sure the velocities match within X%
    np.testing.assert_allclose(velocities,velocities[0],atol=0,rtol=1e-2)
    # just set them all equal now
    v_mean = np.mean(velocities)
    for d in data:
        d.Velocity = v_mean
    # repeat for the spring constant
    spring_constants = [d.SpringConstant for d in data]
    K_key = spring_constants[0]
    K_diff = np.max(np.abs(np.array(spring_constants)-K_key))/\
             np.mean(spring_constants)
    if (K_diff > 1e-2):
        msg ="For {:s}, not all spring constants ({:s}) the same. Replace <K>".\
            format(in_dir,spring_constants)
        warnings.warn(msg)
        # average all the time each K appears
        weighted_mean = np.mean(spring_constants)
        for d in data:
            d.Meta.SpringConstant = weighted_mean
    # get the minimum of the sizes
    np.testing.assert_allclose(data[0].SpringConstant,
                               [d.SpringConstant for d in data],
                               rtol=1e-3)
    max_sizes = [d.Force.size for d in data]
    min_of_max_sizes = min(max_sizes)
    # re-slice each data set so they are exactly the same size (as IWT needs)
    data = [d._slice(slice(0,min_of_max_sizes,1)) for d in data]
    # determine the slices we want for finding the EF helix.
    ex = data[0]
    min_ext_m = 25e-9
    min_idx = [np.where(d.Separation > min_ext_m)[0][0] for d in data]
    max_sizes = [d.Separation.size - (i+1) for i,d  in zip(min_idx,data)]
    max_delta = int(min(max_sizes))
    for i,d in enumerate(data):
        # find where we should start
        converted = RetinalUtil.MetaPulling(d)
        slice_v = [slice(min_idx[i],min_idx[i]+max_delta,1)]
        converted._set_iwt_slices(slice_v)
        assert converted._slice(slice_v[0]).Force.size == max_delta
        yield converted


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir = RetinalUtil._landscape_base()
    step = Pipeline.Step.REDUCED
    in_dir = Pipeline._cache_dir(base=base_dir,
                                 enum=Pipeline.Step.SANITIZED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    limit = None
    functor = lambda : to_iwt(in_dir)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    ProcessingUtil.plot_data(base_dir,step,data,xlim=[-50,150])
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    out_name = plot_subdir + "heatmap.png"
    ProcessingUtil.heatmap_ensemble_plot(data, out_name=out_name)

if __name__ == "__main__":
    run()
