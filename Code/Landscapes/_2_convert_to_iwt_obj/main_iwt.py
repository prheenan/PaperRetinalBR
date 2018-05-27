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
    for i in RetinalUtil._convert_to_iwt(data, in_dir):
        yield i


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
