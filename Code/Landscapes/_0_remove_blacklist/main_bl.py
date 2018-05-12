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


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_input_processing = RetinalUtil._processing_base()
    base_dir = RetinalUtil._landscape_base()
    step = Pipeline.Step.MANUAL
    in_dir = Pipeline._cache_dir(base=base_input_processing,
                                 enum=Pipeline.Step.POLISH)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    data_input = CheckpointUtilities.lazy_multi_load(in_dir)
    force = True
    functor = lambda : ProcessingUtil.\
        _filter_by_bl(data_input,base_input_processing)
    data = CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                          force=force,
                                          name_func=FEC_Util.fec_name_func)
    # plot each individual
    ProcessingUtil.plot_data(base_dir,step,data,xlim=[-50,150])
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    out_name = plot_subdir + "heatmap.png"
    ProcessingUtil.heatmap_ensemble_plot(data, out_name=out_name)


if __name__ == "__main__":
    run()
