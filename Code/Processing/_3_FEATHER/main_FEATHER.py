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
import RetinalUtil, PlotUtil

import warnings
from Lib.AppFEATHER.Code import Detector, Analysis

from multiprocessing import Pool
import multiprocessing

def _align_and_cache(d,out_dir,force=False,**kw):
    return ProcessingUtil._cache_individual(d, out_dir,
                                            RetinalUtil.feather_single,
                                            force,d, **kw)

def func(args):
    x, out_dir, kw = args
    to_ret = _align_and_cache(x,out_dir,**kw)
    return to_ret


def align_data(base_dir,out_dir,n_pool,**kw):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    input_v = [ [d,out_dir,kw] for d in all_data]
    to_ret = ProcessingUtil._multiproc(func, input_v, n_pool)
    to_ret = [r for r in to_ret if r is not None]
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
    step = Pipeline.Step.SANITIZED
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.CORRECTED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    max_n_pool = 6
    n_pool = max_n_pool
    kw_feather = RetinalUtil._def_kw_FEATHER()
    data = align_data(in_dir,out_dir,force=force,n_pool=n_pool,
                      **kw_feather)
    # plot all of the FEATHER information
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    PlotUtil._feather_plot(data,plot_subdir)



if __name__ == "__main__":
    run()
