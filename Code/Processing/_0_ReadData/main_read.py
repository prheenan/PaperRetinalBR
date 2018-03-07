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

def is_valid_file(f):
    """
    :param f: file name
    :return: true if the file is valid
    """
    return "smth" not in f


def hao_grouping_function(str_v):
    pattern = r"""
               (ext|force)     # type (eg: ext,force)
               \D*?            # non-greedy non digits (e.g. "extsmooth")
               (\d+)           # id (e.g. 1131)
               (\D*)           # anything else, who cares
               """
    match = re.match(pattern, str_v, re.VERBOSE)
    assert match is not None, "Whoops! Got a bad string: {:s}".format(str_v)
    ending, id, preamble = match.groups()
    # convert ext to sep
    ending = ending if ending != "ext" else "sep"
    return preamble, id, ending

def read_all_data(base_dir,limit=None):
    data= FEC_Util.read_ibw_directory(base_dir,limit=limit,
                                      f_file_name_valid=is_valid_file,
                                      grouping_function=hao_grouping_function)
    for d in data:
        yield d

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
    step = Pipeline.Step.READ
    cache_dir = Pipeline._cache_dir(base=base_dir, enum=step)
    force = False
    limit = None
    functor = lambda : read_all_data(base_dir)
    data =CheckpointUtilities.multi_load(cache_dir=cache_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    plot_subdir = Pipeline._plot_subdir(base_dir,step)

    markevery = 100
    name_func = FEC_Util.fec_name_func
    x_range = [ [min(d.Separation),max(d.Separation)] for d in data]
    y_range = [ [min(d.Force),max(d.Force)] for d in data]
    xlim = [np.min(x_range),np.max(x_range)]
    ylim = [np.min(y_range),np.max(y_range)]
    for d in data:
        f = PlotUtilities.figure()
        FEC_Plot._fec_base_plot(d.Separation[::markevery], d.Force[::markevery])
        plt.xlim(xlim)
        plt.ylim(ylim)
        PlotUtilities.lazyLabel("Extension (nm)","Force (pN)","")
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + ".png")


if __name__ == "__main__":
    run()
