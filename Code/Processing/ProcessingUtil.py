# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse, enum

from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util,  FEC_Plot
from Lib.UtilForce.UtilGeneral import PlotUtilities


def plot_data(base_dir,step,data,markevery=1):
    """
    :param base_dir: where the data live
    :param step:  what step we are on
    :param data: the actual data; list of TimeSepForce
    :param markevery: how often to mark the data (useful for lowering high
    res to resonable size)
    :return: nothing, plots the data..
    """
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    name_func = FEC_Util.fec_name_func
    x_range = [[min(d.Separation), max(d.Separation)] for d in data]
    y_range = [[min(d.Force), max(d.Force)] for d in data]
    xlim = 1e9 * np.array([np.min(x_range), np.max(x_range)])
    ylim = 1e12 * np.array([np.min(y_range), np.max(y_range)])
    for d in data:
        f = PlotUtilities.figure()
        FEC_Plot._fec_base_plot(d.Separation[::markevery]*1e9,
                                d.Force[::markevery]* 1e12)
        plt.xlim(xlim)
        plt.ylim(ylim)
        PlotUtilities.lazyLabel("Extension (nm)", "Force (pN)", "")
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + ".png")