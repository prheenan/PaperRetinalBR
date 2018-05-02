# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse, enum, copy

from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util,  FEC_Plot
from Lib.UtilForce.UtilIgor.TimeSepForceObj import TimeSepForceObj
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Processing.Util import WLC as WLCHao


class ContourInformation(object):
    def __init__(self,L0,brute_dict,kw_wlc,fit_slice):
        self.L0 = L0
        self.brute_dict = brute_dict
        self.kw_wlc = kw_wlc
        self.fit_slice = fit_slice


class AlignedFEC(TimeSepForceObj):
    def __init__(self,normal_fec,L0_info):
        super(AlignedFEC,self).__init__()
        self.LowResData = copy.deepcopy(normal_fec.LowResData)
        self.L0_info = L0_info

def nm_and_pN_limits(data,f_x):
    x_range = [[min(f_x(d)), max(f_x(d))] for d in data]
    y_range = [[min(d.Force), max(d.Force)] for d in data]
    xlim = 1e9 * np.array([np.min(x_range), np.max(x_range)])
    ylim = 1e12 * np.array([np.min(y_range), np.max(y_range)])
    return xlim,ylim

def plot_single_fec(d,f_x,xlim,ylim,markevery=1):
    FEC_Plot._fec_base_plot(f_x(d)[::markevery] * 1e9,
                            d.Force[::markevery] * 1e12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    PlotUtilities.lazyLabel("Extension (nm)", "Force (pN)", "")

def plot_data(base_dir,step,data,markevery=1,f_x = lambda x: x.Separation):
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
    xlim, ylim = nm_and_pN_limits(data,f_x)
    for d in data:
        f = PlotUtilities.figure()
        plot_single_fec(d, f_x, xlim, ylim,markevery=markevery)
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + ".png")

def make_aligned_plot(base_dir,step,data):
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    f_x = lambda x: x.Separation
    xlim, ylim = nm_and_pN_limits(data,f_x)
    xlim = [xlim[0],200]
    name_func = FEC_Util.fec_name_func
    for d in data:
        f = PlotUtilities.figure()
        # get the fit
        info = d.L0_info
        f_grid = info.f_grid
        ext_grid = info.ext_grid
        x = d.Separation
        f_pred = WLCHao.predicted_f_at_x(x, ext_grid, f_grid)
        # convert to reasonable units for plotting
        f_plot_pred = f_pred * 1e12
        x_plot_pred = (f_x(d))*1e9
        plt.plot(x_plot_pred,f_plot_pred,color='r',linewidth=1.5)
        # plot the fit
        plot_single_fec(d, f_x, xlim, ylim)
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + ".png")

