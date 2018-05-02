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

def plot_data(base_dir,step,data,markevery=1,f_x = lambda x: x.Separation,
              xlim_override=None):
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
    if (xlim_override is not None):
        xlim = xlim_override
    for d in data:
        f = PlotUtilities.figure()
        plot_single_fec(d, f_x, xlim, ylim,markevery=markevery)
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + ".png")

def _aligned_plot(d,f_x,xlim,ylim):
    # get the fit
    # convert to reasonable units for plotting
    # get the fit
    info = d.L0_info
    f_grid = info.f_grid
    # convert to reasonable units for plotting
    f_plot_pred = f_grid * 1e12
    x_plot_pred = (info.ext_grid - info.x_offset)* 1e9
    # convert back to the grid to get rid of the offset

    plt.plot(x_plot_pred, f_plot_pred, color='r', linewidth=1.5)
    # plot the fit
    plot_single_fec(d, f_x, xlim, ylim)

def make_aligned_plot(base_dir,step,data):
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    f_x = lambda x: x.Separation
    xlim, ylim = nm_and_pN_limits(data,f_x)
    xlim = [xlim[0],200]
    name_func = FEC_Util.fec_name_func
    for d in data:
        f = PlotUtilities.figure()
        _aligned_plot(d, f_x, xlim, ylim)
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + ".png")


def heatmap_ensemble_plot(data,out_name,xlim=[-50, 150]):
    """
    makes a heatmap of the ensemble, with the actual data beneath

    :param data: list of FECs
    :param out_name: what to save this as
    :return: na
    """
    fig = PlotUtilities.figure(figsize=(3, 5))
    ax = plt.subplot(2, 1, 1)
    FEC_Plot.heat_map_fec(data, num_bins=(200, 100),
                          use_colorbar=False,
                          separation_max=xlim[1])
    for spine_name in ["bottom", "top"]:
        PlotUtilities.color_axis_ticks(color='w', spine_name=spine_name,
                                       axis_name="x", ax=ax)
    PlotUtilities.xlabel("")
    PlotUtilities.title("")
    PlotUtilities.no_x_label(ax)
    plt.xlim(xlim)
    plt.subplot(2, 1, 2)
    for d in data:
        x, f = d.Separation * 1e9, d.Force * 1e12
        FEC_Plot._fec_base_plot(x, f, style_data=dict(color=None, alpha=0.3,
                                                      linewidth=0.5))
    PlotUtilities.lazyLabel("Extension (nm)", "Force (pN)", "")
    plt.xlim(xlim)
    PlotUtilities.savefig(fig, out_name)
