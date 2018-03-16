# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import re

sys.path.append("../../")
from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Processing import ProcessingUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec

def subdirs(base_dir_analysis):
    raw_dirs = [base_dir_analysis + d for d in os.listdir(base_dir_analysis)]
    filtered_dirs = [r + "/" for r in raw_dirs if os.path.isdir(r)
                     and "cache" not in r]
    return filtered_dirs

def read_in_energy(base_dir):
    """
    :param base_dir: where the landscape lives; should be a series of FECs of
    about the same spring constant (e.g.  /BR+Retinal/300/170321FEC/)
    :return: RetinalUtil.EnergyWithMeta
    """
    landscape_base = RetinalUtil._landscape_dir(base_dir)
    cache_tmp = \
        Pipeline._cache_dir(base=landscape_base,
                            enum=Pipeline.Step.POLISH)
    file_load = cache_tmp + "energy.pkl"
    energy_obj = CheckpointUtilities.lazy_load(file_load)
    obj = RetinalUtil.EnergyWithMeta(file_load,
                                     landscape_base, energy_obj)
    return obj

def get_energy_list(base_dir_analysis):
    """
    :param base_dir_analysis: where we should look (e.g. BR+Retinal)
    :return: list of RetinalUtil.EnergyWithMeta objects
    """
    filtered_dirs = subdirs(base_dir_analysis)
    to_ret = []
    for velocity_directory in filtered_dirs:
        fecs = subdirs(velocity_directory)
        for d in fecs:
            to_ret.append(read_in_energy(base_dir=d))
    return to_ret

def fix_axes(ax_list):
   # loop through all the axes and toss them.
   for axs in ax_list[1:]:
       for j, ax in enumerate(axs):
           PlotUtilities.no_y_label(ax)
           PlotUtilities.ylabel("", ax=ax)
           """
           if (j != 0):
               PlotUtilities.no_x_label(ax)
               PlotUtilities.xlabel("", ax=ax)
           """

def data_plot(fecs,energies):
    n_cols = len(fecs)
    n_rows = 3
    all_ax = []
    gs = gridspec.GridSpec(4,3)
    for i, (data, e) in enumerate(zip(fecs, energies)):
        axs_tmp = [plt.subplot(gs[j,i])
                   for j in range(n_rows)]
        ax1, ax2, ax3 = axs_tmp
        PlotUtil.plot_landscapes(data, e, ax1=ax1, ax2=ax2, ax3=ax3)
        # every axis after the first gets more of the decoaration chopped...
        all_ax.append(axs_tmp)
    fix_axes(all_ax)
    q_interp, splines =  RetinalUtil.interpolating_G0(energies)
    # get an average/stdev of energy
    plot_mean_landscape(q_interp, splines,ax=gs[3,:])

def plot_mean_landscape(q_interp,splines,ax=None):
    values = [s(q_interp) for s in splines]
    mean_energy = np.mean(values, axis=0)
    std_energy = np.std(values, axis=0)
    ax = plt.subplot(1,1,1) if (ax is None) else ax
    plt.plot(q_interp, mean_energy, color='c')
    plt.fill_between(q_interp, mean_energy - std_energy,
                     mean_energy + std_energy,
                     color='c', alpha=0.2)
    PlotUtilities.lazyLabel("q (nm)", "$\Delta G_0$ (kcal/mol)", "")

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir_analysis = RetinalUtil._analysis_base()
    out_dir = Pipeline._cache_dir(base=base_dir_analysis,
                                  enum=Pipeline.Step.CORRECTED)
    force = True
    GenUtilities.ensureDirExists(out_dir)
    energy_list = CheckpointUtilities.getCheckpoint(out_dir + \
                                                    "energies.pkl",
                                                    get_energy_list,force,
                                                    base_dir_analysis)
    fecs = []
    energies = []
    N = len(energy_list)
    for e in energy_list:
        base_tmp = e.base_dir
        in_dir = Pipeline._cache_dir(base=base_tmp,
                                     enum=Pipeline.Step.REDUCED)
        data = CheckpointUtilities.lazy_multi_load(in_dir)
        fecs.append(data)
        energies.append(e)
    n_cols = N
    fig = PlotUtilities.figure(((n_cols * 1.5),3.5))
    data_plot(fecs, energies)
    PlotUtilities.savefig(fig,out_dir + "energies.png")
    # interpolate all the energies to the same grid
    q_interp, splines =  RetinalUtil.interpolating_G0(energy_list)
    # get an average/stdev of energy
    fig = PlotUtilities.figure()
    plot_mean_landscape(q_interp, splines)
    PlotUtilities.savefig(fig,out_dir + "avg.png")




if __name__ == "__main__":
    run()
