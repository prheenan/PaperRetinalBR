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

def subdirs(base_dir_analysis):
    raw_dirs = [base_dir_analysis + d for d in os.listdir(base_dir_analysis)]
    filtered_dirs = [r + "/" for r in raw_dirs if os.path.isdir(r)
                     and "cache" not in r]
    return filtered_dirs



def get_energy_list(base_dir_analysis):
    filtered_dirs = subdirs(base_dir_analysis)
    energies, files, bases = [], [], []
    for velocity_directory in filtered_dirs:
        fecs = subdirs(velocity_directory)
        for d in fecs:
            landscape_base = RetinalUtil._landscape_dir(d)
            cache_tmp = \
                Pipeline._cache_dir(base=landscape_base,
                                    enum=Pipeline.Step.POLISH)
            file_load = cache_tmp + "energy.pkl"
            energy_obj = CheckpointUtilities.lazy_load(file_load)
            bases.append(landscape_base)
            files.append(file_load)
            energies.append(energy_obj)
    return RetinalUtil.EnergyList(files,base_dirs=bases,energies=energies)

def fix_axes(ax_list):
   # loop through all the axes and toss them.
   for axs in ax_list[1:]:
       for j, ax in enumerate(axs):
           PlotUtilities.no_y_label(ax)
           PlotUtilities.ylabel("", ax=ax)
           if (j != 0):
               PlotUtilities.no_x_label(ax)
               PlotUtilities.xlabel("", ax=ax)

def data_plot(fecs,energies):
    n_cols = len(fecs)
    n_rows = 3
    all_ax = []
    for i, (data, e) in enumerate(zip(fecs, energies)):
        axs_tmp = [plt.subplot(n_rows, n_cols, j * n_cols + i + 1)
                   for j in range(n_rows)]
        ax1, ax2, ax3 = axs_tmp
        PlotUtil.plot_landscapes(data, e, ax1=ax1, ax2=ax2, ax3=ax3)
        # every axis after the first gets more of the decoaration chopped...
        all_ax.append(axs_tmp)
    fix_axes(all_ax)

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
    for i in range(energy_list.N):
        base_tmp = energy_list.base_dirs[i]
        in_dir = Pipeline._cache_dir(base=base_tmp,
                                     enum=Pipeline.Step.REDUCED)
        data = CheckpointUtilities.lazy_multi_load(in_dir)
        fecs.append(data)
        energies.append(energy_list.energies[i])
    n_cols = energy_list.N
    fig = PlotUtilities.figure(((n_cols * 1.5),3.5))
    data_plot(fecs, energies)
    PlotUtilities.savefig(fig,out_dir + "energies.png")

if __name__ == "__main__":
    run()
