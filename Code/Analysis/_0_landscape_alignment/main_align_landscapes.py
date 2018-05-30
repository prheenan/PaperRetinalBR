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
from Figures import FigureUtil


def get_energy_list(base_dir_analysis):
    """
    :param base_dir_analysis:  see RetinalUtil._read_all_energies
    :return: list of zeroed retinal energies...
    """
    energy_list = RetinalUtil._read_all_energies(base_dir_analysis)
    return energy_list



def _energy_plot(energy_list,out_dir):
    # interpolate all the energies to the same grid
    energies_plot = [e._iwt_obj for e in energy_list]
    q_interp, splines = RetinalUtil.interpolating_G0(energies_plot)
    # get an average/stdev of energy
    fig = PlotUtilities.figure((7, 7))
    ax = plt.subplot(1, 1, 1)
    PlotUtil.plot_mean_landscape(q_interp, splines, ax=ax)
    max_x_show_nm = int(RetinalUtil.min_sep_landscape_nm() + 20)
    min_x_show_nm = int(np.floor(min(q_interp) - 2))
    plt.xlim([min_x_show_nm, max_x_show_nm])
    plt.xticks([i for i in range(min_x_show_nm, max_x_show_nm)])
    ax.xaxis.set_ticks_position('both')
    ax.grid(True)
    PlotUtilities.savefig(fig, out_dir + "avg.png")

def _fec_demo_plot(energy_list,out_dir):
    fecs = []
    energies = []
    N = len(energy_list)
    for e in energy_list:
        data = RetinalUtil.read_fecs(e)
        fecs.append(data)
        energies.append(e)
    n_cols = N
    fig = PlotUtilities.figure((n_cols * 1,6))
    FigureUtil.data_plot(fecs, energies)
    PlotUtilities.savefig(fig,out_dir + "energies.png",
                          subplots_adjust=dict(hspace=0.02,wspace=0.04))

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
                                                    "energy.pkl",
                                                    get_energy_list,force,
                                                    base_dir_analysis)
    _energy_plot(energy_list, out_dir)
    _fec_demo_plot(energy_list,out_dir)





if __name__ == "__main__":
    run()
