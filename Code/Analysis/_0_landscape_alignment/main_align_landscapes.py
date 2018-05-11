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
import re

def get_energy_list(base_dir_analysis, min_fecs):
    """
    :param base_dir_analysis:  see RetinalUtil._read_all_energies
    :param min_fecs: see RetinalUtil._read_all_energies
    :return: list of zeroed retinal energies...
    """
    energy_list = RetinalUtil._read_all_energies(base_dir_analysis)
    # make sure we have a minimum number of FECS
    energy_list = [e for e in energy_list if e.n_fecs >= min_fecs]
    # the 3000nms BO data is very noisy; discard it.
    energy_list = [e for e in energy_list
                   if "BR-Retinal/3000nms/" not in e.base_dir]

    return energy_list

def get_ranges(ax_list,get_x=True):
    f = lambda x: x.get_xlim() if get_x else x.get_ylim()
    lims = [ [f(ax[i]) for ax in ax_list] for i in range(3)]
    to_ret = [ [np.min(l),np.max(l)] for l in lims]
    return to_ret

def fix_axes(ax_list):
   # loop through all the axes and toss them.
   xlims = get_ranges(ax_list,get_x=True)
   ylims = get_ranges(ax_list,get_x=False)
   for i,axs in enumerate(ax_list):
       for j, ax in enumerate(axs):
           if (i > 0):
               # columns after the first lose their labels
               PlotUtilities.no_y_label(ax)
               PlotUtilities.ylabel("", ax=ax)
           ax.set_ylim(ylims[j])
           if (j != 0):
               PlotUtilities.no_x_label(ax)
               PlotUtilities.xlabel("", ax=ax)

def data_plot(fecs,energies):
    n_cols = len(energies)
    n_rows = 3
    all_ax = []
    gs1 = gridspec.GridSpec(n_rows+1,n_cols)
    for i, (data, e) in enumerate(zip(fecs, energies)):
        axs_tmp = [plt.subplot(gs1[j,i])
                   for j in range(n_rows)]
        ax1, ax2, ax3 = axs_tmp
        PlotUtil.plot_landscapes(data, e, ax1=ax1, ax2=ax2, ax3=ax3)
        # every axis after the first gets more of the decoaration chopped...
        all_ax.append(axs_tmp)
        # XXX should really put this into the meta class...
        plt.sca(ax1)
        tmp_str = e.file_name
        match = re.search(r"""
                          Retinal/([\d\w]+)/([\d\w]+)/
                          """, tmp_str, re.IGNORECASE | re.VERBOSE)
        groups = match.groups()
        velocity, title = groups
        vel_label = velocity.replace("nms","")
        title_label = title.replace("FEC","")
        title = "v={:s}\n {:s}".format(vel_label, title_label)
        PlotUtilities.title(title,fontsize=5)
    fix_axes(all_ax)
    xlim = all_ax[0][0].get_xlim()
    q_interp, splines =  RetinalUtil.interpolating_G0(energies)
    # get an average/stdev of energy
    mean_energy, std_energy = PlotUtil.plot_mean_landscape(q_interp,
                                                           splines,ax=gs1[-1,0])
    q_at_max_energy,_,_ =  \
        PlotUtil.plot_delta_GF(q_interp,mean_energy,std_energy,
                               max_q_nm=RetinalUtil.q_GF_nm())
    plt.axvspan(q_at_max_energy,max(xlim),color='k',alpha=0.3)
    plt.xlim(xlim)

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
    min_fecs = 10
    GenUtilities.ensureDirExists(out_dir)
    energy_list = CheckpointUtilities.getCheckpoint(out_dir + \
                                                    "energies.pkl",
                                                    get_energy_list,force,
                                                    base_dir_analysis,
                                                    min_fecs)
    fecs = []
    energies = []
    N = len(energy_list)
    for e in energy_list:
        data = RetinalUtil.read_fecs(e)
        fecs.append(data)
        energies.append(e)
    n_cols = N
    fig = PlotUtilities.figure((n_cols * 1,6))
    data_plot(fecs, energies)
    PlotUtilities.savefig(fig,out_dir + "energies.png",
                          subplots_adjust=dict(hspace=0.02,wspace=0.04))
    # interpolate all the energies to the same grid
    q_interp, splines =  RetinalUtil.interpolating_G0(energy_list)
    # get an average/stdev of energy
    fig = PlotUtilities.figure((7,7))
    ax = plt.subplot(1,1,1)
    PlotUtil.plot_mean_landscape(q_interp, splines,ax=ax)
    max_x_show_nm= RetinalUtil.q_GF_nm()+ 20
    min_x_show_nm = int(np.floor(min(q_interp)-2))
    plt.xlim([min_x_show_nm,max_x_show_nm])
    plt.xticks([i for i in range(min_x_show_nm,max_x_show_nm)])
    ax.xaxis.set_ticks_position('both')
    ax.grid(True)
    PlotUtilities.savefig(fig,out_dir + "avg.png")




if __name__ == "__main__":
    run()
