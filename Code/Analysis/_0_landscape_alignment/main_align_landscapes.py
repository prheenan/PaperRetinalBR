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
            try:
                tmp = read_in_energy(base_dir=d)
                to_ret.append(tmp)
            except IOError as e:
                print("Couldn't read from (so skipping): {:s}".format(d))
    return to_ret

def get_ranges(ax_list,get_x=True):
    f = lambda x: x.get_xlim() if get_x else x.get_ylim()
    lims = [ [f(ax[i]) for ax in ax_list] for i in range(3)]
    to_ret = [ [np.min(l),np.max(l)] for l in lims]
    return to_ret

def fix_axes(ax_list):
   # loop through all the axes and toss them.
   lims = [[ax[i] for ax in ax_list] for i in range(3)]
   xlims = get_ranges(ax_list,get_x=True)
   xlim_final = [np.min(xlims), np.max(xlims)]
   ylims = get_ranges(ax_list,get_x=False)
   for axs in ax_list[1:]:
       for j, ax in enumerate(axs):
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
    gs = gridspec.GridSpec(2,1)
    gs1 = gridspec.GridSpecFromSubplotSpec(n_rows,n_cols,subplot_spec=gs[0,0],
                                           wspace=0.05,hspace=0.05)
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
        title = "v={:s}\n {:s}".format(velocity, title)
        PlotUtilities.title(title)
    fix_axes(all_ax)
    q_interp, splines =  RetinalUtil.interpolating_G0(energies)
    # get an average/stdev of energy
    mean_energy, std_energy = PlotUtil.plot_mean_landscape(q_interp,
                                                           splines,ax=gs[-1,:])
    q_at_max_energy,_,_ =  \
        PlotUtil.plot_delta_GF(q_interp,mean_energy,std_energy,max_q_nm=25)
    xlim = max(plt.xlim())
    plt.axvspan(q_at_max_energy,xlim,color='k',alpha=0.3)
    PlotUtilities.legend(loc='upper right',frameon=True)


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
    energy_list_raw = CheckpointUtilities.getCheckpoint(out_dir + \
                                                    "energies.pkl",
                                                    get_energy_list,force,
                                                    base_dir_analysis)
    # XXX do this somewhere else
    energy_list = [RetinalUtil.valid_landscape(e) for e in energy_list_raw]
    fecs = []
    energies = []
    N = len(energy_list)
    for e in energy_list:
        base_tmp = e.base_dir
        in_dir = Pipeline._cache_dir(base=base_tmp,
                                     enum=Pipeline.Step.REDUCED)
        dir_exists = os.path.exists(in_dir)
        if (dir_exists and \
            len(GenUtilities.getAllFiles(in_dir,ext=".pkl")) > 0):
            data = CheckpointUtilities.lazy_multi_load(in_dir)
        else:
            data = []
        fecs.append(data)
        energies.append(e)
    n_cols = N
    fig = PlotUtilities.figure(((n_cols * 1.5),5))
    data_plot(fecs, energies)
    PlotUtilities.savefig(fig,out_dir + "energies.png")
    # interpolate all the energies to the same grid
    q_interp, splines =  RetinalUtil.interpolating_G0(energy_list)
    # get an average/stdev of energy
    fig = PlotUtilities.figure()
    PlotUtil.plot_mean_landscape(q_interp, splines)
    PlotUtilities.savefig(fig,out_dir + "avg.png")




if __name__ == "__main__":
    run()
