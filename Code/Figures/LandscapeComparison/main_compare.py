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


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    input_dir = "../../../Data/FECs180307/"
    subdirs_raw = [input_dir + d + "/" for d in os.listdir(input_dir)]
    subdirs = [d for d in subdirs_raw if (os.path.isdir(d))]
    out_dir = "./"
    energy_list_arr =[]
    # get all the energy objects
    for base in subdirs:
        in_dir = Pipeline._cache_dir(base=base,
                                     enum=Pipeline.Step.CORRECTED)
        in_file = in_dir + "energies.pkl"
        try:
            e = CheckpointUtilities.lazy_load(in_file)
            energy_list_arr.append(e)
        except IOError as e:
            print(" ==== Couldn't read in {:s}; skipping ==== ".format(e))
    energy_list_arr = [ [RetinalUtil.valid_landscape(e) for e in list_tmp]
                        for list_tmp in energy_list_arr]
    e_list_flat = [e for list_tmp in energy_list_arr for e in list_tmp ]
    q_interp = RetinalUtil.common_q_interp(energy_list=e_list_flat)
    fig = PlotUtilities.figure()
    ax = plt.subplot(1,1,1)
    style_dicts = [dict(color='c',label="+ Retinal"),
                   dict(color='r',label="- Retinal")]
    markers = ['v','x']
    max_q_nm = 25
    slice_arr = [slice(0,None,1),slice(1,None,1)]
    deltas, deltas_std = [], []
    round_energy = 0
    for i,energy_list_raw in enumerate(energy_list_arr):
        energy_list = [RetinalUtil.valid_landscape(e) for e in energy_list_raw]
        slice_f = slice_arr[i]
        tmp_style = style_dicts[i]
        energy_list = energy_list[slice_f]
        _, splines = RetinalUtil.interpolating_G0(energy_list)
        mean,std = PlotUtil.plot_mean_landscape(q_interp, splines,
                                                 ax=ax,**tmp_style)
        delta_style = dict(color=tmp_style['color'],markersize=5,
                           linestyle='None',marker=markers[i])
        q_at_max_energy, max_energy_mean, max_energy_std =\
            PlotUtil.plot_delta_GF(q_interp,mean,std,max_q_nm=max_q_nm,
                                   round_energy=round_energy,**delta_style)
        deltas.append(max_energy_mean)
        deltas_std.append(max_energy_std)
    delta_delta = np.abs(np.diff(deltas))[0]
    delta_delta_std = np.sqrt(np.sum(np.array(deltas_std)**2))
    delta_delta_fmt = np.round(delta_delta,round_energy)
    delta_delta_std_fmt = np.round(delta_delta_std,-1)
    title = r"$\Delta\Delta G$" +  " = {:.0f} $\pm$ {:.0f} kcal/mol".\
        format(delta_delta_fmt,delta_delta_std_fmt)
    PlotUtilities.lazyLabel("q (nm)","$\Delta G$ (kcal/mol)",title)
    plt.xlim([None,max_q_nm*1.1])
    PlotUtilities.legend()
    PlotUtilities.savefig(fig,out_dir + "avg.png")




if __name__ == "__main__":
    run()
