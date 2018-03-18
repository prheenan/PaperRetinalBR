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
    input_dir = "../../../Data/"
    subdirs_raw = [input_dir + d + "/" for d in os.listdir(input_dir)]
    subdirs = [d for d in subdirs_raw if (os.path.isdir(d))]
    out_dir = "./"
    energy_list_arr =[]
    # get all the energy objects
    for base in subdirs:
        in_dir = Pipeline._cache_dir(base=base,
                                     enum=Pipeline.Step.CORRECTED)
        in_file = in_dir + "energies.pkl"
        e = CheckpointUtilities.lazy_load(in_file)
        energy_list_arr.append(e)
    fig = PlotUtilities.figure()
    for i,energy_list in enumerate(energy_list_arr):
        G0_arr = [tmp.G0_kcal_per_mol for tmp in energy_list]
        G0 = np.mean(G0_arr,axis=0)
        G0_std = np.std(G0_arr,axis=0)
        plt.plot(tmp.q_nm,G0,label=str(i) + tmp.base_dir)
    PlotUtilities.lazyLabel("q (nm)","$\Delta G$ (kcal/mol)","")
    PlotUtilities.legend()
    PlotUtilities.savefig(fig,out_dir + "avg.png")




if __name__ == "__main__":
    run()
