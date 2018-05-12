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

sys.path.append("../../../")
from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec
import re

from Figures import FigureUtil

def force_landscape_N(x):
    return x.A_z_dot

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    input_dir = "../../../../Data/FECs180307/"
    out_dir = "./"
    q_offset_nm = 100
    q_interp, energy_list_arr = FigureUtil.\
        _read_energy_list_and_q_interp(input_dir, q_offset=q_offset_nm)
    means = []
    for i, energy_list in enumerate(energy_list_arr):
        _, splines = RetinalUtil.interpolating_G0(energy_list,
                                                  f=force_landscape_N)
        mean, _ = PlotUtil._mean_and_stdev_landcapes(splines, q_interp)
        means.append(mean)
    # fit a spline to the mean energy
    F_tmp = means[0]
    fig = PlotUtilities.figure()
    plt.plot(q_interp,F_tmp * 1e12)
    PlotUtilities.lazyLabel("q (nm)","$F$ (pN)","")
    PlotUtilities.savefig(fig,"FigureS_A_z.png")
    pass


if __name__ == "__main__":
    run()
