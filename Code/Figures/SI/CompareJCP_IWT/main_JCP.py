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
from Lib.UtilForce.UtilGeneral.Plot import Scalebar

from Processing import ProcessingUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec
from Processing.Util import WLC
from Figures import FigureUtil

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
    q_offset_nm = RetinalUtil.min_sep_landscape() * 1e9
    min_fecs = 8
    q_interp, energy_list_arr = FigureUtil.\
        _read_energy_list_and_q_interp(input_dir, q_offset=q_offset_nm,
                                       min_fecs=min_fecs,remove_noisy=True)
    ex = energy_list_arr[0][0]
    q_start_nm = np.array([e.q_nm[0] for e in ex._other_helices])
    q_target_nm = 0
    helix_idx = np.argmin(np.abs(q_start_nm - q_target_nm))
    helix = ex._other_helices[helix_idx]
    landscape = helix
    data = RetinalUtil.read_fecs(ex)
    _, data_sliced = RetinalUtil.slice_data_for_helix(data,
                                                      min_ext_m=q_target_nm)
    # XXX why is this necessary?? screwing up absolute values
    previous_JCP = FigureUtil.read_non_peg_landscape(base="../../FigData/")
    offset_s = np.mean([d.Separation[0] for d in data_sliced])
    offset_jcp_nm = -35
    G_hao = landscape.G0_kcal_per_mol - landscape.G0_kcal_per_mol[0]
    G_JCP = previous_JCP.G0_kcal_per_mol - previous_JCP.G0_kcal_per_mol[0]
    offset_jcp_kcal = -1 * (np.median(G_hao[-G_hao.size//20:]) - \
                            np.median(G_JCP[-G_JCP.size//10:]))
    landscape_offset_nm = (landscape.q_nm[0]-offset_s * 1e9)
    fig = PlotUtilities.figure()
    xlim, ylim = FigureUtil._limits(data)
    fmt = dict(xlim=xlim,ylim=ylim)
    ax1 = plt.subplot(2,1,1)
    FigureUtil._plot_fec_list(data,color='k',**fmt)
    FigureUtil._plot_fec_list(data_sliced,**fmt)
    FigureUtil._plot_fmt(ax1, **fmt)
    ax2 = plt.subplot(2,1,2)
    plt.plot(landscape.q_nm-landscape_offset_nm,
             G_hao)
    plt.plot(previous_JCP.q_nm-offset_jcp_nm,
             G_JCP-offset_jcp_kcal,'r--')
    FigureUtil._plot_fmt(ax2, ylabel="G (kcal/mol)",is_bottom=True,
                         xlim=xlim,ylim=[None,None])
    PlotUtilities.savefig(fig,"./out.png")
    pass



if __name__ == "__main__":
    run()
