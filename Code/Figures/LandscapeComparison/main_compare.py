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
from Figures.Util import WLC


def read_energy_lists(subdirs):
    energy_list_arr =[]
    # get all the energy objects
    for base in subdirs:
        in_dir = Pipeline._cache_dir(base=base,
                                     enum=Pipeline.Step.CORRECTED)
        in_file = in_dir + "energies.pkl"
        e = CheckpointUtilities.lazy_load(in_file)
        energy_list_arr.append(e)
    energy_list_arr = [ [RetinalUtil.valid_landscape(e) for e in list_tmp]
                        for list_tmp in energy_list_arr]
    return energy_list_arr

def make_comparison_plot(q_interp,energy_list_arr):
    ax = plt.subplot(1, 2, 1)
    common_error = dict(capsize=3)
    style_dicts = [dict(color='c', label=r"$\mathbf{\oplus}$ Retinal"),
                   dict(color='r', label=r"$\mathbf{\ominus}$ Retinal")]
    markers = ['v', 'x']
    max_q_nm = 25
    slice_arr = [slice(0, None, 1), slice(1, None, 1)]
    deltas, deltas_std = [], []
    delta_styles = [dict(color=style_dicts[i]['color'], markersize=5,
                         linestyle='None', marker=markers[i],**common_error)
                    for i in range(len(energy_list_arr))]
    xlim = [0, 27]
    ylim = [-25, 350]
    q_arr = []
    for i, energy_list_raw in enumerate(energy_list_arr):
        energy_list = [RetinalUtil.valid_landscape(e) for e in energy_list_raw]
        slice_f = slice_arr[i]
        tmp_style = style_dicts[i]
        energy_list = energy_list[slice_f]
        _, splines = RetinalUtil.interpolating_G0(energy_list)
        mean, std = PlotUtil.plot_mean_landscape(q_interp, splines,
                                                 fill_between=False,
                                                 ax=ax, **tmp_style)
        delta_style = delta_styles[i]
        q_at_max_energy, max_energy_mean, max_energy_std = \
            PlotUtil.plot_delta_GF(q_interp, mean, std, max_q_nm=max_q_nm,
                                   **delta_style)
        deltas.append(max_energy_mean)
        deltas_std.append(max_energy_std)
        q_arr.append(q_at_max_energy)
    delta_delta = np.abs(np.diff(deltas))[0]
    delta_delta_std = np.sum(np.sqrt(np.array(deltas_std) ** 2))
    delta_delta_fmt = np.round(delta_delta, -1)
    delta_delta_std_fmt = np.round(delta_delta_std, -1)
    title = r"$\Delta\Delta G$" + " = {:.0f} $\pm$ {:.0f} kcal/mol". \
        format(delta_delta_fmt, delta_delta_std_fmt)
    PlotUtilities.lazyLabel("q (nm)", "$\Delta G$ (kcal/mol)", title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    PlotUtilities.legend()
    # add the 'shifted' energies
    peg = WLC.peg_contribution()
    shifts = [peg.W_at_f(f) for f in [250, 100]]
    # can I help-ya, help-ya, help-ya?
    # get the change in the DeltaDeltaG (or, the delta delta delta G)
    delta_delta_delta = np.abs(np.diff(shifts)[0])
    shifted_delta_delta = delta_delta - delta_delta_delta
    shifted_delta_delta_fmt = np.round(shifted_delta_delta, -1)
    title_shift = r"$\Delta\Delta G$" + " = {:.0f} $\pm$ {:.0f} kcal/mol". \
        format(shifted_delta_delta_fmt, delta_delta_std_fmt)
    for i, (q, delta, err) in enumerate(zip(q_arr, deltas, deltas_std)):
        style_uncorrected = dict(**delta_styles[i])
        style_uncorrected['color'] = 'k'
        style_uncorrected['alpha'] = 0.5
        dy = -shifts[i]
        arrow_fudge = dy / 3
        plt.errorbar(x=q, y=delta + dy, yerr=err, **delta_styles[i])
        ax.arrow(x=q, y=deltas[i] - abs(arrow_fudge), dx=0,
                 dy=dy - 2 * arrow_fudge,
                 length_includes_head=True, head_width=0.2, head_length=6)
    plt.xlim(xlim)
    plt.ylim(ylim)
    title_shift = "PEG3400-corrected ($\downarrow$)\n" + title_shift
    PlotUtilities.lazyLabel("Extension (nm)", "$\Delta G$ (kcal/mol)","")

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
    energy_list_arr = read_energy_lists(subdirs)
    e_list_flat = [e for list_tmp in energy_list_arr for e in list_tmp ]
    q_interp = RetinalUtil.common_q_interp(energy_list=e_list_flat)
    fig = PlotUtilities.figure(figsize=(7,3))
    make_comparison_plot(q_interp,energy_list_arr)
    PlotUtilities.savefig(fig,out_dir + "avg.png")




if __name__ == "__main__":
    run()
