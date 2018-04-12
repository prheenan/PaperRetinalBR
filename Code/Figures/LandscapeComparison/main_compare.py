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
from Lib.UtilForce.UtilGeneral.Plot import Scalebar

from Processing import ProcessingUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec
from Figures.Util import WLC

class LandscapeWithError(object):
    def __init__(self,q_nm,G_kcal,G_err_kcal):
        self.q_nm = q_nm
        self.G_kcal = G_kcal
        self.G_err_kcal = G_err_kcal

def read_non_peg_landscape():
    input_file = "../FigData/Fig2c_iwt_diagram.csv"
    arr =  np.loadtxt(input_file,delimiter=",").T
    q, G, G_low, G_upper = arr
    G_std = (G_upper - G_low) * 0.5
    return LandscapeWithError(q_nm=q,G_kcal=G,G_err_kcal=G_std)

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


def make_retinal_subplot(gs,energy_list_arr,shifts):
    q_interp_nm = energy_list_arr[0].q_nm
    means = [e.G_kcal for e in energy_list_arr]
    stdevs = [e.G_err_kcal for e in energy_list_arr]
    ax1 = plt.subplot(gs[0])
    common_error = dict(capsize=3)
    style_dicts = [dict(color='c', label=r"$\mathbf{\oplus}$ Retinal"),
                   dict(color='r', label=r"$\mathbf{\ominus}$ Retinal")]
    markers = ['v', 'x']
    deltas, deltas_std = [], []
    delta_styles = [dict(color=style_dicts[i]['color'], markersize=5,
                         linestyle='None', marker=markers[i], **common_error)
                    for i in range(len(energy_list_arr))]
    xlim = [None, 27]
    ylim = [-25, 400]
    q_arr = []
    round_energy = 2
    max_q_nm = max(q_interp_nm)
    # add the 'shifted' energies
    for i,(mean,stdev) in enumerate(zip(means,stdevs)):
        delta_style = dict(**delta_styles[i])
        plt.plot(q_interp_nm,mean,**style_dicts[i])
        q_at_max_energy, max_energy_mean, max_energy_std = \
            PlotUtil.plot_delta_GF(q_interp_nm, mean, stdev,
                                   max_q_nm=max_q_nm,
                                   round_energy=round_energy,
                                   label_offset=shifts[i],**delta_style)
        deltas.append(max_energy_mean)
        deltas_std.append(max_energy_std)
        q_arr.append(q_at_max_energy)
    delta_delta = np.abs(np.diff(deltas))[0]
    delta_delta_std = np.sqrt(np.sum(np.array(deltas_std) ** 2))
    delta_delta_fmt = np.round(delta_delta, round_energy)
    delta_delta_std_fmt = np.round(delta_delta_std, -1)
    title = r"$\Delta\Delta G$" + " = {:.0f} $\pm$ {:.0f} kcal/mol". \
        format(delta_delta_fmt, delta_delta_std_fmt)
    PlotUtilities.lazyLabel("q (nm)", "$\Delta G$ (kcal/mol)", title)
    plt.xlim([None, max_q_nm * 1.1])
    PlotUtilities.legend()
    # can I help-ya, help-ya, help-ya?
    # get the change in the DeltaDeltaG (or, the delta delta delta G)
    delta_delta_delta = np.abs(np.diff(shifts)[0])
    shifted_delta_delta = delta_delta - delta_delta_delta
    shifted_delta_delta_fmt = np.round(shifted_delta_delta, -1)
    title_shift = r"$\mathbf{\Delta\Delta }G$" + \
                  " = {:.0f} $\pm$ {:.0f} kcal/mol". \
        format(shifted_delta_delta_fmt, delta_delta_std_fmt)
    for i, (q, delta, err) in enumerate(zip(q_arr, deltas, deltas_std)):
        style_uncorrected = dict(**delta_styles[i])
        style_uncorrected['color'] = 'k'
        style_uncorrected['alpha'] = 0.5
        dy = -shifts[i]
        arrow_fudge = dy / 3
        plt.errorbar(x=q, y=delta + dy, yerr=err, **delta_styles[i])
        ax1.arrow(x=q, y=deltas[i] - abs(arrow_fudge), dx=0,
                  dy=dy - 2 * arrow_fudge, color=delta_styles[i]['color'],
                  length_includes_head=True, head_width=0.35, head_length=6)
    plt.xlim(xlim)
    plt.ylim(ylim)
    title_shift = "PEG3400-corrected ($\downarrow$)\n" + title_shift
    PlotUtilities.lazyLabel("Extension (nm)", "$\mathbf{\Delta}G$ (kcal/mol)",
                            title_shift,
                            legend_kwargs=dict(loc='lower right'))
    return ax1, means, stdevs

def make_comparison_plot(q_interp,energy_list_arr,G_no_peg):
    slice_arr = [slice(0, None, 1), slice(1, None, 1)]
    landscpes_with_error = []
    for i, energy_list_raw in enumerate(energy_list_arr):
        energy_list = [RetinalUtil.valid_landscape(e) for e in energy_list_raw]
        slice_f = slice_arr[i]
        energy_list = energy_list[slice_f]
        _, splines = RetinalUtil.interpolating_G0(energy_list)
        mean, stdev = PlotUtil._mean_and_stdev_landcapes(splines, q_interp)
        l = LandscapeWithError(q_nm=q_interp,G_kcal=mean,G_err_kcal=stdev)
        landscpes_with_error.append(l)
    peg = WLC.peg_contribution()
    shifts = [peg.W_at_f(f) for f in [250, 100]]
    gs = gridspec.GridSpec(nrows=1,ncols=1,width_ratios=[1])
    ax1, means, stdevs = make_retinal_subplot(gs,landscpes_with_error,shifts)
    # get the with-retinal max
    ax2 = plt.subplot(gs[0])
    G_offset = np.max([l.G_kcal[-1] for l in landscpes_with_error]) - shifts[0]
    q_offset = 25
    q_nm = G_no_peg.q_nm + q_offset
    G_kcal = G_no_peg.G_kcal + G_offset
    G_err_kcal = G_no_peg.G_err_kcal
    mean_err = np.mean(G_err_kcal)
    idx_errorbar = q_nm.size//2
    common_style = dict(color='k',linewidth=1.5)
    ax2.plot(q_nm,G_kcal,**common_style)
    ax2.errorbar(q_nm[idx_errorbar],G_kcal[idx_errorbar],yerr=mean_err,
                 marker=None,markersize=0,capsize=3,**common_style)
    axes = [ax1,ax2]
    y_limits = [a.get_ylim() for a in axes]
    ylim = [-25,None]
    for a in axes:
        a.set_ylim(ylim)
        a.set_xlim([0,max(q_nm)])
    # add in the scale bar
    ax = axes[0]
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    range_scale_kcal = 200
    x_range_nm = 15
    min_offset, _, rel_delta = Scalebar. \
        offsets_zero_tick(limits=ylim,range_scalebar=range_scale_kcal)
    offset_x = 0.15
    common_kw = dict(add_minor=True)
    scalebar_kw = dict(offset_x=offset_x,offset_y=min_offset,ax=ax,
                       x_on_top=True,
                       x_kwargs=dict(width=x_range_nm,unit="nm",**common_kw),
                       y_kwargs=dict(height=range_scale_kcal,unit="kcal/mol",
                                     **common_kw))
    PlotUtilities.no_x_label(ax=ax)
    PlotUtilities.no_y_label(ax=ax)
    Scalebar.crossed_x_and_y_relative(**scalebar_kw)



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
    q_interp = q_interp[np.where(q_interp  <= 25)]
    G_no_peg = read_non_peg_landscape()
    fig = PlotUtilities.figure(figsize=(3.5,3.25))
    make_comparison_plot(q_interp,energy_list_arr,G_no_peg)
    PlotUtilities.savefig(fig,out_dir + "avg.png")




if __name__ == "__main__":
    run()
