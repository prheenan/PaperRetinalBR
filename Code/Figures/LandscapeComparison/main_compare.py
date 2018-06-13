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
from Lib.UtilForce.UtilGeneral.Plot import Scalebar, Annotations

from Processing import ProcessingUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec
from Processing.Util import WLC
from Figures import FigureUtil


def make_retinal_subplot(gs,energy_list_arr,shifts,skip_arrow=True):
    q_interp_nm = energy_list_arr[0].q_nm
    means = [e.G0_kcal_per_mol for e in energy_list_arr]
    # fit a second order polynomial and subtract from each point
    q_fit_nm_relative = 7
    max_fit_idx = \
        np.argmin(np.abs((q_interp_nm - q_interp_nm[0]) - q_fit_nm_relative))
    fits = []
    fit_pred_arr = []
    for m in means:
        fit = np.polyfit(x=q_interp_nm[:max_fit_idx],y=m[:max_fit_idx],deg=2)
        fits.append(fit)
        fit_pred = np.polyval(fit,x=q_interp_nm)
        fit_pred_arr.append(fit_pred)
    stdevs = [e.G_err_kcal for e in energy_list_arr]
    ax1 = plt.subplot(gs[0])
    common_error = dict(capsize=0)
    style_dicts = [dict(color='c', label=r"with Retinal"),
                   dict(color='r', label=r"w/o  Retinal")]
    markers = ['v', 'x']
    deltas, deltas_std = [], []
    delta_styles = [dict(color=style_dicts[i]['color'], markersize=5,
                         linestyle='None', marker=markers[i], **common_error)
                    for i in range(len(energy_list_arr))]
    xlim = [None, 27]
    ylim = [-25, 450]
    q_arr = []
    round_energy = -1
    max_q_nm = max(q_interp_nm)
    # add the 'shifted' energies
    for i,(mean,stdev) in enumerate(zip(means,stdevs)):
        tmp_style = style_dicts[i]
        style_fit = dict(**tmp_style)
        style_fit['linestyle'] = '--'
        style_fit['label'] = None
        corrected = mean - fit_pred_arr[i]
        plt.plot(q_interp_nm,mean,**tmp_style)
        plt.fill_between(x=q_interp_nm,y1=mean-stdev,y2=mean+stdev,
                         color=tmp_style['color'],linewidth=0,alpha=0.3)
        energy_error = np.mean(stdev)
        max_idx = -1
        q_at_max_energy = q_interp_nm[max_idx]
        max_energy_mean = corrected[max_idx]
        # for the error, use the mean error over all interpolation
        max_energy_std = energy_error
        deltas.append(max_energy_mean)
        deltas_std.append(max_energy_std)
        q_arr.append(q_at_max_energy)
    delta_delta = np.abs(np.diff(deltas))[0]
    delta_delta_std = np.sqrt(np.sum(np.array(deltas_std) ** 2))
    delta_delta_fmt = np.round(delta_delta, round_energy)
    delta_delta_std_fmt = np.round(delta_delta_std, -1)
    title = r"$\mathbf{\Delta\Delta}G$" + " = {:.0f} $\pm$ {:.0f} kcal/mol". \
        format(delta_delta_fmt, delta_delta_std_fmt)
    plt.xlim(xlim)
    plt.ylim(ylim)
    PlotUtilities.lazyLabel("Extension (nm)", "$\mathbf{\Delta}G$ (kcal/mol)",
                            "",legend_kwargs=dict(loc='lower right'))
    return ax1, means, stdevs


def make_comparison_plot(q_interp,energy_list_arr,G_no_peg,q_offset):
    landscpes_with_error = \
        FigureUtil._get_error_landscapes(q_interp, energy_list_arr)
    # get the extension grid we wnt...
    ext_grid = np.linspace(0,25,num=100)
    # read in Hao's energy landscape
    fec_system = WLC._make_plot_inf(ext_grid,WLC.read_haos_data)
    shifts = [fec_system.W_at_f(f) for f in [249, 149]]
    gs = gridspec.GridSpec(nrows=1,ncols=1)
    ax1, means, stdevs = \
        make_retinal_subplot(gs,landscpes_with_error,shifts)
    # get the with-retinal max
    ax1 = plt.subplot(gs[0])
    # get the max of the last point (the retinal energy landscape is greater)
    offsets = [l.G0_kcal_per_mol[-1] for l in landscpes_with_error]
    q_no_PEG_start =  max(q_interp)
    for i,G_offset in enumerate(offsets):
        q_nm = G_no_peg.q_nm + q_no_PEG_start
        G_kcal = G_no_peg.G0_kcal_per_mol + G_offset
        G_err_kcal = G_no_peg.G_err_kcal
        mean_err = np.mean(G_err_kcal)
        idx_errorbar = q_nm.size//2
        common_style = dict(color='grey',linewidth=1.5)
        ax1.plot(q_nm,G_kcal,linestyle='--',**common_style)
        if (i != 0):
            continue
        ax1.errorbar(q_nm[idx_errorbar],G_kcal[idx_errorbar],yerr=mean_err,
                     marker=None,markersize=0,capsize=3,**common_style)
    axes = [ax1]
    ylim = [None,
            np.max(offsets) + max(G_no_peg.G0_kcal_per_mol) + G_err_kcal[-1]*4]
    for a in axes:
        a.set_ylim(ylim)
        a.set_xlim([None,max(q_nm)])
    # add in the scale bar
    ax = axes[0]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = (ylim[1]-ylim[0])
    range_scale_kcal = np.round(y_range/3,-2)
    x_range_scalebar_nm = 20
    min_offset, _, rel_delta = Scalebar. \
        offsets_zero_tick(limits=ylim,range_scalebar=range_scale_kcal)
    min_offset_x, _, rel_delta_x= Scalebar. \
        offsets_zero_tick(limits=xlim,range_scalebar=x_range_scalebar_nm)
    offset_x = 0.25
    offset_y = min_offset + 1 * rel_delta
    common_kw = dict(add_minor=True)
    scalebar_kw = dict(offset_x=offset_x,offset_y=offset_y,ax=ax,
                       x_on_top=True,y_on_right=False,
                       x_kwargs=dict(width=x_range_scalebar_nm,unit="nm",
                                     **common_kw),
                       y_kwargs=dict(height=range_scale_kcal,unit="kcal/mol",
                                     **common_kw))
    PlotUtilities.no_x_label(ax=ax)
    PlotUtilities.no_y_label(ax=ax)
    Scalebar.crossed_x_and_y_relative(**scalebar_kw)
    # add the helical boxes
    offset_boxes = -5
    FigureUtil.add_helical_boxes(ax=ax1,ymax_box=0.97,box_height=0.07,
                                 constant_offset=offset_boxes)
    # draw an arrow depicting the DeltaDeltaG Total
    ax1.annotate(s="",xycoords='data',textcoords='data',
                 xy=(q_no_PEG_start,offsets[0]),
                 arrowprops=dict(arrowstyle="|-|",color='k',mutation_scale=2,
                                 shrinkA=0,shrinkB=0),
                 xytext=(q_no_PEG_start, offsets[1]),annotation_clip=False)
    str_text = "$\mathbf{\Delta\Delta}G_{\mathbf{total}}$"
    x_text = q_no_PEG_start * 1.05
    y_range = np.abs(np.diff(offsets)[0])
    y_text = np.mean(offsets) + y_range * 0.2
    Annotations.relative_annotate(ax=ax1,s=str_text,xy=(x_text,y_text),
                                  xycoords='data',horizontalalignment='left',
                                  verticalalignment='center')



def _giant_debugging_plot(out_dir,energy_list_arr):
    fig = PlotUtilities.figure((8,12))
    gs = gridspec.GridSpec(nrows=2,ncols=1,hspace=0.15)
    n_cols = max([len(list_v) for list_v in energy_list_arr])
    for i,energy_list in enumerate(energy_list_arr):
        fecs = []
        energies = []
        for e in energy_list:
            data = RetinalUtil.read_fecs(e)
            fecs.append(data)
            energies.append(e)
        gs_tmp = gridspec.GridSpecFromSubplotSpec(nrows=3,
                                                  ncols=n_cols,
                                                  subplot_spec=gs[i])
        FigureUtil.data_plot(fecs, energies,gs1=gs_tmp,xlim=[-20,100])
    PlotUtilities.savefig(fig, out_dir + "FigureS_Mega_Debug.png",
                          subplots_adjust=dict(hspace=0.02, wspace=0.04))

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    input_dir = "../../../Data/FECs180307/"
    out_dir = "./"
    q_offset_nm = RetinalUtil.q_GF_nm_plot()
    min_fecs = 2
    q_interp, energy_list_arr = FigureUtil.\
        _read_energy_list_and_q_interp(input_dir, q_offset=q_offset_nm,
                                       min_fecs=min_fecs,remove_noisy=True)
    G_no_peg = FigureUtil.read_non_peg_landscape()
    #_giant_debugging_plot(out_dir, energy_list_arr)
    fig = PlotUtilities.figure(figsize=(3,3))
    make_comparison_plot(q_interp,energy_list_arr,G_no_peg,q_offset_nm)
    PlotUtilities.savefig(fig,out_dir + "FigureX_LandscapeComparison.png")




if __name__ == "__main__":
    run()
