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
sys.path.append("../../../Processing")
from Lib.UtilPipeline import Pipeline
from Lib.AppWLC.Code import WLC
from Lib.AppIWT.Code import InverseWeierstrass,WeierstrassUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Lib.UtilForce.UtilGeneral.Plot import Scalebar
from Lib.UtilForce.FEC import FEC_Plot

from Processing import ProcessingUtil
import RetinalUtil,PlotUtil
from Figures import FigureUtil


class HeatmapJCP:
    def __init__(self,x,f,N):
        self._x = x
        self._f = f
        self._N = N
        self.heatmap = _heatmap(x,f,N).T
    def _extent_nm_and_pN(self,offset_x_nm=0):
        return [min(self._x)+offset_x_nm,max(self._x)+offset_x_nm,
                min(self._f),max(self._f)]

class DataInfo(object):
    def __init__ (self,data_sliced,iwt_obj,wham_obj):
        self.data_sliced = data_sliced
        self.iwt_obj = iwt_obj
        self.wham_obj = wham_obj


def _align_to_EF(data):
    kw_FEATHER = RetinalUtil._def_kw_FEATHER()
    data_FEATHER =   [RetinalUtil.feather_single(d,force_no_adhesion=True,
                                                 **kw_FEATHER) for d in data]
    wlc_kwargs = dict(Lp=0.4e-9,K0=10000e12,kbT=4.1e-21)
    # fit and align to the contour lengths...
    for d in data_FEATHER:
        event_first = d.info_feather.event_idx[0]
        sliced_d = d._slice(slice(0,event_first,1))
        x,f = sliced_d.Separation, sliced_d.Force
        x0,fit = WLC.fit(x,f,brute_dict=dict(Ns=20,
                                             ranges=[slice(1e-9,100e-9,5e-9)]),
                         **wlc_kwargs)
        offset = x0[0] - 18e-9
        d.Separation -= offset
        d.ZSnsr -= offset
    to_ret = data_FEATHER
    return to_ret

def _G0_plot(plot_dir,data_sliced,landscape,fmt):
    # XXX why is this necessary?? screwing up absolute values
    previous_JCP = FigureUtil.read_non_peg_landscape(base="../../FigData/")
    offset_s = np.mean([d.Separation[0] for d in data_sliced])
    G_hao = landscape.G0_kcal_per_mol
    idx_zero = np.where(landscape.q_nm <= 100)
    G_hao = G_hao - landscape.G0_kcal_per_mol[0]
    G_JCP = previous_JCP.G0_kcal_per_mol - previous_JCP.G0_kcal_per_mol[0] + 50
    offset_jcp_nm = min(previous_JCP.q_nm)
    landscape_offset_nm = min(landscape.q_nm)
    q_JCP_nm = previous_JCP.q_nm - offset_jcp_nm + 5
    q_Hao_nm = landscape.q_nm - landscape_offset_nm
    fig = FigureUtil._fig_single(y=6)
    xlim, ylim = FigureUtil._limits(data_sliced)
    ax1 = plt.subplot(2, 1, 1)
    FigureUtil._plot_fec_list(data_sliced, **fmt)
    FigureUtil._plot_fmt(ax1, **fmt)
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(q_Hao_nm,G_hao,label="Aligned, IWT")
    plt.plot(q_JCP_nm,G_JCP, 'r--',label="JCP landscape")
    FigureUtil._plot_fmt(ax2, ylabel="G (kcal/mol)", is_bottom=True,
                         xlim=xlim, ylim=[None, None])
    PlotUtilities.legend(ax=ax2,handlelength=2)
    ax2.set_xlim(fmt['xlim'])
    PlotUtilities.savefig(fig, plot_dir + "FigureSX_LandscapeComparison.png")

def id_fec(d):
    name = d.Meta.Name
    id_v = re.search("\d+",name,re.VERBOSE)
    assert id_v is not None
    return id_v.group(0)

def _heatmap(x,f,N):
    unique_X = np.unique(x)
    unique_F = np.unique(f)
    digitized_N_x = np.digitize(x=x, bins=unique_X)
    digitized_N_F = np.digitize(x=f, bins=unique_F)
    n_x, n_F = unique_X.size, unique_F.size
    digitized_N_x = np.minimum(digitized_N_x, n_x - 1)
    digitized_N_F = np.minimum(digitized_N_F, n_F - 1)
    arr = np.zeros((n_x, n_F))
    for ii, (x_i, f_j) in enumerate(zip(digitized_N_x, digitized_N_F)):
        arr[x_i, f_j] = N[ii]
    return arr


def _read_jcp_heatmap(in_file):
    data_hist_jcp = np.loadtxt(in_file, delimiter=',')
    x, f, N = data_hist_jcp.T
    return HeatmapJCP(x,f,N)

def _slice_to_target(data_sliced,q_target_nm):
    data_sliced_plot = [d._slice(slice(0,None,1)) for d in data_sliced]
    for i in range(len(data_sliced_plot)):
        q_target_m = q_target_nm * 1e-9
        data_sliced_plot[i].Separation -= q_target_m
        data_sliced_plot[i].ZSnsr -= q_target_m
    return data_sliced_plot


def _plot_comparison(plot_dir,heatmap_jcp,iwt_obj,data_sliced_plot):
    fmt = dict(xlim=[-5,55],ylim=[-20,150])
    _G0_plot(plot_dir,data_sliced_plot, iwt_obj,fmt=fmt)
    fig = FigureUtil._fig_single(y=6)
    ax1 = plt.subplot(2,1,1)
    extent = heatmap_jcp._extent_nm_and_pN(offset_x_nm=0)
    plt.imshow(heatmap_jcp.heatmap, origin='lower', aspect='auto',
               extent=extent,cmap=plt.cm.afmhot)
    FigureUtil._plot_fmt(is_bottom=False,ax=ax1,**fmt)
    PlotUtilities.title("Top: JCP.\n Bottom: New data, - PEG3400")
    ax2 = plt.subplot(2,1,2)
    FEC_Plot.heat_map_fec(data_sliced_plot,use_colorbar=False,
                          num_bins=(150, 75),separation_max=fmt['xlim'][1])
    FigureUtil._plot_fmt(is_bottom=True,ax=ax2,**fmt)
    out_name = plot_dir + "FigureSX_jcp_fec_comparison.png"
    PlotUtilities.savefig(fig,out_name,tight=True)

def data_info(data,q_target_nm):
    bl_extra = []
    data = [d for d in data if id_fec(d) not in bl_extra]
    slices = RetinalUtil._get_slice(data,q_target_nm * 1e-9)
    data_sliced = [d._slice(s) for s,d in zip(slices,data)]
    iwt_data = [i for i in RetinalUtil._sanitize_iwt(data_sliced, "")]
    iwt_data = [ WeierstrassUtil.convert_to_iwt(d,offset=d.ZSnsr[0])
                 for d in iwt_data]
    # get the new IWT landscape
    f_iwt = InverseWeierstrass.free_energy_inverse_weierstrass
    iwt_obj = f_iwt(unfolding=iwt_data)
    # XXX wham doesnt work
    wham_data = UtilWHAM.to_wham_input(iwt_data,n_ext_bins=40)
    wham_obj = WeightedHistogram.wham(wham_data)
    data_sliced = _slice_to_target(iwt_data,q_target_nm)
    to_ret = DataInfo(data_sliced, iwt_obj, wham_obj)
    return to_ret

def _read_all_data(energy_list):
    fecs = []
    for e in energy_list:
        data = RetinalUtil.read_fecs(e)
        fecs.append(data)
    return fecs

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    input_dir = "../../../../Data/FECs180307/"
    q_target_nm = RetinalUtil.q_GF_nm_plot() - 5
    q_interp, energy_list_arr = FigureUtil.\
        _read_energy_list_and_q_interp(input_dir, q_offset=q_target_nm,
                                       min_fecs=4,remove_noisy=True)
    data_BR, data_BO = [_read_all_data(e) for e in energy_list_arr]
    # read in the EC histogram...
    in_file = "../../FigData/Fig2a_iwt_diagram.csv"
    heatmap_jcp = _read_jcp_heatmap(in_file)
    plot_base_dir = "./plot/"
    GenUtilities.ensureDirExists(plot_base_dir)
    for i,data in enumerate(data_BR):
        data_info_tmp = data_info(data, q_target_nm)
        meta_src = data[0].Meta.SourceFile
        split_info = meta_src.split("Retinal")
        inf = split_info[2].split("FEC")[0].replace("/", "_")
        plot_dir = plot_base_dir + "_{:d}_{:s}".format(i,inf)
        _plot_comparison(plot_dir, heatmap_jcp, data_info_tmp.iwt_obj,
                         data_info_tmp.data_sliced)
    pass



if __name__ == "__main__":
    run()
