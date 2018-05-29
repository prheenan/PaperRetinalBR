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
from Lib.AppIWT.Code import InverseWeierstrass
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Lib.UtilForce.UtilGeneral.Plot import Scalebar

from Processing import ProcessingUtil
import RetinalUtil,PlotUtil
from Figures import FigureUtil

class FakeMeta(object):
    def __init__(self,SourceFile):
        self.SourceFile = SourceFile

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

def _G0_plot(data_sliced,landscape):
    # XXX why is this necessary?? screwing up absolute values
    previous_JCP = FigureUtil.read_non_peg_landscape(base="../../FigData/")
    offset_s = np.mean([d.Separation[0] for d in data_sliced])
    G_hao = landscape.G0_kcal_per_mol
    idx_zero = np.where(landscape.q_nm <= 100)
    G_hao = G_hao - landscape.G0_kcal_per_mol[0]
    G_JCP = previous_JCP.G0_kcal_per_mol - previous_JCP.G0_kcal_per_mol[0]
    landscape_offset_nm = min(landscape.q_nm)
    offset_jcp_nm = min(previous_JCP.q_nm)
    fig = PlotUtilities.figure()
    xlim, ylim = FigureUtil._limits(data_sliced)
    fmt = dict(xlim=xlim, ylim=ylim)
    ax1 = plt.subplot(2, 1, 1)
    # FigureUtil._plot_fec_list(data,color='k',**fmt)
    FigureUtil._plot_fec_list(data_sliced, **fmt)
    FigureUtil._plot_fmt(ax1, **fmt)
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(landscape.q_nm - landscape_offset_nm,
             G_hao)
    plt.plot(previous_JCP.q_nm - offset_jcp_nm,
             G_JCP, 'r--')
    FigureUtil._plot_fmt(ax2, ylabel="G (kcal/mol)", is_bottom=True,
                         xlim=xlim, ylim=[None, None])
    PlotUtilities.savefig(fig, "./out.png")

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

class HeatmapJCP:
    def __init__(self,x,f,N):
        self._x = x
        self._f = f
        self._N = N
        self.heatmap = _heatmap(x,f,N).T
    @property
    def _extent_nm_and_pN(self):
        return [min(self._x),max(self._x),min(self._f),max(self._f)]

def _read_jcp_heatmap(in_file):
    data_hist_jcp = np.loadtxt(in_file, delimiter=',')
    x, f, N = data_hist_jcp.T
    return HeatmapJCP(x,f,N)


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
    # read in the EC histogram...
    in_file = "../../FigData/Fig2a_iwt_diagram.csv"
    heatmap_jcp = _read_jcp_heatmap(in_file)
    plt.close()
    extent = heatmap_jcp._extent_nm_and_pN
    plt.imshow(heatmap_jcp.heatmap, origin='lower', aspect='equal',
               extent=extent)
    plt.show()
    q_offset_nm = RetinalUtil.min_sep_landscape() * 1e9
    min_fecs = 8
    q_interp, energy_list_arr = FigureUtil.\
        _read_energy_list_and_q_interp(input_dir, q_offset=q_offset_nm,
                                       min_fecs=min_fecs,remove_noisy=True)
    ex = energy_list_arr[0][0]
    q_start_nm = RetinalUtil.min_ext_m() * 1e9
    q_target_nm = 41.5
    helix_idx = np.argmin(np.abs(q_start_nm - q_target_nm))
    helix = ex._other_helices[helix_idx]
    landscape = helix
    data = RetinalUtil.read_fecs(ex,enum=Pipeline.Step.MANUAL)
    bl_extra = ['716', '539']
    data = [d for d in data if id_fec(d) not in bl_extra]
    slices = RetinalUtil._get_slice(data,q_target_nm * 1e-9)
    data_sliced = [d._slice(s) for s,d in zip(slices,data)]
    data_sliced = RetinalUtil.process_helical_slice(data_sliced)
    ef_aligned = _align_to_EF(data_sliced)
    # slice to the appropriate
    min_seps = []
    max_N = np.inf
    for d in ef_aligned:
        sep_fit = RetinalUtil._fit_sep(d)
        min_seps.append(min(sep_fit))
        max_N = min(max_N,sep_fit.size)
    ext_sliced = []
    max_N = 10000
    for d in ef_aligned:
        sep_fit = RetinalUtil._fit_sep(d)
        first_idx = np.where(sep_fit >= max(min_seps))[0][0]
        sliced_tmp = d._slice(slice(first_idx,first_idx + max_N))
        ext_sliced.append(sliced_tmp)
    size_exp = ext_sliced[0].Force.size
    actual_sizes = [e.Force.size for e in ext_sliced]
    np.testing.assert_allclose(size_exp,actual_sizes)
    # convert to IWT objects
    iwt_data = [i for i in RetinalUtil._convert_to_iwt(ext_sliced, "")]
    # get the new IWT landscape
    f_iwt = InverseWeierstrass.free_energy_inverse_weierstrass
    iwt_obj = f_iwt(unfolding=iwt_data)
    _G0_plot(data_sliced, iwt_obj)
    plot_dir = "./plot/"
    GenUtilities.ensureDirExists(plot_dir)
    xlim, ylim = FigureUtil._limits(data)
    fmt = dict(xlim=xlim,ylim=ylim)
    fig = PlotUtilities.figure()
    ax1 = plt.subplot(2,1,1)
    FigureUtil._plot_fec_list(data, color='k',**fmt)
    FigureUtil._plot_fec_list(data_sliced,**fmt)
    FigureUtil._plot_fmt(is_bottom=False,ax=ax1,**fmt)
    ax2 = plt.subplot(2,1,2)
    FigureUtil._plot_fec_list(ext_sliced, **fmt)
    FigureUtil._plot_fmt(is_bottom=True,ax=ax2,**fmt)
    PlotUtilities.savefig(fig,plot_dir + "debug.png")
    PlotUtil._feather_plot(ef_aligned,plot_dir,xlim=[0,60])
    pass



if __name__ == "__main__":
    run()
