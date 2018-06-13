# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../")

from Lib.UtilForce.UtilGeneral import PlotUtilities
from Processing.Util import WLC
from scipy.integrate import cumtrapz
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d


def make_model_plot(plot_inf,title):
    gs = gridspec.GridSpec(nrows=2,ncols=1)
    gs_ext = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                              subplot_spec=gs[0],
                                              width_ratios=[1],
                                              height_ratios=[2,1],hspace=0.02,
                                              wspace=0.02)
    ax1 = plt.subplot(gs_ext[0])
    print("For {:s}...".format(title))
    for k,v in plot_inf.kw.items():
        print("{:s}={:.5g}".format(k,v))
    labels = ["PEG","Unfolded\npeptide"]
    color_final = 'k'
    colors = ['b','r']
    for q, l, c in zip(plot_inf.qs,labels,colors):
        plt.plot(q, plot_inf.f,label=l,color=c)
    plt.plot(plot_inf.q, plot_inf.f,label="Both",color=color_final)
    PlotUtilities.lazyLabel("", "$F$ (pN)", title,
                            legend_kwargs=dict(frameon=True))
    PlotUtilities.no_x_label(ax1)
    W_str = "$W_{\mathbf{PEG}}" if len(labels) == 1 else \
        "$W_{\mathbf{PEG+POLY}}$"
    plt.subplot(gs_ext[1])
    plt.plot(plot_inf.q, plot_inf.w,color=color_final)
    PlotUtilities.lazyLabel("Extension (nm)", W_str + "\n(kcal/mol)", "")
    plt.subplot(gs[1])
    plt.plot(plot_inf.f, plot_inf.w, color=color_final)
    colors = ['m', 'g']
    for i, f_tmp in enumerate([149, 249]):
        W_int = plot_inf.W_at_f(f_tmp)
        label = "{:d} kcal/mol at {:d} pN".format(W_int, f_tmp)
        plt.axhline(W_int, label=label, color=colors[i],linestyle='--')
    xlabel = "$F$ (pN) [! not extension !]"
    PlotUtilities.lazyLabel(xlabel, W_str + "\n(kcal/mol)", "")


def hao_plot_inf(ext_grid):
    fec_system = WLC._make_plot_inf(ext_grid,WLC.read_haos_data,
                                    base="../../FigData/")
    fec_polypeptide = WLC._make_plot_inf(ext_grid,WLC.read_hao_polypeptide,
                                         base="../../FigData/")
    return fec_system, fec_polypeptide

def make_comparison_plot():
    plot_inf = WLC.peg_contribution(model_f=WLC.Hao_PEGModel)
    ext_grid = plot_inf.q
    plot_info_hao,plot_inf_hao_polypeptide = hao_plot_inf(ext_grid)
    # plot all of Hao's data
    label_hao_total = "Hao's FJC+WLC"
    plt.plot(plot_info_hao.q, plot_info_hao.f,'k-',
             label=label_hao_total)
    wlc_color = 'b'
    # plot Hao's polypeptide model
    plt.plot(*WLC.read_hao_polypeptide(base="../../FigData/"),color=wlc_color,
             marker='o',
             linestyle='None',markersize=4,label="Hao's WLC")
    plt.plot(ext_grid,plot_inf.f,color='r',label="My model")
    labels = ["My FJC", "My WLC"]
    colors = ['darkgoldenrod',wlc_color]
    for q,l,c in zip(plot_inf.qs,labels,colors):
        plt.plot(q,plot_inf.f,label=l,color=c)
    plt.xlim([0,50])
    plt.ylim([0,400])
    PlotUtilities.lazyLabel("Extension (nm)","Force (pN)","")

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    model,descr = WLC.Hao_PEGModel,"Hao_Parameters",
    fig = PlotUtilities.figure((4,6))
    plot_inf = WLC._make_plot_inf(ext_grid=np.linspace(0,30,num=3000),
                                  read_functor=WLC.read_haos_data,
                                  base="../../FigData/")
    make_model_plot(plot_inf=plot_inf, title=descr)
    PlotUtilities.savefig(fig,"PEG_{:s}.png".format(descr))
    # make sure what we have matches Hao.
    fig = PlotUtilities.figure((2.5,4))
    make_comparison_plot()
    plt.xlim([0,35])
    plt.ylim([0,300])
    PlotUtilities.legend(frameon=True)
    PlotUtilities.savefig(fig,"out_compare.png")
    pass

if __name__ == "__main__":
    run()
