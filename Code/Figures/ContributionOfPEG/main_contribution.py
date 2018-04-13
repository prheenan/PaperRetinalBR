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

sys.path.append("../../")

from Lib.UtilForce.UtilGeneral import PlotUtilities
from Figures.Util import WLC
from scipy.integrate import cumtrapz
import matplotlib.gridspec as gridspec

def make_model_plot(model_f,title):
    gs = gridspec.GridSpec(nrows=2,ncols=1)
    gs_ext = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                              subplot_spec=gs[0],
                                              width_ratios=[1],
                                              height_ratios=[2,1],hspace=0.02,
                                              wspace=0.02)
    ax1 = plt.subplot(gs_ext[0])
    plot_inf = WLC.peg_contribution(model_f=model_f)
    print("For {:s}...".format(title))
    for k,v in plot_inf.kw.items():
        print("{:s}={:.5g}".format(k,v))
    labels = ["Unfolded\npeptide","PEG"]
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
    for i, f_tmp in enumerate([100, 250]):
        W_int = plot_inf.W_at_f(f_tmp)
        label = "{:d} kcal/mol at {:d} pN".format(W_int, f_tmp)
        plt.axhline(W_int, label=label, color=colors[i],linestyle='--')
    xlabel = "$F$ (pN) [! not extension !]"
    PlotUtilities.lazyLabel(xlabel, W_str + "\n(kcal/mol)", "")


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
    make_model_plot(model_f=model, title=descr)
    PlotUtilities.savefig(fig,"PEG_{:s}.png".format(descr))

if __name__ == "__main__":
    run()
