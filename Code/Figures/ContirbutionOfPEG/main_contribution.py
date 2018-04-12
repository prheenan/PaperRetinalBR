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



def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """

    xlim = [0,27]
    fig = PlotUtilities.figure((3.5,4))
    ax1 = plt.subplot(3,1,1)
    plot_inf = WLC.peg_contribution()
    plt.plot(plot_inf.q,plot_inf.f,color='m')
    PlotUtilities.lazyLabel("","$F$ (pN)","")
    PlotUtilities.no_x_label(ax1)
    plt.xlim(xlim)
    ax2 = plt.subplot(3,1,2)
    plt.plot(plot_inf.q,plot_inf.w,)
    plt.xlim(xlim)
    PlotUtilities.lazyLabel("Extension (nm)","$W_{PEG}$ (kcal/mol)","")
    ax2 = plt.subplot(3,1,3)
    plt.plot(plot_inf.f,plot_inf.w,'--',color='r')
    colors = ['m','g']
    for i,f_tmp in enumerate([100,250]):
        W_int = plot_inf.W_at_f(f_tmp)
        label = "{:d} kcal/mol at {:d} pN".format(W_int,f_tmp)
        plt.axhline(W_int,label=label,color=colors[i])
    xlabel = "$F$ (pN) [! not extension !]"
    PlotUtilities.lazyLabel(xlabel,"$W_{PEG}$ (kcal/mol)","")

    PlotUtilities.savefig(fig,"PEG.png")

if __name__ == "__main__":
    run()
