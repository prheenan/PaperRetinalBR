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
    plt.plot(plot_inf.q,plot_inf.f)
    PlotUtilities.lazyLabel("","$F$ (pN)","")
    PlotUtilities.no_x_label(ax1)
    plt.xlim(xlim)
    ax2 = plt.subplot(3,1,2)
    plt.plot(plot_inf.q,plot_inf.w,label="W$_{\mathbf{PEG}}$")
    plt.xlim(xlim)
    PlotUtilities.lazyLabel("Extension (nm)","$W$ (kcal/mol)","")
    ax2 = plt.subplot(3,1,3)
    plt.plot(plot_inf.f,plot_inf.w,'--',label="W$_{\mathbf{PEG}}$")
    for f_tmp in [150,250]:
        W_int = plot_inf.W_at_f(f_tmp)
        label = "{:d} kcal/mol at {:d} pN".format(W_int,f_tmp)
        plt.axhline(W_int,label=label)
    PlotUtilities.lazyLabel("$F$ (pN)","$W$ (kcal/mol)","")

    PlotUtilities.savefig(fig,"PEG.png")

if __name__ == "__main__":
    run()
