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
from Lib.AppWLC.Code import WLC
from Lib.UtilForce.UtilGeneral import PlotUtilities
from scipy.integrate import cumtrapz

class plot_info:
    def __init__(self,ext,f,w):
        self.q = ext
        self.f = f
        self.w = w

def get_plot_info(ext,F,**kw):
    ext_grid_final,force_grid_final,Force= \
        WLC._inverted_wlc_full(ext=ext,F=F,odjik_as_guess=True,**kw)
    ext = ext_grid_final
    f = force_grid_final
    work = cumtrapz(x=ext,y=f,initial=0)
    # make out plot, units of nanometers, piconewtons, kcal/mol
    ext_plot = ext * 1e9
    f_plot = f * 1e12
    w_plot = (work / 4.1e-21) * 0.593
    return plot_info(ext_plot,f_plot,w_plot)

def peg_contribution():
    pass

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    ext = np.linspace(0,25e-9,num=1000)
    F = np.linspace(0,500e-12,num=1000)
    # see:  Oesterhelt, Rief, and Gaub (New J. Phys. 1, 6.1-6.11, 1999)
    L0_per_monomer = 0.36e-9
    Lp = 0.4e-9
    K0 = np.inf
    kbT = 4.1e-21
    N_monomers = 77
    L0 = N_monomers * L0_per_monomer
    plot_inf = get_plot_info(ext=ext,F=F,Lp=Lp,K0=K0,kbT=kbT,L0=L0)
    xlim = [0,27]
    fig = PlotUtilities.figure((3.5,4))
    ax1 = plt.subplot(3,1,1)
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
        idx_f = np.argmin(np.abs(plot_inf.f - f_tmp))
        W_f = plot_inf.w[idx_f]
        W_int = np.round(int(W_f))
        label = "{:d} kcal/mol at {:d} pN".format(W_int,f_tmp)
        plt.axhline(W_f,label=label)
    PlotUtilities.lazyLabel("$F$ (pN)","$W$ (kcal/mol)","")

    PlotUtilities.savefig(fig,"PEG.png")

if __name__ == "__main__":
    run()
