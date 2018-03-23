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
def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    ext = np.linspace(0,30e-9,num=100)
    F = np.linspace(0,500e-12,num=100)
    # see:  Oesterhelt, Rief, and Gaub (New J. Phys. 1, 6.1-6.11, 1999)
    L0_per_monomer_low_force = 0.28e-9
    L0_per_monomer = 0.36e-9
    Lp = 0.3e-9
    K0 = 1000e-12
    kbT = 4.1e-21
    N_monomers = 77
    L0_3400 = N_monomers * L0_per_monomer
    ext_grid_final,force_grid_final,Force= \
        WLC._inverted_wlc_full(ext=ext,kbT=kbT,Lp=Lp,L0=L0_3400,
                               K0=K0,F=F,odjik_as_guess=True)
    ext = ext_grid_final
    f = force_grid_final
    work = cumtrapz(x=ext,y=f,initial=0)
    # make out plot, units of nanometers, piconewtons, kcal/mol
    ext_plot = ext * 1e9
    f_plot = f * 1e12
    w_plot = (work / 4.1e-21) * 0.593
    fig = PlotUtilities.figure((2.5,4))
    ax1 = plt.subplot(2,1,1)
    plt.plot(ext_plot,f_plot)
    PlotUtilities.lazyLabel("","$F$ (pN)","")
    PlotUtilities.no_x_label(ax1)
    ax2 = plt.subplot(2,1,2)
    plt.plot(ext_plot,w_plot,label="Total work")
    PlotUtilities.lazyLabel("Extension (nm)","$W$ (kcal/mol)","")
    PlotUtilities.savefig(fig,"PEG.png")

if __name__ == "__main__":
    run()
