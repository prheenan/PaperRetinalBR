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

from Lib.AppWLC.Code import WLC
from scipy.integrate import cumtrapz

class plot_info:
    def __init__(self,ext,f,w):
        self.q = ext
        self.f = f
        self.w = w
    def W_at_f(self,f_tmp):
        """
        :param f_tmp: force, in plot units (pN, prolly)
        :return:  work at that force
        """
        idx_f = np.argmin(np.abs(self.f - f_tmp))
        W_f = self.w[idx_f]
        W_int = np.round(int(W_f))
        return W_int

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
    ext = np.linspace(0, 25e-9, num=1000)
    F = np.linspace(0, 500e-12, num=1000)
    # see:  Oesterhelt, Rief, and Gaub (New J. Phys. 1, 6.1-6.11, 1999)
    L0_per_monomer = 0.36e-9
    Lp = 0.4e-9
    K0 = np.inf
    kbT = 4.1e-21
    N_monomers = 77
    L0 = N_monomers * L0_per_monomer
    plot_inf = get_plot_info(ext=ext, F=F, Lp=Lp, K0=K0, kbT=kbT, L0=L0)
    return plot_inf