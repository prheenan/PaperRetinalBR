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


def HaoModel(N_s,L_planar,DeltaG,kbT,L_helical,F,L_K,K):
    """
    :param N_s: number of monomers
    :param L_planar:  planar length per monomer for PEG, units of m
    :param DeltaG: energy change between planar and helical, units of  J
    :param kbT: Botlxzann energy, units of J
    :param L_helical: helical length per monomer for PEG, units of m
    :param F: force, units of N
    :param L_K: kuhn length of PEG
    :param K: PEG's enthalpic stretch modulus, units of N/m
    :return:
    """
    to_ret =  N_s * ((L_planar  / (np.exp(-DeltaG/kbT) + 1)) + \
                      L_helical / (np.exp(+DeltaG/kbT) + 1)) * \
                     (np.tanh(F*L_K/kbT)**-1 - kbT/(F*L_K)) + \
              N_s * F/K
    return to_ret

def Oesterhelt_PEGModel(F):
    kbT = 4.1e-21
    """
    see: 
    Oesterhelt, F., Rief, M., and Gaub, H.E. (1999). 
    Single molecule force spectroscopy by AFM indicates helical structure of 
    poly(ethylene-glycol) in water. New Journal of Physics 1, 6-6.
    
    Particularly...
    
     The Kuhn length LK (= 7 AAA), 
     the stretching modulus (or segment elasticity) KS (= 150 N m-1) and 
     Lhelical (= 2.8 ) are fitted to the experiments with Lplanar = 3.58 
     estimated from bond lengths and angles of the planar 'all-trans' (ttt) 
     structure. 
     This results in DeltaG = (3 +/- 0.3)kBT which is consistent with prior ab 
     initio
      calculations.
    """
    common = dict(N_s = 77,L_planar=0.28e-9,L_helical=0.358e-9,
                   kbT=kbT,DeltaG=3 *kbT,K=150,L_K=0.7e-9)
    to_ret = HaoModel(F=F,**common)
    return to_ret, common

class plot_info:
    def __init__(self,ext,f,w,kw,func):
        self.q = ext
        self.f = f
        self.w = w
        self.kw = kw
        self.func = func
    def W_at_f(self,f_tmp):
        """
        :param f_tmp: force, in plot units (pN, prolly)
        :return:  work at that force
        """
        idx_f = np.argmin(np.abs(self.f - f_tmp))
        W_f = self.w[idx_f]
        W_int = np.round(int(W_f))
        return W_int

def get_plot_info(F,model_f=Oesterhelt_PEGModel):
    x_PEG,kw = model_f(F=F)
    work = cumtrapz(x=x_PEG,y=F,initial=0)
    # make out plot, units of nanometers, piconewtons, kcal/mol
    ext_plot = x_PEG * 1e9
    f_plot = F * 1e12
    w_plot = (work / 4.1e-21) * 0.593
    return plot_info(ext_plot,f_plot,w_plot,func=model_f,kw=kw)

def peg_contribution():
    ext = np.linspace(0, 25e-9, num=1000)
    F = np.linspace(1e-20, 275e-12, num=1000)
    plot_inf = get_plot_info(F=F)
    return plot_inf