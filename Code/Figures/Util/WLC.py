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

from Lib.AppWLC.Code import WLC, WLC_Utils
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


kbT = 4.1e-21


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

def common_peg_params():
    to_ret = dict(L_planar = 0.358e-9, L_helical = 0.28e-9,kbT = kbT,
                  DeltaG = 3 * kbT)
    return to_ret


def Oesterhelt_PEGModel(F):
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
    common = dict(N_s = 77,K=150,L_K=0.7e-9,**common_peg_params())
    to_ret = HaoModel(F=F,**common)
    return [to_ret], common

def grid_interp(points,values,grid):
    interp = interp1d(x=points,y=values,kind='linear',
                      fill_value='extrapolate',bounds_error=False)
    to_ret = interp(grid)
    return to_ret

def grid_both(x,x_a,a,x_b,b):
    """
    :param x: what grid we want, length N
    :param x_a: current grid for a
    :param a: value for a on that grid
    :param x_b: current grid for b
    :param b:  values on b on that grid
    :return: tuple of <grid_a,grid_b>, each length of N
    """
    grid_a = grid_interp(points=x_a,values=a,grid=x)
    grid_b = grid_interp(points=x_b,values=b,grid=x)
    return grid_a, grid_b

def Hao_PEGModel(F):
    """
    see: communication with Hao, 
    """
    common = dict(N_s=25.318,K=906.86,L_K=0.63235e-9,**common_peg_params())
    # get the FJC model of *just* the PEG
    ext_FJC = HaoModel(F=F, **common)
    # get the WLC model of the unfolded polypeptide
    L0 = 27.2e-9
    polypeptide_args = dict(kbT=kbT,Lp=0.4e-9,L0=L0,K0=10000e-12)
    non_ext_polypeptide_args = dict(disable_correction=False,**polypeptide_args)
    non_ext_polypeptide_args['K0'] = np.inf
    ext_wlc = np.linspace(0,L0 * 0.9)
    F_wlc = WLC_Utils.WlcNonExtensible(ext=ext_wlc,
                                       **non_ext_polypeptide_args)
    F_wlc = WLC_Utils.WlcExtensible_Helper(ext=ext_wlc,F=F_wlc,
                                           **polypeptide_args)
    valid_idx = np.where(ext_wlc > 0)
    ext_wlc = ext_wlc[valid_idx]
    F_wlc = F_wlc[valid_idx]
    # create the interpolator of total extension vs force. First, interpolate
    ext_FJC_grid, ext_WLC_grid = grid_both(x=F, x_a=F, a=ext_FJC, x_b=F_wlc,
                                           b=ext_wlc)
    to_ret = [ext_FJC_grid,ext_WLC_grid]
    # the extensions and forces to the same grid
    return to_ret, common

class plot_info:
    def __init__(self,ext,f,w,kw,func):
        self.qs = ext
        self.f = f
        self.w = w
        self.kw = kw
        self.func = func
    @property
    def q(self):
        to_ret = np.sum(self.qs,axis=0)
        return to_ret
    def W_at_f(self,f_tmp):
        """
        :param f_tmp: force, in plot units (pN, prolly)
        :return:  work at that force
        """
        idx_f = np.argmin(np.abs(self.f - f_tmp))
        W_f = self.w[idx_f]
        W_int = np.round(int(W_f))
        return W_int

def _plot_info_helper(x,F,kw,model_f):
    total_x = np.sum(x,axis=0)
    work = cumtrapz(x=total_x,y=F,initial=0)
    # make out plot, units of nanometers, piconewtons, kcal/mol
    ext_plot = np.array(x) * 1e9
    f_plot = F * 1e12
    w_plot = (work / 4.1e-21) * 0.593
    return plot_info(ext_plot,f_plot,w_plot,func=model_f,kw=kw)

def get_plot_info(F,model_f=Hao_PEGModel):
    x_PEG,kw = model_f(F=F)
    return _plot_info_helper(x_PEG,F,kw,model_f)

def peg_contribution(**kw):
    ext = np.linspace(0, 25e-9, num=1000)
    F = np.linspace(1e-20, 275e-12, num=1000)
    plot_inf = get_plot_info(F=F,**kw)
    return plot_inf