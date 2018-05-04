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
from Lib.AppWLC.UtilFit import fit_base
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import minimize

kbT = 4.1e-21


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

class FitFJCandWLC(object):
    def __init__(self, brute_dict, x0, f, minimize_dict, res, output_brute):
        self.brute_dict = brute_dict
        self.x0 = x0
        self.min_f = min(f)
        self.max_f = max(f)
        self.n_f = f.size
        self.minimize_dict = minimize_dict
        self.output_polishing = res
        self.output_brute = output_brute
        self.x_offset = 0

    def set_x_offset(self, x):
        """
        :param x: x, defined with one point at each point in f_grid
        :return: nothing, sets the x ffset
        """
        assert x.size == self.n_f
        self.x_offset = x

    @property
    def f_grid(self):
        return np.linspace(self.min_f, self.max_f, endpoint=True,
                           num=self.n_f)

    @property
    def ext_grid(self, f_grid=None):
        if f_grid is None:
            f_grid = self.f_grid
        # get the force and extension grid again, with the optimized parameters
        ext_grid, _ = _hao_ext_grid(f_grid, *self.x0)
        return ext_grid

    @property
    def component_grid(self,f_grid=None):
        f_grid = self.f_grid if f_grid is None else f_grid
        _, ext_components = _hao_ext_grid(f_grid, *self.x0)
        return ext_components
    @property
    def _Ns(self):
        return self.x0[0]

    @property
    def _K(self):
        return self.x0[1]

    @property
    def _L_K(self):
        return self.x0[2]

    @property
    def L0_c_terminal(self):
        return self.x0[3]

    @property
    def L0_PEG3400(self):
        """
        :return: contour length of the PEG3400
        """
        # align everything to the PEG3400 contour length.
        N_monomers = self._Ns
        L0_PEG3400_per_monomer = common_peg_params()['L_helical']
        L0_PEG3400 = L0_PEG3400_per_monomer * N_monomers
        L0_correct = L0_PEG3400
        return L0_correct


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
    coth = lambda x: 1/np.tanh(x)
    to_ret =  N_s * ((L_planar  / (np.exp(-DeltaG/kbT) + 1)) + \
                      L_helical / (np.exp(+DeltaG/kbT) + 1)) * \
                     (coth(F*L_K/kbT) - kbT/(F*L_K)) + \
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

def Hao_PEGModel(F,N_s=25.318,K=906.86,L_K=0.63235e-9,L0_Protein=27.2e-9,
                 Lp=0.4e-9):
    """
    see: communication with Hao, 
    """
    common = dict(N_s=N_s,K=K,L_K=L_K,**common_peg_params())
    # get the FJC model of *just* the PEG
    ext_FJC = HaoModel(F=F, **common)
    # get the WLC model of the unfolded polypeptide
    L0 = L0_Protein
    polypeptide_args = dict(kbT=kbT,Lp=Lp,L0=L0,K0=10000e-12)
    ext_wlc, F_wlc = WLC._inverted_wlc_helper(F=F,odjik_as_guess=True,
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

def _hao_ext_grid(force_grid,*args,**kw):
    """
    :param force_grid: forces to get the extension at
    :param args: passed to Hao_PEGModel
    :return: tuple of (total extension, [FJC extension, WLC extension))
    """
    ext, _ = Hao_PEGModel(force_grid,*args,**kw)
    # add the FJC and WLC extensions.
    ext_grid = ext[0] + ext[1]
    return ext_grid, ext

def _hao_fit_helper(x,f,force_grid,*args,**kwargs):
    ext_grid,_  =_hao_ext_grid(force_grid,*args,**kwargs)
    l2 = fit_base._l2_grid_to_data(x,f,ext_grid,force_grid)
    return l2

def predicted_f_at_x(x,ext_grid,f_grid):
    to_ret = fit_base._grid_to_data(x,ext_grid,f_grid)
    return to_ret


def _constrained_L2(L2,bounds,*args):
    raw_L2 = L2(*args)
    in_bounds = [(a >= b[0] and a <= b[1]) for a,b in zip(args[0],bounds)]
    if False in in_bounds:
        # articially increase the error
        return raw_L2 * 1000
    else:
        return raw_L2

def hao_fit(x,f):
    # write dfown the ranges for everything
    range_N = (0,250)
    range_K = (50,2500)
    range_L_K = (0.1e-9,4e-9)
    range_L0 = (10e-9,40e-9)
    Lp = 0.4e-9
    f_grid = np.linspace(min(f),max(f),endpoint=True,num=f.size)
    functor_l2 = lambda *args: _hao_fit_helper(x,f,f_grid,*(args[0]),Lp=Lp)
    # how many brute points should we use?
    ranges = (range_N,range_K,range_L_K,range_L0)
    n_pts = [5 for _ in ranges]
    # determine the step sizes in each dimension
    steps = [ (r[1]-r[0])/n_pts[i] for i,r in enumerate(ranges)]
    # determine the slice in each dimension
    slices = [slice(r[0],r[1],step) for r,step in zip(ranges,steps)]
    # bounds...
    bounds = [ (s.start,s.stop) for s in slices]
    # initially, dont polish
    brute_dict = dict(ranges=slices,finish=None,
                      Ns=None,full_output=True)
    output_brute = fit_base._prh_brute(objective=functor_l2,**brute_dict)
    x0_brute = output_brute[0]
    # take the brute result, and polish it within the bounds
    opts = dict(ftol=1e-3,xtol=1e-3,maxfev=int(1e4))
    minimize_dict = dict(x0=x0_brute,method='Nelder-Mead',options=opts)
    # get the bounds we infer from brute; we look within the N-dimensional
    # cube
    bounds_brute = [ (a-step,a+step) for a,step in zip(x0_brute,steps)]
    # make sure the minimum and maximum are within the original bounds
    bounds_brute = [ (max(b1[0],b2[0]),min(b1[1],b2[1]))
                     for (b1,b2) in zip(bounds_brute,bounds)]
    bounded_fun = lambda *args: _constrained_L2(functor_l2,bounds_brute,*args)
    res = minimize(fun=bounded_fun,**minimize_dict)
    x0 = res.x
    for b,x in zip(bounds,x0):
        assert (x <= b[1]) and (x >= b[0])  , "Minimization didn't constrain"
    # get the force and extension grid, interpolated back to the original data
    to_ret = FitFJCandWLC(brute_dict=brute_dict,x0=x0,f=f,
                          output_brute=output_brute,
                          minimize_dict=minimize_dict,res=res)
    return to_ret

def _read_csv_fec(f):
    """
    :param f: filename, should have columns like (ext in nm, force in pN)
    :return:  ext,force
    """
    arr = np.genfromtxt(f,delimiter=",")
    ext, F = arr.T
    return ext,F

def read_haos_data():
    """
    :return: tuple of <ext,F> for Hao's *total* model
    """
    hao_file = "../FigData/HaosFEC.csv"
    return _read_csv_fec(hao_file)

def read_hao_polypeptide():
    """
    :return: see  read_haos_data, except just the WLC (polypeptide) part
    """
    file_polypeptide = "../FigData/HaosFEC_Polypeptide.csv"
    return _read_csv_fec(file_polypeptide)

def _make_plot_inf(ext_grid,read_functor):
    """
    :param ext_grid: grid we want the extension on
    :param read_functor: no arguments, call to get ext, F
    :return:
    """
    ext, F = read_functor()
    interp = interp1d(x=ext*1e-9, y=F*1e-12, kind='linear',
                      fill_value='extrapolate')
    Hao_F = interp(ext_grid * 1e-9)
    to_ret =  _plot_info_helper(x=[ext_grid*1e-9], F=Hao_F,
                                kw=dict(), model_f=read_haos_data)
    return to_ret



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
    F = np.linspace(1e-20, 300e-12, num=1000)
    plot_inf = get_plot_info(F=F,**kw)
    return plot_inf