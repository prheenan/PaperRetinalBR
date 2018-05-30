# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import re

sys.path.append("../../../")
from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
from Lib.AppWHAM.Code.UtilLandscape import BidirectionalUtil
import RetinalUtil,PlotUtil
import matplotlib.gridspec as gridspec
import re
from Lib.UtilPipeline import Pipeline

from Figures import FigureUtil
from scipy.integrate import trapz
import warnings

def Jaryzynski_A(x):
    return x.A_z

def _mean_f(f,energy_list_arr,q_interp):
    means = []
    for i, energy_list in enumerate(energy_list_arr):
        _, splines = RetinalUtil.interpolating_G0(energy_list,
                                                  f=f)
        mean, _ = PlotUtil._mean_and_stdev_landcapes(splines, q_interp)
        means.append(mean)
    return means

def mean_A_jarzynski(energy_list_arr,q_interp):
    f = lambda x_tmp : x_tmp.A_z
    return _mean_f(f,energy_list_arr, q_interp)


def mean_G_iwt(energy_list_arr, q_interp):
    f = lambda x_tmp : x_tmp.G0
    return _mean_f(f, energy_list_arr, q_interp)

def mean_A_dot_iwt(energy_list_arr, q_interp):
    f = lambda x_tmp : x_tmp.A_z_dot
    return _mean_f(f, energy_list_arr, q_interp)

def Exp(arg,tol=700):
    """
    :param arg: argument to exp
    :param tol: maximum exp
    :return:  exp(arg), except where  |arg| > tol
    """
    high_idx = np.where(arg > tol)
    low_idx = np.where(arg < -tol)
    safe_idx = np.where( (arg < tol) & (arg > -tol))
    to_ret = np.zeros(arg.shape)
    if (low_idx[0].size > 0):
        to_ret[low_idx] = 1
    if (high_idx[0].size > 0):
        to_ret[high_idx] = np.inf
    if (safe_idx[0].size > 0):
        to_ret[safe_idx] = np.exp(arg[safe_idx])
    else:
        warnings.warn("No safe idx..")
    return to_ret

def _G_iter(G0,A,q,z,k,beta):
    """
    :param G0: previous G, in J
    :param A: jarzynski A, in J
    :param q: extension, in m
    :param z: stage position, in m
    :param k: stiffness, N/m
    :param beta: 1/kT
    :return:
    """
    n_z = z.size
    n_q = q.size
    qq, zz = np.meshgrid(q,z)
    V = k * (qq - zz)**2/2
    arg_denom = -beta * (G0 + V)
    arg_numer = -beta * (A + V)
    # use a constant offset at each z...
    offset = np.mean([np.mean(a,axis=1) for a in [arg_denom,arg_numer]],
                     axis=0)
    sanit = lambda x_tmp: (x_tmp.T - 0).T
    arg_denom_offset = sanit(arg_denom)
    arg_numer_offset = sanit(arg_numer)
    boltz_denom = Exp(arg_denom_offset)
    boltz_numer = Exp(arg_numer_offset)
    assert boltz_denom.shape == (n_z,n_q)
    denom_int = trapz(x=q, y=boltz_denom,axis=1)
    assert denom_int.size == n_z
    fraction = (boltz_numer.T / denom_int).T
    assert fraction.shape == (n_z,n_q)
    fraction_int = trapz(x=z, y=fraction,axis=0)
    assert fraction_int.size == n_q
    G_next = G0 - (1/beta) * np.log(fraction_int)
    return G_next

def lucy_richardson(G0,A,q,z,k,beta,n_iters=100):
    """
    :param G0: see LucyRichardson object
    :param A: see LucyRichardson object
    :param q: see LucyRichardson object
    :param z: see LucyRichardson object
    :param k: see LucyRichardson object
    :param beta: see LucyRichardson object
    :param n_iters: number of iterations to perform
    :return:
    """
    iters = []
    G_tmp = G0.copy()
    for i in range(n_iters):
        G_prev = G_tmp.copy()
        iters.append(G_prev)
        G_tmp = _G_iter(G_prev, A, q, z, k, beta)
    Gf = G_tmp.copy()
    iters.append(Gf)
    return LucyRichardson(A=A,q=q,z=z,k=k,beta=beta,G0=G0,Gf=Gf,G_iters=iters)

class LucyRichardson(BidirectionalUtil._BaseLandscape):
    def __init__(self,A,q,z,k,beta,G0,Gf,G_iters):
        """
        :param A: jarzynski free energy in J
        :param q: extension, in m
        :param z: control parameter (e.g. stage pos), in m
        :param k: stiffness, in pN/nm
        :param beta: 1/kT, in J
        :param G0: initial guess, in J
        :param Gf: final energy, in J
        :param G_iters: energies from and including G0 to Gf
        """
        self.A = A
        self.q = q
        self.z = z
        self.k = k
        self.beta = beta
        self.Gf = Gf
        self.kw_base = dict(beta=beta,q=q)
        self._G_iters = G_iters
        self._G0_initial_lr = self.landscape_at_idx(0)
        # set up this as a landscape with the final itertation
        super(LucyRichardson,self).__init__(G0=Gf,**self.kw_base)
    def landscape_at_idx(self,idx):
        return BidirectionalUtil._BaseLandscape(G0=self._G_iters[idx],
                                                **self.kw_base)
    @property
    def mean_diffs(self,):
        to_ret = [ np.mean(np.abs(G2-G1))
                   for G1,G2 in zip(self._G_iters[:-1],self._G_iters[1:])]
        return to_ret
    @property
    def n_iters(self):
        # we include the first
        return len(self._G_iters) - 1

def _deconvoled_lr(**kw):
    lr = lucy_richardson(**kw)
    return lr

def _plot_f_at_iter_idx(lr,i,**kw):
    converged = lr.landscape_at_idx(i)
    q = lr.q
    fit_conv = RetinalUtil.spline_fit(q=q, G0=converged.G0)
    fit_landscape = fit_conv(q)
    # re-zero
    fit_landscape -= min(fit_landscape)
    F = fit_conv.derivative(n=1)(q)
    plt.plot(q * 1e9, F * 1e12,**kw)

def run():
    input_dir = "../../../../Data/FECs180307/"
    out_dir = "./"
    q_offset_nm = 100
    q_interp, energy_list_arr = FigureUtil.\
        _read_energy_list_and_q_interp(input_dir, q_offset=q_offset_nm,
                                       min_fecs=9)
    # read in some example data
    base_dir_BR = input_dir + "BR+Retinal/"
    names_BR = ["/BR+Retinal/3000nms/170503FEC/landscape_"]
    PEG600 = FigureUtil._read_samples(base_dir_BR,names_BR)[0]
    # read in the FECs
    fecs = FigureUtil._snapsnot(PEG600.base_dir,
                                step=Pipeline.Step.REDUCED).fec_list
    args_mean = (energy_list_arr, q_interp)
    mean_A = mean_A_jarzynski(*args_mean)
    mean_G = mean_G_iwt(*args_mean)
    mean_F_from_dot = mean_A_dot_iwt(*args_mean)[0]
    ex = fecs[0]
    q = q_interp * 1e-9
    # xxx offset q and z... probably a little off!
    z = ex.ZFunc(ex) + q[0]
    k = ex.SpringConstant
    beta = ex.Beta
    A = mean_A[0]
    G = mean_G[0]
    n_iters = 5000
    force =True
    kw_lr = dict(G0=G, A=A, q=q, z=z, k=k, beta=beta,n_iters=n_iters)
    lr = CheckpointUtilities.getCheckpoint("./lr_deconv.pkl",
                                           _deconvoled_lr,force,**kw_lr)
    diff_kT = np.array(lr.mean_diffs) * beta
    min_idx = np.argmin(diff_kT)
    idx_search = np.logspace(start=3,stop=np.floor(np.log2(min_idx)),
                             endpoint=True,num=10,base=2)
    idx_to_use = [int(i) for i in idx_search]
    # fit a spline to the converged G to get the mean restoring force
    fig = PlotUtilities.figure((4,6))
    xlim = [min(q_interp)-5, max(q_interp)]
    fmt_kw = dict(xlim=xlim,ylim=[None,None])
    ax1 = plt.subplot(3,1,1)
    plt.plot(diff_kT)
    plt.axvline(min_idx)
    FigureUtil._plot_fmt(ax1,is_bottom=True,xlabel="iter #",
                         ylabel="diff G (kT)",xlim=[None,None],ylim=[None,None])
    ax2 = plt.subplot(3,1,2)
    plt.plot(q_interp,lr._G0_initial_lr.G0_kT,color='b',linewidth=3)
    plt.plot(q_interp,lr.G0_kT,color='r',linewidth=3)
    FigureUtil._plot_fmt(ax2,is_bottom=False,ylabel="G (kT)",**fmt_kw)
    ax1 = plt.subplot(3,1,3)
    FigureUtil._plot_fec_list(fecs, xlim, ylim=[None,None])
    for i in idx_to_use:
        _plot_f_at_iter_idx(lr, i)
    _plot_f_at_iter_idx(lr, 0,label="F from G0",linewidth=4)
    plt.plot(q_interp,mean_F_from_dot*1e12,label="F from A_z_dot",linewidth=2)
    # also plot the force we expect from the original A_z_dot
    FigureUtil._plot_fmt(ax1,is_bottom=True,xlim=xlim,ylim=[None,None])
    PlotUtilities.legend()
    PlotUtilities.savefig(fig,"FigureS_A_z.png")
    pass


if __name__ == "__main__":
    run()
