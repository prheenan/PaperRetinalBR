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
from Lib.AppIWT.Code.UtilLandscape import BidirectionalUtil
from Lib.AppWLC.UtilFit import fit_base

from Processing import ProcessingUtil
import RetinalUtil,PlotUtil
from Figures import FigureUtil
from Lib.UtilForce.UtilGeneral import PlotUtilities, CheckpointUtilities
from scipy.optimize import minimize, LinearConstraint
from scipy.integrate import trapz
from scipy.integrate import trapz


class BootstrapInfo(object):
    def __init__(self,F_J_hat,W_mean,n_samples_per_round):
        self._F_J_hat = F_J_hat
        self._W_mean = W_mean
        self.n_samples_per_round = n_samples_per_round
    @property
    def W_dis(self):
        return self._W_mean - self._F_J_hat

class BootstrapSeries(object):
    def __init__(self,list_of_info):
        self.list_of_info = list_of_info
    @property
    def W_dis_bar(self):
        return np.mean([e.W_dis for e in self.list_of_info],axis=0)

class PalassiniFit(object):
    def __init__(self,brute_dict,kw_common,res_polish,Ws_kT,res_brute):
        self.brute_dict = brute_dict
        self.kw_common = kw_common
        self.res_polish = res_polish
        self.Ws_kT = Ws_kT
        self.res_brute = res_brute
    @property
    def x_final(self):
        return self.res_polish.x
    @property
    def W_c(self):
        return self.x_final[0]
    @property
    def Omega(self):
        return self.x_final[1]
    @property
    def delta(self):
        return self.kw_common['delta']
    @property
    def alpha(self):
        return self.kw_common['alpha']
    @property
    def N(self):
        return self.Ws_kT.size

def _F_J_hat(W_choice,beta):
    exp_arg = BidirectionalUtil.Exp(-W_choice * beta, tol_v=0)
    F_J_hat = -1 / beta * (np.log(np.mean(exp_arg, axis=0)))
    return F_J_hat

def _bootstrap_W_dis(W,beta,n_samples_per_round,n_rounds):
    to_ret = []
    idx = [i for i in range(len(W))]
    choice_idx = np.random.choice(idx, size=(n_rounds,n_samples_per_round),
                                  replace=True)
    for i in range(n_rounds):
        W_choice = np.array([W[j] for j in choice_idx[i]])
        W_mean = np.mean(W_choice,axis=0)
        F_J_hat = _F_J_hat(W_choice,beta)
        to_ret.append(BootstrapInfo(F_J_hat=F_J_hat,W_mean=W_mean,
                                    n_samples_per_round=n_samples_per_round))
    return BootstrapSeries(to_ret)

def _bootstrap_W_dif(W,beta,n_sample_arr,n_rounds):
    to_ret = []
    for n in n_sample_arr:
        tmp = _bootstrap_W_dis(W=W,beta=beta,n_samples_per_round=n,
                               n_rounds=n_rounds)
        to_ret.append(tmp)
    return to_ret

def bias_difference(C,W_dis,beta):
    alpha = np.log(2 * beta * C * W_dis) / \
            np.log(C * (np.exp(2 * beta * W_dis) - 1))
    N_c = C * (np.exp(2 * beta * W_dis) - 1)
    large_N_bias_kT = (np.exp(2 * beta * W_dis) - 1) / (2 * N_c)
    small_N_bias_kT = beta * W_dis / N_c**(alpha)
    return small_N_bias_kT - large_N_bias_kT

def unnormalized_prob(W,W_c,Omega,alpha,delta):
    to_ret = (Omega **(alpha-1) / np.abs(W-W_c)**alpha) * \
        np.exp(-((np.abs(W-W_c)/Omega)**delta))
    return to_ret

def _ll_model(W,W_int,*params,**kwargs):
    args = params[0]
    probabilities = unnormalized_prob(W,*args,**kwargs)
    P_total = unnormalized_prob(W_int,*args,**kwargs)
    integ_constant = trapz(x=W_int,y=P_total)
    P = probabilities / integ_constant
    P[np.where(P >= 1)] = 1
    log_P = np.log(P)
    ll = -1 * sum(log_P)
    if (not np.isfinite(ll)):
        return np.inf
    return ll

class CallableFitLL(object):
    def __init__(self,Ws_kT,W_int,kw_common):
        self.Ws_kT = Ws_kT
        self.W_int = W_int
        self.kw_common = kw_common
    def __call__(self,*args):
        return _ll_model(self.Ws_kT,self.W_int,*args,**self.kw_common)


def fit_W_dist(Ws,beta):
    Ws_kT = Ws * beta
    min_W_kT, max_W_kT = min(Ws_kT),max(Ws_kT)
    range_W_kT = (max_W_kT - min_W_kT)
    W_mean = np.mean(Ws_kT)
    W_std = np.std(Ws_kT)
    W_int = np.linspace(min_W_kT-range_W_kT,max_W_kT + range_W_kT)
    kw_common = dict(delta=2,alpha=0)
    objective = CallableFitLL(Ws_kT,W_int,kw_common)
    f_Omega = 4
    range_Omega = [W_std / f_Omega, W_std * f_Omega]
    range_alpha = [2,10]
    range_W_c = [min_W_kT,max_W_kT]
    range_delta = [2,10]
    ranges = [range_W_c,range_Omega]
    N_per_slice = 50
    steps = [ (f-i)/N_per_slice for i,f in ranges]
    slices  = [ slice(i,f, delta) for delta,(i,f) in zip(steps,ranges)]
    brute_dict = dict(objective=objective, disp=False,
                      full_output=True,ranges=slices,
                      Ns=N_per_slice,finish=None)
    res_brute = fit_base._prh_brute(**brute_dict)
    new_bounds = [ [(r-s/2),(r+s/2)] for r,s in zip(res_brute[0],steps)]
    # # polish the results
    upper_bound = [ s[0] for s in new_bounds]
    lower_bound = [ s[1] for s in new_bounds]
    n = len(upper_bound)
    # get the constraints matrix
    matrix = np.zeros((n,n))
    np.fill_diagonal(matrix,val=1)
    x0 = np.mean(new_bounds,axis=1)
    min_dict = dict(fun=objective,method='Nelder-Mead',x0=x0,
                    options=dict(maxiter=int(10e3)))
    res_polish = minimize(**min_dict)
    to_ret = PalassiniFit(brute_dict=brute_dict,kw_common=kw_common,
                          res_polish=res_polish,Ws_kT=Ws_kT,
                          res_brute=res_brute)
    return to_ret

def palassini_2011_eq_5(fit_info):
    Omega = fit_info.Omega
    delta = fit_info.delta
    alpha = fit_info.alpha
    N = fit_info.N
    # calculate all the terms needed..
    gamma_E = np.euler_gamma
    mu = (delta - 1) * ((Omega / delta) ** (delta / (delta - 1)))
    lambda_scaling = (delta - 1) * np.log(N) / mu
    # XXX calculate these
    D_c = mu
    q = np.sqrt(np.pi)
    B_REM = D_c + np.log(N) - Omega * (np.log(N)) ** (1 / delta)
    eq_5_sq_bracket = gamma_E + \
                      ((1 - alpha - delta) / delta) * np.log(np.log(N)) + \
                      np.log(q / delta)
    eq_5 = B_REM - (lambda_scaling ** ((1 - delta) / delta)) * eq_5_sq_bracket
    return eq_5

def bias_fit_info(cache_file,W_of_interest,beta):
    fit_info = CheckpointUtilities.getCheckpoint(cache_file,fit_W_dist,True,
                                                 W_of_interest,beta)
    return fit_info

def _target_W(data_all,target_q_m):
    work_list = [e.Work for e in data_all]
    ext_list = [e.Extension for e in data_all]
    work_all = np.array(work_list)
    idx_of_interest = [ np.argmin(np.abs(e-target_q_m)) for e in ext_list]
    W_of_interest = np.array([w[i] for i,w in zip(idx_of_interest,work_all)])
    return W_of_interest, ext_list, work_list

def _make_plots(pre_str,W_of_interest,beta,fit_info,ext_list,work_list):
    work_all = np.array(work_list)
    Ws_kT = W_of_interest * beta
    min_W_kT, max_W_kT = min(Ws_kT), max(Ws_kT)
    x_final = fit_info.x_final
    kw_common = fit_info.kw_common
    eq_5_B_REM = palassini_2011_eq_5(fit_info)
    range_W_kT = (max_W_kT - min_W_kT)
    model_W = np.linspace(min_W_kT - range_W_kT, max_W_kT + range_W_kT, num=50)
    model_P_not_norm = unnormalized_prob(model_W, *x_final, **kw_common)
    model_P = model_P_not_norm / trapz(x=model_W, y=model_P_not_norm)
    fig = PlotUtilities.figure()
    plt.hist(Ws_kT, normed=True)
    plt.plot(model_W, model_P,label="Model")
    title = "Bias = {:.2f} kT".format(eq_5_B_REM)
    PlotUtilities.lazyLabel("W (kT)","$N$",title)
    PlotUtilities.savefig(fig,pre_str + "outhist.png")
    fig = PlotUtilities.figure()
    for e,w in zip(ext_list,work_all):
        plt.plot(e,w * beta)
    PlotUtilities.lazyLabel("Ext (m)","W (kT)","")
    PlotUtilities.savefig(fig,pre_str + "out.png")

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    input_dir = "../../../../Data/FECs180307/"
    q_target_nm = RetinalUtil.q_GF_nm_plot() - 2
    q_interp, energy_list_arr = FigureUtil.\
        _read_energy_list_and_q_interp(input_dir, q_offset=q_target_nm,
                                       min_fecs=4,remove_noisy=True)
    step = Pipeline.Step.ALIGNED
    for list_v in energy_list_arr:
        for e in list_v:
            pass
    data_BR, data_BO = [RetinalUtil._read_all_data(e) for e in energy_list_arr]
    # read in the EC histogram...
    data_all = [e for list_tmp in data_BR for e in list_tmp]
    beta = 1 / 4.1e-21
    target_q_m = 15e-9
    W_of_interest, ext_list, work_list = _target_W(data_all,
                                                   target_q_m=target_q_m)
    fit_info = bias_fit_info("./cache.pkl", W_of_interest,beta)
    _make_plots("./",W_of_interest,beta,fit_info,ext_list,work_list)



if __name__ == "__main__":
    run()
