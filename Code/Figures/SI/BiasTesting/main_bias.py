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

from Processing import ProcessingUtil
import RetinalUtil,PlotUtil
from Figures import FigureUtil

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
    data_all = [e for e in data_BO[0]]
    work_list = [e.Work for e in data_all]
    from Code.Lib.AppIWT.Code.UtilLandscape import BidirectionalUtil
    beta = 1 / 4.1e-21
    N = len(work_list)
    work_all = np.array(work_list)
    sigma_w = np.std(work_all, axis=0)
    mean_w = np.mean(work_all, axis=0)
    n_sample_arr = np.array([i for i in range(1,11)])
    n_rounds = 50
    bootstrap = _bootstrap_W_dif(work_list, beta, n_sample_arr=n_sample_arr,
                                 n_rounds=n_rounds)
    idx_of_interest = 1400
    W_dis_of_n_sample = np.array([b.W_dis_bar[idx_of_interest]
                                  for b in bootstrap])
    W_dis = np.array(W_dis_of_n_sample)
    C = np.logspace(10,130,base=10,num=500,endpoint=True)
    diffs = np.sum([np.abs(bias_difference(C, W_tmp, beta))
                    for W_tmp in W_dis],axis=0)
    abs_diffs = np.abs(diffs)
    C_best_idx = np.argmin(abs_diffs)
    C_best = C[C_best_idx]
    plt.close()
    plt.semilogx(C, np.log(diffs))
    plt.axvline(C_best)
    plt.show()
    F_J_hat = _F_J_hat(work_all,beta)
    # just before eq 18
    W_bar_hat_dis = mean_w - F_J_hat
    # just before eq 19
    C = C_best
    W_bar_dis = 0.5 * beta * sigma_w ** 2
    alpha = np.log(2 * beta * C * W_bar_dis) /\
            np.log(C * (np.exp(2 * beta * W_bar_dis) - 1))
    alpha[0] = 0
    W_bar_hat_dis_2 = mean_w - F_J_hat + W_bar_hat_dis / (N ** alpha)
    B_J_2 = W_bar_hat_dis_2 / N**(alpha)
    B_J_1 = W_bar_hat_dis / N**alpha
    plt.close()
    for w in work_all:
        plt.plot(w * beta)
    plt.plot(sigma_w * beta, 'r--')
    plt.plot(mean_w * beta, 'b:')
    plt.plot(F_J_hat * beta, 'g-', linewidth=3)
    plt.plot((F_J_hat-B_J_2)*  beta,'m')
    plt.show()

if __name__ == "__main__":
    run()
