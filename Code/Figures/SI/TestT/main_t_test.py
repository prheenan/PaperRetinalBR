# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.stats import t as t_distribution
from scipy.stats import ttest_ind_from_stats

def simulated_mean_stdev(mu,stdev,N,exact=True):
    to_ret = np.random.standard_normal(N)
    to_ret = to_ret * stdev + mu
    if exact:
        to_ret = exact_mean_and_stdev(to_ret,mu,stdev)
    return to_ret

def repeated_simulation(N_repeats,*args,**kwargs):
    to_ret = [simulated_mean_stdev(*args,**kwargs) for _ in range(N_repeats)]
    return to_ret

def exact_mean_and_stdev(samples,mu,sigma):
    samples_zero_mean = samples - np.mean(samples)
    samples_zero_mean_one_stdev = samples_zero_mean/np.std(samples_zero_mean)
    return samples_zero_mean_one_stdev * sigma + mu

def errs_and_N():
    # write down the mean and error in the force (pN) for wild type (WT) BR
    F_WT_err = np.array([3.3045661,2.3268523,4.248373])
    F_WT_mean = np.array([99.661331,120.63016,139.46082])
    N_WT = np.array([17,32,21])
    F_WT_loading = np.array([303.33878,2657.5464,23427.143])
    # convert to stdev
    F_WT_err *= np.sqrt(N_WT-1)
    # same for photocleaved ('PC') BR
    F_PC_mean = np.array([96.828041,110.24906,138.62259])
    F_PC_err = np.array([2.6289949,3.9497223,3.8225975])
    N_PC = np.array([16,25,17])
    F_PC_err *= np.sqrt(N_PC-1)
    F_PC_loading = np.array([317.28918,2512.8311,19653.908])
    return F_WT_err, F_WT_mean, N_WT, F_WT_loading,F_PC_mean, F_PC_err, N_PC, F_PC_loading

def print_individual_p_values():
    F_WT_err, F_WT_mean, N_WT, _, F_PC_mean, F_PC_err, N_PC, _ = errs_and_N()
    # Since we have approximately normally distributed rupture force for a
    # fixed loading rate, we can
    # use Welch's formula for getting the t-test value with different
    # population variances . See: Welch, Biometrika, 1947
    # t = mean_WT - mean_PO / sqrt( stdev_WT**2/N_WT + stdev_PC**2/N_PC)
    t_denom = np.sqrt( F_WT_err**2/N_WT + F_PC_err**2/N_PC)
    t = (F_WT_mean - F_PC_mean) / t_denom
    # get the degrees of freedom asscioated with the system using the
    # Welch-satterthwaite eq. See: Satterthwaite, 1946, Biometrics Bulletin
    v_denom = (F_WT_err**4/(N_WT**2 * (N_WT-1)) + F_PC_err**4/(N_PC**2*(N_PC-1)))
    v = t_denom**4 / v_denom
    # determine the p value based on degrees of freedom and the t statistic
    p_value_one_sided = 1-t_distribution.cdf(t,df=v)
    p_value_two_sided = 2 * p_value_one_sided
    # as a check, use scientific python to calculate the same thing
    t_stat_and_p = [ttest_ind_from_stats(\
        mean1=F_WT_mean[i],std1=F_WT_err[i],nobs1=N_WT[i],
        mean2=F_PC_mean[i],std2=F_PC_err[i],nobs2=N_PC[i],equal_var=False)
                    for i in range(3)]
    t_stat = [ele[0] for ele in t_stat_and_p]
    p_values = [ele[1] for ele in t_stat_and_p]
    print("Manually calculated p-values: " + \
          ",".join((["{:.3g}".format(p) for p in p_value_two_sided])))
    print("Automatically calculated p-values: " + \
          ",".join(["{:.3g}".format(p) for p in p_values]))

def run():
    """
    Calculates the two-sided p values for the ED helix
    """
    # print the individual p values
    print_individual_p_values()



if __name__ == "__main__":
    run()
