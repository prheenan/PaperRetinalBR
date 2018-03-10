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
import re

sys.path.append("../../")
from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Processing import ProcessingUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil

def generate_landscape(in_dir):
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    data = UtilWHAM.to_wham_input(data)
    data.z -= min(data.z)
    data.z += np.mean([min(e) for e in data.extensions])
    energy_obj = WeightedHistogram.wham(fwd_input=data)
    return energy_obj




def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir = RetinalUtil._landscape_base()
    step = Pipeline.Step.POLISH
    in_dir = Pipeline._cache_dir(base=base_dir,
                                 enum=Pipeline.Step.REDUCED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    limit = None
    functor = lambda : generate_landscape(in_dir)
    energy_obj = CheckpointUtilities.\
        getCheckpoint(filePath=out_dir + "energy.pkl",
                      orCall=functor,force=force)
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    # also load all the data
    q = energy_obj.q
    q_nm = q * 1e9
    xlim_nm = [min(q_nm),max(q_nm)]
    G0_kT = energy_obj.G0/4.1e-21
    from scipy.interpolate import LSQUnivariateSpline
    knots = np.linspace(min(q),max(q),num=50,endpoint=True)[1:-1]
    spline_G0 = LSQUnivariateSpline(x=q,y=energy_obj.G0,
                                    t=knots,k=3)
    fig = PlotUtilities.figure((3,6))

    ax1 = plt.subplot(3,1,1)
    for d in data:
        plt.plot(d.Separation*1e9,d.Force*1e12,markevery=50)
    plt.xlim(xlim_nm)
    PlotUtilities.lazyLabel("","Force (pN)","")
    PlotUtilities.no_x_label(ax=ax1)
    ax2= plt.subplot(3,1,2)
    plt.plot(q_nm,G0_kT)
    plt.plot(q_nm,spline_G0(q)/4.1e-21,'r--')
    PlotUtilities.lazyLabel("","$\Delta G_\mathrm{0}$ (kbT)","")
    PlotUtilities.no_x_label(ax=ax2)
    plt.xlim(xlim_nm)
    plt.subplot(3,1,3)
    k_N_per_m = spline_G0.derivative(2)(q)
    k_pN_per_nm = k_N_per_m * 1e3
    plt.plot(q_nm,k_pN_per_nm)
    PlotUtilities.lazyLabel("q (nm)","k (pN/nm)","")
    lim = 75
    plt.ylim(-lim,lim)
    PlotUtilities.savefig(fig,out_dir + "out_G.png")
    pass

if __name__ == "__main__":
    run()
