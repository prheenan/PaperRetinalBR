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
from Lib.AppIWT.Code import WeierstrassUtil, InverseWeierstrass
import RetinalUtil

def generate_landscape(in_dir):
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    energy_obj = \
        InverseWeierstrass.free_energy_inverse_weierstrass(unfolding=data)
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
    q_nm = energy_obj.q * 1e9
    xlim_nm = [min(q_nm),max(q_nm)]
    fig = PlotUtilities.figure((3,6))
    ax1 = plt.subplot(2,1,1)
    for d in data:
        plt.plot(d.Separation*1e9,d.Force*1e12,markevery=50)
    plt.xlim(xlim_nm)
    PlotUtilities.lazyLabel("","Force (pN)","")
    PlotUtilities.no_x_label(ax=ax1)
    plt.subplot(2,1,2)
    plt.plot(q_nm,energy_obj.G_0/4.1e-21)
    plt.xlim(xlim_nm)
    PlotUtilities.lazyLabel("Extension (nm)","$G_0$ (kbT)","")
    PlotUtilities.savefig(fig,out_dir + "out_G.png")
    pass

if __name__ == "__main__":
    run()
