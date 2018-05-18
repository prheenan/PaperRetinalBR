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
from Lib.AppIWT.Code import InverseWeierstrass
import RetinalUtil
import PlotUtil


def generate_landscape(in_dir):
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    data_wham = UtilWHAM.to_wham_input(data)
    energy_wham = WeightedHistogram.wham(fwd_input=data_wham)
    f_iwt = InverseWeierstrass.free_energy_inverse_weierstrass
    iwt_obj = f_iwt(unfolding=data)
    # offset the IWT so that it matches the offset of WHAM...
    iwt_obj.q -= iwt_obj.q[0]
    offset_q = energy_wham.q[0]
    iwt_obj.q += offset_q
    iwt_obj._z += offset_q
    n = len(data[0]._iwt_slices)
    iwt_EF = []
    for i in range(n):
        data_sliced = [d._slice(d._iwt_slices[i]) for d in data]
        data_sliced = RetinalUtil.process_helical_slice(data_sliced)
        # zero the separations
        iwt_tmp = f_iwt(unfolding=data_sliced)
        iwt_tmp.q += offset_q
        iwt_tmp._z += offset_q
        iwt_EF.append(iwt_tmp)
    to_ret = RetinalUtil.DualLandscape(wham_obj=energy_wham,iwt_obj=iwt_obj,
                                       other_helices=iwt_EF)
    return to_ret

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
    fig = PlotUtilities.figure((3, 6))
    PlotUtil.plot_landscapes(data,energy_obj)
    PlotUtilities.savefig(fig,out_dir + "out_G.png")
    pass

if __name__ == "__main__":
    run()
