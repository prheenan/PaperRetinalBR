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

sys.path.append("../../")
from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Processing import ProcessingUtil
from Lib.AppWHAM.Code import WeightedHistogram, UtilWHAM
import RetinalUtil


def get_energy_list(base_dir_analysis):
    raw_dirs = [base_dir_analysis + d for d in os.listdir(base_dir_analysis)]
    filtered_dirs = [r + "/" for r in raw_dirs if os.path.isdir(r)
                     and "cache" not in r]
    energies, files = [], []
    for d in filtered_dirs:
        base_tmp = RetinalUtil._landscape_base(default_base=d)
        cache_tmp = \
            Pipeline._cache_dir(base=base_tmp, enum=Pipeline.Step.POLISH)
        file_load = cache_tmp + "energy.pkl"
        energy_obj = CheckpointUtilities.lazy_load(file_load)
        files.append(file_load)
        energies.append(energy_obj)
    return RetinalUtil.EnergyList(files,energies)

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir_analysis = RetinalUtil._analysis_base()
    out_dir = Pipeline._cache_dir(base=base_dir_analysis,
                                  enum=Pipeline.Step.CORRECTED)
    energy_list = CheckpointUtilities.getCheckpoint(out_dir + "energies.pkl",
                                                    get_energy_list,True,
                                                    base_dir_analysis)
    for e in energy_list.energies:
        plt.plot(e.q,e.G0,'r.')
    plt.show()

if __name__ == "__main__":
    run()
