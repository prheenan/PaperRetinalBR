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

from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
import RetinalUtil
sys.path.append("../")

class SnapshotFEC(object):
    def __init__(self,step,fec_list):
        self.step = step
        self.fec_list = fec_list

class AlignmentInfo(object):
    def __init__(self,e,zeroed,polished,blacklisted):
        self.landscape = e
        self.zeroed = zeroed
        self.polished = polished
        self.blacklisted = blacklisted
    @property
    def _all_fecs(self):
        all_lists = [self.zeroed,self.polished,self.blacklisted]
        to_ret = [f for list_v in all_lists for f in list_v.fec_list]
        return to_ret

class LandscapeGallery(object):
    def __init__(self,PEG600,PEG3400):
        self.PEG600 = PEG600
        self.PEG3400 = PEG3400

def _snapsnot(base_dir,step):
    corrected_dir = Pipeline._cache_dir(base=base_dir,
                                        enum=step)
    data = CheckpointUtilities.lazy_multi_load(corrected_dir)
    return SnapshotFEC(step,data)


def _alignment_pipeline(e):
    base_dir_landscapes = e.base_dir
    base_dir = base_dir_landscapes.split("landscape_")[0]
    # get the corrected directory (this is *zeroed*)
    zeroed = _snapsnot(base_dir,step=Pipeline.Step.CORRECTED)
    # get the polished / aligned dir
    polished = _snapsnot(base_dir,step=Pipeline.Step.POLISH)
    # get the directory after blacklisting bad curves
    base_landscape = RetinalUtil._landscape_dir(base_dir)
    blacklist = _snapsnot(base_landscape, step=Pipeline.Step.MANUAL)
    to_ret = AlignmentInfo(e,zeroed,polished,blacklist)
    return to_ret


def read_sample_landscapes(base_dir):
    """
    :param base_dir: input to RetinalUtil._read_all_energies
    :return:
    """
    energies = RetinalUtil._read_all_energies(base_dir)
    names = [e.base_dir.split("FECs180307")[1] for e in energies]
    str_PEG600_example = "/BR+Retinal/300nms/170511FEC/landscape_"
    str_PEG3400_example = "/BR+Retinal/3000nms/170503FEC/landscape_"
    idx_PEG600 = names.index(str_PEG600_example)
    idx_PEG3400 = names.index(str_PEG3400_example)
    assert (idx_PEG600 >= 0) and (idx_PEG3400 >= 0)
    to_ret = LandscapeGallery(PEG600=energies[idx_PEG600],
                              PEG3400=energies[idx_PEG3400])
    return to_ret