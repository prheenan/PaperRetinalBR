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
import os

sys.path.append("../")
from Processing import ProcessingUtil

from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities, \
    PlotUtilities
import RetinalUtil


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
    def __init__(self,PEG600,PEG3400,BO_PEG3400):
        self.PEG600 = PEG600
        self.PEG3400 = PEG3400
        self.BO_PEG3400 = BO_PEG3400

def _snapsnot(base_dir,step):
    corrected_dir = Pipeline._cache_dir(base=base_dir,
                                        enum=step)
    data = CheckpointUtilities.lazy_multi_load(corrected_dir)
    return SnapshotFEC(step,data)

def _plot_fmt(ax,xlim,ylim,is_bottom=False,color=False,is_left=True,
              ylabel="$F$ (pN)",xlabel="Extension (nm)"):
    PlotUtilities.tickAxisFont(ax=ax)
    plt.xlim(xlim)
    plt.ylim(ylim)
    PlotUtilities.title("")
    PlotUtilities.ylabel(ylabel)
    PlotUtilities.xlabel(xlabel)
    if (not is_bottom):
        PlotUtilities.no_x_label(ax=ax)
        PlotUtilities.xlabel("")
    if (not is_left):
        PlotUtilities.no_y_label(ax=ax)
        PlotUtilities.ylabel("")
    if color:
        color_kw = dict(ax=ax,color='w',label_color='k')
        PlotUtilities.color_x(**color_kw)
        PlotUtilities.color_y(**color_kw)


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

def _plot_fec_list(list_v,xlim,ylim,label=None,color=None,linewidth=0.3,**kw):
    f_x = lambda x_tmp : x_tmp.Separation
    for i,d in enumerate(list_v):
        label_tmp = label if i == 0 else None
        ProcessingUtil.plot_single_fec(d, f_x, xlim, ylim,label=label_tmp,
                                       style_data=dict(color=color, alpha=0.3,
                                                       linewidth=linewidth),
                                       **kw)
def _limits(_all_fecs):
    xlim = [-20,120]
    max_y = np.max([max(f.Force) for f in _all_fecs]) * 1e12
    ylim = [-50,max(max_y,300)]
    return xlim,ylim

def _read_samples(base_dir_input,sample_names):
    energies = RetinalUtil._read_all_energies(base_dir_input)
    names = [e.base_dir.split("FECs180307")[1] for e in energies]
    to_ret = []
    for n in sample_names:
        idx_tmp = names.index(n)
        to_ret.append(energies[idx_tmp])
    return to_ret

def read_sample_landscapes(base_dir):
    """
    :param base_dir: input to RetinalUtil._read_all_energies
    :return:
    """
    base_dir_BR = base_dir + "BR+Retinal/"
    names_BR = ["/BR+Retinal/300nms/170511FEC/landscape_",
                "/BR+Retinal/3000nms/170503FEC/landscape_"]
    PEG600, PEG3400 = _read_samples(base_dir_BR,names_BR)
    base_dir_BO = base_dir + "BR-Retinal/"
    names_BO = ["/BR-Retinal/300nms/170327FEC/landscape_"]
    BO_PEG3400 = _read_samples(base_dir_BO,names_BO)[0]
    to_ret = LandscapeGallery(PEG600=PEG600,
                              PEG3400=PEG3400,
                              BO_PEG3400=BO_PEG3400)
    return to_ret


def read_energy_lists(subdirs):
    energy_list_arr =[]
    # get all the energy objects
    for base in subdirs:
        in_dir = Pipeline._cache_dir(base=base,
                                     enum=Pipeline.Step.CORRECTED)
        in_file = in_dir + "energy.pkl"
        e = CheckpointUtilities.lazy_load(in_file)
        energy_list_arr.append(e)
    return energy_list_arr

def _read_energy_list_and_q_interp(input_dir,q_offset):
    """
    :param input_dir: where all the data live, e.g.  Data/FECs180307/"
    :param q_offset: how much of the landscape to use...
    :return:  tuple of (q to interpolate to, list, each element is a list
    of landcapes associated with one of the dirctories under input_dir)
    """
    subdirs_raw = [input_dir + d + "/" for d in os.listdir(input_dir)]
    subdirs = [d for d in subdirs_raw if (os.path.isdir(d))
               and "David" not in d]
    energy_list_arr = read_energy_lists(subdirs)
    energy_list_arr = [ [e._iwt_obj for e in list_v]
                        for list_v in energy_list_arr]
    e_list_flat = [e for list_tmp in energy_list_arr for e in list_tmp ]
    q_interp = RetinalUtil.common_q_interp(energy_list=e_list_flat)
    q_interp = q_interp[np.where(q_interp-q_interp[0]  <= q_offset)]
    return q_interp,energy_list_arr