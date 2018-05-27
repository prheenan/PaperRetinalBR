# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse, enum, copy

from Lib.UtilPipeline import Pipeline
from Lib.UtilForce.FEC import FEC_Util,  FEC_Plot
from Lib.UtilForce.UtilIgor.TimeSepForceObj import TimeSepForceObj
from Lib.UtilForce.UtilGeneral import PlotUtilities, CheckpointUtilities
from Processing.Util import WLC as WLCHao

import multiprocessing
from multiprocessing import Pool
import re
max_n_pool = multiprocessing.cpu_count() - 1


class AlignedFEC(TimeSepForceObj):
    def __init__(self,normal_fec,info_fit,feather_info):
        super(AlignedFEC,self).__init__()
        self.LowResData = copy.deepcopy(normal_fec.LowResData)
        self.info_fit = info_fit
        self.info_feather = feather_info
    @property
    def L0_info(self):
        return self.info_fit

str_BR = "BR+Retinal"
str_BO = "BR-Retinal"
f_v = lambda v: "{:d}nms".format(v)
f_date = lambda s: "{:s}FEC".format(s)

class Blacklist(object):
    def __init__(self,str_pm_bR,str_vel,str_date,list_ids):
        self.str_pm_bR = str_pm_bR
        self.str_vel = str_vel
        self.str_date = str_date
        self.list_ids = list_ids

blacklist_tuples = [ \
    # all the blacklisted BR data
    [str_BR,f_v(50),f_date("170502"),[1245,2160]],
    [str_BR,f_v(50),f_date("170503"),[819,837]],
    [str_BR, f_v(300), f_date("170321"), [463,471,500,532,760,786,821,833]],
    [str_BR, f_v(300), f_date("170501"), [203,870,1353]],
    [str_BR, f_v(300), f_date("170502"), []],# this one is OK
    [str_BR, f_v(300), f_date("170511"), [316,731]],
    [str_BR, f_v(3000), f_date("170502"), [717]],
    [str_BR, f_v(3000), f_date("170503"), [231,]],
    # all the blacklisted BO data
    [str_BO, f_v(50), f_date("170523"), [117,176,203,204,223]],
    [str_BO, f_v(300), f_date("170327"), [121,236,310,347,383,386,473,484,486]],
    [str_BO, f_v(3000), f_date("170523"), [18,20,22,69,349,741,773]],
]

blacklists = [Blacklist(*t) for t in blacklist_tuples]
blacklist_dict_vels = dict([((b.str_pm_bR,b.str_vel,b.str_date),b.list_ids)
                            for b in blacklists])



def _filter_by_bl(data,base_input_processing):
    """
    :param data: list of FECS
    :param base_input_processing: input associated with the data
    :return: list of FECs which arent blacklists
    """
    # get the meta data associated with this data
    pattern = \
        r"""
        (BR[+-]Retinal)/
        ([^/]+)/
        ([^/]+)/
        """
    match = re.search(pattern, base_input_processing, re.VERBOSE)
    str_br_type, str_vel, str_data = match.groups()
    # determine the blacklist...
    blacklist_tmp = blacklist_dict_vels[(str_br_type,str_vel,str_data)]
    ids_groups = [re.search("(\d+)", d.Meta.Name) for d in data]
    for i in ids_groups:
        assert i is not None
    # POST: found all ids
    ids = [int(i.group(0)) for i in ids_groups]
    # make sure all ids in the blacklist are actually in this group
    for tmp in blacklist_tmp:
        assert tmp in ids
    # POST: all ids in blacklist are in the data
    to_ret = []
    for i, d in zip(ids, data):
        if i not in blacklist_tmp:
            to_ret.append(d)
    return to_ret

def _multiproc(func,input_v,n_pool=max_n_pool):
    """
    :param func: function to map; should take a single arugment (element of
    input_v)
    :param input_v: each argument passed to func
    :param n_pool: number to use in pool; defaults to max
    :return:
    """
    p = Pool(n_pool)
    if (n_pool > 1):
        to_ret = p.map(func,input_v)
    else:
        to_ret = [func(d) for d in input_v]
    return to_ret

def _cache_individual(d,out_dir,f,force,*args,**kw):
    """
    :param d: fec, or something that can be applied to  FEC_Util.fec_name_func
    :param out_dir: where the pkl fie should live
    :param f: function to call
    :param force: if we should force re-caching
    :param args: to f
    :param kw:  to f
    :return: cache or new function run
    """
    name = out_dir + FEC_Util.fec_name_func(0,d) + ".pkl"
    data = CheckpointUtilities.getCheckpoint(name,f,force,*args,**kw)
    return data


def nm_and_pN_limits(data,f_x):
    x_range = [[min(f_x(d)), max(f_x(d))] for d in data]
    y_range = [[min(d.Force), max(d.Force)] for d in data]
    xlim = 1e9 * np.array([np.min(x_range), np.max(x_range)])
    ylim = 1e12 * np.array([np.min(y_range), np.max(y_range)])
    return xlim,ylim

def plot_single_fec(d,f_x,xlim,ylim,markevery=1,**kw):
    FEC_Plot._fec_base_plot(f_x(d)[::markevery] * 1e9,
                            d.Force[::markevery] * 1e12,**kw)
    plt.xlim(xlim)
    plt.ylim(ylim)
    PlotUtilities.lazyLabel("Extension (nm)", "$F$ (pN)", "")

def plot_data(base_dir,step,data,markevery=1,f_x = lambda x: x.Separation,
              xlim=[-20,175]):
    """
    :param base_dir: where the data live
    :param step:  what step we are on
    :param data: the actual data; list of TimeSepForce
    :param markevery: how often to mark the data (useful for lowering high
    res to resonable size)
    :return: nothing, plots the data..
    """
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    name_func = FEC_Util.fec_name_func
    xlim_tmp , ylim = nm_and_pN_limits(data,f_x)
    if (xlim is not None):
        xlim = xlim
    else:
        xlim = xlim_tmp
    for d in data:
        f = PlotUtilities.figure()
        plot_single_fec(d, f_x, xlim, ylim,markevery=markevery)
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + ".png")

def _aligned_plot(d,f_x,xlim,ylim,use_shift=False):
    # get the fit
    # convert to reasonable units for plotting
    # get the fit
    info = d.L0_info
    f_grid = info.f_grid
    # convert to reasonable units for plotting
    offset =  info._L_shift  if use_shift else 0
    ext_grid = info.ext_grid() - offset
    f_plot_pred = f_grid * 1e12
    x_plot_pred = (ext_grid)* 1e9
    # convert back to the grid to get rid of the offset
    plt.plot(x_plot_pred, f_plot_pred, color='r', linewidth=1.5,
             label="Total")
    # get the two components (FJC and WLC)
    components = info.component_grid()
    component_offset = offset
    for ext,label in [ [components[1],"C-term"],[components[0],"PEG3400"] ]:
        ext_plot = (ext - component_offset) * 1e9
        plt.plot(ext_plot,f_plot_pred,label=label,linestyle='--')
    # plot the fit
    plot_single_fec(d, f_x, xlim, ylim)

def make_aligned_plot(base_dir,step,data,xlim=None,post="",**kw):
    plot_subdir = Pipeline._plot_subdir(base_dir, step)
    f_x = lambda x: x.Separation
    xlim_tmp, ylim = nm_and_pN_limits(data,f_x)
    xlim = [xlim[0],200] if xlim is None else xlim
    name_func = FEC_Util.fec_name_func
    for d in data:
        f = PlotUtilities.figure()
        _aligned_plot(d, f_x, xlim, ylim,**kw)
        PlotUtilities.savefig(f, plot_subdir + name_func(0, d) + post + ".png")


def heatmap_ensemble_plot(data,out_name,xlim=[-50, 150]):
    """
    makes a heatmap of the ensemble, with the actual data beneath

    :param data: list of FECs
    :param out_name: what to save this as
    :return: na
    """
    fig = PlotUtilities.figure(figsize=(3, 5))
    ax = plt.subplot(2, 1, 1)
    FEC_Plot.heat_map_fec(data, num_bins=(200, 100),
                          use_colorbar=False,
                          separation_max=xlim[1])
    for spine_name in ["bottom", "top"]:
        PlotUtilities.color_axis_ticks(color='w', spine_name=spine_name,
                                       axis_name="x", ax=ax)
    PlotUtilities.xlabel("")
    PlotUtilities.title("")
    PlotUtilities.no_x_label(ax)
    plt.xlim(xlim)
    plt.subplot(2, 1, 2)
    for d in data:
        x, f = d.Separation * 1e9, d.Force * 1e12
        FEC_Plot._fec_base_plot(x, f, style_data=dict(color=None, alpha=0.3,
                                                      linewidth=0.5))
    PlotUtilities.lazyLabel("Extension (nm)", "Force (pN)", "")
    plt.xlim(xlim)
    PlotUtilities.savefig(fig, out_name)
