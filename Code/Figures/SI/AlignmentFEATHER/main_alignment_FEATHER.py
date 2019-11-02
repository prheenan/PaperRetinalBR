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
from Lib.UtilForce.FEC import FEC_Util, FEC_Plot
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Processing import ProcessingUtil
from Lib.UtilPipeline.UtilGeneral import PlotUtilities
from Lib.UtilPipeline.UtilGeneral.Plot import Scalebar
from Lib.UtilPipeline.FEC import Plotting

def make_plot(dir_in,out_name):
    dir_aligned = dir_in + "cache_8_aligned/"
    dir_polished = dir_in + "cache_10_polish/"
    data_unaligned = CheckpointUtilities.lazy_multi_load(dir_aligned)
    data_polished = CheckpointUtilities.lazy_multi_load(dir_polished)
    data_unaligned = ProcessingUtil._filter_by_bl(data_unaligned, dir_in)
    data_polished = ProcessingUtil._filter_by_bl(data_polished, dir_in)
    # scootch the data over slightly
    for d in data_polished:
        d.Separation -= 20e-9
    to_x = lambda _x: _x * 1e9
    to_y = lambda _y: _y * 1e12
    xlim = [-10, 100]
    ylim = [-20, 350]
    plt.close()
    fig = PlotUtilities.figure(figsize=(3.5, 4.5))
    kw_plot = dict(linewidth=0.75)
    ax1 = plt.subplot(2, 2, 1)
    for d in data_unaligned:
        plt.plot(to_x(d.Separation), to_y(d.Force), **kw_plot)
    ax2 = plt.subplot(2, 2, 2)
    for d in data_polished:
        plt.plot(to_x(d.Separation), to_y(d.Force), **kw_plot)
    for a in [ax1, ax2]:
        a.set_ylim(ylim)
        a.set_xlim(xlim)
        PlotUtilities.lazyLabel("Extension (nm)", "$F$ (pN)", "", ax=a)
    PlotUtilities.ylabel("", ax=ax2)
    height_pN = 100
    min_y, max_y, delta_y = Scalebar.offsets_zero_tick(limits=ylim,
                                                       range_scalebar=height_pN)
    kw_scale = dict(offset_x=0.7,
                    offset_y=max_y ,
                    x_kwargs=dict(width=20, unit="nm"),
                    y_kwargs=dict(height=height_pN, unit="pN"))
    for a in [ax1, ax2]:
        Scalebar.crossed_x_and_y_relative(ax=a, **kw_scale)
        PlotUtilities.no_y_label(ax=a)
        PlotUtilities.no_x_label(ax=a)
        PlotUtilities.x_label_on_top(ax=a)
    num=200
    bins_x = np.linspace(xlim[0], xlim[-1], endpoint=True, num=num)
    bins_y = np.linspace(ylim[0], ylim[-1], endpoint=True, num=num)
    kw = dict(color='w', use_colorbar=False,
              bins=(bins_x, bins_y), title="")
    kw_scale_heat = dict(**kw_scale)
    line_kw_def = dict(color='w', linewidth=2)
    font_x, font_y = Scalebar.font_kwargs_modified(x_kwargs=dict(color='w'),
                                                   y_kwargs=dict(color='w'))
    kw_scale_heat['x_kwargs']['line_kwargs'] = line_kw_def
    kw_scale_heat['x_kwargs']['font_kwargs'] = font_x
    kw_scale_heat['y_kwargs']['line_kwargs'] = line_kw_def
    kw_scale_heat['y_kwargs']['font_kwargs'] = font_y
    ax3 = plt.subplot(2, 2, 3)
    Plotting.formatted_heatmap(data=data_unaligned, **kw)
    Scalebar.crossed_x_and_y_relative(ax=ax3, **kw_scale_heat)
    ax4 = plt.subplot(2, 2, 4)
    Plotting.formatted_heatmap(data=data_polished, **kw)
    Scalebar.crossed_x_and_y_relative(ax=ax4, **kw_scale_heat)
    for a in [ax3, ax4]:
        PlotUtilities.no_y_label(ax=a)
        PlotUtilities.no_x_label(ax=a)
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        PlotUtilities.xlabel("",ax=a)
    PlotUtilities.ylabel("", ax=ax4)
    PlotUtilities.ylabel("$F$ (pN)", ax=ax3)
    PlotUtilities.savefig(fig, out_name)
    pass

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    data_base = "../../../../Data/FECs180307/"
    data_ex = ["BR+Retinal/50nms/170502FEC/",
               "BR+Retinal/50nms/170503FEC/",
               "BR+Retinal/300nms/170321FEC/"]
    for i, d in enumerate(data_ex):
        make_plot(data_base + d,
                  out_name="FEATHER_Alignment_{:d}.png".format(i))
    pass

if __name__ == "__main__":
    run()
