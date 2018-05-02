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
from Lib.AppWLC.Code import WLC
from Processing.Util import WLC as WLCHao
from Lib.AppWLC.UtilFit import fit_base


def polish_data(base_dir,**kw):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    for d in all_data:
        # filter the data, making a copy
        to_ret = d._slice(slice(0, None, 1))
        # get the slice we are fitting
        inf = to_ret.L0_info
        fit_slice = inf.fit_slice
        x, f = to_ret.Separation.copy(), to_ret.Force.copy()
        # get a grid over all possible forces
        f_grid = np.linspace(min(f), max(f), num=f.size, endpoint=True)
        # get the extension components
        ext_total, ext_components = WLCHao._hao_ext_grid(f_grid, *inf.x0)
        ext_FJC = ext_components[0]
        # make the extension at <= force be zero
        where_f_le = np.where(f_grid <= 0)
        ext_FJC[where_f_le] = 0
        ext_total[where_f_le] = 0
        # we now have X_FJC as a function of force. Therefore, we can subtract
        # off the extension of the PEG3400 to determining the force-extension
        # associated with only the protein (*including* its C-term)
        # note we are getting ext_FJC(f), where f is each point in the original
        # data.
        ext_FJC_all_forces = fit_base._grid_to_data(x=f, x_grid=f_grid,
                                                    y_grid=ext_FJC,
                                                    bounds_error=False)
        # remove the extension associated with the PEG
        to_ret.Separation -= ext_FJC_all_forces
        # align by the contour length of the protein
        L0 = inf.L0_c_terminal
        print(L0)
        to_ret.Separation -= L0
        to_ret.ZSnsr -= L0
        yield to_ret

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    default_base = "../../../Data/170321FEC/"
    base_dir = Pipeline._base_dir_from_cmd(default=default_base)
    step = Pipeline.Step.POLISH
    in_dir = Pipeline._cache_dir(base=base_dir, enum=Pipeline.Step.ALIGNED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    plot_dir = Pipeline._plot_subdir(base=base_dir, enum=step)
    force = True
    limit = None
    functor = lambda : polish_data(in_dir)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    fig = PlotUtilities.figure()
    ax = plt.subplot(2,1,1)
    heatmap = FEC_Plot.heat_map_fec(data, num_bins=(200, 100),
                                    use_colorbar=False)
    for spine_name in ["bottom","top"]:
        PlotUtilities.color_axis_ticks(color='w',spine_name=spine_name,
                                       axis_name="x",ax=ax)
    xlim = plt.xlim()
    PlotUtilities.xlabel("")
    PlotUtilities.title("")
    PlotUtilities.no_x_label(ax)
    plt.xlim(xlim)
    plt.subplot(2,1,2)
    for d in data:
        x,f = d.Separation*1e9,d.Force*1e12
        FEC_Plot._fec_base_plot(x,f,style_data=dict(color=None,alpha=0.3,
                                                    linewidth=0.5))
    PlotUtilities.lazyLabel("Extension (nm)","Force (pN)","")
    plt.xlim(xlim)
    PlotUtilities.savefig(fig,plot_dir + "heat_map.png")
    # plot each individual
    ProcessingUtil.plot_data(base_dir,step,data)
    """
    XXX work into plot...
        plt.subplot(2, 1, 1)
        plt.plot(x, f, color='k', alpha=0.3)
        plt.plot(x[fit_slice], f[fit_slice], 'r')
        plt.plot(ext_total, f_grid, 'b--')
        plt.plot(ext_FJC, f_grid, 'g:')
        plt.subplot(2, 1, 2)
        plt.plot(to_ret.Separation, to_ret.Force)
        plt.show()
    """

if __name__ == "__main__":
    run()
