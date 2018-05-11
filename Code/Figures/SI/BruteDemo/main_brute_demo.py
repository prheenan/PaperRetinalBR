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

sys.path.append("../../../")
from Lib.UtilForce.UtilGeneral import PlotUtilities
from Lib.UtilForce.UtilGeneral.Plot import Annotations


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    n = 1000
    pts = np.linspace(-1,1,num=n)
    f = lambda x: -1 * x**2 - 0.5 * x + 3 * x**3 + 10 * x**4 + \
        np.sin(2 * np.pi * x * 3) * 0.5 / (1+abs(x/3))
    f_pts = f(pts)
    max_f = max(f(pts))
    f_pts /= max_f
    brute_sample_n = 15
    dict_brute = dict(markersize=3)
    brute_pts = pts[::int(n/brute_sample_n)]
    brute_f_pts = f(brute_pts) / max_f
    # get the 'zoom in' region
    best_idx = np.argmin(brute_f_pts)
    pts_zoom = np.linspace(brute_pts[best_idx-1],brute_pts[best_idx+1],
                            num=10,endpoint=True)
    f_zoom = f(pts_zoom) / max_f
    fig = PlotUtilities.figure((3,4))
    ax1 = plt.subplot(2,1,1)
    plt.plot(pts,f_pts)
    plt.plot(brute_pts,brute_f_pts,'ro',**dict_brute)
    PlotUtilities.lazyLabel("","Error","")
    PlotUtilities.no_x_label(ax1)
    ax2 = plt.subplot(2,1,2)
    plt.plot(pts,f_pts)
    xmin, xmax = min(pts_zoom),max(pts_zoom)
    ymin, ymax = min(f_zoom),max(f_zoom)
    y_range = ymax-ymin
    plt.xlim(xmin,xmax)
    plt.ylim(ymin - y_range * 0.1,ymax + y_range * 0.1)
    plt.axvline(pts_zoom[np.argmin(f_zoom)],label="Found\nmin",
                color='g',linestyle='--')
    Annotations.zoom_effect01(ax1, ax2, xmin, xmax)
    PlotUtilities.lazyLabel("X (au)","Error","")
    PlotUtilities.savefig(fig,"./out.png")

if __name__ == "__main__":
    run()
