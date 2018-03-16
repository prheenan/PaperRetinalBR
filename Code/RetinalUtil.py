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
from Lib.UtilForce.UtilGeneral import PlotUtilities
from scipy.interpolate import LSQUnivariateSpline

from Lib.AppIWT.Code.InverseWeierstrass import FEC_Pulling_Object

class MetaPulling(FEC_Pulling_Object):
    def __init__(self,time_sep_force,kbT=4.1e-21,**kw):
        kw_time_sep_f = dict(Time=time_sep_force.Time,
                              Extension=time_sep_force.Separation,
                              Force=time_sep_force.Force,
                              SpringConstant=time_sep_force.SpringConstant,
                              Velocity=time_sep_force.Velocity,
                              kT=kbT,
                              **kw)
        super(MetaPulling,self).__init__(**kw_time_sep_f)
        self.Meta = time_sep_force.Meta

class EnergyList(object):
    def __init__(self,file_names,base_dirs,energies):
        self.files_name = file_names
        self.base_dirs = base_dirs
        self.energies = energies
    @property
    def N(self):
        return len(self.files_name)

def _processing_base(default_base="../../../Data/BR+Retinal/170321FEC/",**kw):
    return Pipeline._base_dir_from_cmd(default=default_base,**kw)

def _landscape_dir(dir_base):
    return dir_base + "landscape_"

def _landscape_base(**kw):
    to_ret = _landscape_dir(_processing_base(**kw))
    return to_ret
    
def _analysis_base(default_base="../../../Data/BR+Retinal/50/",**kw):
    return _processing_base(default_base=default_base,**kw)


def spline_fit(q, G0, k=3, knots=None,num=100):
    if (knots is None):
        knots = np.linspace(min(q), max(q), num=num, endpoint=True)[1:-1]
    spline_G0 = LSQUnivariateSpline(x=q, y=G0,t=knots, k=k)
    return spline_G0