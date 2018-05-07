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
from Lib.AppWHAM.Code import WeightedHistogram
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

class EnergyWithMeta(WeightedHistogram.LandscapeWHAM):
    def __init__(self,file_name,base_dir,energy):
        self.file_name = file_name
        self.base_dir = base_dir
        self.n_fecs = None
        self.__init__energy(energy)
    def __init__energy(self,energy):
        offset = energy._offset_G0_of_q
        super(EnergyWithMeta,self).__init__(q=energy._q,
                                            G0=energy._G0,
                                            offset_G0_of_q=offset,
                                            beta=energy.beta)
    def _slice(self,*args,**kw):
        sliced = super(EnergyWithMeta,self)._slice(*args,**kw)
        self.__init__energy(sliced)
        return self
    def set_n_fecs(self,n):
        self.n_fecs = n


def _processing_base(default_base="../../../Data/BR+Retinal/170321FEC/",**kw):
    return Pipeline._base_dir_from_cmd(default=default_base,**kw)

def _landscape_dir(dir_base):
    return dir_base + "landscape_"

def _landscape_base(**kw):
    to_ret = _landscape_dir(_processing_base(**kw))
    return to_ret
    
def _analysis_base(default_base="../../../Data/BR+Retinal/50/",**kw):
    return _processing_base(default_base=default_base,**kw)

def common_q_interp(energy_list,num_q=200):
    q_mins = [min(e.q_nm) for e in energy_list]
    q_maxs = [max(e.q_nm) for e in energy_list]
    q_limits = [np.max(q_mins), np.min(q_maxs)]
    q_interp = np.linspace(*q_limits,num=num_q)
    return q_interp

def interpolating_G0(energy_list,num_q=200,num_splines=75):
    """
    :param energy_list: list of energy objects to get splines from
    :param num_q: h
    :param num_splines:
    :return: tuple of <q (nm) in range of all energy objects,
                      splines for objects' energy in kcal/mol(q in nm)>
    """
    q_interp =  common_q_interp(energy_list,num_q=num_q)
    # get all the splines
    splines = [spline_fit(q=e.q_nm, G0=e.G0_kcal_per_mol)
               for e in energy_list]
    return q_interp, splines

def spline_fit(q, G0, k=3, knots=None,num=None):
    if num is None:
        num = min(75,q.size//(k)-1)
    if (knots is None):
        step = q.size//num
        assert step > 0 and step
        knots = q[1:-1:step]
    spline_G0 = LSQUnivariateSpline(x=q, y=G0,t=knots, k=k)
    return spline_G0

def valid_landscape(e):
    """
    :param e: Landsape object
    :return: new landscape object, all the places where it is finite-valued
    """
    good_idx = np.where(np.isfinite(e.G0) & ~np.isnan(e.G0))[0]
    assert good_idx.size > 0
    tmp_e = e._slice(good_idx)
    return tmp_e