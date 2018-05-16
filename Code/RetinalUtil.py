# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse, enum, copy, os

from Lib.UtilPipeline import Pipeline
from Lib.AppWHAM.Code import WeightedHistogram
from scipy.interpolate import LSQUnivariateSpline

from Lib.AppIWT.Code import InverseWeierstrass
from Lib.AppIWT.Code.InverseWeierstrass import FEC_Pulling_Object
from Lib.AppIWT.Code.UtilLandscape import BidirectionalUtil
from Processing.Util import WLC as WLCHao
from Lib.AppWLC.UtilFit import fit_base
from Lib.AppFEATHER.Code import Detector, Analysis
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities

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


class DualLandscape(BidirectionalUtil._BaseLandscape):
    def __init__(self, wham_obj, iwt_obj,other_helices):
        self._iwt_obj = iwt_obj
        self._wham_obj = wham_obj
        self._other_helices = other_helices
        super(DualLandscape, self).__init__(iwt_obj._q, iwt_obj._G0,
                                            iwt_obj.beta)
    def _slice(self,*args,**kwargs):
        slice_iwt = self._iwt_obj
        wham_obj = self._wham_obj._slice(*args,**kwargs)
        return DualLandscape(wham_obj=wham_obj,iwt_obj=slice_iwt,
                             other_helices=self._other_helices)

class EnergyWithMeta(DualLandscape):
    def __init__(self,file_name,base_dir,energy):
        self.file_name = file_name
        self.base_dir = base_dir
        self.n_fecs = None
        self.__init__energy(energy)
    def __init__energy(self,energy):
        super(EnergyWithMeta,self).__init__(wham_obj=energy._wham_obj,
                                            iwt_obj=energy._iwt_obj,
                                            other_helices=energy._other_helices)
    def _slice(self,*args,**kw):
        sliced = super(EnergyWithMeta,self)._slice(*args,**kw)
        self.__init__energy(sliced)
        return self
    def set_n_fecs(self,n):
        self.n_fecs = n

class HelicalSearch(object):
    def __init__(self,data,min_ext_m):
        self.min_ext_m = min_ext_m
        N_GF, data_iwt_EF = slice_data_for_helix(data, min_ext_m)
        plt.subplot(2,1,1)
        for i in data_iwt_EF:
            plt.plot(i.Separation,i.Force)
        plt.show()
        iwt_EF = InverseWeierstrass.free_energy_inverse_weierstrass(data_iwt_EF)
        plt.subplot(2,1,2)
        plt.plot(iwt_EF.q,iwt_EF.G0)
        plt.show()
        self._landscape = iwt_EF

def _to_pts(d,meters):
    t = d.Time
    v = d.Velocity
    idx_gt = np.where(d.Separation >= meters)[0]
    assert idx_gt.size > 0
    N_GF = idx_gt[0]
    return N_GF

def _slice_single(d,min_ext_m):
    N_GF = 0 if min_ext_m is None else _to_pts(d,min_ext_m)
    N_final = None
    data_iwt_EF = d._slice(slice(N_GF, N_final, 1))
    return N_GF, N_final, data_iwt_EF

def slice_data_for_helix(data,min_ext_m):
    data_iwt_EF = [_slice_single(d,min_ext_m) for d in data]
    return data_iwt_EF[0][0], [d[-1] for d in data_iwt_EF]

def q_GF_nm():
    return 35

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


def subdirs(base_dir_analysis):
    raw_dirs = [base_dir_analysis + d for d in os.listdir(base_dir_analysis)]
    filtered_dirs = [r + "/" for r in raw_dirs if os.path.isdir(r)
                     and "cache" not in r]
    return filtered_dirs


def read_fecs(e):
    base_tmp = e.base_dir
    in_dir = Pipeline._cache_dir(base=base_tmp,
                                 enum=Pipeline.Step.REDUCED)
    dir_exists = os.path.exists(in_dir)
    if (dir_exists and \
            len(GenUtilities.getAllFiles(in_dir, ext=".pkl")) > 0):
        data = CheckpointUtilities.lazy_multi_load(in_dir)
    else:
        data = []
    return data


def read_in_energy(base_dir):
    """
    :param base_dir: where the landscape lives; should be a series of FECs of
    about the same spring constant (e.g.  /BR+Retinal/300/170321FEC/)
    :return: RetinalUtil.EnergyWithMeta
    """
    landscape_base = _landscape_dir(base_dir)
    cache_tmp = \
        Pipeline._cache_dir(base=landscape_base,
                            enum=Pipeline.Step.POLISH)
    file_load = cache_tmp + "energy.pkl"
    energy_obj = CheckpointUtilities.lazy_load(file_load)
    obj = EnergyWithMeta(file_load,
                                     landscape_base, energy_obj)
    # read in the data, determine how many curves there are
    data_tmp = read_fecs(obj)
    n_data = len(data_tmp)
    obj.set_n_fecs(n_data)
    return obj


def _read_all_energies(base_dir_analysis):
    """
    :param base_dir_analysis: where we should look (e.g. BR+Retinal)
    :return: list of RetinalUtil.EnergyWithMeta objects
    """
    filtered_dirs = subdirs(base_dir_analysis)
    to_ret = []
    for velocity_directory in filtered_dirs:
        fecs = subdirs(velocity_directory)
        for d in fecs:
            try:
                tmp = read_in_energy(base_dir=d)
                to_ret.append(tmp)
            except (IOError, AssertionError) as e:
                print("Couldn't read from (so skipping): {:s}".format(d))
    energy_list_raw = to_ret
    # get the valid points in the landscape
    energy_list = [valid_landscape(e) for e in energy_list_raw]
    # zero everything
    for e in energy_list:
        n_pts = e.G0.size
        e._G0 -= min(e.G0[:n_pts//2])
        e._iwt_obj._G0 -= e._iwt_obj._G0[0]
    return energy_list_raw

def interpolating_G0(energy_list,num_q=200,
                     f=lambda x_tmp: x_tmp.G0_kcal_per_mol ):
    """
    :param energy_list: list of energy objects to get splines from
    :param num_q: h
    :param f: takes in a function, returns what property we want splines of
    :return: tuple of <q (nm) in range of all energy objects,
                      splines for objects' energy in kcal/mol(q in nm)>
    """
    q_interp =  common_q_interp(energy_list,num_q=num_q)
    # get all the splines
    splines = [spline_fit(q=e.q_nm, G0=f(e))
               for e in energy_list]
    return q_interp, splines

def spline_fit(q, G0, k=3, knots=None,num=None):
    if num is None:
        num = min(150,q.size//(k)-1)
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
    # XXX dont slice anything
    return e._slice(slice(0,None,1))


def min_sep_landscape():
    return 15e-9

def offset_L(info):
    # align by the contour length of the protein
    offset_m = 20e-9
    L0 = info.L0_c_terminal - offset_m
    return L0


def _detect_retract_FEATHER(d,pct_approach,tau_f,threshold,f_refs=None):
    """
    :param d:  TimeSepForce
    :param pct_approach: how much of the retract, starting from the end,
    to use as an effective approach curve
    :param tau_f: fraction for tau
    :param threshold: FEATHERs probability threshold
    :return:
    """
    force_N = d.Force
    # use the last x% as a fake 'approach' (just for noise)
    n = force_N.size
    n_approach = int(np.ceil(n * pct_approach))
    tau_n_points = int(np.ceil(n * tau_f))
    # slice the data for the approach, as described above
    n_approach_start = n - (n_approach + 1)
    fake_approach = d._slice(slice(n_approach_start, n, 1))
    fake_dwell = d._slice(slice(n_approach_start - 1, n_approach_start, 1))
    # make a 'custom' split fec (this is what FEATHER needs for its noise stuff)
    split_fec = Analysis.split_force_extension(fake_approach, fake_dwell, d,
                                               tau_n_points)
    # set the 'approach' number of points for filtering to the retract.
    split_fec.set_tau_num_points_approach(split_fec.tau_num_points)
    # set the predicted retract surface index to a few tau. This avoids looking
    #  at adhesion
    split_fec.get_predicted_retract_surface_index = lambda: 2*tau_n_points
    split_fec.get_predicted_approach_surface_index = lambda : 3*tau_n_points
    pred_info = Detector._predict_split_fec(split_fec, threshold=threshold,
                                            f_refs=f_refs)
    return pred_info, tau_n_points

def _polish_helper(d):
    """
    :param d: AlignedFEC to use
    :return: new FEC, with separation adjusted appropriately
    """
    to_ret = d._slice(slice(0, None, 1))
    # get the slice we are fitting
    inf = to_ret.L0_info
    min_idx = inf.fit_slice.start
    slice_fit = slice(min_idx,None,1)
    to_ret.Separation = to_ret.Separation[slice_fit]
    to_ret.Force = to_ret.Force[slice_fit]
    x,f = to_ret.Separation, to_ret.Force
    # get a grid over all possible forces
    f_grid = np.linspace(min(f), max(f), num=f.size, endpoint=True)
    info_fit = d.info_fit
    ext_FJC = info_fit.ext_FJC(f_grid)
    # we now have X_FJC as a function of force. Therefore, we can subtract
    # off the extension of the PEG3400 to determining the force-extension
    # associated with only the protein (*including* its C-term)
    # note we are getting ext_FJC(f), where f is each point in the original
    # data.
    ext_FJC_all_forces = fit_base._grid_to_data(x=f, x_grid=f_grid,
                                                y_grid=ext_FJC,
                                                bounds_error=False)
    ext_FJC_all_forces[np.isnan(ext_FJC_all_forces)] = 0
    # remove the extension associated with the PEG
    const_offset_x_m = 0
    to_ret.Separation -= ext_FJC_all_forces + const_offset_x_m
    to_ret.ZSnsr -= const_offset_x_m
    # make sure the fitting object knows about the change in extensions...
    ext_FJC_correct_info = info_fit.ext_FJC(info_fit.f_grid)
    to_ret.L0_info.set_x_offset(const_offset_x_m + ext_FJC_correct_info)
    return to_ret