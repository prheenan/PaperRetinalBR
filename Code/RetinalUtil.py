# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse, enum, copy, os, warnings

from Lib.UtilPipeline import Pipeline
from Lib.AppWHAM.Code import WeightedHistogram
from scipy.interpolate import LSQUnivariateSpline
from Processing import ProcessingUtil

from Lib.AppIWT.Code import InverseWeierstrass
from Lib.AppIWT.Code.InverseWeierstrass import FEC_Pulling_Object
from Lib.AppIWT.Code.UtilLandscape import BidirectionalUtil
from Processing.Util import WLC as WLCHao
from Lib.AppWLC.UtilFit import fit_base
from Lib.AppFEATHER.Code.UtilFEATHER import _detect_retract_FEATHER
from Lib.UtilForce.UtilGeneral import CheckpointUtilities, GenUtilities
from Lib.AppFEATHER.Code import Detector

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
        self._iwt_slices = []
    def _set_iwt_slices(self,s):
        self._iwt_slices = s


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

def min_ext_m():
    to_ret =  np.arange(30,50,step=1)*1e-9
    return to_ret

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

def _fit_sep(d):
    idx = np.arange(d.Separation.size)
    return spline_fit(q=idx, G0=d.Separation)(idx)

def _get_slice(data,min_ext_m):
    fits_d = [ _fit_sep(d) for d in data]
    min_idx = [np.where(d <= min_ext_m)[0][-1] for d in fits_d]
    max_sizes = [d.Separation.size - (i+1) for i,d  in zip(min_idx,data)]
    max_delta = int(min(max_sizes))
    slices = [slice(i,i+max_delta,1) for i in min_idx]
    return slices

def process_helical_slice(data_sliced):
    return data_sliced

def q_GF_nm_plot():
    return _offset_L_m()

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


def read_dir(base_dir,enum):
    in_dir = Pipeline._cache_dir(base=base_dir,
                                 enum=enum)
    dir_exists = os.path.exists(in_dir)
    if (dir_exists and \
            len(GenUtilities.getAllFiles(in_dir, ext=".pkl")) > 0):
        data = CheckpointUtilities.lazy_multi_load(in_dir)
    else:
        data = []
    return data

def read_fecs(e,enum=Pipeline.Step.REDUCED):
    base_tmp = e.base_dir
    return read_dir(base_tmp,enum)


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


def _get_slices(data,exts):
    slices_by_exts = [_get_slice(data,e) for e in exts]
    slices_by_data = [ [s_list[i] for s_list in slices_by_exts ]
                       for i in range(len(data))]
    return slices_by_data


def _sanitize_iwt(data,in_dir):
    velocities = [d.Velocity for d in data]
    # make sure the velocities match within X%
    np.testing.assert_allclose(velocities, velocities[0], atol=0, rtol=1e-2)
    # just set them all equal now
    v_mean = np.mean(velocities)
    for d in data:
        d.Velocity = v_mean
    # repeat for the spring constant
    spring_constants = [d.SpringConstant for d in data]
    K_key = spring_constants[0]
    K_diff = np.max(np.abs(np.array(spring_constants) - K_key)) / \
             np.mean(spring_constants)
    if (K_diff > 1e-2):
        msg = "For {:s}, not all spring constants ({:s}) the same. Replace <K>". \
            format(in_dir, spring_constants)
        warnings.warn(msg)
        # average all the time each K appears
        weighted_mean = np.mean(spring_constants)
        for d in data:
            d.Meta.SpringConstant = weighted_mean
    # get the minimum of the sizes
    np.testing.assert_allclose(data[0].SpringConstant,
                               [d.SpringConstant for d in data],
                               rtol=1e-3)
    return data

def _convert_to_iwt(data,in_dir):
    data = _sanitize_iwt(data,in_dir)
    max_sizes = [d.Force.size for d in data]
    min_of_max_sizes = min(max_sizes)
    # re-slice each data set so they are exactly the same size (as IWT needs)
    data = [d._slice(slice(0, min_of_max_sizes, 1)) for d in data]
    # determine the slices we want for finding the EF helix.
    ex = data[0]
    min_ext_m_tmp = min_ext_m()
    slices = _get_slices(data, min_ext_m_tmp)
    for i, d in enumerate(data):
        # find where we should start
        converted = MetaPulling(d)
        converted._set_iwt_slices(slices[i])
        yield converted


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
    return _offset_L_m() + 7e-9

def min_sep_landscape_nm():
    return min_sep_landscape() * 1e9

def _offset_L_m():
    return -(WLCHao._L0_tail())

def _const_offset(inf):
    offset = -inf._L_shift
    const_offset_x_m = offset - _offset_L_m()
    return const_offset_x_m

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
    const_offset_x_m = _const_offset(inf)
    # XXX remove the extension changes.
    sep_FJC_force = ext_FJC_all_forces
    to_ret.Separation -= sep_FJC_force + const_offset_x_m
    to_ret.ZSnsr -= const_offset_x_m
    # make sure the fitting object knows about the change in extensions...
    ext_FJC_correct_info = info_fit.ext_FJC(info_fit.f_grid)
    to_ret.L0_info.set_x_offset(const_offset_x_m + ext_FJC_correct_info)
    return to_ret


def _def_kw_FEATHER():
    return dict(pct_approach=0.3, tau_f=0.01, threshold=1e-3)

def _is_PEG600(d):
    PEG600_keys = ["BR+Retinal/300nms/170511FEC/"]
    src = d.Meta.SourceFile
    for key in PEG600_keys:
        if key in src:
            return True
    return False


def feather_single(d,force_no_adhesion=False,**kw):
    """
    :param d: FEC to get FJC+WLC fit of
    :param min_F_N: minimum force, in Newtons, for fitting event. helps avoid
     occasional small force events
    :param kw: keywords to use for fitting...
    :return:
    """
    force_N = d.Force
    where_above_surface = np.where(force_N >= 0)[0]
    assert where_above_surface.size > 0, "Force never above surface "
    # use FEATHER; fit to the first event, don't look for adhesion
    d_pred_only = d._slice(slice(0,None,1))
    # first, try removing surface adhesions
    is_600 = _is_PEG600(d)
    skip_adhesion = force_no_adhesion or is_600
    f_refs_initial = [Detector.delta_mask_function] if skip_adhesion else None
    feather_kw =  dict(d=d_pred_only,**kw)
    pred_info,tau_n = _detect_retract_FEATHER(f_refs=f_refs_initial,
                                              **feather_kw)
    # if we removed more than 20nm or we didnt find any events, then
    # FEATHER got confused by a near-surface BR. Tell it not to look for
    # surface adhesions
    expected_surface_m = d.Separation[pred_info.slice_fit.start]
    expected_gf_m = 20e-9
    if ((len(pred_info.event_idx) == 0) or (expected_surface_m > expected_gf_m)):
        f_refs = [Detector.delta_mask_function]
        pred_info,tau_n = _detect_retract_FEATHER(f_refs=f_refs,
                                                              **feather_kw)
    pred_info.tau_n = tau_n
    assert len(pred_info.event_idx) > 0 , "FEATHER can't find an event..."
    to_ret = ProcessingUtil.AlignedFEC(d,info_fit=None,feather_info=pred_info)
    return to_ret


def GF2_event_idx(d,min_F_N):
    pred_info = d.info_feather
    tau_n = pred_info.tau_n
    # POST: FEATHER found something; we need to screen for lower-force events..
    event_idx = [i for i in pred_info.event_idx]
    event_slices = [slice(i - tau_n * 2, i, 1) for i in event_idx]
    # determine the coefficients of the fit
    t, f = d.Time, d.Force
    # loading rate helper has return like:
    # fit_x, fit_y, pred, _, _, _
    list_v = [Detector._loading_rate_helper(t, f, e)
              for e in event_slices]
    # get the predicted force (rupture force), which is the last element of the
    # predicted force.
    pred = [e[2] if len(e[0]) > 0 else [0] for e in list_v]
    f_at_idx = [p[-1] for p in pred]
    valid_events = [i for i, f in zip(event_idx, f_at_idx) if f > min_F_N]
    if (len(valid_events) == 0):
        warnings.warn("Couldn't find high-force events for {:s}". \
                      format(d.Meta.Name))
        # just take the maximum
        valid_events = [event_idx[np.argmax(f_at_idx)]]
    # make sure the event makes sense
    max_fit_idx = valid_events[0]
    return max_fit_idx

def align_single(d,min_F_N,**kw):
    """
    :param d: FEC to get FJC+WLC fit of
    :param min_F_N: minimum force, in Newtons, for fitting event. helps avoid
     occasional small force events
    :param kw: keywords to use for fitting...
    :return:
    """
    force_N = d.Force
    pred_info = d.info_feather
    max_fit_idx = GF2_event_idx(d,min_F_N)
    where_above_surface = np.where(force_N >= 0)[0]
    first_time_above_surface = where_above_surface[0]
    assert first_time_above_surface < max_fit_idx , \
        "Couldn't find fitting region"
    # start the fit after any potential adhesions
    fit_start = max(first_time_above_surface,pred_info.slice_fit.start)
    fit_slice = slice(fit_start,max_fit_idx,1)
    # slice the object to just the region we want
    obj_slice = d._slice(fit_slice)
    # fit wlc to the f vs x of that slice
    info_fit = WLCHao.hao_fit(obj_slice.Separation,obj_slice.Force,**kw)
    info_fit.fit_slice = fit_slice
    to_ret = ProcessingUtil.AlignedFEC(d,info_fit,feather_info=pred_info)
    return to_ret

def _align_and_cache(d,out_dir,force=False,**kw):
    return ProcessingUtil._cache_individual(d, out_dir, align_single,
                                            force,d, **kw)

def func(args):
    x, out_dir, kw = args
    to_ret = _align_and_cache(x,out_dir,**kw)
    return to_ret

def _multi_align(out_dir,kw,all_data,n_pool):
    input_v = [ [d,out_dir,kw] for d in all_data]
    to_ret = ProcessingUtil._multiproc(func, input_v, n_pool)
    to_ret = [r for r in to_ret if r is not None]
    return to_ret

def align_data(base_dir,out_dir,n_pool,**kw):
    all_data = CheckpointUtilities.lazy_multi_load(base_dir)
    return  _multi_align(out_dir,kw,all_data,n_pool)