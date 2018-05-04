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
import RetinalUtil

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
    [str_BR,f_v(50),f_date("170502"),[1372,1374,2160]],
    [str_BR,f_v(50),f_date("170503"),[1268]],
    [str_BR, f_v(300), f_date("170321"), [500,760,786,821]],
    [str_BR, f_v(300), f_date("170501"), [203]],
    [str_BR, f_v(300), f_date("170502"), []], # yep, this one is OK.
    [str_BR, f_v(300), f_date("170511"), []], # this one too.
    [str_BR, f_v(3000), f_date("170502"), [717]],
    [str_BR, f_v(3000), f_date("170503"), [231,]],
    # all the blacklisted BO data
    [str_BO, f_v(50), f_date("170523"), [176,223]],
    [str_BO, f_v(300), f_date("170327"), [386,]],
    [str_BO, f_v(3000), f_date("170523"), [18,69,509,773]],
]

blacklists = [Blacklist(*t) for t in blacklist_tuples]
blacklist_dicts = dict([((b.str_pm_bR,b.str_vel,b.str_date),b.list_ids)
                       for b in blacklists])

def filter_by_bl(data,base_input_processing):
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
    blacklist_tmp = blacklist_dicts[(str_br_type,str_vel,str_data)]
    ids_groups = [re.search("(\d+)", d.Meta.Name) for d in data]
    for i in ids_groups:
        assert i is not None
    # POST: found all ids
    ids = [int(i.group(0)) for i in ids_groups]
    # make sure all ids in the blacklist are actually in this group
    for tmp in blacklist_tmp:
        assert tmp in ids
    # POST: all ids in blacklist are in the data
    for i, d in zip(ids, data):
        if i not in blacklist_tmp:
            yield d

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_input_processing = RetinalUtil._processing_base()
    base_dir = RetinalUtil._landscape_base()
    step = Pipeline.Step.MANUAL
    in_dir = Pipeline._cache_dir(base=base_input_processing,
                                 enum=Pipeline.Step.POLISH)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    force = True
    functor = lambda : filter_by_bl(data,base_input_processing)
    CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                   force=force,
                                   name_func=FEC_Util.fec_name_func)

if __name__ == "__main__":
    run()
