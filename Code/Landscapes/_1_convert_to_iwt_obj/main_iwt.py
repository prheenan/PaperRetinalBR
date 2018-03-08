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
from Lib.AppIWT.Code import WeierstrassUtil
import RetinalUtil

def to_iwt(in_dir):
    data = CheckpointUtilities.lazy_multi_load(in_dir)
    # make sure they all have the same velocity
    velocities = [d.Velocity for d in data]
    # make sure the velocities match within X%
    np.testing.assert_allclose(velocities,velocities[0],atol=0,rtol=1e-3)
    # repeat for the spring constant
    spring_constants = [d.SpringConstant for d in data]
    np.testing.assert_allclose(spring_constants,spring_constants[0],
                               atol=0,rtol=1e-3)
    # get the minimum of the sizes
    max_sizes = [d.Force.size for d in data]
    min_of_max_sizes = min(max_sizes)
    # re-slice each data set so they are exactly the same size (as IWT needs)
    data = [d._slice(slice(0,min_of_max_sizes,1)) for d in data]
    for d in data:
        # find where we should start
        converted = RetinalUtil.MetaPulling(d)
        yield converted


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    base_dir = RetinalUtil._landscape_base()
    step = Pipeline.Step.REDUCED
    in_dir = Pipeline._cache_dir(base=base_dir,
                                 enum=Pipeline.Step.SANITIZED)
    out_dir = Pipeline._cache_dir(base=base_dir,enum=step)
    force = True
    limit = None
    functor = lambda : to_iwt(in_dir)
    data =CheckpointUtilities.multi_load(cache_dir=out_dir,load_func=functor,
                                         force=force,
                                         limit=limit,
                                         name_func=FEC_Util.fec_name_func)
    pass

if __name__ == "__main__":
    run()
