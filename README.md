# cupy_packages

cupy packages

sepconv: https://github.com/sniklaus/revisiting-sepconv

flowprojection: https://github.com/baowenbo/DAIN/tree/master/my_package/FlowProjection

interpolation: https://github.com/baowenbo/DAIN/tree/master/my_package/Interpolation

depthflowprojection: https://github.com/baowenbo/DAIN/tree/master/my_package/DepthFlowProjection

filterinterpolation: https://github.com/baowenbo/DAIN/tree/master/my_package/FilterInterpolation

# CUPY
It can be installed using pip install cupy or alternatively using one of the provided binary packages as outlined in the CuPy repository.

# Usage, for example depthflowprojection:
  1. copy [DepthFlowProjection/depthflowprojection_cupy.py] into original folder [DepthFlowProjection]
  2. change the import of [networks/DAIN.py] 
     from: 
     from my_package.DepthFlowProjection import DepthFlowProjectionModule
     to:
     from my_package.DepthFlowProjection.depthflowprojection_cupy import ModuleDepthFlowProj as DepthFlowProjectionModule
