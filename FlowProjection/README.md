cupy version of https://github.com/baowenbo/DAIN/tree/master/my_package/FlowProjection

test_module.py illustrate how to use it.

original files : [__init__.py, flowprojection_cuda.cc, flowprojection_cuda_kernel.cu, flowprojection_cuda_kernel.cuh, FlowProjectionLayer.py, FlowProjectionModule.py, setup.py]

cupy version file : [flowprojection_cupy.py]

CUDA:
from  FlowProjectionModule import  FlowProjectionModule

CUPY:
from flowprojection_cupy import ModuleFlowProj as FlowProjectionModule
