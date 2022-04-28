cupy version of https://github.com/baowenbo/DAIN/tree/master/my_package/Interpolation

test_module.py illustrate how to use it.

original files : [InterpolationModule.py]

cupy version file : [interpolation_cupy.py]

CUDA:
from InterpolationModule import InterpolationModule

CUPY:
from interpolation_cupy import ModuleInterpolationCUPY as InterpolationModule


My env: gpu 2080Ti, pytorch13

run test_module.py
obtain:  

   CUDA GPU Forard and backward time is : 0.03236222267150879ms    0.0023119449615478516s
   
   CUPY GPU Forward and backward time is : 0.2577095031738281ms    0.00464224815368652341s
   
   Check the output between Ori and My...  output pass 
   
   Check the grad_inp between Ori and My...        grad_In pass 
   
   Check the grad_flow between Ori and My...       grad_In pass 
