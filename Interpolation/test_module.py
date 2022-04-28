from os import terminal_size
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import gradcheck
from InterpolationModule import InterpolationModule
from interpolation_cupy import ModuleInterpolationCUPY
import time
import numpy

RequireGrad = True

def test_module(tenOne, tenTwo):


    eps = 1e-3

    tenOne_cuda = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenTwo_cuda = Variable(tenTwo.data.type(torch.cuda.FloatTensor), requires_grad = True)

    Net = InterpolationModule()
    t1 = time.time()
    for x in range(1000):
        tenOut_cuda = Net(tenOne_cuda, tenTwo_cuda)
    t2 = time.time()
    if RequireGrad:
        for x in range(1):
            tenOut_cuda.backward(tenOut_cuda.data)
    t3 = time.time()
    print("CUDA GPU Forard and backward time is : " + str(t2-t1) +"ms\t" + str(t3-t2) +"s\t")
    # test = gradcheck(Project, (tenOne_cuda), eps=1e-2, atol=1e-2,raise_exception=True)
    # print('original test:', test)


    tenOne_cuda_cupy = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenTwo_cuda_cupy = Variable(tenTwo.data.type(torch.cuda.FloatTensor), requires_grad = True)
    cupy_Net = ModuleInterpolationCUPY()
    t1 = time.time()
    for x in range(1000):
        cupy_tenOut_cuda = cupy_Net(tenOne_cuda_cupy,tenTwo_cuda_cupy)
    t2 = time.time()
    if RequireGrad:
        for x in range(1):
            cupy_tenOut_cuda.backward(cupy_tenOut_cuda.data)
    t3 = time.time()
    print("CUPY GPU Forward and backward time is : " + str(t2-t1) +"ms\t" + str(t3-t2) +"1s\t")
    # test = gradcheck(cupy_Project, (tenOne_cuda_cupy), eps=1e-2, atol=1e-2,raise_exception=True)
    # print('original test:', test)


    print("Check the output between Ori and My...",end='\t')
    x = tenOut_cuda - cupy_tenOut_cuda
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy() > eps):
        print(x)
        print(torch.mean(torch.abs(tenOut_cuda - cupy_tenOut_cuda)))
        print(torch.mean((tenOut_cuda - cupy_tenOut_cuda)))
    else:
        print("output pass \n")
    

    if RequireGrad:
        print("Check the grad_inp between Ori and My...",end='\t')
        x = tenOne_cuda.grad - tenOne_cuda_cupy.grad
        x = torch.max(torch.abs(x))
        if(x.cpu().data.numpy() > eps):
            print(x)
            print(torch.mean(torch.abs(tenOne_cuda.grad - tenOne_cuda_cupy.grad.cuda())))
            print(torch.mean((tenOne_cuda.grad - tenOne_cuda_cupy.grad.cuda())))
        else:
            print("grad_In pass \n")

        print("Check the grad_flow between Ori and My...",end='\t')
        x = tenTwo_cuda.grad - tenTwo_cuda_cupy.grad
        x = torch.max(torch.abs(x))
        if(x.cpu().data.numpy() > eps):
            print(x)
            print(torch.mean(torch.abs(tenTwo_cuda.grad - tenTwo_cuda_cupy.grad.cuda())))
            print(torch.mean((tenTwo_cuda.grad - tenTwo_cuda_cupy.grad.cuda())))
        else:
            print("grad_In pass \n")

    
    print('tenOut_cuda:\n', tenOut_cuda[0][-1][0][0:10])
    print('cupy_tenOut_cuda:\n', cupy_tenOut_cuda[0][-1][0][0:10])
    if RequireGrad:
        print('tenOne_cuda_gradin:\n', tenOne_cuda.grad[0][-1][0][0:10])
        print('tenOne_cuda_cupy_gradin:\n', tenOne_cuda_cupy.grad[0][-1][0][0:10])
        print('tenFlow_cuda_gradin:\n', tenTwo_cuda.grad[0][-1][0][0:10])
        print('tenFlow_cuda_cupy_gradin:\n', tenTwo_cuda_cupy.grad[0][-1][0][0:10])
    

# B,C,H,W = 1,2,128,128
# B,C,H,W = 1,2,4,4
input = Variable(torch.rand(1,3,64,64).type(torch.FloatTensor))
flow = Variable(torch.rand(1,2,64,64).type(torch.FloatTensor))
for i in range(1):

    input = Variable(input.clone().data, requires_grad = True)  
    flow = Variable(flow.clone().data, requires_grad = True)  
    test_module(input, flow)
