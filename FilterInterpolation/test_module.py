from os import terminal_size
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import gradcheck
from FilterInterpolationModule import FilterInterpolationModule
from filterinterpolation_cupy import ModuleFilterInterpolation
import time
import numpy

RequireGrad = True

def test_module(tenOne, tenTwo, tenThree):


    eps = 1e-3

    tenOne_cuda = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenTwo_cuda = Variable(tenTwo.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenThree_cuda = Variable(tenThree.data.type(torch.cuda.FloatTensor), requires_grad = True)

    Net = FilterInterpolationModule()
    t1 = time.time()
    tenOut_cuda = Net(tenOne_cuda, tenTwo_cuda, tenThree_cuda)
    t2 = time.time()
    if RequireGrad:
        tenOut_cuda.backward(tenOut_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # test = gradcheck(Project, (tenOne_cuda), eps=1e-2, atol=1e-2,raise_exception=True)
    # print('original test:', test)


    tenOne_cuda_cupy = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenTwo_cuda_cupy = Variable(tenTwo.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenThree_cuda_cupy = Variable(tenThree.data.type(torch.cuda.FloatTensor), requires_grad = True)

    cupy_Net = ModuleFilterInterpolation()
    t1 = time.time()
    cupy_tenOut_cuda = cupy_Net(tenOne_cuda_cupy,tenTwo_cuda_cupy, tenThree_cuda_cupy)
    t2 = time.time()
    if RequireGrad:
        cupy_tenOut_cuda.backward(cupy_tenOut_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
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
            print("grad_Flow pass \n")

        print("Check the grad_flow between Ori and My...",end='\t')
        x = tenThree_cuda.grad - tenThree_cuda_cupy.grad
        x = torch.max(torch.abs(x))
        if(x.cpu().data.numpy() > eps):
            print(x)
            print(torch.mean(torch.abs(tenThree_cuda.grad - tenThree_cuda_cupy.grad.cuda())))
            print(torch.mean((tenThree_cuda.grad - tenThree_cuda_cupy.grad.cuda())))
        else:
            print("grad_Filter pass \n")

    
    print('tenOut_cuda:\n', tenOut_cuda[0][-1][0][0:10])
    print('cupy_tenOut_cuda:\n', cupy_tenOut_cuda[0][-1][0][0:10])
    if RequireGrad:
        print('tenOne_cuda_gradin:\n', tenOne_cuda.grad[0][-1][0][0:10])
        print('tenOne_cuda_cupy_gradin:\n', tenOne_cuda_cupy.grad[0][-1][0][0:10])
        print('tenFlow_cuda_gradin:\n', tenTwo_cuda.grad[0][-1][0][0:10])
        print('tenFlow_cuda_cupy_gradin:\n', tenTwo_cuda_cupy.grad[0][-1][0][0:10])
        print('tenFilter_cuda_gradin:\n', tenThree_cuda.grad[0][-1][0][0:10])
        print('tenFilter_cuda_cupy_gradin:\n', tenThree_cuda_cupy.grad[0][-1][0][0:10])
        
    

H,W = 32,32
filtersize = 4
input = Variable(torch.rand(1,3,H,W).type(torch.FloatTensor))
flow = Variable(torch.rand(1,2,H,W).type(torch.FloatTensor))
filter = Variable(torch.rand(1,filtersize**2,H,W).type(torch.FloatTensor))
for i in range(1):
    input.data.uniform_(0, 1)
    flow.data.uniform_(-1, 1)
    filter.data.uniform_(0,1)
    input = Variable(input.clone().data, requires_grad = True)  
    flow = Variable(flow.clone().data, requires_grad = True)
    filter = Variable(filter.clone().data, requires_grad = True)  
    test_module(input, flow, filter)