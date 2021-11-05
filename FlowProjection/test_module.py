from os import terminal_size
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import gradcheck
from  FlowProjectionModule import  FlowProjectionModule
from flowprojection_cupy import ModuleFlowProj
import time
import numpy

TEST_FILLHOLE = False

def test_module(tenOne):

    if TEST_FILLHOLE == True:
        reqs_grad = False
    else:
        reqs_grad = True

    eps = 1e-3

    tenOne_cuda = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = reqs_grad)
    Project = FlowProjectionModule()
    t1 = time.time()
    tenOut_cuda = Project(tenOne_cuda)
    t2 = time.time()
    if reqs_grad:
        tenOut_cuda.backward(tenOut_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # test = gradcheck(Project, (tenOne_cuda), eps=1e-2, atol=1e-2,raise_exception=True)
    # print('original test:', test)


    tenOne_cuda_cupy = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = reqs_grad)
    cupy_Project = ModuleFlowProj()
    t1 = time.time()
    cupy_tenOut_cuda = cupy_Project(tenOne_cuda_cupy)
    t2 = time.time()
    if reqs_grad:
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
        print(torch.mean(torch.abs(tenOut_cuda - cupy_tenOut_cuda.cuda())))
        print(torch.mean((tenOut_cuda - cupy_tenOut_cuda.cuda())))
    else:
        print("output pass \n")
    

    if reqs_grad:
        print("Check the grad_inp between Ori and My...",end='\t')
        x = tenOne_cuda.grad - tenOne_cuda_cupy.grad
        x = torch.max(torch.abs(x))
        if(x.cpu().data.numpy() > eps):
            print(x)
            print(torch.mean(torch.abs(tenOne_cuda.grad - tenOne_cuda_cupy.grad.cuda())))
            print(torch.mean((tenOne_cuda.grad - tenOne_cuda_cupy.grad.cuda())))
        else:
            print("grad_In pass \n")

    
    print('tenOut_cuda:\n', tenOut_cuda[0][-1][0][0:10])
    print('cupy_tenOut_cuda:\n', cupy_tenOut_cuda[0][-1][0][0:10])
    if reqs_grad:
        print('tenOne_cuda_gradin:\n', tenOne_cuda.grad[0][-1][0][0:10])
        print('tenOne_cuda_cupy_gradin:\n', tenOne_cuda_cupy.grad[0][-1][0][0:10])
    

B,C,H,W = 1,2,512,704
# B,C,H,W = 1,2,4,4
input1 = Variable(torch.arange(0.0, B*C * H * W).view(B, C, H, W), requires_grad=True)
for i in range(1):
    input1.data.uniform_(-1.0, 1.0)
    input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in 
    test_module(input1)