from os import terminal_size
from numpy.core.numeric import True_
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import gradcheck
from  DepthFlowProjectionModule import  DepthFlowProjectionModule
from depthflowprojection_cupy import ModuleDepthFlowProj 
import time
import numpy

TEST_FILLHOLE = False

def test_module(tenOne, tenTwo):

    # tenOne: input flow
    # tenTwo: input depth

    if TEST_FILLHOLE == True:
        reqs_grad = False
    else:
        reqs_grad = True

    eps = 1e-3

    tenOne_cuda = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = reqs_grad)
    tenTwo_cuda = Variable(tenTwo.data.type(torch.cuda.FloatTensor), requires_grad = reqs_grad)

    Project = DepthFlowProjectionModule()
    t1 = time.time()
    tenOut_cuda = Project(tenOne_cuda, tenTwo_cuda)
    t2 = time.time()
    if reqs_grad:
        tenOut_cuda.backward(tenOut_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # test = gradcheck(Project, (tenOne_cuda, tenTwo_cuda), eps=1e-2, atol=1e-2,raise_exception=True)
    # print('original test:', test)


    tenOne_cuda_cupy = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = reqs_grad)
    tenTwo_cuda_cupy = Variable(tenTwo.data.type(torch.cuda.FloatTensor), requires_grad = reqs_grad)
    cupy_Project = ModuleDepthFlowProj()
    t1 = time.time()
    cupy_tenOut_cuda = cupy_Project(tenOne_cuda_cupy, tenTwo_cuda_cupy)
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
        print("Check the grad_Flow between Ori and My...",end='\t')
        x = tenOne_cuda.grad - tenOne_cuda_cupy.grad
        x = torch.max(torch.abs(x))
        if(x.cpu().data.numpy() > eps):
            print(x)
            print(torch.mean(torch.abs(tenOne_cuda.grad - tenOne_cuda_cupy.grad.cuda())))
            print(torch.mean((tenOne_cuda.grad - tenOne_cuda_cupy.grad.cuda())))
        else:
            print("grad_Flow pass \n")

        print("Check the grad_Depth between Ori and My...",end='\t')
        x = tenTwo_cuda.grad - tenTwo_cuda_cupy.grad
        x = torch.max(torch.abs(x))
        if(x.cpu().data.numpy() > eps):
            print(x)
            print(torch.mean(torch.abs(tenTwo_cuda.grad - tenTwo_cuda_cupy.grad.cuda())))
            print(torch.mean((tenTwo_cuda.grad - tenTwo_cuda_cupy.grad.cuda())))
        else:
            print("grad_Depth pass \n")

    
    print('tenOut_cuda:\n', tenOut_cuda[0][-1][0][0:10])
    print('cupy_tenOut_cuda:\n', cupy_tenOut_cuda[0][-1][0][0:10])
    if reqs_grad:
        print('Grad_Flow:\n', tenOne_cuda.grad[0][-1][0][0:10])
        print('Grad_Flow_Cupy:\n', tenOne_cuda_cupy.grad[0][-1][0][0:10])
        print('Grad_Depth:\n', tenTwo_cuda.grad[0][-1][0][0:10])
        print('Grad_Depth_Cupy:\n', tenTwo_cuda_cupy.grad[0][-1][0][0:10])
    


# input1 = Variable(torch.rand(1,2,64,64).type(torch.FloatTensor))
# input2 = Variable(torch.rand(1,1,64,64).type(torch.FloatTensor))

input1 = Variable(torch.rand(1,2,8,8).type(torch.FloatTensor))
input2 = Variable(torch.rand(1,1,8,8).type(torch.FloatTensor))

for i in range(1):
    input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in 
    input2 = Variable(input2.clone().data, requires_grad = True) # to delete the graph in 
    test_module(input1, input2)