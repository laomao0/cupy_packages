import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import gradcheck

from sepconv import my_sepconv # the custom separable convolution layer
from sepconv import sepconv # the custom separable convolution layer


import time
import numpy

def test_sepconv(tenOne, tenVer, tenHor):

    eps = 1e-2

    module_sepconv = sepconv.ModuleSepConv()

    tenOne_cuda = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenVer_cuda = Variable(tenVer.data.type(torch.cuda.FloatTensor), requires_grad = True)
    tenHor_cuda = Variable(tenHor.data.type(torch.cuda.FloatTensor), requires_grad = True)

    tenout_cuda = module_sepconv(tenOne_cuda, tenVer_cuda, tenHor_cuda)
    
    tenout_cuda.backward(tenout_cuda.data)

    # print('tenOne_cuda:', tenOne_cuda.grad[0][0])

    # RuntimeError: Jacobian mismatch for output 0 with respect to input 0,
    # The default values for the step in the gradcheck function are for double precision numbers.
    # If you cast all your tensors and modules to double, the check will work.
    # If you want to work with single precision, you will need to increase eps to 1e-3 or 1e-4.
    
    # test = gradcheck(module_sepconv, (tenOne_cuda, tenVer_cuda, tenHor_cuda), eps=1e-2, atol=1e-2,raise_exception=True)

    # print('original test:', test)

    # test my sepconv 

    my_module_sepconv = my_sepconv.ModuleSepConv()

    my_tenOne_cuda = Variable(tenOne.data.type(torch.cuda.FloatTensor), requires_grad = True)
    my_tenVer_cuda = Variable(tenVer.data.type(torch.cuda.FloatTensor), requires_grad = True)
    my_tenHor_cuda = Variable(tenHor.data.type(torch.cuda.FloatTensor), requires_grad = True)

    my_tenout_cuda = my_module_sepconv(my_tenOne_cuda, my_tenVer_cuda, my_tenHor_cuda)
    
    # test = gradcheck(my_module_sepconv, (my_tenOne_cuda, my_tenVer_cuda, my_tenHor_cuda), eps=1e-2, atol=1e-2,raise_exception=True)

    # print('my_implementation test:', test)

    my_tenout_cuda.backward(my_tenout_cuda.data)

    ############################################################
    
    
    print("Check the grad_inp between Ori and My...",end='\t')
    x = tenOne_cuda.grad - my_tenOne_cuda.grad
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy() > eps):
        print(x)
        print(torch.mean(torch.abs(tenOne_cuda.grad - my_tenOne_cuda.grad.cuda())))
        print(torch.mean((tenOne_cuda.grad - my_tenOne_cuda.grad.cuda())))
    else:
        print("grad_In pass \n")

    ############################################################

    print("Check the grad_ver between Ori and My...",end='\t')
    x = tenVer_cuda.grad - my_tenVer_cuda.grad
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy() > eps):
        print(x)
        print(torch.mean(torch.abs(tenVer_cuda.grad - my_tenVer_cuda.grad.cuda())))
        print(torch.mean((tenVer_cuda.grad - my_tenVer_cuda.grad.cuda())))
    else:
        print("grad_ver pass \n")


    ############################################################
    
    print("Check the grad_hor between Ori and My...",end='\t')
    x = tenHor_cuda.grad - my_tenHor_cuda.grad
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy() > eps):
        print(x)
        print(torch.mean(torch.abs(tenHor_cuda.grad - my_tenHor_cuda.grad.cuda())))
        print(torch.mean((tenHor_cuda.grad - my_tenHor_cuda.grad.cuda())))
    else:
        print("grad_hor pass \n")

    
    print('test over~')
    print('----------------------------------------')

    print('tenOne_cuda:\n', tenOne_cuda.grad[0][-1])
    print('my_tenout_cuda:\n', my_tenOne_cuda.grad[0][-1])
    print('tenVer_cuda:\n', tenVer_cuda.grad[0][-1])
    print('my_tenVer_cuda:\n', my_tenVer_cuda.grad[0][-1])
    print('tenHor_cuda:\n', tenHor_cuda.grad[0][-1])
    print('my_tenHor_cuda:\n', my_tenHor_cuda.grad[0][-1])


    



B, C1, C2, H1, W1, H2, W2 = 1, 4, 51, 438, 634, 388, 584

# B, C1, C2, H1, W1, H2, W2 = 1, 4, 51, 8+50, 8+50, 8, 8

tenOne = Variable(torch.arange(0.0, B * C1 * H1 * W1).view(B, C1, H1, W1), requires_grad=True)
tenVer = Variable(torch.arange(0.0, B * C2 * H2 * W2).view(B, C2, H2, W2), requires_grad=True)
tenHor = Variable(torch.arange(0.0, B * C2 * H2 * W2).view(B, C2, H2, W2), requires_grad=True)

ftimes = []
btimes = []

for i in range(1):
    tenOne.data.uniform_(-1.0, 1.0)
    tenVer.data.uniform_(-1.0, 1.0)
    tenHor.data.uniform_(-1.0, 1.0)

    tenOne = Variable(tenOne.clone().data, requires_grad = True) 
    tenVer = Variable(tenVer.clone().data, requires_grad = True) 
    tenHor = Variable(tenHor.clone().data, requires_grad = True) 

    test_sepconv(tenOne, tenVer, tenHor)


