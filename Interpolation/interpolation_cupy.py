#!/usr/bin/env python

import torch
import cupy
import re

kernel_forward_interpolation_UpdateOutput= '''
    extern "C" __global__ void kernel_forward_interpolation_UpdateOutput(
        const int n,
        const float* input,
        const float* flow,
        float* output
    ){  for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output)   ) % SIZE_0(output);
            const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                    ) % SIZE_1(output);
            const int intY = ( intIndex / SIZE_3(output)                                     ) % SIZE_2(output);
            const int intX = ( intIndex                                                      ) % SIZE_3(output);

            float fx = (float)VALUE_4(flow, intN, 0, intY, intX);
            float fy = (float)VALUE_4(flow, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int Xsize = SIZE_3(output);
            int Ysize = SIZE_2(output);
        
            if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 < Xsize  && y2 <  Ysize ){
                int ix2_L = (int) (x2);
                int iy2_T = (int) (y2);
                int ix2_R =  (((ix2_L + 1)<(Xsize - 1))?(ix2_L + 1):(Xsize - 1));
                int iy2_B =  (((iy2_T + 1)<(Ysize - 1))?(iy2_T + 1):(Ysize - 1));

                float alpha = x2 - ix2_L;
                float beta  = y2 - iy2_T;

                float TL = (float)VALUE_4(input, intN, intC, iy2_T, ix2_L);
                float TR = (float)VALUE_4(input, intN, intC, iy2_T, ix2_R);
                float BL = (float)VALUE_4(input, intN, intC, iy2_B, ix2_L);
                float BR = (float)VALUE_4(input, intN, intC, iy2_B, ix2_R);
                
                output[intIndex] = (1-alpha)*(1-beta)*TL + alpha*(1-beta)*TR + (1-alpha)*beta*BL + alpha*beta*BR;
            
            } else{
                output[intIndex]  = 0.0f;
            }
            
        }
    }
'''



kernel_flowproj_updateGradInput= '''
extern "C" __global__ void kernel_flowproj_updateGradInput(
        const int n,
        const float* input,
        const float* flow,
        const float* gradOutput,
        float* gradInput,
        float* gradFlow
    ){  for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput)   ) % SIZE_0(gradInput);
            const int intC = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                       ) % SIZE_1(gradInput);
            const int intY = ( intIndex / SIZE_3(gradInput)                                           ) % SIZE_2(gradInput);
            const int intX = ( intIndex                                                               ) % SIZE_3(gradInput);

            float fx = (float)VALUE_4(flow, intN, 0, intY, intX);
            float fy = (float)VALUE_4(flow, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int Xsize = SIZE_3(gradInput);
            int Ysize = SIZE_2(gradInput);
        
            if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 < Xsize  && y2 <  Ysize ){
                int ix2_L = (int) (x2);
                int iy2_T = (int) (y2);
                int ix2_R =  (((ix2_L + 1)<(Xsize - 1))?(ix2_L + 1):(Xsize - 1));
                int iy2_B =  (((iy2_T + 1)<(Ysize - 1))?(iy2_T + 1):(Ysize - 1));

                float alpha = x2 - ix2_L;
                float beta  = y2 - iy2_T;

                float gradoutput_value = VALUE_4(gradOutput, intN, intC, intY, intX);

                atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, iy2_T, ix2_L)],  gradoutput_value*(1-alpha)*(1-beta));
                atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, iy2_T, ix2_R)],  gradoutput_value*alpha*(1-beta));
                atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, iy2_B, ix2_L)],  gradoutput_value*(1-alpha)*beta);
                atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, iy2_B, ix2_R)],  gradoutput_value*alpha*beta);
            
            }
            
        }
    }
'''

kernel_flowproj_updateGradFlow= '''
extern "C" __global__ void kernel_flowproj_updateGradFlow(
        const int n,
        const float* input,
        const float* flow,
        const float* gradOutput,
        float* gradInput,
        float* gradFlow
    ){  for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            // use input, since shape of input equals output
            const int intN = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow) / SIZE_1(gradFlow)  ) % SIZE_0(gradFlow);
            const int intC = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow)                     ) % SIZE_1(gradFlow);
            const int intY = ( intIndex / SIZE_3(gradFlow)                                        ) % SIZE_2(gradFlow);
            const int intX = ( intIndex                                                           ) % SIZE_3(gradFlow);

            float fx = (float)VALUE_4(flow, intN, 0, intY, intX);
            float fy = (float)VALUE_4(flow, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int Xsize = SIZE_3(gradOutput);
            int Ysize = SIZE_2(gradOutput);

            if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 < (float)Xsize  && y2 <  (float)Ysize ){

                int ix2_L = (int) (x2);
                int iy2_T = (int) (y2);
                int ix2_R =  (((ix2_L + 1)<(Xsize - 1))?(ix2_L + 1):(Xsize - 1));
                int iy2_B =  (((iy2_T + 1)<(Ysize - 1))?(iy2_T + 1):(Ysize - 1));

                float alpha = x2 - ix2_L;
                float beta  = y2 - iy2_T;

                
                
                float fltGradFlow = 0.0;

                if (intC == 0)
                {   
                    for (int intChannel  = 0; intChannel  < SIZE_1(gradOutput); intChannel ++) {
    
                        fltGradFlow += VALUE_4(gradOutput, intN, intChannel, intY, intX) *
                         (
                            (1 - beta) * ( VALUE_4(input, intN, intChannel, iy2_T, ix2_R) - VALUE_4(input, intN, intChannel, iy2_T, ix2_L) )
                            + beta     * ( VALUE_4(input, intN, intChannel, iy2_B, ix2_R) - VALUE_4(input, intN, intChannel, iy2_B, ix2_L) )
                         );
                    }
                }

                if (intC == 1)
                {
                    for (int intChannel = 0; intChannel < SIZE_1(gradOutput); intChannel++) {

                        fltGradFlow += VALUE_4(gradOutput, intN, intChannel, intY, intX) * 
                        (
                            (1 - alpha) * ( VALUE_4(input, intN, intChannel, iy2_B, ix2_L) - VALUE_4(input, intN, intChannel, iy2_T, ix2_L) )
                            +  alpha    * ( VALUE_4(input, intN, intChannel, iy2_B, ix2_R) - VALUE_4(input, intN, intChannel, iy2_T, ix2_R) )
                        );
                    }
                }

                gradFlow[OFFSET_4(gradFlow, intN, intC, intY, intX)] = fltGradFlow;


            
            }
            
        }
    }
'''




def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel
# end

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class _FunctionInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(self, input, flow):
        intSamples = input.shape[0]  # N
        intInputDepth, intInputHeight, intInputWidth  = input.shape[1], input.shape[2], input.shape[3] # C H W
        intFlowDepth,  intFlowHeight,  intFlowWidth   = flow.shape[1],  flow.shape[2],  flow.shape[3]

        assert(intFlowDepth==2)
        assert(intInputHeight == intFlowHeight)
        assert(intInputWidth == intFlowWidth)

        input = input.contiguous()
        assert(input.is_cuda == True)
        flow = flow.contiguous()
        assert(flow.is_cuda == True)

        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])
    

        if input.is_cuda == True:

            cupy_launch('kernel_forward_interpolation_UpdateOutput', cupy_kernel('kernel_forward_interpolation_UpdateOutput', {
                'input': input,
                'flow': flow,
                'output': output
            }))(
                grid=tuple([ int((output.nelement() + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ cupy.int32(output.nelement()), input.data_ptr(), flow.data_ptr(), output.data_ptr() ]
            )

        elif input.is_cuda == False:
            raise NotImplementedError()
        # end

        self.save_for_backward(input, flow)

        return output
    
    # end

    @staticmethod
    def backward(self, gradOutput):

        input, flow = self.saved_tensors

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth,  intFlowHeight,  intFlowWidth   = flow.shape[1],  flow.shape[2],  flow.shape[3]

        assert(intFlowDepth==2)
        assert(intInputHeight == intFlowHeight)
        assert(intInputWidth == intFlowWidth)

        gradOutput = gradOutput.contiguous(); assert(gradOutput.is_cuda == True)

        gradInput = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth]) if self.needs_input_grad[0] == True else None
        gradFlow  = input.new_zeros([ intSamples, intFlowDepth,  intFlowHeight,  intFlowWidth ]) if self.needs_input_grad[1] == True else None

        if input.is_cuda == True:

            if gradInput is not None:
                n = gradInput.nelement()
                cupy_launch('kernel_flowproj_updateGradInput', cupy_kernel('kernel_flowproj_updateGradInput',{
                    'input': input,
                    'flow': flow,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradFlow': gradFlow
                }))(
                    grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ cupy.int32(n), input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr(), None]
                )
            # end


            if gradFlow is not None:
                n = gradFlow.nelement()
                cupy_launch('kernel_flowproj_updateGradFlow', cupy_kernel('kernel_flowproj_updateGradFlow',{
                    'input': input,
                    'flow': flow,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradFlow': gradFlow
                }))(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([512, 1, 1 ]),
					args=[ cupy.int32(n), input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), None, gradFlow.data_ptr()]
                )
            # end

        elif input.is_cuda == False:
            raise NotImplementedError()
        # end

        return gradInput, gradFlow
# end



def FunctionInterpolation(teninput, tenflow):
    tenOutput = _FunctionInterpolation.apply(teninput, tenflow)
    return tenOutput

class ModuleInterpolationCUPY(torch.nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, teninput, tenflow):
        return _FunctionInterpolation.apply(teninput, tenflow)