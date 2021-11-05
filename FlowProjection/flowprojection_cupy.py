#!/usr/bin/env python

import torch
import cupy
import re

kernel_flowproj_updateOutput= '''
    extern "C" __global__ void kernel_flowproj_updateOutput(
        const int n,
        const float* input,
        float* output,
        float* count
    ){ for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(input) / SIZE_2(input) / SIZE_1(input)   ) % SIZE_0(input);
            const int intC = ( intIndex / SIZE_3(input) / SIZE_2(input)                   ) % SIZE_1(input);
            const int intY = ( intIndex / SIZE_3(input)                                   ) % SIZE_2(input);
            const int intX = ( intIndex                                                   ) % SIZE_3(input);

            float fx = (float)VALUE_4(input, intN, 0, intY, intX);
            float fy = (float)VALUE_4(input, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int Xsize = (int)(SIZE_3(output));
            int Ysize = (int)(SIZE_2(output));

            if (  intC == 0 ){

                if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 <= (float)(Xsize-1)  && y2 <=  (float)(Ysize-1) ){
                    int ix2_L = (int) (x2);
                    int iy2_T = (int) (y2);
                    int ix2_R =  (((ix2_L + 1)<(Xsize - 1))?(ix2_L + 1):(Xsize - 1));   //min(ix2_L + 1, Xsize - 1);
                    int iy2_B =  (((iy2_T + 1)<(Ysize - 1))?(iy2_T + 1):(Ysize - 1));   //min(iy2_T + 1, Ysize - 1);

                  
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_T, ix2_L)],  -fx);
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_T, ix2_R)],  -fx);
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_B, ix2_L)],  -fx);
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_B, ix2_R)],  -fx);

                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_T, ix2_L)],  -fy);
                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_T, ix2_R)],  -fy);
                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_B, ix2_L)],  -fy);
                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_B, ix2_R)],  -fy);

                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_T, ix2_L)],  +1);
                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_T, ix2_R)],  +1);
                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_B, ix2_L)],  +1);
                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_B, ix2_R)],  +1);
                }
            }
        
    }
    }
'''

kernel_flowproj_updataAvg= '''
    extern "C" __global__ void kernel_flowproj_updataAvg(
        const int n,
        const float* input,
        float* output,
        float* count
    ){ for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
        const int intN = ( intIndex / SIZE_3(input) / SIZE_2(input) / SIZE_1(input)   ) % SIZE_0(input);
        const int intC = ( intIndex / SIZE_3(input) / SIZE_2(input)                   ) % SIZE_1(input);
        const int intY = ( intIndex / SIZE_3(input)                                   ) % SIZE_2(input);
        const int intX = ( intIndex                                                   ) % SIZE_3(input);

        float temp = VALUE_4(count, intN, 0, intY, intX);
        if( temp > 0.0f){
            output[OFFSET_4(output, intN, intC, intY, intX)] /= temp;
        }
        
    }
    }


'''

kernel_flowproj_updaFillhole= '''
    extern "C" __global__ void kernel_flowproj_updaFillhole(
        const int n,
        const float* input,
        float* output,
        float* count
    ){ for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
        const int intN = ( intIndex / SIZE_3(input) / SIZE_2(input) / SIZE_1(input)   ) % SIZE_0(input);
        const int intC = ( intIndex / SIZE_3(input) / SIZE_2(input)                   ) % SIZE_1(input);
        const int intY = ( intIndex / SIZE_3(input)                                   ) % SIZE_2(input);
        const int intX = ( intIndex                                                   ) % SIZE_3(input);
        
        int Xsize = (int)(SIZE_3(input));
        int Ysize = (int)(SIZE_2(input));

        float temp = VALUE_4(count, intN, 0, intY, intX);

        if (temp <= 0.0f){

            //search along the four directions,0/90/180/270, until finding at least one
            int left_offset = intX;
            float left_temp = 0.0f;
            while(left_temp == 0.0f && left_offset - 1 >= 0){
                left_offset = left_offset -1;
                left_temp = VALUE_4(count, intN, 0, intY, left_offset);
            }

            int right_offset = intX;
            float right_temp = 0.0f;
            while(right_temp ==0.0f && right_offset + 1 <= Xsize - 1 ){
                right_offset  = right_offset + 1 ;
                right_temp = VALUE_4(count, intN, 0, intY, right_offset);
            }

            int up_offset = intY ;            
            float up_temp = 0.0f;
            while(up_temp == 0.0f && up_offset - 1 >=0){
                up_offset = up_offset - 1;
                up_temp =  VALUE_4(count, intN, 0, up_offset, intX);
            }

            int down_offset = intY;            
            float down_temp = 0.0f;
            while(down_temp == 0.0f && down_offset + 1 <= Ysize - 1 ){
                down_offset = down_offset + 1;
                down_temp =  VALUE_4(count, intN, 0, down_offset, intX);
            }

            if(left_temp + right_temp + up_temp + down_temp <=0.0f){
                return;
            }

            left_temp = (left_temp > 0.0f)?1:0;
            right_temp = (right_temp > 0.0f)?1:0;
            up_temp = (up_temp > 0.0f)?1:0;
            down_temp = (down_temp > 0.0f)?1:0;


            output[intIndex] = (  left_temp  * VALUE_4(output, intN, intC, intY, left_offset) + right_temp * VALUE_4(output, intN, intC, intY, right_offset) + up_temp    * VALUE_4(output, intN, intC, up_offset, intX) + down_temp  * VALUE_4(output, intN, intC, down_offset, intX) ) / (left_temp + right_temp + up_temp + down_temp);

        }
    
    }
    }
'''

kernel_flowproj_updateGradInput= '''
    extern "C" __global__ void kernel_flowproj_updateGradInput(
            const int n,
            const float* input,
            const float* count,
            float* gradOutput,
            float* gradInput
        ){ for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput)   ) % SIZE_0(gradInput);
            const int intC = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                       ) % SIZE_1(gradInput);
            const int intY = ( intIndex / SIZE_3(gradInput)                                           ) % SIZE_2(gradInput);
            const int intX = ( intIndex                                                               ) % SIZE_3(gradInput);
            
            float fx = (float)VALUE_4(input, intN, 0, intY, intX);
            float fy = (float)VALUE_4(input, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int Xsize = (int)(SIZE_3(gradInput));
            int Ysize = (int)(SIZE_2(gradInput));

           

            if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 <= (float)(Xsize-1)  && y2 <=  (float)(Ysize-1) ){
                int ix2_L = (int) (x2);
                int iy2_T = (int) (y2);
                int ix2_R =  (((ix2_L + 1)<(Xsize - 1))?(ix2_L + 1):(Xsize - 1));   
                int iy2_B =  (((iy2_T + 1)<(Ysize - 1))?(iy2_T + 1):(Ysize - 1)); 

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_T, ix2_L) / VALUE_4(count, intN, 0, iy2_T, ix2_L);

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_T, ix2_R) / VALUE_4(count, intN, 0, iy2_T, ix2_R);

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_B, ix2_L) / VALUE_4(count, intN, 0, iy2_B, ix2_L);

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_B, ix2_R) / VALUE_4(count, intN, 0, iy2_B, ix2_R);
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

class _FunctionFlowProj(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        intSamples = input.shape[0]  # N
        intInputDepth, intInputHeight, intInputWidth  = input.shape[1], input.shape[2], input.shape[3] # C H W

        assert(intInputDepth==2)
        # assert()

        input = input.contiguous()
        assert(input.is_cuda == True)

        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])
        count  = input.new_zeros([intSamples, 1            , intInputHeight, intInputWidth])

        if input.is_cuda == True:
            # n = output.nelement()

            cupy_launch('kernel_flowproj_updateOutput', cupy_kernel('kernel_flowproj_updateOutput', {
                'input': input,
                'output': output,
                'count': count
            }))(
                grid=tuple([ int((input.nelement() + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ cupy.int32(input.nelement()), input.data_ptr(), output.data_ptr(), count.data_ptr() ]
            )

            cupy_launch('kernel_flowproj_updataAvg', cupy_kernel('kernel_flowproj_updataAvg', {
                'input': input,
                'output': output,
                'count': count
            }))(
                grid=tuple([ int((input.nelement() + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ cupy.int32(input.nelement()), input.data_ptr(), output.data_ptr(), count.data_ptr() ]
            )

            fillhole = 1 if input.requires_grad == False else 0

            if fillhole == True:
                cupy_launch('kernel_flowproj_updaFillhole', cupy_kernel('kernel_flowproj_updaFillhole', {
                'input': input,
                'output': output,
                'count': count
                }))(
                    grid=tuple([ int((output.nelement() + 512 - 1) / 512), 1, 1 ]),
                    block=tuple([ 512, 1, 1 ]),
                    args=[ cupy.int32(output.nelement()), input.data_ptr(), output.data_ptr(), count.data_ptr() ]
                )
        elif input.is_cuda == False:
            raise NotImplementedError()
        # end


        self.save_for_backward(input, count)

        return output
    
    # end

    @staticmethod
    def backward(self, gradOutput):

        input, count = self.saved_tensors

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]

        assert( intInputDepth == 2)

        gradOutput = gradOutput.contiguous(); assert(gradOutput.is_cuda == True)

        gradInput = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth ]) 
        
        # if self.needs_input_grad[0] == True else None

        if input.is_cuda == True:
            if gradInput is not None:
                n = gradInput.nelement()
                cupy_launch('kernel_flowproj_updateGradInput', cupy_kernel('kernel_flowproj_updateGradInput',{
                    'input': input,
                    'count': count,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput
                }))(
                    grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ cupy.int32(n), input.data_ptr(), count.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr() ]
                )
            # end
        elif input.is_cuda == False:
            raise NotImplementedError()
        # end

        return gradInput
# end



def FunctionFlowProj(tenflow):
    fillhole = tenflow.requires_grad
    tenOutput = _FunctionFlowProj.apply(input, fillhole=fillhole)
    return tenOutput

class ModuleFlowProj(torch.nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, tenOne):
        return _FunctionFlowProj.apply(tenOne)