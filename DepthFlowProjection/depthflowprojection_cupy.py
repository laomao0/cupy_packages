#!/usr/bin/env python

import torch
import cupy
import re

kernel_depthflowproj_updateOutput= '''
    extern "C" __global__ void kernel_depthflowproj_updateOutput(
        const int n,
        const float* input,
        const float* depth,
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

                    float fltdepth = VALUE_4(depth, intN, 0, intY, intX);
                  
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_T, ix2_L)],  -fx * fltdepth);
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_T, ix2_R)],  -fx * fltdepth);
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_B, ix2_L)],  -fx * fltdepth);
                    atomicAdd( & output[OFFSET_4(output, intN, 0, iy2_B, ix2_R)],  -fx * fltdepth);

                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_T, ix2_L)],  -fy * fltdepth);
                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_T, ix2_R)],  -fy * fltdepth);
                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_B, ix2_L)],  -fy * fltdepth);
                    atomicAdd( & output[OFFSET_4(output, intN, 1, iy2_B, ix2_R)],  -fy * fltdepth);

                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_T, ix2_L)],  +1 * fltdepth);
                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_T, ix2_R)],  +1 * fltdepth);
                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_B, ix2_L)],  +1 * fltdepth);
                    atomicAdd( & count[OFFSET_4(count, intN, 0, iy2_B, ix2_R)],  +1 * fltdepth);
                }
            }
        
    }
    }
'''

kernel_depthflowproj_updataAvg= '''
    extern "C" __global__ void kernel_depthflowproj_updataAvg(
        const int n,
        const float* input,
        const float* depth,
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

kernel_depthflowproj_updaFillhole= '''
    extern "C" __global__ void kernel_depthflowproj_updaFillhole(
        const int n,
        const float* input,
        const float* depth,
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


            output[intIndex] = (  left_temp  * VALUE_4(output, intN, intC, intY, left_offset) 
                    + right_temp * VALUE_4(output, intN, intC, intY, right_offset) 
                    + up_temp    * VALUE_4(output, intN, intC, up_offset, intX) 
                    + down_temp  * VALUE_4(output, intN, intC, down_offset, intX) )
                    / (left_temp + right_temp + up_temp + down_temp);

        }
    
    }
    }
'''

kernel_depthflowproj_updateGradInput= '''
    extern "C" __global__ void kernel_depthflowproj_updateGradInput(
            const int n,
            const float* input,
            const float* depth,
            const float* count,
            float* gradOutput,
            float* gradInput,
            float* gradDepth
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

                float fltdepth = VALUE_4(depth, intN, 0, intY, intX);

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_T, ix2_L) * fltdepth 
                                        / VALUE_4(count, intN, 0, iy2_T, ix2_L);

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_T, ix2_R) * fltdepth
                                        / VALUE_4(count, intN, 0, iy2_T, ix2_R);

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_B, ix2_L) * fltdepth
                                        / VALUE_4(count, intN, 0, iy2_B, ix2_L);

                gradInput[intIndex] += -VALUE_4(gradOutput, intN, intC, iy2_B, ix2_R) * fltdepth
                                        / VALUE_4(count, intN, 0, iy2_B, ix2_R);
            }
            

        }
        }


'''

kernel_depthflowproj_updateGradDepth= '''
    extern "C" __global__ void kernel_depthflowproj_updateGradDepth(
            const int n,
            const float* input,
            const float* depth,
            const float* count,
            const float* output,
            float* gradOutput,
            float* gradInput,
            float* gradDepth
        ){ for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(gradDepth) / SIZE_2(gradDepth) / SIZE_1(gradDepth)   ) % SIZE_0(gradDepth);
            const int intC = ( intIndex / SIZE_3(gradDepth) / SIZE_2(gradDepth)                       ) % SIZE_1(gradDepth);
            const int intY = ( intIndex / SIZE_3(gradDepth)                                           ) % SIZE_2(gradDepth);
            const int intX = ( intIndex                                                               ) % SIZE_3(gradDepth);
            
            float fx = (float)VALUE_4(input, intN, 0, intY, intX);
            float fy = (float)VALUE_4(input, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int Xsize = (int)(SIZE_3(gradDepth));
            int Ysize = (int)(SIZE_2(gradDepth));

           

            if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 <= (float)(Xsize-1)  && y2 <=  (float)(Ysize-1) ){
                int ix2_L = (int) (x2);
                int iy2_T = (int) (y2);
                int ix2_R =  (((ix2_L + 1)<(Xsize - 1))?(ix2_L + 1):(Xsize - 1));   
                int iy2_B =  (((iy2_T + 1)<(Ysize - 1))?(iy2_T + 1):(Ysize - 1)); 
                
                // gradient for output[N,0,Y,X]

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 0, iy2_T, ix2_L) 
                                        / VALUE_4(count, intN, 0, iy2_T, ix2_L) 
                                        * (fx - VALUE_4(output, intN, 0, iy2_T, ix2_L) );

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 0, iy2_T, ix2_R) 
                                        / VALUE_4(count, intN, 0, iy2_T, ix2_R) 
                                        * (fx - VALUE_4(output, intN, 0, iy2_T, ix2_R) );

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 0, iy2_B, ix2_L) 
                                        / VALUE_4(count, intN, 0, iy2_B, ix2_L)
                                        * (fx - VALUE_4(output, intN, 0, iy2_B, ix2_L) );

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 0, iy2_B, ix2_R) 
                                        / VALUE_4(count, intN, 0, iy2_B, ix2_R)
                                        * (fx - VALUE_4(output, intN, 0, iy2_B, ix2_R) );

                // gradient for output[N,1,Y,X]

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 1, iy2_T, ix2_L) 
                                        / VALUE_4(count, intN, 0, iy2_T, ix2_L) 
                                        * (fy - VALUE_4(output, intN, 1, iy2_T, ix2_L) );

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 1, iy2_T, ix2_R) 
                                        / VALUE_4(count, intN, 0, iy2_T, ix2_R) 
                                        * (fy - VALUE_4(output, intN, 1, iy2_T, ix2_R) );

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 1, iy2_B, ix2_L) 
                                        / VALUE_4(count, intN, 0, iy2_B, ix2_L)
                                        * (fy - VALUE_4(output, intN, 1, iy2_B, ix2_L) );

                gradDepth[intIndex] += -VALUE_4(gradOutput, intN, 1, iy2_B, ix2_R) 
                                        / VALUE_4(count, intN, 0, iy2_B, ix2_R)
                                        * (fy - VALUE_4(output, intN, 1, iy2_B, ix2_R) );
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
    def forward(self, input, depth):
        intSamples = input.shape[0]  # N
        intInputDepth, intInputHeight, intInputWidth  = input.shape[1], input.shape[2], input.shape[3] # C H W
        intDepthDepth, intDepthHeight, intDepthWidth  = depth.shape[1], depth.shape[2], depth.shape[3]

        assert(intInputDepth==2)
        assert(intDepthDepth==1)
        assert(intInputHeight == intDepthHeight)
        assert(intInputWidth == intDepthWidth)

        input = input.contiguous()
        assert(input.is_cuda == True)
        depth = depth.contiguous()
        assert(input.is_cuda == True)

        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])
        count  = input.new_zeros([intSamples, 1            , intInputHeight, intInputWidth])

        if input.is_cuda == True:
            # n = output.nelement()

            cupy_launch('kernel_depthflowproj_updateOutput', cupy_kernel('kernel_depthflowproj_updateOutput', {
                'input': input,
                'depth': depth,
                'output': output,
                'count': count
            }))(
                grid=tuple([ int((input.nelement() + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ cupy.int32(input.nelement()), 
                    input.data_ptr(), depth.data_ptr(),
                    output.data_ptr(), count.data_ptr() ]
            )

            cupy_launch('kernel_depthflowproj_updataAvg', cupy_kernel('kernel_depthflowproj_updataAvg', {
                'input': input,
                'depth': depth,
                'output': output,
                'count': count
            }))(
                grid=tuple([ int((input.nelement() + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ cupy.int32(input.nelement()), 
                    input.data_ptr(),  depth.data_ptr(), 
                    output.data_ptr(), count.data_ptr() ]
            )

            fillhole = 1 if input.requires_grad == False else 0

            if fillhole == True:
                cupy_launch('kernel_depthflowproj_updaFillhole', cupy_kernel('kernel_depthflowproj_updaFillhole', {
                'input': input,
                'depth': depth,
                'output': output,
                'count': count
                }))(
                    grid=tuple([ int((output.nelement() + 512 - 1) / 512), 1, 1 ]),
                    block=tuple([ 512, 1, 1 ]),
                    args=[ cupy.int32(output.nelement()), input.data_ptr(), 
                            depth.data_ptr(), output.data_ptr(), count.data_ptr() ]
                )

                print('fill hole')
        elif input.is_cuda == False:
            raise NotImplementedError()
        # end


        self.save_for_backward(input, depth, count, output)

        return output
    
    # end

    @staticmethod
    def backward(self, gradOutput):

        input, depth, count, output = self.saved_tensors

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intDepthDepth, intDepthHeight, intDepthWidth  = depth.shape[1], depth.shape[2], depth.shape[3]

        assert(intInputDepth==2)
        assert(intDepthDepth==1)
        assert(intInputHeight == intDepthHeight)
        assert(intInputWidth == intDepthWidth)

        gradOutput = gradOutput.contiguous(); assert(gradOutput.is_cuda == True)

        gradInput = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth ]) 
        gradDepth = input.new_zeros([ intSamples, intDepthDepth, intDepthHeight, intDepthWidth])
        
        # if self.needs_input_grad[0] == True else None

        if input.is_cuda == True:
            if gradInput is not None:
                n = gradInput.nelement()
                cupy_launch('kernel_depthflowproj_updateGradInput', cupy_kernel('kernel_depthflowproj_updateGradInput',{
                    'input': input,
                    'depth': depth,
                    'count': count,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradDepth': gradDepth
                }))(
                    grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ cupy.int32(n), input.data_ptr(), depth.data_ptr(),
                    count.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr(), None]
                )
            # end

            if gradDepth is not None:
                n = gradDepth.nelement()
                cupy_launch('kernel_depthflowproj_updateGradDepth', cupy_kernel('kernel_depthflowproj_updateGradDepth',{
                    'input': input,
                    'depth': depth,
                    'count': count,
                    'output': output,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradDepth': gradDepth
                }))(
                    grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ cupy.int32(n), input.data_ptr(), depth.data_ptr(),
                    count.data_ptr(), output.data_ptr(),
                    gradOutput.data_ptr(), None, gradDepth.data_ptr()]
                )
            # end

        elif input.is_cuda == False:
            raise NotImplementedError()
        # end

        return gradInput, gradDepth
# end



def FunctionFlowProj(tenflow, tendepth):
    fillhole = tenflow.requires_grad
    tenOutput = _FunctionFlowProj.apply(tenflow , tendepth, fillhole=fillhole)
    return tenOutput

class ModuleDepthFlowProj(torch.nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, flow, depth):
        return _FunctionFlowProj.apply(flow, depth)