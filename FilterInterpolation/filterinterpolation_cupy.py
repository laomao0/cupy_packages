#!/usr/bin/env python

import torch
import cupy
import re
   

kernel_forward_interpolation_UpdateOutput= '''
#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))
    extern "C" __global__ void kernel_forward_interpolation_UpdateOutput(
        const int n,
        const float* input,
        const float* flow,
        const float* filter,
        float* output
    ){ for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output)   ) % SIZE_0(output);
            const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                    ) % SIZE_1(output);
            const int intY = ( intIndex / SIZE_3(output)                                     ) % SIZE_2(output);
            const int intX = ( intIndex                                                       ) % SIZE_3(output);

            

            int filter_size = (int)(sqrt(float(SIZE_1(filter))));

            float fx = (float)VALUE_4(flow, intN, 0, intY, intX);
            float fy = (float)VALUE_4(flow, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int w = SIZE_3(output);
            int h = SIZE_2(output);

            if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 <= (float)(w-1)  && y2 <= (float)(h-1)
                && fabs(fx) < (float)(w)/2.0f  && fabs(fy) < (float)(h)/2.0f  ){
                    int ix2_L = (int) (x2) + 1 - (int)(filter_size / 2);
                    int iy2_T = (int) (y2) + 1 - (int)(filter_size / 2);
                    int ix2_R = ix2_L + filter_size;
                    int iy2_B = iy2_T + filter_size;

                    float alpha = x2 - (int)(x2);
                    float beta  = y2 - (int)(y2);

                    float TL = 0.0f;
                    for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                        int _filter_j = min(max(0, filter_j), h - 1);
                        for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i ), w - 1);
                            int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                            TL += VALUE_4(input, intN, intC, _filter_j, _filter_i) *
                                VALUE_4(filter, intN, filterc, intY, intX); 
                        }
                    }

                    float TR = 0.0f;
                    for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                            TR += VALUE_4(input, intN, intC, _filter_j, _filter_i) *
                                VALUE_4(filter, intN, filterc, intY, intX);
                        }
                    }
            
                    float BL = 0.0f;
                    for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                            BL += VALUE_4(input, intN, intC, _filter_j, _filter_i) *
                                VALUE_4(filter, intN, filterc, intY, intX);
                        }
                    }

                    float BR = 0.0f;
                    for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                            BR += VALUE_4(input, intN, intC, _filter_j, _filter_i) *
                                VALUE_4(filter, intN, filterc, intY, intX);
                        }
                    }

                    output[intIndex] = (1-alpha)*(1-beta)*TL + alpha*(1-beta)*TR + (1-alpha)*beta*BL + alpha*beta*BR;
            }
            else{
                output[intIndex]  = input[intIndex];
            }
        
    }
    }
'''



kernel_flowproj_updateGradIn_Filter= '''
#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))
extern "C" __global__ void kernel_flowproj_updateGradIn_Filter(
        const int n,
        const float* input,
        const float* flow,
        const float* filter,
        const float* gradOutput,
        float* gradInput,
        float* gradFlow,
        float* gradFilter
    ){  for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(input) / SIZE_2(input) / SIZE_1(input)   ) % SIZE_0(input);
            const int intC = ( intIndex / SIZE_3(input) / SIZE_2(input)                   ) % SIZE_1(input);
            const int intY = ( intIndex / SIZE_3(input)                                   ) % SIZE_2(input);
            const int intX = ( intIndex                                                   ) % SIZE_3(input);

            int filter_size = (int)(sqrt(float(SIZE_1(filter))));

            float fx = (float)VALUE_4(flow, intN, 0, intY, intX);
            float fy = (float)VALUE_4(flow, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int w = SIZE_3(input);
            int h = SIZE_2(input);

            
            

            if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 <= (float)(w-1)  && y2 <= (float)(h-1)
                && fabs(fx) < (float)(w)/2.0f  && fabs(fy) < (float)(h)/2.0f  ){
                int ix2_L = (int) (x2) + 1 - (int)(filter_size / 2);
                int iy2_T = (int) (y2) + 1 - (int)(filter_size / 2);
                int ix2_R = ix2_L + filter_size;
                int iy2_B = iy2_T + filter_size;

                float alpha = x2 - (int)(x2);
                float beta  = y2 - (int)(y2);

                

                float gradoutput_value = VALUE_4(gradOutput, intN, intC, intY, intX);

                float TL_grad = gradoutput_value * (1-alpha ) * (1-beta);
                for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                    int _filter_j = min(max(0, filter_j), h - 1);
                    for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                        int _filter_i = min(max(0, filter_i ), w - 1);
                        int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);

                        atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, _filter_j, _filter_i)], 
                            TL_grad * VALUE_4(filter, intN, filterc, intY, intX) );

                        atomicAdd(&gradFilter[OFFSET_4(gradFilter, intN, filterc, intY, intX)],
                            TL_grad * VALUE_4(input, intN, intC, _filter_j, _filter_i));
                    }
                }


                float TR_grad= gradoutput_value * alpha * ( 1- beta);
                for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                    for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                        
                        atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, _filter_j, _filter_i)], 
                            TR_grad * VALUE_4(filter, intN, filterc, intY, intX) );

                        atomicAdd(&gradFilter[OFFSET_4(gradFilter, intN, filterc, intY, intX)],
                            TR_grad * VALUE_4(input, intN, intC, _filter_j, _filter_i));

                    }
                }


                float BL_grad = gradoutput_value * ( 1 - alpha ) * beta;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                    for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                        
                        atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, _filter_j, _filter_i)], 
                            BL_grad * VALUE_4(filter, intN, filterc, intY, intX) );

                        atomicAdd(&gradFilter[OFFSET_4(gradFilter, intN, filterc, intY, intX)],
                            BL_grad * VALUE_4(input, intN, intC, _filter_j, _filter_i));

                    }
                }


                float BR_grad = gradoutput_value * alpha * beta;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                    for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);

                        atomicAdd(&gradInput[OFFSET_4(gradInput, intN, intC, _filter_j, _filter_i)], 
                            BR_grad * VALUE_4(filter, intN, filterc, intY, intX) );

                        atomicAdd(&gradFilter[OFFSET_4(gradFilter, intN, filterc, intY, intX)],
                            BR_grad * VALUE_4(input, intN, intC, _filter_j, _filter_i));
                    }
                }


            }
            
        }
    }
'''


kernel_flowproj_updateGradFlow= '''
#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))
extern "C" __global__ void kernel_flowproj_updateGradFlow(
        const int n,
        const float* input,
        const float* flow,
        const float* filter,
        const float* gradOutput,
        float* gradInput,
        float* gradFlow,
        float* gradFilter
    ){  for(int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x){
            const int intN = ( intIndex / SIZE_3(input) / SIZE_2(input) / SIZE_1(input)   ) % SIZE_0(input);
            const int intC = ( intIndex / SIZE_3(input) / SIZE_2(input)                   ) % SIZE_1(input);
            const int intY = ( intIndex / SIZE_3(input)                                   ) % SIZE_2(input);
            const int intX = ( intIndex                                                   ) % SIZE_3(input);

            

            int filter_size = (int)(sqrt(float(SIZE_1(filter))));

            float fx = (float)VALUE_4(flow, intN, 0, intY, intX);
            float fy = (float)VALUE_4(flow, intN, 1, intY, intX);

            float x2 = (float)(intX) + fx;
            float y2 = (float)(intY) + fy;

            int w = SIZE_3(input);
            int h = SIZE_2(input);

            if(intC == 0)
            {

                if( x2 >= 0.0f  &&  y2>= 0.0f  &&  x2 <= (float)(w-1)  && y2 <= (float)(h-1)
                    && fabs(fx) < (float)(w)/2.0f  && fabs(fy) < (float)(h)/2.0f  ){
                    int ix2_L = (int) (x2) + 1 - (int)(filter_size / 2);
                    int iy2_T = (int) (y2) + 1 - (int)(filter_size / 2);
                    int ix2_R = ix2_L + filter_size;
                    int iy2_B = iy2_T + filter_size;

                    float alpha = x2 - (int)(x2);
                    float beta  = y2 - (int)(y2);

                    
                    
                    float gamma = 1.0f - beta; 
                    float bot_diff = 0.0f;

                    for (int intChannel =0; intChannel < SIZE_1(input); intChannel++)
                    {
                            float gradoutput_value = VALUE_4(gradOutput, intN, intChannel, intY, intX);

                            float TL = 0.0f;
                            for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                                int _filter_j = min(max(0, filter_j), h - 1);
                                for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i ), w - 1);
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    TL += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX); 
                                }
                            }


                            float TR = 0.0f;
                            for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                                int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                                for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    TR += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX);
                                }
                            }


                            float BL = 0.0f;
                            for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                                int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                                for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    BL += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX);
                                }
                            }


                            float BR = 0.0f;
                            for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                                int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                                for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    BR += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX);
                                }
                            }

                            float temp = 0.0f;
                            temp += gamma * (TR - TL);
                            temp += (1-gamma) * (BR - BL);
                            bot_diff += gradoutput_value * temp;

                    }
                        
                    gradFlow[OFFSET_4(gradFlow, intN, 0, intY, intX)] = bot_diff;

                   


                    gamma = 1.0f - alpha; 
                    bot_diff = 0.0f;
                    for (int intChannel =0; intChannel < SIZE_1(input); intChannel++)
                        {
                            float gradoutput_value = VALUE_4(gradOutput, intN, intChannel, intY, intX);

                            float TL = 0.0f;
                            for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                                int _filter_j = min(max(0, filter_j), h - 1);
                                for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i ), w - 1);
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    TL += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX); 
                                }
                            }


                            float TR = 0.0f;
                            for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                                int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                                for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    TR += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX);
                                }
                            }


                            float BL = 0.0f;
                            for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                                int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                                for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    BL += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX);
                                }
                            }


                            float BR = 0.0f;
                            for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                                int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                                for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                    int filterc = (filter_j - iy2_T) * filter_size + (filter_i - ix2_L);
                                    BR += VALUE_4(input, intN, intChannel, _filter_j, _filter_i) *
                                        VALUE_4(filter, intN, filterc, intY, intX);
                                }
                            }

                            float temp = 0.0f;
                            temp += gamma * (BL - TL);
                            temp += (1.0f - gamma) * ( BR - TR);
                            bot_diff += gradoutput_value * temp;




                    }
                    
                    gradFlow[OFFSET_4(gradFlow, intN, 1, intY, intX)] = bot_diff;

                    

                }

                
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
    def forward(self, input, flow, filter):
        intSamples = input.shape[0]  # N
        intInputDepth, intInputHeight, intInputWidth  = input.shape[1], input.shape[2], input.shape[3] # C H W
        intFlowDepth,  intFlowHeight,  intFlowWidth   = flow.shape[1],  flow.shape[2],  flow.shape[3]
        intfiltDepth,  intfiltHeight,  intFiltWidth   = filter.shape[1], filter.shape[2], filter.shape[3]

        assert(intFlowDepth==2)
        assert(intInputHeight == intFlowHeight)
        assert(intInputWidth == intFlowWidth)
        assert(intInputHeight == intfiltHeight)
        assert(intInputWidth == intFiltWidth)


        input = input.contiguous()
        assert(input.is_cuda == True)
        flow = flow.contiguous()
        assert(flow.is_cuda == True)
        filter = filter.contiguous()
        assert(filter.is_cuda == True)

        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])
    

        if input.is_cuda == True:

            cupy_launch('kernel_forward_interpolation_UpdateOutput', cupy_kernel('kernel_forward_interpolation_UpdateOutput', {
                'input': input,
                'flow': flow,
                'filter': filter,
                'output': output
            }))(
                grid=tuple([ int((output.nelement() + 512 - 1) / 512), 1, 1 ]),
                block=tuple([ 512, 1, 1 ]),
                args=[ cupy.int32(output.nelement()), input.data_ptr(), 
                flow.data_ptr(), filter.data_ptr(), output.data_ptr() ]
            )

        elif input.is_cuda == False:
            raise NotImplementedError()
        # end

        self.save_for_backward(input, flow, filter)

        return output
    
    # end

    @staticmethod
    def backward(self, gradOutput):

        input, flow, filter = self.saved_tensors

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth,  intFlowHeight,  intFlowWidth   = flow.shape[1],  flow.shape[2],  flow.shape[3]
        intfiltDepth,  intfiltHeight,  intFiltWidth   = filter.shape[1], filter.shape[2], filter.shape[3]

        assert(intFlowDepth==2)
        assert(intInputHeight == intFlowHeight)
        assert(intInputWidth == intFlowWidth)
        assert(intInputHeight == intfiltHeight)
        assert(intInputWidth == intFiltWidth)

        gradOutput = gradOutput.contiguous(); assert(gradOutput.is_cuda == True)

        gradInput =   input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth]) #if self.needs_input_grad[0] == True else None
        gradFlow  =   input.new_zeros([ intSamples, intFlowDepth,  intFlowHeight,  intFlowWidth ]) #if self.needs_input_grad[1] == True else None
        gradFilter  = input.new_zeros([ intSamples, intfiltDepth,  intfiltHeight,  intFiltWidth ]) #if self.needs_input_grad[2] == True else None

        if input.is_cuda == True:

            if (gradInput is not None) and (gradFilter is not None) :
                n = input.nelement()
                cupy_launch('kernel_flowproj_updateGradIn_Filter', cupy_kernel('kernel_flowproj_updateGradIn_Filter',{
                    'input': input,
                    'flow': flow,
                    'filter': filter,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradFlow': gradFlow,
                    'gradFilter': gradFilter
                }))(
                    grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
                    block=tuple([ 512, 1, 1 ]),
                    args=[ cupy.int32(n), input.data_ptr(),
                        flow.data_ptr(), filter.data_ptr(),
                        gradOutput.data_ptr(), gradInput.data_ptr(), 
                        None, gradFilter.data_ptr()]
                )
            # end

            if (gradFlow is not None) :
                n = gradFlow.nelement()
                cupy_launch('kernel_flowproj_updateGradFlow', cupy_kernel('kernel_flowproj_updateGradFlow',{
                    'input': input,
                    'flow': flow,
                    'filter': filter,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradFlow': gradFlow,
                    'gradFilter': gradFilter
                }))(
                    grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
                    block=tuple([ 512, 1, 1 ]),
                    args=[ cupy.int32(n), input.data_ptr(),
                        flow.data_ptr(), filter.data_ptr(),
                        gradOutput.data_ptr(), None, 
                        gradFlow.data_ptr(), None]
                )
            # end


        elif input.is_cuda == False:
            raise NotImplementedError()
        # end

        return gradInput, gradFlow, gradFilter
# end



def FunctionInterpolation(teninput, tenflow, tenfilter):
    tenOutput = _FunctionInterpolation.apply(teninput, tenflow, tenfilter)
    return tenOutput

class ModuleFilterInterpolation(torch.nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, teninput, tenflow, tenfilter):
        return _FunctionInterpolation.apply(teninput, tenflow, tenfilter)