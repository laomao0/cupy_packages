Myenv: GPU2080Ti, ubuntu 18.04, pytorch1.3, cuda 11.1, TensorRT-8.4.0.6

1. Ensure you could build TensorRT OSS : https://github.com/NVIDIA/TensorRT

   i.e., 
    cd $TRT_OSSPATH
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
    make -j$(nproc)

   here, $TRT_OSSPATH is the downloaed src path, mine is /DATA/wangshen_data/CODES/TensorRT (use git download)
   $TRT_LIBPATH is TRT path, mine is /usr/local/TensorRT-8.4.0.6/lib

2. Copy folder from this repository to src path, mine is /DATA/wangshen_data/CODES/TensorRT/plugin,
   
   including [CMakeLists.txt, dcn_v2_xx.cu, dcn_v2_xx.h, dcn_v2xx.cpp, dcn_v2xx.h]

3. Modify CmakeLists.txt at /DATA/wangshen_data/CODES/TensorRT/plugin/CMakeLists.txt,
   
    Line32 : set(PLUGIN_LISTS  batchedNMSPlugin xxx ) ----> add dcn_v2Plugin 
          to set(PLUGIN_LISTS dcn_v2Plugin batchedNMSPlugin xxx ) 

4. Modify InferPlugin.cpp at /DATA/wangshen_data/CODES/TensorRT/plugin/InferPlugin.cpp

   a)   Line57: add header file : #include "dcn_v2Plugin.h"
   b)   Line198: add initializePlugin<nvinfer1::plugin::DCNv2PluginCreator>(logger, libNamespace);

5. Now, you can build the src again, 
   
   a) first remove folder build at /DATA/wangshen_data/CODES/TensorRT/build
   b) follow step 1. to rebuild the src
   c) when built over, you can see [100%] Built target xxx from terminal,
      then you could obtain  libnvinfer_plugin.so at /DATA/wangshen_data/CODES/TensorRT/build/out
 
 
Notice, src file DCNv2 for TensorRT 7 can be found at https://github.com/lesliejackson/TensorRT-DCNv2-Plugin

I make two modifications to adapt it to TensorRT 8.4., including:
  
   1. add noexcept for several functions for dcn_v2Plugin.cpp and dcn_v2Plugin.h, due to c++
   2. modify Line 118 dcn_v2Plugin.cpp [nvinfer1::TensorFormat::kNCHW] to [nvinfer1::TensorFormat::kLINEAR],
      since [kNCHW :Deprecated name of kLINEAR, provided for backwards compatibility and will be removed in TensorRT 8.0.]
      cite: https://www.ccoderun.ca/programming/doxygen/tensorrt/namespacenvinfer1.html#ac3e115b1a2b1e578e8221ef99d27cd45acaf70f83fa10041f93bb2ee89848d4b9


