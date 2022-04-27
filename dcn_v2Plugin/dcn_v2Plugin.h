//
// Created by cao on 19-12-20.
//

#ifndef TRT_DCNV2_PLUGIN_H
#define TRT_DCNV2_PLUGIN_H

#include "plugin.h"
#include "serialize.hpp"
#include <cudnn.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda.h>

namespace nvinfer1 {
namespace plugin {

class DCNv2Plugin final : public nvinfer1::IPluginV2DynamicExt {
private:
    int _in_channel;
    int _out_channel;
    int _kernel_H;
    int _kernel_W;
    int _deformable_group;
    int _dilation;
    int _groups; // not use
    int _padding;
    int _stride;
    int _output_height;
    int _output_width;
    std::vector<float> _h_weight;
    std::vector<float> _h_bias;
    float* _d_weight;
    float* _d_bias;
    float* _d_ones;
    float *_d_columns;
    cublasHandle_t _cublas_handle;
    const char* _plugin_namespace;

    bool _initialized;

public:

    void deserialize(void const* serialData, size_t serialLength) {
        deserialize_value(&serialData, &serialLength, &_in_channel);
        deserialize_value(&serialData, &serialLength, &_out_channel);
        deserialize_value(&serialData, &serialLength, &_kernel_H);
        deserialize_value(&serialData, &serialLength, &_kernel_W);
        deserialize_value(&serialData, &serialLength, &_deformable_group);
        deserialize_value(&serialData, &serialLength, &_dilation);
        deserialize_value(&serialData, &serialLength, &_groups);
        deserialize_value(&serialData, &serialLength, &_padding);
        deserialize_value(&serialData, &serialLength, &_stride);
        deserialize_value(&serialData, &serialLength, &_h_weight);
        deserialize_value(&serialData, &serialLength, &_h_bias);
        deserialize_value(&serialData, &serialLength, &_output_height);
        deserialize_value(&serialData, &serialLength, &_output_width);
    }


    // size_t getSerializationSize() const override {
    size_t getSerializationSize() const noexcept override {
        return (serialized_size(_in_channel) +
                serialized_size(_out_channel) +
                serialized_size(_kernel_H) +
                serialized_size(_kernel_W) +
                serialized_size(_deformable_group) +
                serialized_size(_dilation) +
                serialized_size(_groups) +
                serialized_size(_padding) +
                serialized_size(_stride) +
                serialized_size(_h_weight) +
                serialized_size(_h_bias) +
                serialized_size(_output_height) +
                serialized_size(_output_width)
               );
    }

    // void serialize(void *buffer) const override {
    void serialize(void *buffer) const noexcept override {
        serialize_value(&buffer, _in_channel);
        serialize_value(&buffer, _out_channel);
        serialize_value(&buffer, _kernel_H);
        serialize_value(&buffer, _kernel_W);
        serialize_value(&buffer, _deformable_group);
        serialize_value(&buffer, _dilation);
        serialize_value(&buffer, _groups);
        serialize_value(&buffer, _padding);
        serialize_value(&buffer, _stride);
        serialize_value(&buffer, _h_weight);
        serialize_value(&buffer, _h_bias);
        serialize_value(&buffer, _output_height);
        serialize_value(&buffer, _output_width);
    }

    DCNv2Plugin(int in_channel,
                int out_channel,
                int kernel_H,
                int kernel_W,
                int deformable_group,
                int dilation,
                int groups,
                int padding,
                int stride,
                nvinfer1::Weights const& weight,
                nvinfer1::Weights const& bias);

    DCNv2Plugin(int in_channel,
                int out_channel,
                int kernel_H,
                int kernel_W,
                int deformable_group,
                int dilation,
                int groups,
                int padding,
                int stride,
                const std::vector<float> &weight,
                const std::vector<float> &bias);

    DCNv2Plugin(void const* serialData, size_t serialLength) : _initialized(false) {
        this->deserialize(serialData, serialLength);
        cublasCreate(&_cublas_handle);
     }

    DCNv2Plugin() = delete;

    // const char* getPluginType() const override { return "DCNv2"; }
    const char* getPluginType() const noexcept override { return "DCNv2"; }

    // const char* getPluginVersion() const override { return "001"; }
    const char* getPluginVersion() const noexcept override { return "001"; }

    // void destroy() override;
    void destroy() noexcept override;

    // int getNbOutputs() const override { return 1; }
    int getNbOutputs() const noexcept override { return 1; }


    // nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
    //                                         const nvinfer1::DimsExprs* inputs,
    //                                         int nbInputs,
    //                                         nvinfer1::IExprBuilder& exprBuilder) override;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;


    // int initialize() override;
    int initialize() noexcept override;

    // void terminate() override;
    void terminate() noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace,  cudaStream_t stream) noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override {_plugin_namespace = pluginNamespace;};

    const char* getPluginNamespace() const noexcept override {return _plugin_namespace;};

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override {};

    void detachFromContext() noexcept override {};

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    ~DCNv2Plugin();
};



class DCNv2PluginCreator : public BaseCreator
{
public:
  DCNv2PluginCreator();

  ~DCNv2PluginCreator() override = default;

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const PluginFieldCollection* getFieldNames() noexcept override;

  IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

  IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};

}
}
#endif //TRT_DCNV2_PLUGIN_H
