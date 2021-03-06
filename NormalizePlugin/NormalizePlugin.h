/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_NORMALIZE_PLUGIN_H
#define TRT_NORMALIZE_PLUGIN_H
#include "cudnn.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
// #include "kernel.h"
// #include "plugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <cstdlib>

//plugin.h
typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1
{
// namespace plugin
// {

class BasePlugin : public IPluginV2
{
protected:
    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    std::string mNamespace;
};

class BaseCreator : public IPluginCreator
{
public:
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace;
};

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    // std::memcpy(buffer, &val, sizeof(T));
    memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val{};
    // std::memcpy(&val, buffer, sizeof(T));
    memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

// } // namespace plugin
} // namespace nvinfer1




namespace nvinfer1
{
// namespace plugin
// {

class Normalize : public IPluginV2Ext
{
public:
    Normalize(const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps);

    Normalize(
        const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps, int C, int H, int W);

    Normalize(const void* buffer, size_t length);

    ~Normalize() override = default;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    Weights copyToDevice(const void* hostData, size_t count);
    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;
    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    cublasHandle_t mCublas;

    Weights mWeights{};
    int mNbWeights{};
    bool acrossSpatial{};
    bool channelShared{};
    float eps{};
    int C{};
    int H{};
    int W{};

    std::string mPluginNamespace;
};

class NormalizePluginCreator_1 : public BaseCreator
{
public:
    NormalizePluginCreator_1();

    ~NormalizePluginCreator_1();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    bool mAcrossSpatial{};
    bool mChannelShared{};
    float mEps{};
    int mNbWeights{};
    static std::vector<PluginField> mPluginAttributes;
};
// } // namespace plugin
} // namespace nvinfer1

#endif // TRT_NORMALIZE_PLUGIN_H
