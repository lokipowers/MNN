//
//  ConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvExecution.hpp"
#include "ConvWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "ConvLowMemoryExecution.hpp"
#include <cstdlib>

namespace MNN {
namespace OpenCL {

ConvCommonExecution::ConvCommonExecution(const Convolution2D *conv2dParams, Backend *backend) {
    mResource.reset(new ConvResource);
    mOpenCLBackend           = (OpenCLBackend *)backend;
    auto runtime             = mOpenCLBackend->getOpenCLRuntime();
    int biasSize             = conv2dParams->bias()->size();
    const float *biasDataPtr = conv2dParams->bias()->data();
    
    int buffer_size = ALIGN_UP8(biasSize) * sizeof(float);
    cl::Buffer biasBuffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto biasPtrCL = runtime->commandQueue().enqueueMapBuffer(biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(biasPtrCL != nullptr && error == CL_SUCCESS){
        ::memset(biasPtrCL, 0, ALIGN_UP8(biasSize) * sizeof(float));
        ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    runtime->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
    if (!(backend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC))) {
        mConvComValid = false;
        return;
    }
    copyBufferToImage(runtime, biasBuffer, openCLImage(mResource->mBias.get()), UP_DIV(biasSize, 4), 1, mOpenCLBackend->getPrecision());
}

ConvCommonExecution::ConvCommonExecution(const Op *op, Backend *backend, bool isExtra) {
    mResource.reset(new ConvResource);
    mOpenCLBackend           = (OpenCLBackend *)backend;
    auto runtime             = mOpenCLBackend->getOpenCLRuntime();
    const Convolution2D *conv2dParams = nullptr;
    if(isExtra){
        conv2dParams = flatbuffers::GetRoot<Convolution2D>(op->main_as_Extra()->attr()->GetAs<Attribute>(0)->tensor()->uint8s()->data());
    }else{
        conv2dParams = op->main_as_Convolution2D();
    }
    int biasSize             = conv2dParams->bias()->size();
    const float *biasDataPtr = conv2dParams->bias()->data();
    
    int buffer_size = ALIGN_UP8(biasSize) * sizeof(float);
    cl::Buffer biasBuffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto biasPtrCL = runtime->commandQueue().enqueueMapBuffer(biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(biasPtrCL != nullptr && error == CL_SUCCESS){
        ::memset(biasPtrCL, 0, ALIGN_UP8(biasSize) * sizeof(float));
        ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    runtime->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
    if (!(backend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC))) {
        mConvComValid = false;
        return;
    }
    copyBufferToImage(runtime, biasBuffer, openCLImage(mResource->mBias.get()), UP_DIV(biasSize, 4), 1, mOpenCLBackend->getPrecision());

    if(isExtra){
        const PRelu* preluParam = flatbuffers::GetRoot<PRelu>(op->main_as_Extra()->attr()->GetAs<Attribute>(1)->tensor()->uint8s()->data());
        const float *slopeDataPtr = preluParam->slope()->data();
        cl::Buffer slopeBuffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        cl_int error;
        auto slopePtrCL = runtime->commandQueue().enqueueMapBuffer(slopeBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(slopePtrCL != nullptr && error == CL_SUCCESS){
            ::memset(slopePtrCL, 0, ALIGN_UP8(biasSize) * sizeof(float));
            ::memcpy(slopePtrCL, slopeDataPtr, biasSize * sizeof(float));
        }else{
            MNN_ERROR("Map error slopePtrCL == nullptr \n");
        }
        runtime->commandQueue().enqueueUnmapMemObject(slopeBuffer, slopePtrCL);
        mResource->mSlope.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
        if (!(backend->onAcquireBuffer(mResource->mSlope.get(), Backend::STATIC))) {
            mConvComValid = false;
            return;
        }
        copyBufferToImage(runtime, slopeBuffer, openCLImage(mResource->mSlope.get()), UP_DIV(biasSize, 4), 1, mOpenCLBackend->getPrecision());
    }
}

ConvCommonExecution::~ConvCommonExecution() {
    // Do nothinng
}

ConvExecution::ConvExecution(std::shared_ptr<ConvResource> resource, const MNN::Op* op, Backend *backend)
    : CommonExecution(backend, op), ConvCommonExecution(backend) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool ConvExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvExecution(mResource, op, bn);
    return true;
}

ConvExecution::ConvExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend, bool isExtra)
: CommonExecution(backend, op), ConvCommonExecution(op, backend, isExtra) {
    if (!mConvComValid) {
        mValid = false;
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    const Convolution2D* conv2dParams = nullptr;
    if(isExtra){
        conv2dParams = flatbuffers::GetRoot<Convolution2D>(op->main_as_Extra()->attr()->GetAs<Attribute>(0)->tensor()->uint8s()->data());
        mResource->mPrelu = true;
    }else{
        conv2dParams       = op->main_as_Convolution2D();
    }
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dCommonParams = conv2dCommonParams;
    mResource->mStrides            = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations          = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    mResource->mRelu               = conv2dCommonParams->relu();
    mResource->mRelu6              = conv2dCommonParams->relu6();
        
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mResource->mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;
        
    int kernelWidth   = conv2dCommonParams->kernelX();
    int kernelHeight  = conv2dCommonParams->kernelY();
    int outputChannel = conv2dCommonParams->outputCount();
    auto gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
#ifndef MNN_OPENCL_BUFFER_CLOSED
    mResource->mWeightUseBuffer = gpuType == GpuType::MALI;
#endif
    
    int weightSize             = 0;
    const float *filterDataPtr = nullptr;
        
    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2dParams->quanParameter()) {
        quanCommon = ConvolutionCommon::load(op, backend, true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    } else if (nullptr == conv2dParams->weight() || nullptr == conv2dParams->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", op->name()->c_str());
    }
        
    if (nullptr == filterDataPtr) {
        weightSize    = conv2dParams->weight()->size();
        filterDataPtr = conv2dParams->weight()->data();
    }
    int inputChannel = weightSize / (kernelWidth * kernelHeight * outputChannel);
    
    //select opt conv method
    std::string kernelName = "conv_2d_c4h1w4";
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        mResource->mConv1x1Opt = (mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && gpuType == GpuType::MALI && !mResource->mWeightUseBuffer);
        if(mResource->mConv1x1Opt){
            kernelName = "conv_2d_1x1_mali";
        }else{
            kernelName = "conv_2d_1x1";
        }
    }
    
    if(mResource->mConv1x1Opt){
        cl_int error;
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({UP_DIV(outputChannel, 4)*4, UP_DIV(inputChannel, 4)*4, kernelWidth, kernelHeight}));
        
        int buffer_size = filterBuffer->elementSize();
        if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(kernelBufferPtr != nullptr && error == CL_SUCCESS){
            ::memset(kernelBufferPtr, 0, buffer_size);
            for(int o = 0; o < outputChannel; o++){
                for(int i = 0 ; i < inputChannel; i++){
                    int bufferIdx = (o/4) * ROUND_UP(inputChannel, 4)*4 + (i/4)*16 + (o%4)*4 + (i%4);
                    int filterIdx = o*inputChannel + i;
                    if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High){
                        ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                    }else{
                        ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelBuffer.get()), kernelBufferPtr);
        
    }else if(kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0 && mResource->mWeightUseBuffer){
        cl_int error;
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({UP_DIV(outputChannel, 4), ROUND_UP(inputChannel, 4), 4}));
        
        int buffer_size = filterBuffer->elementSize();
        if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(kernelBufferPtr != nullptr && error == CL_SUCCESS){
            ::memset(kernelBufferPtr, 0, buffer_size);
            for(int o = 0; o < outputChannel; o++){
                for(int i = 0 ; i < inputChannel; i++){
                    int bufferIdx = (o/4) * ROUND_UP(inputChannel, 4)*4 + i*4 + (o%4);
                    int filterIdx = o*inputChannel + i;
                    if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High){
                        ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                    }else{
                        ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelBuffer.get()), kernelBufferPtr);
    }else{
        std::vector<int> filterImageShape{(int)ROUND_UP(inputChannel, 4), (int)(UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(
                                             Tensor::createDevice<float>({outputChannel, ROUND_UP(inputChannel, 4), kernelWidth, kernelHeight}));
        
        size_t buffer_size = filterBuffer->elementSize() * sizeof(float);
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        
        cl_int error;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(ptrCL != nullptr && error == CL_SUCCESS) {
            ::memset(ptrCL, 0, buffer_size);
            int cpySrcNum = inputChannel * kernelWidth * kernelHeight;
            int cpyDstNum = ROUND_UP(inputChannel, 4) * kernelWidth * kernelHeight;
            int cpysize = cpySrcNum * sizeof(float);
            for(int o = 0; o < outputChannel; ++o){
                ::memcpy((float*)ptrCL + o * cpyDstNum, filterDataPtr + o * cpySrcNum, cpysize);
            }
            
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
#ifndef MNN_OPENCL_BUFFER_CLOSED
        if(mResource->mWeightUseBuffer){
            mResource->mFilter.reset(Tensor::createDevice<float>({UP_DIV(inputChannel, 4)*4, UP_DIV(outputChannel, 4), kernelWidth * kernelHeight, 4}));
            int kernel_buffer_size = UP_DIV(outputChannel, 4)*4* UP_DIV(inputChannel, 4)*4* kernelWidth* kernelHeight;
            if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
                kernel_buffer_size *= sizeof(half_float::half);
            } else {
                kernel_buffer_size *= sizeof(float);
            }
            mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, kernel_buffer_size));
            mResource->mFilter.get()->buffer().device = (uint64_t)mResource->mKernelBuffer.get();
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            
            bool needTrans = true;
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), mOpenCLBackend->getPrecision(), needTrans);
        } else
#endif
        {
            mResource->mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            OPENCL_CHECK_ALLOC_CTOR(mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC));
            MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            
            std::string buildOption = "-DBUFFER_INP_FP32";
            imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), mOpenCLBackend->getPrecision(), false, buildOption);
        }
    }
    
    // Create Kernel
    if (mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && mResource->mDilations[0] == 1 && mResource->mDilations[1] == 1) {
        mResource->mBuildOptions.emplace("-DMNN_CONV_S1D1");
    }
    mResource->mBuildOptions.emplace("-DBIAS");
    if (mResource->mRelu) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (mResource->mRelu6) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }else if(mResource->mPrelu){
        mResource->mBuildOptions.emplace("-DPRELU");
    }

    if(mResource->mWeightUseBuffer){
        mResource->mBuildOptions.emplace("-DUSE_BUFFER");
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvExecution::~ConvExecution() {
    // Do nothing
}

ErrorCode ConvExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto input  = inputs[0];
    auto output = outputs[0];
    
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int channel            = outputShape.at(3);
    
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    int kernelHeight = mResource->mConv2dCommonParams->kernelY();
    int kernelWidth  = mResource->mConv2dCommonParams->kernelX();
    
    auto pad = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;
    
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(channel) + "_" + std::to_string(kernelHeight) + "_" + std::to_string(kernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        if(mResource->mConv1x1Opt){
            
            std::string kernelName = "conv_2d_1x1_mali";
            unit.kernel             = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName, mResource->mBuildOptions, mOpenCLBackend->getPrecision());
            uint32_t idx            = 0;
            
            mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            unit.kernel->get().setArg(idx++, UP_DIV(width, 4));
            unit.kernel->get().setArg(idx++, openCLImage(input));
            unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
            unit.kernel->get().setArg(idx++, openCLImage(output));
            unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            unit.kernel->get().setArg(idx++, height);
            unit.kernel->get().setArg(idx++, width);
            if(mResource->mPrelu){
                unit.kernel->get().setArg(idx++, openCLImage(mResource->mSlope.get()));
            }
            
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mResource->mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "conv_2d").first;
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }else{
            int inputImageShape[2]  = {inputHeight, inputWidth};
            int outputImageShape[2] = {height, width};
            int stideShape[2]       = {mResource->mStrides[0], mResource->mStrides[1]};
            const int total_kernel = 2;
            std::string kernelName[total_kernel] = {"conv_2d_1x1", "conv_2d_1x1_c8h1w4"};
            int itemC[total_kernel] = {4, 8};
            int itemH[total_kernel] = {1, 1};
            int itemW[total_kernel] = {4, 4};
            
            int actual_kernel = total_kernel;
            
            std::shared_ptr<KernelWrap> kernel[total_kernel];
            std::vector<uint32_t> globalWorkSize[total_kernel];
            std::vector<uint32_t> localWorkSize[total_kernel];
            std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
            
            for(int knl_idx = 0; knl_idx < 1; knl_idx++) {
                std::set<std::string> buildOption = mResource->mBuildOptions;
                if(itemC[knl_idx] == 8 && outputShape.at(3) % itemC[knl_idx] > 0 && outputShape.at(3) % itemC[knl_idx] <= 4){
                    buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
                }
                kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], buildOption, mOpenCLBackend->getPrecision());
                uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                
                globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
                uint32_t idx            = 0;
                kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
                kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
                kernel[knl_idx]->get().setArg(idx++, openCLImage(input));
                if(mResource->mWeightUseBuffer){
                    kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelBuffer.get());
                }else{
                    kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
                }
                kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mBias.get()));
                kernel[knl_idx]->get().setArg(idx++, openCLImage(output));
                kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
                kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
                kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
                kernel[knl_idx]->get().setArg(idx++, sizeof(stideShape), stideShape);
                kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, 4));
                kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
                if(mResource->mPrelu){
                    kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mSlope.get()));
                }
                
                std::pair<std::vector<uint32_t>, uint32_t> retTune;
                retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx], mOpenCLBackend->getCLTuneLevel(), "conv_2d");
                
                //printf("conv1x1 kernel_%d = %d  [%d, %d]\n", knl_idx, retTune.second, retTune.first[0], retTune.first[1]);
                if(min_cost.first > retTune.second) {
                    min_cost.first = retTune.second;
                    min_cost.second = knl_idx;
                    mLocalWorkSize = {retTune.first[0], retTune.first[1]};
                }
            }
            int min_index  = min_cost.second;
            //printf("min_index = %d  %d\n", min_index, min_cost.first);
            mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
            std::set<std::string> buildOption = mResource->mBuildOptions;
            if(itemC[min_index] == 8 && outputShape.at(3) % itemC[min_index] > 0 && outputShape.at(3) % itemC[min_index] <= 4){
                buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
            }
            unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], buildOption, mOpenCLBackend->getPrecision());
            
            uint32_t idx = 0;
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            unit.kernel->get().setArg(idx++, openCLImage(input));
            if(mResource->mWeightUseBuffer){
                unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            }else{
                unit.kernel->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
            }
            unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
            unit.kernel->get().setArg(idx++, openCLImage(output));
            unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
            unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
            unit.kernel->get().setArg(idx++, sizeof(stideShape), stideShape);
            unit.kernel->get().setArg(idx++, UP_DIV(width, 4));
            unit.kernel->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
            if(mResource->mPrelu){
                unit.kernel->get().setArg(idx++, openCLImage(mResource->mSlope.get()));
            }
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }
    }else {
        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {height, width};
        int kernelShape[2]      = {kernelHeight, kernelWidth};
        int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
        int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
        int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
        
        const int total_kernel = 3;
        std::string kernelName[total_kernel] = {"conv_2d_c4h1w4", "conv_2d_c4h4w1", "conv_2d_c8h4w1" };
        int itemC[total_kernel] = {4, 4, 8};
        int itemH[total_kernel] = {1, 4, 4};
        int itemW[total_kernel] = {4, 1, 1};
        
        
        int actual_kernel = total_kernel;
        
        std::shared_ptr<KernelWrap> kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        
        // Nova/UNO Q workaround: MNN 3.6.1's h4w1 image kernels are
        // incorrect/failing for the S3 tokenizer's tall, width-1 conv2.
        // Restrict only this exact geometry to the conservative h1w4 kernel.
        if (inputChannels == 1280 || channel == 1280) {
            MNN_PRINT("[nova-opencl-dim] IC=%d C=%d IH=%d IW=%d H=%d W=%d KH=%d KW=%d S=%d,%d D=%d,%d\\n",
                      inputChannels, channel, inputHeight, inputWidth,
                      height, width, kernelHeight, kernelWidth,
                      mResource->mStrides[0], mResource->mStrides[1],
                      mResource->mDilations[0], mResource->mDilations[1]);
        }

        MNN_PRINT("[nova-opencl-buffer] input_dims=%d output_dims=%d\n",
                  input->buffer().dimensions, output->buffer().dimensions);

        MNN_PRINT("[nova-opencl-buffer] input_extents=");
        for (int i = 0; i < input->buffer().dimensions; ++i) {
            MNN_PRINT("%s%d", i ? "," : "", input->buffer().dim[i].extent);
        }
        MNN_PRINT(" output_extents=");
        for (int i = 0; i < output->buffer().dimensions; ++i) {
            MNN_PRINT("%s%d", i ? "," : "", output->buffer().dim[i].extent);
        }
        MNN_PRINT("\n");

        MNN_PRINT("[nova-opencl-tensor] input_dims=%d output_dims=%d input_lengths=",
                  input->dimensions(), output->dimensions());
        for (int i = 0; i < input->dimensions(); ++i) {
            MNN_PRINT("%s%d", i ? "," : "", input->length(i));
        }
        MNN_PRINT(" output_lengths=");
        for (int i = 0; i < output->dimensions(); ++i) {
            MNN_PRINT("%s%d", i ? "," : "", output->length(i));
        }
        MNN_PRINT("\n");

        MNN_PRINT("[nova-opencl-shapes] input=%d,%d,%d,%d output=%d,%d,%d,%d\n",
                  inputShape.at(0), inputShape.at(1),
                  inputShape.at(2), inputShape.at(3),
                  outputShape.at(0), outputShape.at(1),
                  outputShape.at(2), outputShape.at(3));

        const bool novaForceS3Conv =
            (
                inputChannels == 128 && channel == 1280 &&
                inputHeight == 1000 && inputWidth == 1 &&
                height == 500 && width == 1 &&
                kernelHeight == 3 && kernelWidth == 1 &&
                mResource->mStrides[0] == 2 && mResource->mStrides[1] == 1 &&
                mResource->mDilations[0] == 1 && mResource->mDilations[1] == 1
            ) ||
            (
                inputChannels == 1280 && channel == 1280 &&
                inputHeight == 500 && inputWidth == 1 &&
                height == 250 && width == 1 &&
                kernelHeight == 3 && kernelWidth == 1 &&
                mResource->mStrides[0] == 2 && mResource->mStrides[1] == 1 &&
                mResource->mDilations[0] == 1 && mResource->mDilations[1] == 1
            );

        const bool novaForceC4H4W1 =
            inputChannels == 1280 && channel == 1280 &&
            inputHeight == 500 && inputWidth == 1 &&
            height == 250 && width == 1 &&
            kernelHeight == 3 && kernelWidth == 1 &&
            mResource->mStrides[0] == 2 && mResource->mStrides[1] == 1 &&
            mResource->mDilations[0] == 1 && mResource->mDilations[1] == 1;

        // Pin only the first S3 convolution for a controlled kernel A/B.
        // Otherwise the tuner bypass overwrites min_index for every candidate,
        // leaving c8h4w1 selected without any measured execution cost.
        int novaConv1KernelIndex = -1;
        const char* novaConv1Kernel = std::getenv("NOVA_S3_CONV1_KERNEL");
        if (novaForceS3Conv && !novaForceC4H4W1 && novaConv1Kernel && *novaConv1Kernel) {
            for (int i = 0; i < total_kernel; ++i) {
                if (kernelName[i] == novaConv1Kernel) novaConv1KernelIndex = i;
            }
            if (novaConv1KernelIndex < 0) {
                MNN_ERROR("[nova-opencl] invalid NOVA_S3_CONV1_KERNEL=%s\n", novaConv1Kernel);
                return INVALID_VALUE;
            }
            MNN_PRINT("[nova-opencl] pinning S3 conv1 kernel=%s LWS=1,1\n", novaConv1Kernel);
        }

        if (novaForceC4H4W1) {
            kernelName[0] = "conv_2d_c4h4w1";
            itemC[0] = 4;
            itemH[0] = 4;
            itemW[0] = 1;
            MNN_PRINT("[nova-opencl] forcing conv_2d_c4h4w1 for 1280x500x1 -> 1280x250x1 conv2\\n");
        }

        const int novaKernelCount = novaForceC4H4W1 ? 1 : total_kernel;
        for(int knl_idx = 0; knl_idx < novaKernelCount; knl_idx++) {
            if (novaConv1KernelIndex >= 0 && knl_idx != novaConv1KernelIndex) continue;
            std::set<std::string> buildOption = mResource->mBuildOptions;
            if(itemC[knl_idx] == 8 && outputShape.at(3) % itemC[knl_idx] > 0 && outputShape.at(3) % itemC[knl_idx] <= 4){
                buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
            }
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], buildOption, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
            
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
            uint32_t idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(input));
            if(mResource->mWeightUseBuffer){
                ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
            }else{
                ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
            }
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mBias.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(output));
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, inputChannelBlocks);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(kernelShape), kernelShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(strideShape), strideShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(paddingShape), paddingShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(dilationShape), dilationShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, itemW[knl_idx]));
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(height, itemH[knl_idx]));
            if(mResource->mPrelu){
                ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mSlope.get()));
            }
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvExecution Kernel Select");
            
            if (novaForceS3Conv) {
                // Nova/UNO Q diagnostic: bypass MNN's LWS tuner for both
                // tokenizer stride-2 convolutions. The tuner enqueues many
                // real kernel executions and can poison the in-order queue.
                min_cost.first = 0;
                min_cost.second = knl_idx;
                mLocalWorkSize = {1, 1};
                MNN_PRINT("[nova-opencl] bypassing LWS tuner for S3 stride-2 conv LWS 1x1\\n");
            } else {
                std::pair<std::vector<uint32_t>, uint32_t> retTune;
                retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx], mOpenCLBackend->getCLTuneLevel(), "conv_2d");

                if(min_cost.first > retTune.second) {
                    min_cost.first = retTune.second;
                    min_cost.second = knl_idx;
                    mLocalWorkSize = {retTune.first[0], retTune.first[1]};
                }
            }
        }
        int min_index  = min_cost.second;

        MNN_PRINT("[nova-opencl] CONV SELECT kernel=%s index=%d "
                  "GWS=%u,%u LWS=%u,%u "
                  "shape_in=%d,%d shape_out=%d,%d "
                  "stride=%d,%d kernel=%d,%d\\n",
                  kernelName[min_index].c_str(),
                  min_index,
                  globalWorkSize[min_index][0],
                  globalWorkSize[min_index][1],
                  mLocalWorkSize[0],
                  mLocalWorkSize[1],
                  inputHeight, inputWidth,
                  height, width,
                  mResource->mStrides[0], mResource->mStrides[1],
                  kernelHeight, kernelWidth);

        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        if (novaForceC4H4W1) {
            // Nova/UNO Q diagnostic: expanded accumulator variants of c4h1w4
            // hit CL event failures with MNN's tuned local workgroup. Force a
            // single work-item per group to test whether this is register /
            // workgroup resource pressure rather than arithmetic correctness.
            mLocalWorkSize = {1, 1};
            MNN_PRINT("[nova-opencl] forcing LWS 1x1 for S3 conv2 diagnostic\n");
        }
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(itemC[min_index] == 8 && outputShape.at(3) % itemC[min_index] > 0 && outputShape.at(3) % itemC[min_index] <= 4){
            buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
        }
        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], buildOption, mOpenCLBackend->getPrecision());
        
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
        if(mResource->mWeightUseBuffer){
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        }else{
            ret |= unit.kernel->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= unit.kernel->get().setArg(idx++, inputChannelBlocks);
        ret |= unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(width, itemW[min_index]));
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(height, itemH[min_index]));
        if(mResource->mPrelu){
            ret |= unit.kernel->get().setArg(idx++, openCLImage(mResource->mSlope.get()));
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvExecution");
        if (ret != CL_SUCCESS) return INVALID_VALUE;
        const char* novaFailFast = std::getenv("NOVA_OPENCL_FAIL_FAST");
        if (novaFailFast && novaFailFast[0] == '1' && novaForceS3Conv) {
            auto runtime = mOpenCLBackend->getOpenCLRuntime();
            const auto maxImage = runtime->getMaxImage2DSize();
            cl_int argsRes = CL_SUCCESS;
            const auto numArgs = unit.kernel->get().getInfo<CL_KERNEL_NUM_ARGS>(&argsRes);
            MNN_PRINT("[nova-conv-bind] device=%s precision=%d fp16_supported=%d "
                      "max_image=%zu,%zu kernel_max_wg=%llu device_local_mem=%llu "
                      "num_args=%u args_res=%d bound_args=%u weight_buffer=%d\n",
                      runtime->getDeviceName().c_str(), mOpenCLBackend->getPrecision(),
                      runtime->isSupportedFP16(), maxImage[0], maxImage[1],
                      (unsigned long long)runtime->getMaxWorkGroupSize(unit.kernel),
                      (unsigned long long)runtime->getMaxLocalMem(), numArgs, argsRes, idx,
                      mResource->mWeightUseBuffer);
            auto logImage = [](const char* label, const cl::Image& image) {
                size_t width = 0, height = 0, bytes = 0;
                cl_image_format format = {};
                cl_mem_object_type type = 0;
                const cl_mem mem = image();
                const cl_int wr = clGetImageInfo(mem, CL_IMAGE_WIDTH, sizeof(width), &width, nullptr);
                const cl_int hr = clGetImageInfo(mem, CL_IMAGE_HEIGHT, sizeof(height), &height, nullptr);
                const cl_int fr = clGetImageInfo(mem, CL_IMAGE_FORMAT, sizeof(format), &format, nullptr);
                const cl_int tr = clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(type), &type, nullptr);
                const cl_int br = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(bytes), &bytes, nullptr);
                MNN_PRINT("[nova-conv-bind] %s handle=%p width=%zu height=%zu bytes=%zu "
                          "type=0x%x order=0x%x channel_type=0x%x query_res=%d,%d,%d,%d,%d\n",
                          label, (void*)mem, width, height, bytes, (unsigned)type,
                          (unsigned)format.image_channel_order, (unsigned)format.image_channel_data_type,
                          wr, hr, fr, tr, br);
            };
            logImage("input", openCLImage(input));
            if (!mResource->mWeightUseBuffer) logImage("filter", openCLImage(mResource->mFilter.get()));
            logImage("bias", openCLImage(mResource->mBias.get()));
            logImage("output", openCLImage(output));
            MNN_PRINT("[nova-conv-bind] input_shape=%d,%d output_shape=%d,%d "
                      "kernel=%d,%d stride=%d,%d pad=%d,%d dilation=%d,%d "
                      "input_c4=%d output_c4=%d width_blocks=%d height_blocks=%d\n",
                      inputImageShape[0], inputImageShape[1], outputImageShape[0], outputImageShape[1],
                      kernelShape[0], kernelShape[1], strideShape[0], strideShape[1],
                      paddingShape[0], paddingShape[1], dilationShape[0], dilationShape[1],
                      inputChannelBlocks, UP_DIV(outputShape.at(3), 4), UP_DIV(width, itemW[min_index]),
                      UP_DIV(height, itemH[min_index]));
        }
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    }

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}

class ConvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto conv2D = op->main_as_Convolution2D();
        std::vector<int> inputShape  = tensorShapeFormat(inputs[0]);
        const int inputChannels = inputShape.at(3);
#if defined(MNN_LOW_MEMORY) && not defined(MNN_OPENCL_BUFFER_CLOSED)
        if (static_cast<OpenCLBackend *>(backend)->getMemory() == BackendConfig::Memory_Low){
            auto conv2dParams = op->main_as_Convolution2D();
            if (conv2dParams->quanParameter() != nullptr) {
                if (((conv2dParams->quanParameter()->type() == 4) ||
                     (conv2dParams->quanParameter()->type() == 1) ||
                     (conv2dParams->quanParameter()->type() == 2))) {
                    if ((1 == conv2dParams->quanParameter()->type() || 2 == conv2dParams->quanParameter()->type()) && conv2dParams->quanParameter()->has_scaleInt()) {
                        // Don't support IDST-int8 because of error
                        return nullptr;
                    }
                    OPENCL_CREATOR_CHECK(new ConvLowMemoryExecution(inputs, outputs, op, backend));
                } else {
                    //MNN_ERROR("OpenCL Conv buf low memory init error. For Opencl Backend, only support low memory mode of int8 or int4 dequantization currently.\n");
                    return nullptr;
                }
            }
        }
#endif
        if(op->main_as_Convolution2D()->common()->group() > 1){
            // Don't support group > 1 now
            return nullptr;
        }
        
        if (inputs.size() > 1) {
            return nullptr;
        }
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }
        auto clBackend = static_cast<OpenCLBackend *>(backend);
        int maxWidth  = clBackend->getOpenCLRuntime()->getMaxImage2DSize()[0];
        int maxHeight = clBackend->getOpenCLRuntime()->getMaxImage2DSize()[1];
        if (ConvWinograd::valid(conv2D->common(), inputs[0], outputs[0], maxWidth, maxHeight, 8192,
                                clBackend->fpBytes(), clBackend->getMemory())) {
            OPENCL_CREATOR_CHECK(new ConvWinograd(op, backend));
        }
        
        OPENCL_CREATOR_CHECK(new ConvExecution(inputs, outputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(ConvolutionCreator, OpType_Convolution, IMAGE);

} // namespace OpenCL
} // namespace MNN
