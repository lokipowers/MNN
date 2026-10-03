//
//  CommonExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include <cstdlib>
namespace MNN {
namespace OpenCL {

CommonExecution::CommonExecution(Backend *backend, const MNN::Op *Op)
    : Execution(backend), mOp(Op) {
    mOpType = Op->type();
}

ErrorCode CommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
    openCLBackend->startRecord(mRecording);
    
    auto error = onEncode(inputs, outputs);
    if(NO_ERROR != error){
        return error;
    }
    
    for (auto &unit : mUnits) {
        bool lws_null = true;
        for (size_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
            unit.globalWorkSize.get()[i] = ROUND_UP(unit.globalWorkSize.get()[i], std::max((size_t)1, unit.localWorkSize.get()[i]));
            if(unit.localWorkSize.get()[i] != 0) {
                lws_null = false;
            }
        }
        if(lws_null){
            unit.localWorkSize = cl::NullRange;
        }
    }
    openCLBackend->endRecord(mRecording);
    return NO_ERROR;
}

ErrorCode CommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
    const char* diagnostic = std::getenv("NOVA_OPENCL_FAIL_FAST");
    const bool failFast = diagnostic && diagnostic[0] == '1';
#ifndef ENABLE_OPENCL_TIME_PROFILER
    if (!failFast && openCLBackend->isUseRecordQueue()) {
        openCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
#endif
    const char* opName = mOp->name() ? mOp->name()->c_str() : "<unnamed>";
    if (failFast) {
        // Resize/tuning and other execution classes may enqueue work too.
        // Do not attribute a previously failed queue to this operation.
        const cl_int prior = runtime->commandQueue().finish();
        if (prior != CL_SUCCESS) {
            MNN_ERROR("[nova-exec] FIRST FAILURE phase=before-op op=%s name=%s queue_res=%d\n",
                      EnumNameOpType(mOpType), opName, prior);
            return INVALID_VALUE;
        }
    }
    int unitIndex = 0;
    for (auto &unit : mUnits) {
        cl::Event event;
        std::string kernelName;
        if (failFast) {
            cl_int nameRes = CL_SUCCESS;
            kernelName = unit.kernel->get().getInfo<CL_KERNEL_FUNCTION_NAME>(&nameRes);
            if (nameRes != CL_SUCCESS) kernelName = "<unknown>";
            MNN_PRINT("[nova-exec] op=%s name=%s unit=%d kernel=%s GWS=",
                      EnumNameOpType(mOpType), opName, unitIndex, kernelName.c_str());
            for (size_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
                MNN_PRINT("%s%zu", i ? "," : "", unit.globalWorkSize.get()[i]);
            }
            MNN_PRINT(" LWS=");
            if (unit.localWorkSize.dimensions() == 0) MNN_PRINT("auto");
            for (size_t i = 0; i < unit.localWorkSize.dimensions(); ++i) {
                MNN_PRINT("%s%zu", i ? "," : "", unit.localWorkSize.get()[i]);
            }
            MNN_PRINT("\n");
        }
#ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event* eventPtr = &event;
#else
        cl::Event* eventPtr = failFast ? &event : nullptr;
#endif
        // Diagnostic only: partition independent 1x1 output rows. Preserve
        // the kernel arguments and LWS; global offsets retain output indices.
        // The listed image kernels address independent outputs with global IDs.
        // Bound each dispatch without changing arguments or accumulation order.
        // Issue 99 review candidate: FD702 FP32 S3 stride-2 and MLP geometries.
        // Other convolutions retain their dispatch size. Diagnostic serialization stays.
        bool rowIndependentConv = false;
        if (failFast && mOpType == OpType_Convolution &&
            mOp->main_type() == OpParameter_Convolution2D &&
            !inputs.empty() && !outputs.empty()) {
            const auto in = tensorShapeFormat(inputs[0]);
            const auto out = tensorShapeFormat(outputs[0]);
            const auto common = mOp->main_as_Convolution2D()->common();
            if (common) {
                MNN_PRINT("[nova-scope] name=%s in=%d,%d,%d,%d out=%d,%d,%d,%d kernel=%d,%d stride=%d,%d dilation=%d,%d group=%d\n",
                    opName, in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3],
                    common->kernelY(), common->kernelX(), common->strideY(), common->strideX(),
                    common->dilateY(), common->dilateX(), common->group());
                const bool unit2d = unit.globalWorkSize.dimensions() == 2 &&
                    unit.localWorkSize.dimensions() == 2 &&
                    unit.localWorkSize.get()[0] == 1 && unit.localWorkSize.get()[1] == 1;
                const bool conv1 = in == std::vector<int>({1,1000,1,128}) &&
                    out == std::vector<int>({1,500,1,1280}) &&
                    kernelName == "conv_2d_c8h4w1" && unit2d &&
                    unit.globalWorkSize.get()[0] == 160 && unit.globalWorkSize.get()[1] == 125;
                const bool conv2 = in == std::vector<int>({1,500,1,1280}) &&
                    out == std::vector<int>({1,250,1,1280}) &&
                    kernelName == "conv_2d_c4h4w1" && unit2d &&
                    unit.globalWorkSize.get()[0] == 320 && unit.globalWorkSize.get()[1] == 63;
                const bool mlp = in[0] == 250 && out[0] == 250 &&
                    in[1] == 1 && out[1] == 1 && in[2] == 1 && out[2] == 1 &&
                    ((in[3] == 1280 && out[3] == 5120) || (in[3] == 5120 && out[3] == 1280)) &&
                    common->kernelY() == 1 && common->kernelX() == 1 &&
                    common->strideY() == 1 && common->strideX() == 1 &&
                    kernelName == "conv_2d_1x1" && unit.globalWorkSize.dimensions() == 2 &&
                    unit.globalWorkSize.get()[0] == static_cast<size_t>(out[3] / 4) &&
                    unit.globalWorkSize.get()[1] == 250 && unit.localWorkSize.dimensions() == 0;
                rowIndependentConv = runtime->getDeviceName() == "FD702" &&
                    openCLBackend->getPrecision() == BackendConfig::Precision_High &&
                    common->dilateY() == 1 && common->dilateX() == 1 && common->group() == 1 &&
                    ((common->kernelY() == 3 && common->kernelX() == 1 &&
                      common->strideY() == 2 && common->strideX() == 1 && (conv1 || conv2)) || mlp);
            }
        }
        const char* convTileEnv = std::getenv("NOVA_OPENCL_CONV_ROWS");
        const char* tileEnv = kernelName == "conv_2d_1x1" && !convTileEnv ?
            std::getenv("NOVA_OPENCL_1X1_ROWS") : convTileEnv;
        const int requestedRows = tileEnv ? std::atoi(tileEnv) : 0;
        // Preserve the pre-existing explicitly requested 1x1 diagnostic.
        // NOVA_OPENCL_CONV_ROWS applies only to the FD702 S3 gate above.
        const bool legacy1x1 = !convTileEnv && kernelName == "conv_2d_1x1";
        const bool tileConv = failFast && requestedRows > 0 &&
            (rowIndependentConv || legacy1x1) && unit.globalWorkSize.dimensions() == 2;
        const size_t totalRows = tileConv ? unit.globalWorkSize.get()[1] : 1;
        const size_t localRows = unit.localWorkSize.dimensions() == 2 ?
            std::max(size_t(1), unit.localWorkSize.get()[1]) : 1;
        const size_t tileRows = tileConv ?
            ROUND_UP(static_cast<size_t>(requestedRows), localRows) : 1;
        for (size_t row = 0; row < totalRows; row += tileRows) {
            cl::NDRange dispatchOffset = cl::NullRange;
            cl::NDRange dispatchSize = unit.globalWorkSize;
            if (tileConv) {
                const size_t rows = std::min(tileRows, totalRows - row);
                dispatchOffset = cl::NDRange(0, row);
                dispatchSize = cl::NDRange(unit.globalWorkSize.get()[0], rows);
                MNN_PRINT("[nova-exec] conv tile kernel=%s row=%zu rows=%zu total=%zu\n", kernelName.c_str(), row, rows, totalRows);
            }
            const cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                unit.kernel->get(), dispatchOffset, dispatchSize,
                unit.localWorkSize, nullptr, eventPtr);
            if (res != CL_SUCCESS) {
                MNN_ERROR("[nova-exec] FIRST FAILURE phase=enqueue op=%s name=%s unit=%d res=%d\n",
                          EnumNameOpType(mOpType), opName, unitIndex, res);
                return INVALID_VALUE;
            }
            if (failFast) {
                const cl_int waitRes = event.wait();
                cl_int infoRes = CL_SUCCESS;
                const cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(&infoRes);
                MNN_PRINT("[nova-exec] op=%s name=%s unit=%d wait_res=%d post_status=%d info_res=%d\n",
                          EnumNameOpType(mOpType), opName, unitIndex, waitRes, status, infoRes);
                if (waitRes != CL_SUCCESS || infoRes != CL_SUCCESS || status != CL_COMPLETE) {
                    MNN_ERROR("[nova-exec] FIRST FAILURE phase=execute op=%s name=%s unit=%d "
                              "kernel=%s wait_res=%d post_status=%d info_res=%d\n",
                              EnumNameOpType(mOpType), opName, unitIndex, kernelName.c_str(),
                              waitRes, status, infoRes);
                    return INVALID_VALUE;
                }
            }
    #ifdef ENABLE_OPENCL_TIME_PROFILER
            runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(unitIndex), event});
    #endif
        }
        ++unitIndex;
    }
    return NO_ERROR;
}

} // namespace OpenCL
}; // namespace MNN
