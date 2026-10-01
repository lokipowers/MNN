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
        const cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
            unit.kernel->get(), cl::NullRange, unit.globalWorkSize,
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
        ++unitIndex;
    }
    return NO_ERROR;
}

} // namespace OpenCL
}; // namespace MNN
