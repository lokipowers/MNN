//
//  CommonExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/CommonExecution.hpp"
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
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#else
    if(openCLBackend->isUseRecordQueue()){
        openCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
#endif
    auto res = CL_SUCCESS;
    int novaUnitIndex = 0;

    for (auto &unit : mUnits) {
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize,
                                                    nullptr,
                                                    &event);
        runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(idx++), event});
    #else
        cl::Event novaEvent;

        MNN_PRINT("[nova-exec] op=%s unit=%d GWS=%u,%u LWS=%u,%u\\n",
                  EnumNameOpType(mOpType),
                  novaUnitIndex,
                  (uint32_t)unit.globalWorkSize.get()[0],
                  (uint32_t)unit.globalWorkSize.get()[1],
                  (uint32_t)unit.localWorkSize.get()[0],
                  (uint32_t)unit.localWorkSize.get()[1]);

        res = runtime->commandQueue().enqueueNDRangeKernel(
            unit.kernel->get(),
            cl::NullRange,
            unit.globalWorkSize,
            unit.localWorkSize,
            nullptr,
            &novaEvent);

        MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));

        if (res == CL_SUCCESS) {
            cl_int statusRes = CL_SUCCESS;
            cl_int preStatus =
                novaEvent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(&statusRes);

            MNN_PRINT("[nova-exec] op=%s unit=%d prewait_status=%d info_res=%d\\n",
                      EnumNameOpType(mOpType),
                      novaUnitIndex,
                      preStatus,
                      statusRes);

            cl_int waitRes = novaEvent.wait();

            cl_int postStatusRes = CL_SUCCESS;
            cl_int postStatus =
                novaEvent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(&postStatusRes);

            MNN_PRINT("[nova-exec] op=%s unit=%d wait_res=%d post_status=%d info_res=%d\\n",
                      EnumNameOpType(mOpType),
                      novaUnitIndex,
                      waitRes,
                      postStatus,
                      postStatusRes);

            if (waitRes != CL_SUCCESS || postStatus < 0) {
                MNN_ERROR("[nova-exec] FIRST FAILURE op=%s unit=%d "
                          "GWS=%u,%u LWS=%u,%u "
                          "wait_res=%d post_status=%d info_res=%d\\n",
                          EnumNameOpType(mOpType),
                          novaUnitIndex,
                          (uint32_t)unit.globalWorkSize.get()[0],
                          (uint32_t)unit.globalWorkSize.get()[1],
                          (uint32_t)unit.localWorkSize.get()[0],
                          (uint32_t)unit.localWorkSize.get()[1],
                          waitRes,
                          postStatus,
                          postStatusRes);
                return OUT_OF_MEMORY;
            }
        }
    #endif

        ++novaUnitIndex;
    }
    return NO_ERROR;
}
} // namespace OpenCL
}; // namespace MNN
