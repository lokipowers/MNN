#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/TensorUtils.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>

using namespace MNN;
using namespace MNN::Express;


static std::string dimsString(const MNN::Tensor* t) {
    std::ostringstream os;
    for (int i = 0; i < t->dimensions(); ++i) {
        if (i) os << "x";
        os << t->length(i);
    }
    return os.str();
}


static void printTensorMeta(const char* phase, const MNN::OperatorInfo* info,
                            const std::vector<MNN::Tensor*>& tensors) {
    if (!info) return;
    const char* filterEnv = std::getenv("NOVA_MNN_META");
    if (!filterEnv || !*filterEnv) return;
    const std::string filter(filterEnv);
    const std::string name = info->name();
    if (name.find(filter) == std::string::npos) return;

    for (size_t ti = 0; ti < tensors.size(); ++ti) {
        const auto* t = tensors[ti];
        if (!t) continue;
        auto* desc = MNN::TensorUtils::getDescribe(t);
        std::ostringstream strides;
        for (int d = 0; d < t->dimensions(); ++d) {
            if (d) strides << "x";
            strides << t->stride(d);
        }
        std::fprintf(stdout,
            "[s3-meta] phase=%s op=%s type=%s tensor=%zu dims=%s dimType=%d "
            "format=%d device=0x%llx host=%p strides=%s elements=%d bytes=%zu\n",
            phase,
            info->name().c_str(),
            info->type().c_str(),
            ti,
            dimsString(t).c_str(),
            static_cast<int>(t->getDimensionType()),
            desc ? static_cast<int>(desc->dimensionFormat) : -1,
            static_cast<unsigned long long>(t->deviceId()),
            static_cast<void*>(t->host<void>()),
            strides.str().c_str(),
            t->elementSize(),
            static_cast<size_t>(t->size()));
    }
    std::fflush(stdout);
}

template <typename T>
static void summarizeTyped(const T* p, int n, double& minv, double& maxv,
                           double& sum, double& sumsq, double& weighted) {
    if (n <= 0 || p == nullptr) {
        minv = maxv = sum = sumsq = weighted = 0.0;
        return;
    }
    minv = maxv = static_cast<double>(p[0]);
    sum = sumsq = weighted = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(p[i]);
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
        sum += v;
        sumsq += v * v;
        weighted += v * static_cast<double>((i % 251) + 1);
    }
}

static void installTrace(const std::shared_ptr<Executor::RuntimeManager>& rtmgr,
                         const char* path, int maxOps) {
    const char* logFromEnv = std::getenv("NOVA_MNN_LOG_OPS_FROM");
    const char* logToEnv = std::getenv("NOVA_MNN_LOG_OPS_TO");
    const bool wantTrace = path && *path;
    const bool wantLog = logFromEnv && *logFromEnv;
    const char* requestedDumpOp = std::getenv("NOVA_MNN_DUMP_OP");
    const char* requestedDumpPath = std::getenv("NOVA_MNN_DUMP_PATH");
    const bool wantDump = requestedDumpOp && *requestedDumpOp &&
                          requestedDumpPath && *requestedDumpPath;
    if (!wantTrace && !wantLog && !wantDump) return;

    rtmgr->setMode(Interpreter::Session_Debug);
    std::shared_ptr<std::ofstream> out;
    if (wantTrace) {
        out = std::make_shared<std::ofstream>(path);
        if (!out->good()) {
            std::fprintf(stderr, "[s3-mnn] unable to open trace file: %s\n", path);
            return;
        }
        *out << "op_index\top_name\top_type\ttensor_index\tdims\ttype_code\ttype_bits\telements\tmin\tmax\tmean\tsum\tsumsq\tweighted\n";
    }
    auto counter = std::make_shared<int>(0);
    const int logFrom = logFromEnv ? std::max(0, std::atoi(logFromEnv)) : -1;
    const int logTo = logToEnv ? std::max(logFrom, std::atoi(logToEnv)) : -1;
    const char* dumpOpEnv = std::getenv("NOVA_MNN_DUMP_OP");
    const char* dumpPathEnv = std::getenv("NOVA_MNN_DUMP_PATH");
    const std::string dumpOp = dumpOpEnv ? dumpOpEnv : "";
    const std::string dumpPath = dumpPathEnv ? dumpPathEnv : "";

    MNN::TensorCallBackWithInfo before =
        [](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
            printTensorMeta("before", info, tensors);
            return true;
        };

    MNN::TensorCallBackWithInfo after =
        [out, counter, maxOps, logFrom, logTo, dumpOp, dumpPath](const std::vector<MNN::Tensor*>& tensors,
                                                                                const MNN::OperatorInfo* info) {
            const int opIndex = (*counter)++;
            printTensorMeta("after", info, tensors);
            if (logFrom >= 0 && opIndex >= logFrom && (logTo < 0 || opIndex <= logTo)) {
                std::fprintf(stdout, "[s3-op] index=%d name=%s type=%s tensors=%zu\n",
                             opIndex,
                             info ? info->name().c_str() : "<null>",
                             info ? info->type().c_str() : "<null>",
                             tensors.size());
                std::fflush(stdout);
            }
            const bool selectedDump = info && !dumpPath.empty() && !dumpOp.empty() &&
                                      info->name() == dumpOp;
            const bool traceThis = out && opIndex < maxOps;
            if (!traceThis && !selectedDump) return true;

            for (size_t ti = 0; ti < tensors.size(); ++ti) {
                MNN::Tensor* src = tensors[ti];
                if (!src) continue;

                std::shared_ptr<MNN::Tensor> host(
                    new MNN::Tensor(src, selectedDump ? MNN::Tensor::CAFFE : src->getDimensionType()));
                if (!src->copyToHostTensor(host.get())) {
                    std::fprintf(stderr, "[s3-mnn] tensor readback failed op=%s\n", info->name().c_str());
                    return false;
                }
                src = host.get();

                const int n = src->elementSize();
                const auto type = src->getType();
                double minv = 0, maxv = 0, sum = 0, sumsq = 0, weighted = 0;
                bool numeric = false;

                if (n > 0 && type.code == halide_type_float && type.bits == 32) {
                    summarizeTyped(src->host<float>(), n, minv, maxv, sum, sumsq, weighted);
                    numeric = true;
                } else if (n > 0 && type.code == halide_type_int && type.bits == 32) {
                    summarizeTyped(src->host<int32_t>(), n, minv, maxv, sum, sumsq, weighted);
                    numeric = true;
                } else if (n > 0 && type.code == halide_type_int && type.bits == 8) {
                    summarizeTyped(src->host<int8_t>(), n, minv, maxv, sum, sumsq, weighted);
                    numeric = true;
                } else if (n > 0 && type.code == halide_type_uint && type.bits == 8) {
                    summarizeTyped(src->host<uint8_t>(), n, minv, maxv, sum, sumsq, weighted);
                    numeric = true;
                }

                if (!dumpPath.empty() && !dumpOp.empty() &&
                    info->name() == dumpOp && ti == 0 &&
                    n > 0 && type.code == halide_type_float && type.bits == 32) {
                    std::ofstream bin(dumpPath, std::ios::binary | std::ios::trunc);
                    if (bin.good()) {
                        bin.write(reinterpret_cast<const char*>(src->host<float>()),
                                  static_cast<std::streamsize>(n * sizeof(float)));
                        bin.close();
                        if (bin.fail()) {
                            std::fprintf(stderr, "[s3-mnn] tensor dump write failed: %s\n", dumpPath.c_str());
                            return false;
                        }
                        std::fprintf(stdout,
                                     "[s3-mnn] dumped op=%s tensor=%zu dims=%s format=NCHW elements=%d path=%s\n",
                                     info->name().c_str(), ti, dimsString(src).c_str(), n, dumpPath.c_str());
                        std::fflush(stdout);
                    } else {
                        std::fprintf(stderr, "[s3-mnn] unable to dump tensor to %s\n",
                                     dumpPath.c_str());
                        return false;
                    }
                }

                if (!traceThis) continue;
                const double mean = (numeric && n > 0) ? (sum / static_cast<double>(n)) : 0.0;
                *out << opIndex << '\t'
                     << info->name() << '\t'
                     << info->type() << '\t'
                     << ti << '\t'
                     << dimsString(src) << '\t'
                     << type.code << '\t'
                     << type.bits << '\t'
                     << n << '\t'
                     << minv << '\t'
                     << maxv << '\t'
                     << mean << '\t'
                     << sum << '\t'
                     << sumsq << '\t'
                     << weighted << '\n';
            }
            if (out) out->flush();
            return true;
        };

    ExecutorScope::Current()->setCallBack(std::move(before), std::move(after));
    if (wantTrace) {
        std::printf("[s3-mnn] trace=%s max_ops=%d\n", path, maxOps);
    }
    if (wantLog) {
        std::printf("[s3-mnn] op-log from=%d to=%d trace_readback=%d\n",
                    logFrom, logTo, wantTrace ? 1 : 0);
    }
    std::fflush(stdout);
}

static void mark(const char* msg) {
    std::fprintf(stdout, "[s3-mnn] %s\n", msg);
    std::fflush(stdout);
    std::fflush(stderr);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL backend frames [precision] [frozen_length] [threads]\n", argv[0]);
        return 2;
    }

    const char* modelPath = argv[1];
    const int backend = std::atoi(argv[2]);
    const int frames = std::atoi(argv[3]);
    const int precision = argc >= 5 ? std::atoi(argv[4]) : (backend == 3 ? 2 : 0);
    const bool frozenLength = argc >= 6 ? (std::atoi(argv[5]) != 0) : false;
    const int threads = argc >= 7 ? std::max(1, std::atoi(argv[6])) : 4;
    if (frames <= 0) return 2;

    std::printf("[s3-mnn] model=%s backend=%d frames=%d precision=%d\n",
                modelPath, backend, frames, precision);
    std::fflush(stdout);

    ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(backend);
    // Let unsupported OpenCL ops fall back to CPU. Using the requested backend
    // as its own backup prevents MNN's normal heterogeneous fallback path.
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = threads;

    BackendConfig backendConfig;
    backendConfig.precision =
        static_cast<BackendConfig::PrecisionMode>(precision);
    config.backendConfig = &backendConfig;

    mark("create runtime manager");
    std::shared_ptr<Executor::RuntimeManager> rtmgr(
        Executor::RuntimeManager::createRuntimeManager(config),
        Executor::RuntimeManager::destroy
    );
    if (!rtmgr) {
        std::fprintf(stderr, "[s3-mnn] failed to create runtime manager\n");
        return 3;
    }

    const char* tracePath = std::getenv("NOVA_MNN_TRACE");
    const char* maxOpsEnv = std::getenv("NOVA_MNN_TRACE_MAX_OPS");
    const int traceMaxOps = maxOpsEnv ? std::max(1, std::atoi(maxOpsEnv)) : 120;
    installTrace(rtmgr, tracePath, traceMaxOps);

    const std::vector<std::string> inputNames = frozenLength
        ? std::vector<std::string>{"feats"}
        : std::vector<std::string>{"feats", "feats_length"};
    const std::vector<std::string> outputNames{"indices"};

    Module::Config moduleConfig;
    moduleConfig.shapeMutable = true;

    mark("load module");
    std::shared_ptr<Module> module(
        Module::load(inputNames, outputNames, modelPath, rtmgr, &moduleConfig),
        Module::destroy
    );
    if (!module) {
        std::fprintf(stderr, "[s3-mnn] Module::load returned null\n");
        return 4;
    }

    mark("create inputs");
    VARP feats = _Input({1, 128, frames}, NCHW, halide_type_of<float>());
    if (feats == nullptr) return 5;

    mark("zero feats");
    float* featsPtr = feats->writeMap<float>();
    if (!featsPtr) {
        std::fprintf(stderr, "[s3-mnn] feats writeMap failed\n");
        return 6;
    }
    std::memset(featsPtr, 0, static_cast<size_t>(128) * frames * sizeof(float));

    const char* featurePath = std::getenv("NOVA_S3_FEATURES");
    if (featurePath && *featurePath) {
        const size_t count = static_cast<size_t>(128) * frames;
        const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
        std::ifstream featureFile(featurePath, std::ios::binary | std::ios::ate);
        if (!featureFile || featureFile.tellg() != bytes) {
            std::fprintf(stderr, "[s3-mnn] feature file must contain exactly %zu float32 values: %s\n",
                         count, featurePath);
            return 12;
        }
        featureFile.seekg(0);
        featureFile.read(reinterpret_cast<char*>(featsPtr), bytes);
        if (!featureFile) return 12;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(featsPtr[i])) {
                std::fprintf(stderr, "[s3-mnn] nonfinite feature at index=%zu\n", i);
                return 12;
            }
        }
        std::printf("[s3-mnn] features=%s layout=1x128x%d float32 elements=%zu\n",
                    featurePath, frames, count);
    }

    std::vector<VARP> inputs{feats};
    if (!frozenLength) {
        mark("set feats_length");
        VARP featsLength = _Input({1}, NCHW, halide_type_of<int32_t>());
        if (featsLength == nullptr) return 7;
        int32_t* lenPtr = featsLength->writeMap<int32_t>();
        if (!lenPtr) {
            std::fprintf(stderr, "[s3-mnn] feats_length writeMap failed\n");
            return 7;
        }
        lenPtr[0] = frames;
        inputs.push_back(featsLength);
    } else {
        mark("feats_length frozen in model");
    }

    mark("module forward");
    const auto start = std::chrono::steady_clock::now();
    auto outputs = module->onForward(inputs);
    if (outputs.size() != 1 || outputs[0] == nullptr) {
        std::fprintf(stderr, "[s3-mnn] onForward returned %zu outputs\n", outputs.size());
        return 8;
    }

    const auto* outputInfo = outputs[0]->getInfo();
    if (!outputInfo || outputInfo->type != halide_type_of<int32_t>() || outputInfo->size <= 0) {
        std::fprintf(stderr, "[s3-mnn] expected nonempty int32 indices output\n");
        return 10;
    }

    mark("materialize output");
    const int32_t* outputPtr = outputs[0]->readMap<int32_t>();
    const auto end = std::chrono::steady_clock::now();
    const double secs =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    if (!outputPtr) {
        std::fprintf(stderr, "[s3-mnn] output readMap failed\n");
        return 9;
    }

    const auto* info = outputs[0]->getInfo();
    if (!info) {
        std::fprintf(stderr, "[s3-mnn] output info unavailable\n");
        return 10;
    }

    std::printf("[s3-mnn] elapsed=%.3fs output dims=", secs);
    for (size_t i = 0; i < info->dim.size(); ++i) {
        std::printf("%s%d", i ? "x" : "", info->dim[i]);
    }
    std::printf(" elements=%zu first_index=%d\n",
                info->size,
                info->size > 0 ? outputPtr[0] : -1);
    std::fflush(stdout);

    // Dump all materialized tokens, one decimal int32 per line, for exact
    // CPU/OpenCL comparison. A requested dump must succeed to report success.
    const char* tokenDumpPath = std::getenv("NOVA_S3_TOKEN_DUMP");
    if (tokenDumpPath && *tokenDumpPath) {
        std::ofstream tokens(tokenDumpPath, std::ios::out | std::ios::trunc);
        if (!tokens.is_open()) {
            std::fprintf(stderr, "[s3-mnn] unable to open token dump: %s\n", tokenDumpPath);
            return 11;
        }
        for (size_t i = 0; i < info->size; ++i) {
            tokens << outputPtr[i] << '\n';
        }
        tokens.close();
        if (tokens.fail()) {
            std::fprintf(stderr, "[s3-mnn] token dump write failed: %s\n", tokenDumpPath);
            return 11;
        }
        std::printf("[s3-mnn] token_dump=%s elements=%zu\n", tokenDumpPath, info->size);
        std::fflush(stdout);
    }

    float memoryMB = 0.0f;
    if (rtmgr->getInfo(Interpreter::MEMORY, &memoryMB)) {
        std::printf("[s3-mnn] runtime memory=%.1f MB\n", memoryMB);
        std::fflush(stdout);
    }

    mark("OUTPUT_MATERIALIZED (token correctness requires CPU comparison)");
    return 0;
}
