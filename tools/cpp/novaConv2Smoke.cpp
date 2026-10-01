#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "MNN_generated.h"
#include "half.hpp"
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

static inline float fp16RoundTrip(float v) {
    const half_float::half h(v);
    return static_cast<float>(h);
}

int main(int argc, char** argv) {
    const int backend = argc >= 2 ? std::atoi(argv[1]) : 0;
    const int precision = argc >= 3 ? std::atoi(argv[2]) : 0;
    const int threads = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 1;
    const char* modelPath = argc >= 5 ? argv[4] : nullptr;
    const char* inputPath = argc >= 6 ? argv[5] : nullptr;
    if (modelPath && std::string(modelPath) == "-") modelPath = nullptr;
    const bool fp16Sim = std::getenv("NOVA_CONV2_FP16_SIM") != nullptr;
    const bool halfAccumSim = std::getenv("NOVA_CONV2_HALF_ACCUM_SIM") != nullptr;
    int splitChunks = 1;
    if (const char* splitEnv = std::getenv("NOVA_CONV2_SPLIT_CHUNKS")) {
        splitChunks = std::max(1, std::atoi(splitEnv));
    } else if (std::getenv("NOVA_CONV2_SPLIT_GRAPH") != nullptr) {
        splitChunks = 2;
    }

    BackendConfig cfg;
    cfg.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    Executor::getGlobalExecutor()->setGlobalExecutorConfig(
        static_cast<MNNForwardType>(backend), cfg, threads);

    constexpr int N = 1;
    constexpr int IC = 1280;
    constexpr int OC = 1280;
    constexpr int H = 500;
    constexpr int W = 1;
    constexpr int KW = 1;
    constexpr int KH = 3;
    constexpr int OUT_W = 250;
    constexpr int INPUT_COUNT = N * IC * H * W;
    constexpr int WEIGHT_COUNT = OC * IC * KH * KW;
    constexpr int OUTPUT_COUNT = N * OC * OUT_W;

    auto input = _Input({N, IC, H, W}, NCHW, halide_type_of<float>());
    float* ip = input->writeMap<float>();
    if (!ip) {
        std::fprintf(stderr, "[conv2-smoke] input writeMap failed\n");
        return 2;
    }
    std::string inputSource = "synthetic";
    if (inputPath) {
        std::ifstream bin(inputPath, std::ios::binary);
        if (!bin) {
            std::fprintf(stderr, "[conv2-smoke] unable to open input %s\n", inputPath);
            return 8;
        }
        bin.read(reinterpret_cast<char*>(ip), static_cast<std::streamsize>(INPUT_COUNT * sizeof(float)));
        if (bin.gcount() != static_cast<std::streamsize>(INPUT_COUNT * sizeof(float))) {
            std::fprintf(stderr, "[conv2-smoke] input size mismatch bytes=%lld expected=%zu\n",
                         static_cast<long long>(bin.gcount()),
                         static_cast<size_t>(INPUT_COUNT * sizeof(float)));
            return 9;
        }
        char extra = 0;
        if (bin.read(&extra, 1)) {
            std::fprintf(stderr, "[conv2-smoke] input file has trailing bytes\n");
            return 10;
        }
        inputSource = "dump";
    } else {
        for (int i = 0; i < INPUT_COUNT; ++i) {
            const int k = i % 257;
            ip[i] = (float(k) - 128.0f) / 256.0f;
        }
    }

    if (fp16Sim) {
        for (int i = 0; i < INPUT_COUNT; ++i) {
            ip[i] = fp16RoundTrip(ip[i]);
        }
    }

    std::vector<float> weight;
    std::vector<float> bias;
    std::string weightSource = "synthetic";

    if (modelPath) {
        std::ifstream in(modelPath, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "[conv2-smoke] unable to open model %s\n", modelPath);
            return 5;
        }
        std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const MNN::Net* net = buf.empty() ? nullptr : MNN::GetNet(buf.data());
        const MNN::Convolution2D* found = nullptr;
        if (net && net->oplists()) {
            for (auto op : *net->oplists()) {
                if (!op || !op->name()) continue;
                if (op->name()->str() == "/conv2/Conv_output_0" &&
                    op->main_type() == MNN::OpParameter_Convolution2D) {
                    found = op->main_as_Convolution2D();
                    break;
                }
            }
        }
        if (!found || !found->weight() || !found->bias()) {
            std::fprintf(stderr, "[conv2-smoke] conv2 weights/bias not found in model\n");
            return 6;
        }
        weight.assign(found->weight()->begin(), found->weight()->end());
        bias.assign(found->bias()->begin(), found->bias()->end());
        if (weight.size() != WEIGHT_COUNT || bias.size() != OC) {
            std::fprintf(stderr, "[conv2-smoke] unexpected conv2 constants weight=%zu bias=%zu\n",
                         weight.size(), bias.size());
            return 7;
        }
        weightSource = "model";
    } else {
        weight.resize(WEIGHT_COUNT);
        for (int i = 0; i < WEIGHT_COUNT; ++i) {
            const int k = i % 31;
            weight[i] = (float(k) - 15.0f) / 4096.0f;
        }
        bias.resize(OC);
        for (int i = 0; i < OC; ++i) {
            bias[i] = float((i % 17) - 8) / 512.0f;
        }
    }

    if (fp16Sim) {
        for (auto& v : weight) v = fp16RoundTrip(v);
        for (auto& v : bias) v = fp16RoundTrip(v);
    }


    if (halfAccumSim) {
        const int sampleIndices[5] = {0, 1, OUTPUT_COUNT / 2, OUTPUT_COUNT - 2, OUTPUT_COUNT - 1};

        std::printf("[conv2-smoke] manual_fp32_samples");
        for (int si = 0; si < 5; ++si) {
            const int outIndex = sampleIndices[si];
            const int oc = outIndex / OUT_W;
            const int oh = outIndex % OUT_W;
            float acc = bias[oc];
            for (int ic = 0; ic < IC; ++ic) {
                for (int kh = 0; kh < KH; ++kh) {
                    const int ih = oh * 2 - 1 + kh;
                    if (ih < 0 || ih >= H) continue;
                    const float xv = ip[ic * H + ih];
                    const float wv = weight[(oc * IC + ic) * KH + kh];
                    acc += xv * wv;
                }
            }
            std::printf(" %.9g", acc);
        }
        std::printf("\n");

        std::printf("[conv2-smoke] halfacc_samples");
        for (int si = 0; si < 5; ++si) {
            const int outIndex = sampleIndices[si];
            const int oc = outIndex / OUT_W;
            const int oh = outIndex % OUT_W;
            float acc = fp16RoundTrip(bias[oc]);
            for (int ic = 0; ic < IC; ++ic) {
                for (int kh = 0; kh < KH; ++kh) {
                    const int ih = oh * 2 - 1 + kh;
                    if (ih < 0 || ih >= H) continue;
                    const float xv = fp16RoundTrip(ip[ic * H + ih]);
                    const float wv = fp16RoundTrip(weight[(oc * IC + ic) * KH + kh]);
                    acc = fp16RoundTrip(acc + xv * wv);
                }
            }
            std::printf(" %.9g", acc);
        }
        std::printf("\n");
    }

    VARP output;
    if (splitChunks > 1) {
        if (IC % splitChunks != 0) {
            std::fprintf(stderr, "[conv2-smoke] split chunk count %d does not divide IC=%d\n", splitChunks, IC);
            return 12;
        }
        const int chunkIC = IC / splitChunks;
        std::vector<VARP> partials;
        partials.reserve(splitChunks);

        for (int chunk = 0; chunk < splitChunks; ++chunk) {
            auto chunkInput = _Input({N, chunkIC, H, W}, NCHW, halide_type_of<float>());
            float* cp = chunkInput->writeMap<float>();
            if (!cp) {
                std::fprintf(stderr, "[conv2-smoke] split input writeMap failed chunk=%d\n", chunk);
                return 11;
            }

            const size_t inputOffset = static_cast<size_t>(chunk) * chunkIC * H * W;
            std::copy(ip + inputOffset,
                      ip + inputOffset + static_cast<size_t>(chunkIC) * H * W,
                      cp);

            std::vector<float> chunkWeight(static_cast<size_t>(OC) * chunkIC * KH * KW);
            const size_t perOcCount = static_cast<size_t>(chunkIC) * KH * KW;
            for (int oc = 0; oc < OC; ++oc) {
                const size_t srcBase =
                    (static_cast<size_t>(oc) * IC + static_cast<size_t>(chunk) * chunkIC) * KH * KW;
                const size_t dstBase = static_cast<size_t>(oc) * perOcCount;
                std::copy(weight.begin() + srcBase,
                          weight.begin() + srcBase + perOcCount,
                          chunkWeight.begin() + dstBase);
            }

            std::vector<float> chunkBias = chunk == 0 ? bias : std::vector<float>(OC, 0.0f);
            partials.push_back(_Conv(std::move(chunkWeight), std::move(chunkBias), chunkInput,
                                     {chunkIC, OC}, {KW, KH}, CAFFE,
                                     {1, 2}, {1, 1}, 1, {0, 1}, false, false));
        }

        output = partials[0];
        for (int chunk = 1; chunk < splitChunks; ++chunk) {
            output = _Add(output, partials[chunk]);
        }
    } else {
        output = _Conv(std::move(weight), std::move(bias), input,
                       {IC, OC}, {KW, KH}, CAFFE,
                       {1, 2}, {1, 1}, 1, {0, 1}, false, false);
    }

    const float* q = output->readMap<float>();
    if (!q) {
        std::fprintf(stderr, "[conv2-smoke] output readMap failed\n");
        return 3;
    }
    const auto* info = output->getInfo();
    if (!info) {
        std::fprintf(stderr, "[conv2-smoke] output info unavailable\n");
        return 4;
    }

    double minv = q[0], maxv = q[0], sum = 0.0, sumsq = 0.0, weighted = 0.0;
    int count = info->size;
    for (int i = 0; i < count; ++i) {
        const double v = q[i];
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
        sum += v;
        sumsq += v * v;
        weighted += v * double((i % 251) + 1);
    }

    std::printf("[conv2-smoke] backend=%d precision=%d threads=%d weights=%s input=%s fp16_sim=%d half_accum_sim=%d split_chunks=%d\n",
                backend, precision, threads, weightSource.c_str(), inputSource.c_str(), fp16Sim ? 1 : 0, halfAccumSim ? 1 : 0, splitChunks);
    std::printf("[conv2-smoke] input=1x1280x500x1 kernel=3x1 stride=2x1 pad=1x0 output_elements=%d expected=%d\n",
                count, OUTPUT_COUNT);
    std::printf("[conv2-smoke] min=%.9g max=%.9g mean=%.9g sum=%.9g sumsq=%.9g weighted=%.9g\n",
                minv, maxv, sum / count, sum, sumsq, weighted);
    std::printf("[conv2-smoke] samples %.9g %.9g %.9g %.9g %.9g\n",
                q[0], q[1], q[count / 2], q[count - 2], q[count - 1]);
    return 0;
}
