#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/MathOp.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace MNN;
using namespace MNN::Express;

int main(int argc, char** argv) {
    const int backend = argc >= 2 ? std::atoi(argv[1]) : 0;
    const int precision = argc >= 3 ? std::atoi(argv[2]) : 0;
    const int threads = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 1;

    BackendConfig cfg;
    cfg.precision = static_cast<BackendConfig::PrecisionMode>(precision);

    auto exe = Executor::getGlobalExecutor();
    exe->setGlobalExecutorConfig(static_cast<MNNForwardType>(backend), cfg, threads);

    constexpr int N = 1;
    constexpr int C = 1280;
    constexpr int H = 500;
    constexpr int W = 1;
    constexpr int COUNT = N * C * H * W;

    auto input = _Input({N, C, H, W}, NCHW, halide_type_of<float>());
    float* p = input->writeMap<float>();
    if (!p) {
        std::fprintf(stderr, "[gelu-smoke] input writeMap failed\n");
        return 2;
    }

    for (int i = 0; i < COUNT; ++i) {
        p[i] = -0.467529f + (0.315186f + 0.467529f) * float(i % 4096) / 4095.0f;
    }

    auto output = _Gelu(input);
    const float* q = output->readMap<float>();
    if (!q) {
        std::fprintf(stderr, "[gelu-smoke] output readMap failed\n");
        return 3;
    }

    double minv = q[0], maxv = q[0], sum = 0.0, sumsq = 0.0, weighted = 0.0;
    for (int i = 0; i < COUNT; ++i) {
        const double v = q[i];
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
        sum += v;
        sumsq += v * v;
        weighted += v * double((i % 251) + 1);
    }

    std::printf("[gelu-smoke] backend=%d precision=%d threads=%d shape=1x1280x500x1\n",
                backend, precision, threads);
    std::printf("[gelu-smoke] min=%.9g max=%.9g mean=%.9g sum=%.9g sumsq=%.9g weighted=%.9g\n",
                minv, maxv, sum / COUNT, sum, sumsq, weighted);
    std::printf("[gelu-smoke] samples %.9g %.9g %.9g %.9g %.9g\n",
                q[0], q[1], q[2047], q[4095], q[COUNT - 1]);
    return 0;
}
