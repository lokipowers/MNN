#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Interpreter.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

static int runCase(const char* label, bool pattern, const std::vector<float>* dump = nullptr) {
    constexpr int N = 1;
    const int C = dump ? 1280 : 128;
    const int H = dump ? 500 : 1000;
    constexpr int W = 1;
    const int COUNT = N * C * H * W;

    auto input = _Input({N, C, H, W}, NCHW, halide_type_of<float>());
    float* p = input->writeMap<float>();
    if (!p) {
        std::fprintf(stderr, "[raster-smoke] %s input writeMap failed\n", label);
        return 2;
    }

    for (int i = 0; i < COUNT; ++i) {
        p[i] = dump ? (*dump)[i] : (pattern ? static_cast<float>((i % 257) - 128) / 128.0f : 0.0f);
    }

    auto packed = _Convert(input, NC4HW4);
    auto output = _Convert(packed, NCHW);

    const float* q = output->readMap<float>();
    if (!q) {
        std::fprintf(stderr, "[raster-smoke] %s output readMap failed\n", label);
        return 3;
    }

    double maxAbs = 0.0;
    double maxErr = 0.0;
    double sum = 0.0;
    int firstBad = -1;
    for (int i = 0; i < COUNT; ++i) {
        const double expected = dump ? static_cast<double>((*dump)[i]) : (pattern ? static_cast<double>((i % 257) - 128) / 128.0 : 0.0);
        const double got = q[i];
        const double err = std::abs(got - expected);
        maxAbs = std::max(maxAbs, std::abs(got));
        maxErr = std::max(maxErr, err);
        sum += got;
        if (firstBad < 0 && err > 1e-6) firstBad = i;
    }

    std::printf("[raster-smoke] %s max_abs=%.9g max_err=%.9g sum=%.9g first_bad=%d\n",
                label, maxAbs, maxErr, sum, firstBad);
    std::fflush(stdout);

    return firstBad < 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    const int backend = argc >= 2 ? std::atoi(argv[1]) : 0;
    const int precision = argc >= 3 ? std::atoi(argv[2]) : 0;

    BackendConfig cfg;
    cfg.precision = static_cast<BackendConfig::PrecisionMode>(precision);

    auto exe = Executor::getGlobalExecutor();
    exe->setGlobalExecutorConfig(static_cast<MNNForwardType>(backend), cfg, 4);

    const char* dumpPath = argc >= 4 ? argv[3] : nullptr;
    std::vector<float> dump;
    if (dumpPath) {
        std::ifstream in(dumpPath, std::ios::binary | std::ios::ate);
        if (!in) {
            std::fprintf(stderr, "[raster-smoke] failed to open dump: %s\n", dumpPath);
            return 2;
        }
        const auto bytes = static_cast<size_t>(in.tellg());
        constexpr size_t expectedBytes = 1280ull * 500ull * sizeof(float);
        if (bytes != expectedBytes) {
            std::fprintf(stderr, "[raster-smoke] dump bytes=%zu expected=%zu\n", bytes, expectedBytes);
            return 2;
        }
        dump.resize(1280 * 500);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(dump.data()), bytes);
    }

    std::printf("[raster-smoke] backend=%d precision=%d shape=%s\n",
                backend, precision, dumpPath ? "1x1280x500x1" : "1x128x1000x1");
    std::fflush(stdout);

    const int z = runCase("zeros", false);
    const int p = runCase("pattern", true);
    const int d = dumpPath ? runCase("dump", false, &dump) : 0;

    if (z == 0 && p == 0 && d == 0) {
        std::puts("[raster-smoke] SUCCESS");
        return 0;
    }
    std::puts("[raster-smoke] FAIL");
    return 1;
}
