// Regression for the ARM64 NEON CPU/OpenCL image FP32 contract used by the S3 tokenizer.
#include <cmath>
#include <cstring>
#include <MNN/MNNForwardType.h>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;

class GeluCpuApproximationTest : public MNNTestCase {
public:
    bool run(int precision) override {
#if (!defined(__aarch64__) && !defined(_M_ARM64)) || !defined(MNN_USE_NEON)
        MNN_PRINT("GeluCpuApproximationTest skipped: requires ARM64 NEON.\n");
        return true;
#else
        // ARM64 goldens are not a universal GELU contract for other backends.
        const auto& status = MNNTestSuite::get()->pStaus;
        const auto backend = status.forwardType;
        const bool openclBuffer = backend == MNN_FORWARD_OPENCL && (status.thread & MNN_GPU_MEMORY_BUFFER);
        if (precision != 1 || (backend != MNN_FORWARD_CPU && backend != MNN_FORWARD_OPENCL) || openclBuffer) {
            MNN_PRINT("GeluCpuApproximationTest skipped: requires CPU/OpenCL image FP32.\n");
            return true;
        }
        // Cover the observed speech range, both clamp boundaries, central values,
        // signed zero and tails. Goldens were captured from ARM64 MNNGelu, rather
        // than computed with the OpenCL formula under test.
        const float input[] = {
            -12.f, -8.f, -5.006524f, -4.914062f, -4.f, -3.81f, -3.802f, -3.8f,
            -3.79f, -3.f, -2.5f, -2.f, -1.1126f, -.9805f, -.467529f, -.1101f,
            -0.f, 0.f, .1681f, .315186f, .5264f, 1.5541f, 2.f, 2.544455f,
            3.f, 3.79f, 3.8f, 3.802f, 3.81f, 4.f, 8.f, 12.f,
        };
        const float expected[] = {
            -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f,
            -3.61442562e-6f, -.0036329627f, -.0150839984f, -.0454022884f,
            -.148099363f, -.160378844f, -.149651125f, -.0502238162f,
            -0.f, 0.f, .0952700302f, .196575254f, .368825048f, 1.46052754f,
            1.95459771f, 2.5309844f, 2.99636698f, 3.78999639f, 3.79999995f,
            3.80200005f, 3.80999994f, 4.f, 8.f, 12.f,
        };
        auto x = _Input({1, 4, 8, 1}, NCHW, halide_type_of<float>());
        auto data = x->writeMap<float>();
        if (!data) {
            return false;
        }
        std::memcpy(data, input, sizeof(input));
        auto y = _Gelu(x);
        auto output = y->readMap<float>();
        if (!output) {
            return false;
        }
        for (int i = 0; i < 32; ++i) {
            // Absolute floor catches the negative-tail regression; a small
            // relative term allows FP32 division differences between devices.
            const float tolerance = 5e-7f * (1.f + std::fabs(expected[i]));
            if (!std::isfinite(output[i]) || std::fabs(output[i] - expected[i]) > tolerance) {
                MNN_ERROR("GeluCpuApproximationTest i=%d input=%.9g expected=%.9g actual=%.9g tolerance=%.9g\n",
                          i, input[i], expected[i], output[i], tolerance);
                return false;
            }
        }
        return true;
#endif
    }
};

MNNTestSuiteRegister(GeluCpuApproximationTest, "op/unary/gelu_cpu_approximation");
