#include "MNN_generated.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL [name-substring]\n", argv[0]);
        return 2;
    }
    const std::string modelPath = argv[1];
    const std::string needle = argc >= 3 ? argv[2] : "conv2";

    std::ifstream in(modelPath, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "[mnn-op-inspect] unable to open %s\n", modelPath.c_str());
        return 3;
    }
    std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (buf.empty()) {
        std::fprintf(stderr, "[mnn-op-inspect] empty model\n");
        return 4;
    }

    const MNN::Net* net = MNN::GetNet(buf.data());
    if (!net || !net->oplists()) {
        std::fprintf(stderr, "[mnn-op-inspect] invalid model\n");
        return 5;
    }

    int found = 0;
    for (auto op : *net->oplists()) {
        if (!op || !op->name()) continue;
        const std::string name = op->name()->str();
        if (!needle.empty() && name.find(needle) == std::string::npos) continue;

        std::printf("[mnn-op-inspect] name=%s type=%s\n",
                    name.c_str(), MNN::EnumNameOpType(op->type()));
        if (op->main_type() == MNN::OpParameter_Convolution2D) {
            auto conv = op->main_as_Convolution2D();
            auto c = conv ? conv->common() : nullptr;
            if (!c) {
                std::printf("  convolution common=null\n");
            } else {
                std::printf("  inputCount=%d outputCount=%d group=%d\n",
                            c->inputCount(), c->outputCount(), c->group());
                std::printf("  kernel=%dx%d stride=%dx%d dilation=%dx%d pad=%dx%d\n",
                            c->kernelX(), c->kernelY(),
                            c->strideX(), c->strideY(),
                            c->dilateX(), c->dilateY(),
                            c->padX(), c->padY());
                std::printf("  padMode=%d relu=%d relu6=%d hasOutputShape=%d\n",
                            static_cast<int>(c->padMode()),
                            c->relu() ? 1 : 0,
                            c->relu6() ? 1 : 0,
                            c->hasOutputShape() ? 1 : 0);
                if (conv->weight()) {
                    std::printf("  weight_count=%u\n", conv->weight()->size());
                }
                if (conv->bias()) {
                    std::printf("  bias_count=%u\n", conv->bias()->size());
                }
            }
        }
        if (op->inputIndexes()) {
            std::printf("  inputs=");
            for (unsigned i = 0; i < op->inputIndexes()->size(); ++i) {
                std::printf("%s%d", i ? "," : "", op->inputIndexes()->Get(i));
            }
            std::printf("\n");
        }
        if (op->outputIndexes()) {
            std::printf("  outputs=");
            for (unsigned i = 0; i < op->outputIndexes()->size(); ++i) {
                std::printf("%s%d", i ? "," : "", op->outputIndexes()->Get(i));
            }
            std::printf("\n");
        }
        ++found;
    }

    std::printf("[mnn-op-inspect] matched=%d needle=%s\n", found, needle.c_str());
    return found ? 0 : 6;
}
