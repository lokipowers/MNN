#include "MNN_generated.h"
#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace MNN;

static std::unique_ptr<OpT> cloneOp(const OpT* src) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto packed = Op::Pack(builder, src);
    builder.Finish(packed);
    return std::unique_ptr<OpT>(
        flatbuffers::GetRoot<Op>(builder.GetBufferPointer())->UnPack());
}

static bool writeNet(const NetT* net, const char* path) {
    flatbuffers::FlatBufferBuilder builder(1024);
    builder.ForceDefaults(true);
    auto packed = Net::Pack(builder, net);
    builder.Finish(packed);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
              static_cast<std::streamsize>(builder.GetSize()));
    return out.good();
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr,
            "usage: %s INPUT.mnn OUTPUT.mnn [op_name] [chunks]\n",
            argv[0]);
        return 2;
    }

    const char* inputPath = argv[1];
    const char* outputPath = argv[2];
    const std::string targetName =
        argc >= 4 ? argv[3] : "/conv2/Conv_output_0";
    const int chunks = argc >= 5 ? std::max(2, std::atoi(argv[4])) : 4;

    std::ifstream in(inputPath, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "[mnn-split-conv] unable to open %s\n", inputPath);
        return 3;
    }
    std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    if (buf.empty()) {
        std::fprintf(stderr, "[mnn-split-conv] empty input model\n");
        return 4;
    }

    const auto* root = GetNet(buf.data());
    if (!root) {
        std::fprintf(stderr, "[mnn-split-conv] invalid MNN model\n");
        return 5;
    }
    std::unique_ptr<NetT> net(root->UnPack());
    if (!net) {
        std::fprintf(stderr, "[mnn-split-conv] unable to unpack model\n");
        return 6;
    }

    int targetIndex = -1;
    for (int i = 0; i < static_cast<int>(net->oplists.size()); ++i) {
        const auto& op = net->oplists[i];
        if (op && op->name == targetName &&
            op->type == OpType_Convolution &&
            op->main.type == OpParameter_Convolution2D) {
            targetIndex = i;
            break;
        }
    }
    if (targetIndex < 0) {
        std::fprintf(stderr, "[mnn-split-conv] target convolution not found: %s\n",
                     targetName.c_str());
        return 7;
    }

    const OpT* original = net->oplists[targetIndex].get();
    const auto* conv = original->main.AsConvolution2D();
    if (!conv || !conv->common) {
        std::fprintf(stderr, "[mnn-split-conv] target convolution parameters missing\n");
        return 8;
    }
    if (original->inputIndexes.size() != 1 || original->outputIndexes.size() != 1) {
        std::fprintf(stderr,
            "[mnn-split-conv] expected one input/output, got %zu/%zu\n",
            original->inputIndexes.size(), original->outputIndexes.size());
        return 9;
    }

    const int ic = conv->common->inputCount;
    const int oc = conv->common->outputCount;
    const int kh = conv->common->kernelY;
    const int kw = conv->common->kernelX;
    if (ic <= 0 || oc <= 0 || kh <= 0 || kw <= 0 || ic % chunks != 0) {
        std::fprintf(stderr,
            "[mnn-split-conv] unsupported geometry ic=%d oc=%d kh=%d kw=%d chunks=%d\n",
            ic, oc, kh, kw, chunks);
        return 10;
    }

    const size_t expectedWeights =
        static_cast<size_t>(oc) * ic * kh * kw;
    if (conv->weight.size() != expectedWeights ||
        conv->bias.size() != static_cast<size_t>(oc)) {
        std::fprintf(stderr,
            "[mnn-split-conv] unexpected constants weights=%zu expected=%zu bias=%zu expected_bias=%d\n",
            conv->weight.size(), expectedWeights, conv->bias.size(), oc);
        return 11;
    }

    const int chunkIC = ic / chunks;
    const int originInputIndex = original->inputIndexes[0];
    const int originOutputIndex = original->outputIndexes[0];
    const std::string originName = original->name;

    auto newTensor = [&](const std::string& name) -> int {
        const int idx = static_cast<int>(net->tensorName.size());
        net->tensorName.emplace_back(name);
        return idx;
    };

    std::vector<int> splitInputs;
    splitInputs.reserve(chunks);
    for (int i = 0; i < chunks; ++i) {
        splitInputs.push_back(
            newTensor(originName + "__nova_input_chunk" + std::to_string(i)));
    }

    std::vector<int> convOutputs;
    convOutputs.reserve(chunks);
    for (int i = 0; i < chunks; ++i) {
        convOutputs.push_back(
            newTensor(originName + "__nova_conv_chunk" + std::to_string(i)));
    }

    std::vector<std::unique_ptr<OpT>> replacement;
    replacement.reserve(1 + chunks + std::max(0, chunks - 1));

    {
        std::unique_ptr<OpT> slice(new OpT);
        slice->type = OpType_Slice;
        slice->name = originName + "__nova_slice";
        slice->main.type = OpParameter_Slice;
        slice->main.value = new SliceT;
        slice->main.AsSlice()->axis = 1;
        slice->main.AsSlice()->sourceType = NetSource_TORCH;
        slice->inputIndexes = {originInputIndex};
        slice->outputIndexes = splitInputs;
        replacement.emplace_back(std::move(slice));
    }

    const size_t chunkWeightPerOC =
        static_cast<size_t>(chunkIC) * kh * kw;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        auto sub = cloneOp(original);
        if (!sub || !sub->main.AsConvolution2D() ||
            !sub->main.AsConvolution2D()->common) {
            std::fprintf(stderr, "[mnn-split-conv] failed to clone convolution\n");
            return 12;
        }

        auto* subConv = sub->main.AsConvolution2D();
        sub->name = originName + "__nova_chunk" + std::to_string(chunk);
        sub->inputIndexes = {splitInputs[chunk]};
        sub->outputIndexes = {convOutputs[chunk]};
        subConv->common->inputCount = chunkIC;

        std::vector<float> subWeight(
            static_cast<size_t>(oc) * chunkIC * kh * kw);
        for (int outChannel = 0; outChannel < oc; ++outChannel) {
            const size_t srcBase =
                (static_cast<size_t>(outChannel) * ic +
                 static_cast<size_t>(chunk) * chunkIC) * kh * kw;
            const size_t dstBase =
                static_cast<size_t>(outChannel) * chunkWeightPerOC;
            std::copy(conv->weight.begin() + srcBase,
                      conv->weight.begin() + srcBase + chunkWeightPerOC,
                      subWeight.begin() + dstBase);
        }
        subConv->weight = std::move(subWeight);
        if (chunk == 0) {
            subConv->bias = conv->bias;
        } else {
            subConv->bias.assign(oc, 0.0f);
        }

        replacement.emplace_back(std::move(sub));
    }

    int running = convOutputs[0];
    for (int chunk = 1; chunk < chunks; ++chunk) {
        std::unique_ptr<OpT> add(new OpT);
        add->type = OpType_BinaryOp;
        add->main.type = OpParameter_BinaryOp;
        add->main.value = new BinaryOpT;
        add->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
        add->inputIndexes = {running, convOutputs[chunk]};

        if (chunk == chunks - 1) {
            add->name = originName;
            add->outputIndexes = {originOutputIndex};
        } else {
            add->name =
                originName + "__nova_add" + std::to_string(chunk);
            const int addOut =
                newTensor(originName + "__nova_add_tensor" + std::to_string(chunk));
            add->outputIndexes = {addOut};
            running = addOut;
        }
        replacement.emplace_back(std::move(add));
    }

    auto pos = net->oplists.begin() + targetIndex;
    pos = net->oplists.erase(pos);
    net->oplists.insert(pos,
                        std::make_move_iterator(replacement.begin()),
                        std::make_move_iterator(replacement.end()));

    if (!writeNet(net.get(), outputPath)) {
        std::fprintf(stderr, "[mnn-split-conv] unable to write %s\n", outputPath);
        return 13;
    }

    std::printf(
        "[mnn-split-conv] patched op=%s chunks=%d ic=%d chunk_ic=%d oc=%d kernel=%dx%d\n",
        originName.c_str(), chunks, ic, chunkIC, oc, kh, kw);
    std::printf("[mnn-split-conv] wrote %s ops=%zu tensors=%zu\n",
                outputPath, net->oplists.size(), net->tensorName.size());
    return 0;
}
