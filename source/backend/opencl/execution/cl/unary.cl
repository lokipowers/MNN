#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
inline float4 gelu(float4 in){
    // Match the FP32 ARM64 MNNGelu operation order and clamped rational tanh.
    #pragma OPENCL FP_CONTRACT OFF
    float4 value = in * in;
    value = value * in;
    value = value * 0.044715f;
    value = value + in;
    value = value * 0.79788458f;
    value = clamp(value, (float4)(-5.0f), (float4)(5.0f));
    float4 x2 = value * value;
    float4 a = x2 + 378.0f;
    a = a * x2; a = a + 17325.0f;
    a = a * x2; a = a + 135135.0f;
    a = a * value;
    float4 b = x2 * 28.0f;
    b = b + 3150.0f; b = b * x2;
    b = b + 62370.0f; b = b * x2;
    b = b + 135135.0f;
    float4 approx = clamp(a / b, (float4)(-1.0f), (float4)(1.0f));
    return ((approx + 1.0f) * in) * 0.5f;
}
// MNN: erfinv via TensorFlow's two-branch polynomial (kept in lockstep with
// CPU's UnaryErfinv in source/backend/cpu/UnaryUtils.hpp). Avoids the OpenCL
// CPU-fallback path which is broken when the runtime is in IMAGE memtype.
inline float4 erfinv4(float4 x){
    float4 w = -log((float4)1.0f - x * x + (float4)1e-12f);
    int4 lt = isless(w, (float4)5.0f);
    float4 w_sel = select(sqrt(w) - (float4)3.0f, w - (float4)2.5f, lt);
    float4 p_lt = (float4)2.81022636e-08f;
    p_lt = (float4)3.43273939e-07f + p_lt * w_sel;
    p_lt = (float4)(-3.5233877e-06f) + p_lt * w_sel;
    p_lt = (float4)(-4.39150654e-06f) + p_lt * w_sel;
    p_lt = (float4)0.00021858087f + p_lt * w_sel;
    p_lt = (float4)(-0.00125372503f) + p_lt * w_sel;
    p_lt = (float4)(-0.00417768164f) + p_lt * w_sel;
    p_lt = (float4)0.246640727f + p_lt * w_sel;
    p_lt = (float4)1.50140941f + p_lt * w_sel;
    float4 p_ge = (float4)(-0.000200214257f);
    p_ge = (float4)0.000100950558f + p_ge * w_sel;
    p_ge = (float4)0.00134934322f + p_ge * w_sel;
    p_ge = (float4)(-0.00367342844f) + p_ge * w_sel;
    p_ge = (float4)0.00573950773f + p_ge * w_sel;
    p_ge = (float4)(-0.0076224613f) + p_ge * w_sel;
    p_ge = (float4)0.00943887047f + p_ge * w_sel;
    p_ge = (float4)1.00167406f + p_ge * w_sel;
    p_ge = (float4)2.83297682f + p_ge * w_sel;
    return select(p_ge, p_lt, lt) * x;
}

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void unary(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int width = global_size_dim1;

    const int pos  = mad24(channel_block_idx, width, w);
    float4 in  = convert_float4(RI_DATA(input, SAMPLER, (int2)(pos, hb)));
    OUTPUT_TYPE_I4 out = CONVERT_OUTPUT_I4(OPERATOR);
    
    WI_DATA(output, (int2)(pos, hb), out);
}
