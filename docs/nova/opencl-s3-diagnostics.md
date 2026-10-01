# Nova S3 OpenCL event diagnostic

The first failed profiled event in the supplied log is Raster0, not conv2.
Conv selection occurs during resize and does not establish execution order.
The existing CommonExecution diagnostic was under the non-profiler branch,
so ENABLE_OPENCL_TIME_PROFILER bypassed its waits and error returns.
The conv selection log also interchanged its stride and kernel arguments.

The supplied c4h4w1 kernel bounds its final 250-row output tile (two remaining
rows). Its work-size calculation agrees with the forced selection: 320 x 63.
No kernel arithmetic or forced selection is changed by this diagnostic.
The tuner bypass still selects the last candidate for conv1; it is not a
measured performance choice. Other operators' tuners remain enabled.

Set NOVA_OPENCL_FAIL_FAST=1 to synchronously check CommonExecution units in
both profiling and non-profiling builds. It bypasses recorded-queue replay,
checks previously queued work before each operation, and logs actual kernel
name, op name, unit, and all work-size dimensions (including automatic LWS).
It returns INVALID_VALUE on enqueue, wait, status-query, or incomplete-event
failure and does not enqueue the remaining units. A before-op failure points
to previously queued work, potentially resize/tuning or another execution class;
it must not be attributed to the named operation. A clean preflight does not
prove that other execution classes have individually validated every event.

This is an opt-in localization tool, not a performance benchmark or a confirmed
GPU fix. Without the switch, ordinary asynchronous execution and profiling
remain enabled. Enqueue failures now propagate in both modes.

The tokenizer probe checks that indices are nonempty int32 before reading them.
It reports OUTPUT_MATERIALIZED rather than SUCCESS because it does not compare
against CPU or verify vocabulary bounds. Output length 250 is the number of
tokens, not a vocabulary-size bound on each token ID.

## Device validation

Apply only the diagnostic commit's patch to the uploaded UNO source snapshot,
using git apply --check first. Keep the existing build configuration (including
profiling), then build:

```bash
cd ~/nova-mnn/build
make -j2 novaS3TokenizerProbe.out novaConv2Smoke.out novaRasterSmoke.out
set -o pipefail
NOVA_OPENCL_FAIL_FAST=1 ./novaS3TokenizerProbe.out \
  /tmp/s3tokenizer-v2-len1000.mnn 3 1000 0 0 1 2>&1 \
  | tee /tmp/nova-opencl-failfast.log
```

This runs one inference with 1000 input frames. Do not substitute 100 frames:
the supplied model has a fixed attention-mask length and rejects that shape.
Run each comparison as a separate process so a failed OpenCL queue is not reused:

```bash
./novaS3TokenizerProbe.out /tmp/s3tokenizer-v2-len1000.mnn \
  0 1000 0 0 1 2>&1 | tee /tmp/nova-s3-cpu.log
NOVA_OPENCL_FAIL_FAST=1 ./novaRasterSmoke.out 3 0 2>&1 \
  | tee /tmp/nova-raster-failfast.log
NOVA_OPENCL_FAIL_FAST=1 ./novaConv2Smoke.out 3 0 1 2>&1 \
  | tee /tmp/nova-conv2-failfast.log
```

CPU and GPU tokenizer logs only provide first-ID summaries; they do not establish
full numerical parity. The synthetic conv smoke also reports summaries rather
than asserting parity. The raster smoke covers layout conversion, not necessarily
the exact failing graph region. These probes help localize the failure; they do
not replace full-model testing on the UNO's OpenCL driver.

Host checks: CommonExecution compiles with and without profiling using MNN's
OpenCL headers; ConvExecution and tokenizer probe pass syntax compilation.
Fault injection against the actual CommonExecution source checks prior-queue,
enqueue, wait, negative-status, status-query, incomplete-status failures, success,
record-replay bypass, automatic LWS, and 3D work sizes in both profiler modes.
No UNO hardware run has been performed by this change's author.

## UNO result and conv1 A/B

The first serial fail-fast hardware run passed all three initial Raster units
(image_to_nchw_buffer, raster_buffer, nchw_buffer_to_image) and then failed
/conv1/Conv_output_0, conv_2d_c8h4w1, GWS 160,125, LWS 1,1 with wait=-14,
status=-5. Conv2 was not executed. This localizes the first failing execution
in that run; it does not establish whether driver resource pressure, indexing,
or another kernel/driver issue is the underlying cause.

Set NOVA_S3_CONV1_KERNEL to conv_2d_c4h1w4, conv_2d_c4h4w1, or
conv_2d_c8h4w1 to pin ONLY the first exact S3 geometry (128 channels,
1000x1 input -> 1280 channels, 500x1 output, 3x1 kernel, stride 2x1,
dilation 1x1). LWS remains 1x1; conv2 and all other geometries are unchanged.
An invalid name is rejected for this geometry. With the variable unset,
the uploaded selection behavior is preserved.

First compare c4h1w4 against the already observed c8h4w1 failure:

```bash
cd ~/nova-mnn/build
set -o pipefail
make -j2 novaS3TokenizerProbe.out &&
NOVA_OPENCL_FAIL_FAST=1 NOVA_S3_CONV1_KERNEL=conv_2d_c4h1w4 \
  ./novaS3TokenizerProbe.out /tmp/s3tokenizer-v2-len1000.mnn \
  3 1000 0 0 1 2>&1 | tee /tmp/nova-conv1-c4h1w4.log
```

Expected conv1 selection is GWS 320,500 and LWS 1,1. A completed conv1 may
expose a later failure; this is not a promise of full-model correctness.

## FP32 result and binding diagnostic

UNO CPU full-model inference materialized 250 int32 indices in 13.061 seconds,
first ID 2648. This is a reference summary, not a complete token correctness check.
The synthetic conv2 probe fails with Raster units passing, independently of model
weights and attention. Both precision 0 (Normal) and 1 (High/FP32) fail at the same
convolutions. Precision selection alone did not resolve the observed failure.

With NOVA_OPENCL_FAIL_FAST=1, the exact S3 convolutions now log the effective
precision, device, kernel argument count, maximum image dimensions and work-group
size, device local-memory limit, and the actual bound image handles, dimensions,
formats, sizes and query return codes. Geometry/stride/padding parameters are
printed beside these bindings. Failed setArg now returns INVALID_VALUE rather
than allowing execution with incomplete bindings. Kernel math is unchanged.

Rebuild and run only the synthetic probe first:

```bash
cd ~/nova-mnn/build
set -o pipefail
make -j2 novaConv2Smoke.out &&
NOVA_OPENCL_FAIL_FAST=1 ./novaConv2Smoke.out 3 1 1 2>&1 \
  | tee /tmp/nova-conv-bindings.log
```

The binding diagnostic passes object compilation; all 12 OpenCL C entry points
referenced by ConvExecution are defined by the compiled dynamic wrapper. Device
validation remains pending.

## Work-group hint A/B

The FD702 binding run queried all four images successfully, bound all 16 kernel
arguments and reported kernel_max_wg=32. The output image is physically larger
than its logical shape, which can be legitimate pooled-image reuse; the reported
image dimensions alone do not prove invalid indexing or adequate memory lifetime.

The runtime emits -DSET_ATTRIBUTE=false when it intends to disable attributes,
but conv_2d kernels use #ifdef SET_ATTRIBUTE, so the 16x16 work-group hint remains
present. A hint is not a required work-group size, so this discrepancy alone does
not prove the cause of CL_OUT_OF_RESOURCES.

NOVA_OPENCL_NO_WORKGROUP_HINT=1 omits this definition entirely in both cached and
source kernel-build paths. Existing default build options are preserved. The
changed build-option string uses a distinct in-memory program-cache key. Test
in a fresh process, keeping precision, kernel choice and LWS the same:

```bash
cd ~/nova-mnn/build
set -o pipefail
make -j2 novaConv2Smoke.out &&
NOVA_OPENCL_FAIL_FAST=1 NOVA_OPENCL_NO_WORKGROUP_HINT=1 \
  ./novaConv2Smoke.out 3 1 1 2>&1 | tee /tmp/nova-conv-no-hint.log
```

Host checks compile OpenCLRuntime with the dynamic wrapper and preprocess the
actual conv_2d.cl under default versus omitted definitions, confirming the hint
is present with SET_ATTRIBUTE=false and absent with the diagnostic switch.
Hardware validation remains pending.

## Unsplit conv2 with driver-selected local work size

`NOVA_S3_CONV2_C4H1W4_AUTO=1` pins only the exact 1280x500x1 to
1280x250x1 stride-2 conv2 to `conv_2d_c4h1w4` and uses null local
work size (logged as `LWS=auto` during execution). It bypasses tuning
submissions. The default forced c4h4w1 / LWS 1x1 path is unchanged.
This compares the unsplit calculation against successful 2/4/8-chunk
smoke runs; those runs matched all printed CPU summaries and samples,
but do not constitute an element-by-element comparison.

```sh
NOVA_OPENCL_FAIL_FAST=1 NOVA_OPENCL_NO_WORKGROUP_HINT=1 \
NOVA_S3_CONV2_C4H1W4_AUTO=1 ./novaConv2Smoke.out 3 1 1
```
