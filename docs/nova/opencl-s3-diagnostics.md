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
