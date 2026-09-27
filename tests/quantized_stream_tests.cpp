// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

TEST_CASE("test affine quantized matmul casts preserve explicit stream") {
  for (auto device : {Device::cpu, Device::gpu}) {
    if (!is_available(device)) {
      continue;
    }
    auto stream = new_stream(device);
    REQUIRE_FALSE(stream == default_stream(device));
    for (int bits : {2, 3, 4, 5, 6, 8}) {
      for (int group_size : {32, 64}) {
        for (int rows : {1, 3}) {
          CAPTURE(device);
          CAPTURE(bits);
          CAPTURE(group_size);
          CAPTURE(rows);
          // Seven output rows exclude the mixed BF16/F16 QMV fast path.
          // Both activations and affine metadata must promote to F32 here.
          auto weights = reshape(
              sin(arange(7 * 512, float32, stream), stream), {7, 512}, stream);
          auto q = quantize(
              astype(weights, float16, stream),
              group_size,
              bits,
              "affine",
              std::nullopt,
              stream);
          auto x = astype(
              reshape(
                  cos(arange(rows * 512, float32, stream), stream),
                  {rows, 512},
                  stream),
              bfloat16,
              stream);
          eval(x, q[0], q[1], q[2]);
          auto y = quantized_matmul(
              x, q[0], q[1], q[2], true, group_size, bits, "affine", stream);
          REQUIRE(y.has_primitive());
          CHECK(y.primitive().stream() == stream);
          CHECK(y.dtype() == float32);
          REQUIRE(y.inputs().size() == 4);
          for (int input : {0, 2, 3}) {
            REQUIRE(y.inputs()[input].has_primitive());
            CHECK(y.inputs()[input].dtype() == float32);
            CHECK(y.inputs()[input].primitive().stream() == stream);
          }
          // Scheduling must not change arithmetic: compare against explicitly
          // promoted inputs on the requested stream, with exact equality.
          auto reference = quantized_matmul(
              astype(x, float32, stream),
              q[0],
              astype(q[1], float32, stream),
              astype(q[2], float32, stream),
              true,
              group_size,
              bits,
              "affine",
              stream);
          CHECK(array_equal(y, reference, false, stream).item<bool>());
          synchronize(stream);
        }
      }
    }
  }
}

TEST_CASE("test mixed bf16 and f16 affine inputs on a cpu stream") {
  // Only the Metal kernels read the mixed path's unconverted f16 scales and
  // biases. A CPU stream must promote them, as the general path does.
  auto stream = new_stream(Device::cpu);
  for (int bits : {4, 8}) {
    CAPTURE(bits);
    // Eight output rows, 512 inputs, group 64 and one activation row: every
    // shape condition of the mixed path holds.
    auto weights = reshape(
        sin(arange(8 * 512, float32, stream), stream), {8, 512}, stream);
    auto q = quantize(
        astype(weights, float16, stream),
        64,
        bits,
        "affine",
        std::nullopt,
        stream);
    auto x = astype(
        reshape(cos(arange(512, float32, stream), stream), {1, 512}, stream),
        bfloat16,
        stream);
    auto x32 = astype(x, float32, stream);
    auto s32 = astype(q[1], float32, stream);
    auto b32 = astype(q[2], float32, stream);

    auto y =
        quantized_matmul(x, q[0], q[1], q[2], true, 64, bits, "affine", stream);
    auto reference =
        quantized_matmul(x32, q[0], s32, b32, true, 64, bits, "affine", stream);
    CHECK(y.dtype() == float32);
    CHECK(array_equal(y, reference, false, stream).item<bool>());

    // gather_qmm without indices is quantized_matmul. One expert, selected by
    // an index, reaches gather_qmm's own copy of the condition.
    auto expert = array({0}, uint32);
    auto gathered = gather_qmm(
        x,
        expand_dims(q[0], 0, stream),
        expand_dims(q[1], 0, stream),
        expand_dims(q[2], 0, stream),
        std::nullopt,
        expert,
        true,
        64,
        bits,
        "affine",
        false,
        stream);
    auto gathered_reference = gather_qmm(
        x32,
        expand_dims(q[0], 0, stream),
        expand_dims(s32, 0, stream),
        expand_dims(b32, 0, stream),
        std::nullopt,
        expert,
        true,
        64,
        bits,
        "affine",
        false,
        stream);
    CHECK(gathered.dtype() == float32);
    CHECK(
        array_equal(gathered, gathered_reference, false, stream).item<bool>());
  }
}
