// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

TEST_CASE("test empty gather output constructs without dividing by zero") {
  for (auto device : {Device::cpu, Device::gpu}) {
    if (!is_available(device)) {
      continue;
    }
    auto stream = new_stream(device);
    auto output = gather_qmm(
        ones({16, 1, 512}, bfloat16, stream),
        zeros({4, 0, 64}, uint32, stream),
        zeros({4, 0, 8}, float16, stream),
        zeros({4, 0, 8}, float16, stream),
        std::nullopt,
        zeros({16}, uint32, stream),
        true,
        64,
        4,
        "affine",
        std::nullopt,
        true,
        stream);
    CHECK(output.shape() == Shape{16, 1, 0});
    CHECK(output.dtype() == float32);
    CHECK(output.primitive().stream() == stream);
  }
}

TEST_CASE("test mixed affine gather storage follows the selected consumer") {
  for (auto device : {Device::cpu, Device::gpu}) {
    if (!is_available(device)) {
      continue;
    }
    auto stream = new_stream(device);
    for (int bits : {4, 8}) {
      for (int experts : {2, 4}) {
        for (int rows : {1, 15, 16}) {
          for (bool sorted : {false, true}) {
            CAPTURE(device);
            CAPTURE(bits);
            CAPTURE(experts);
            CAPTURE(rows);
            CAPTURE(sorted);
            auto x = ones({rows, 1, 512}, bfloat16, stream);
            auto w = zeros({experts, 8, 512 * bits / 32}, uint32, stream);
            auto scales = zeros({experts, 8, 8}, float16, stream);
            // Packed zeros plus exact biases expose half/BF16 reinterpretation
            // independently of matmul reduction order.
            auto bias = broadcast_to(
                reshape(
                    multiply(
                        add(arange(experts, float32, stream), array(1.0f)),
                        array(0.03125f),
                        stream),
                    {experts, 1, 1},
                    stream),
                {experts, 8, 8},
                stream);
            bias = astype(bias, float16, stream);
            auto ids = floor_divide(
                multiply(arange(rows, uint32, stream), array(experts, uint32)),
                array(rows, uint32),
                stream);
            auto output = gather_qmm(
                x,
                w,
                scales,
                bias,
                std::nullopt,
                ids,
                true,
                64,
                bits,
                "affine",
                std::nullopt,
                sorted,
                stream);
            const bool matrix = sorted && rows >= 16 && rows / experts >= 4;
            const bool mixed_vector =
                device == Device::gpu && metal::is_available() && !matrix;
            CHECK(output.dtype() == (mixed_vector ? bfloat16 : float32));
            CHECK(output.primitive().stream() == stream);
            auto expected = broadcast_to(
                reshape(
                    multiply(
                        add(astype(ids, float32, stream), array(1.0f)),
                        array(16.0f),
                        stream),
                    {rows, 1, 1},
                    stream),
                {rows, 1, 8},
                stream);
            CHECK(array_equal(
                      astype(output, float32, stream), expected, false, stream)
                      .item<bool>());
            synchronize(stream);
          }
        }
      }
    }
  }
}

TEST_CASE("test explicit gather lhs indices preserve mixed vector dispatch") {
  if (!is_available(Device::gpu) || !metal::is_available()) {
    return;
  }
  auto stream = new_stream(Device::gpu);
  auto x = ones({16, 1, 512}, bfloat16, stream);
  auto w = zeros({2, 8, 64}, uint32, stream);
  auto scales = zeros({2, 8, 8}, float16, stream);
  auto biases = full({2, 8, 8}, 0.03125f, float16, stream);
  auto lhs = arange(16, uint32, stream);
  auto rhs = zeros({16}, uint32, stream);
  auto output = gather_qmm(
      x,
      w,
      scales,
      biases,
      lhs,
      rhs,
      true,
      64,
      4,
      "affine",
      std::nullopt,
      true,
      stream);
  CHECK(output.dtype() == bfloat16);
  CHECK(array_equal(
            output, full({16, 1, 8}, 16.0f, bfloat16, stream), false, stream)
            .item<bool>());
  synchronize(stream);
}

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

TEST_CASE("test qqmm on the cpu applies the nvfp4 global scales") {
  auto s = Device::cpu;
  // Values on the e2m1 grid. Any 16 consecutive i give all 15 indices, so
  // each group holds 6 and -6, and all the steps below are exact.
  auto grid = array(
      {0.0f,
       0.5f,
       1.0f,
       1.5f,
       2.0f,
       3.0f,
       4.0f,
       6.0f,
       -0.5f,
       -1.0f,
       -1.5f,
       -2.0f,
       -3.0f,
       -4.0f,
       -6.0f});
  auto values = [&](int size, int offset) {
    auto i =
        add(multiply(arange(size, int32, s), array(7), s), array(offset), s);
    return take(grid, remainder(i, array(15), s), s);
  };
  auto v_x = reshape(values(32, 1), {1, 32}, s);
  auto w = reshape(values(4 * 32, 4), {4, 32}, s);

  struct Case {
    std::string name;
    array x;
    std::optional<float> g_x;
    std::optional<float> g_w;
  };
  // A: g_w = 336 makes the w block scales 8. B: x = v * 2^-12 has block
  // scales that round to 0 without g_x. C: no global scales. D: float16 x;
  // g_w = 10.5 makes the w block scales 256, too large for a float16 sum.
  auto x_d = astype(slice(w, {0, 0}, {1, 32}, s), float16, s);
  for (const auto& c :
       {Case{"A", v_x, 2688.0f, 336.0f},
        Case{"B", multiply(v_x, array(0x1p-12f), s), 0.65625f, 2688.0f},
        Case{"C", v_x, std::nullopt, std::nullopt},
        Case{"D", x_d, 2688.0f, 10.5f}}) {
    CAPTURE(c.name);
    const auto& x = c.x;
    std::optional<array> g_x;
    std::optional<array> g_w;
    if (c.g_x) {
      g_x = array(*c.g_x);
      g_w = array(*c.g_w);
    }
    auto w_q = quantize(w, 16, 4, "nvfp4", g_w, s);
    auto x_q = quantize(x, 16, 4, "nvfp4", g_x, s);
    // The reference uses the op-level dequantize, not the qqmm kernels.
    auto w_hat = dequantize(
        w_q[0], w_q[1], std::nullopt, 16, 4, "nvfp4", g_w, float32, s);
    auto x_hat = dequantize(
        x_q[0], x_q[1], std::nullopt, 16, 4, "nvfp4", g_x, float32, s);
    CHECK(array_equal(w_hat, w, false, s).item<bool>());
    CHECK(array_equal(x_hat, x, false, s).item<bool>());
    auto expected = matmul(x_hat, transpose(w_hat, s), s);
    auto y = qqmm(x, w_q[0], w_q[1], 16, 4, "nvfp4", g_x, g_w, s);
    CHECK(array_equal(y, expected, false, s).item<bool>());
  }
}
