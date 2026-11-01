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

TEST_CASE("test quantized matmul on the cpu with no rows or no K") {
  auto s = Device::cpu;
  int N = 64;
  for (std::string mode : {"affine", "mxfp4"}) {
    bool affine = mode == "affine";
    int group_size = affine ? 64 : 32;
    for (bool transpose : {true, false}) {
      // M = 0 gives an empty output. K = 0 gives an empty sum, which is 0.
      for (int M : {0, 2}) {
        int K = M == 0 ? 64 : 0;
        CAPTURE(mode);
        CAPTURE(transpose);
        CAPTURE(M);
        CAPTURE(K);
        int rows = transpose ? N : K;
        int cols = transpose ? K : N;
        auto w = zeros({rows, cols * 4 / 32}, uint32, s);
        auto scales =
            zeros({rows, cols / group_size}, affine ? float32 : uint8, s);
        std::optional<array> biases;
        if (affine) {
          biases = zeros({rows, cols / group_size}, float32, s);
        }
        auto x = ones({M, K}, float32, s);
        if (K == 0) {
          // Leave a freed buffer of ones of the output's size as the only one
          // in the cache, so an output that is not written is not 0.
          clear_cache();
          auto dirty = full({M, N}, 1.0f, float32, s);
          CHECK_EQ(min(dirty, s).item<float>(), 1.0f);
        }
        auto y = quantized_matmul(
            x, w, scales, biases, transpose, group_size, 4, mode, s);
        CHECK(y.shape() == Shape{M, N});
        // Compare with a scalar, so that no other array takes the buffer.
        CHECK(all(equal(y, array(0.0f), s), s).item<bool>());
      }
    }
  }
}

TEST_CASE("test mixed bf16 and f16 quantized matmul with K = 0") {
  // The mixed path counts the rows of x, which must not divide by K.
  auto s = Device::cpu;
  auto x = ones({1, 0}, bfloat16, s);
  auto w = zeros({64, 0}, uint32, s);
  auto scales = zeros({64, 0}, float16, s);
  auto biases = zeros({64, 0}, float16, s);
  auto y = quantized_matmul(x, w, scales, biases, true, 64, 4, "affine", s);
  CHECK(y.shape() == Shape{1, 64});
  CHECK_EQ(y.dtype(), float32);
  CHECK(array_equal(y, zeros({1, 64}, float32, s), s).item<bool>());
}

TEST_CASE("test qqmm on the cpu with K = 0 or N = 0") {
  auto s = Device::cpu;
  // K = 0 gives an empty sum, which is 0.
  auto y = qqmm(
      ones({1, 0}, float32, s),
      zeros({64, 0}, uint32, s),
      zeros({64, 0}, uint8, s),
      32,
      4,
      "mxfp4",
      std::nullopt,
      std::nullopt,
      s);
  CHECK(y.shape() == Shape{1, 64});
  CHECK(array_equal(y, zeros({1, 64}, float32, s), s).item<bool>());

  // N = 0 gives an empty output.
  y = qqmm(
      ones({1, 32}, float32, s),
      zeros({0, 4}, uint32, s),
      zeros({0, 1}, uint8, s),
      32,
      4,
      "mxfp4",
      std::nullopt,
      std::nullopt,
      s);
  CHECK(y.shape() == Shape{1, 0});
  CHECK(array_equal(y, zeros({1, 0}, float32, s), s).item<bool>());
}

TEST_CASE("test gather_qmm on the cpu with no rows or no N or no K") {
  auto s = Device::cpu;
  for (std::string mode : {"affine", "mxfp4"}) {
    bool affine = mode == "affine";
    int group_size = affine ? 64 : 32;
    // M = 0 or N = 0 gives an empty output. K = 0 gives zeros.
    for (auto mnk :
         {std::array{0, 64, 64}, std::array{1, 0, 64}, std::array{1, 64, 0}}) {
      int M = mnk[0];
      int N = mnk[1];
      int K = mnk[2];
      CAPTURE(mode);
      CAPTURE(M);
      CAPTURE(N);
      CAPTURE(K);
      auto w = zeros({2, N, K * 4 / 32}, uint32, s);
      auto scales = zeros({2, N, K / group_size}, affine ? float32 : uint8, s);
      std::optional<array> biases;
      if (affine) {
        biases = zeros({2, N, K / group_size}, float32, s);
      }
      auto y = gather_qmm(
          ones({2, M, K}, float32, s),
          w,
          scales,
          biases,
          std::nullopt,
          array({1u, 0u}),
          true,
          group_size,
          4,
          mode,
          std::nullopt,
          false,
          s);
      CHECK(y.shape() == Shape{2, M, N});
      CHECK(array_equal(y, zeros({2, M, N}, float32, s), s).item<bool>());
    }
  }
}

TEST_CASE("test quantized matmul with a 2-D x and an empty batch of w") {
  // The output has the shape of x, from one matrix of w. Here w has none.
  auto s = Device::cpu;
  for (std::string mode : {"affine", "mxfp4"}) {
    bool affine = mode == "affine";
    int group_size = affine ? 64 : 32;
    CAPTURE(mode);
    auto w = zeros({0, 64, 8}, uint32, s);
    auto scales = zeros({0, 64, 64 / group_size}, affine ? float32 : uint8, s);
    std::optional<array> biases;
    if (affine) {
      biases = zeros({0, 64, 1}, float32, s);
    }
    auto x = ones({2, 64}, float32, s);
    CHECK_THROWS_AS(
        sum(quantized_matmul(
                x, w, scales, biases, true, group_size, 4, mode, s),
            s)
            .item<float>(),
        std::invalid_argument);
  }
}
