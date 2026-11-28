// Copyright © 2023-2026 Apple Inc.

#include <numeric>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test matmul") {
  auto a = array(1);
  auto b = array({1.0});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = array({1.0});
  b = array({1.0});
  auto out = matmul(a, b);
  CHECK_EQ(out.shape(), Shape{});
  CHECK_EQ(out.size(), 1);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.item<float>(), 1.0f);

  a = ones({2, 4});
  b = ones({2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  a = ones({2, 4}, int32);
  b = ones({4, 2}, float32);
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  // Check single dimensions
  a = ones({4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({2, 4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({}, 4.0f)).item<bool>());

  // Test transposed arrays
  a = array({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4});
  b = array({1.0f, 1.0f, 1.0f, 1.0f}, {4, 1});
  out = matmul(transpose(a), transpose(b));
  CHECK(array_equal(out, ones({4, 4})).item<bool>());

  a = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  b = array({1.0f, 2.0f, 1.0f, 2.0f}, {2, 2});
  out = matmul(transpose(a), b);
  CHECK(
      array_equal(out, array({4.0f, 8.0f, 6.0f, 12.0f}, {2, 2})).item<bool>());

  out = matmul(a, transpose(b));
  CHECK(
      array_equal(out, array({5.0f, 5.0f, 11.0f, 11.0f}, {2, 2})).item<bool>());

  out = matmul(transpose(a), transpose(b));
  CHECK(
      array_equal(out, array({7.0f, 7.0f, 10.0f, 10.0f}, {2, 2})).item<bool>());

  // Test broadcasting for both arrays
  a = ones({5, 4, 2});
  b = ones({2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 4, 3}, 2.0f)).item<bool>());

  a = ones({5, 1, 4, 2});
  b = ones({1, 7, 2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 7, 4, 3}, 2.0f)).item<bool>());

  // Test batched matmul with transpose
  a = ones({2, 2, 4});
  b = ones({2, 4, 2});
  out = matmul(transpose(a, {0, 2, 1}), transpose(b, {0, 2, 1}));
  CHECK(array_equal(out, full({2, 4, 4}, 2.0f)).item<bool>());
}

TEST_CASE("test matmul large transposed lhs") {
  constexpr int M = 32;
  constexpr int N = 32;
  constexpr int K = 64;

  auto check = [](Dtype dtype, float rtol, float atol) {
    auto a = astype(reshape(arange(K * M, float32) * 0.001f, {K, M}), dtype);
    auto b = astype(reshape(arange(K * N, float32) * 0.002f, {K, N}), dtype);
    auto lhs = transpose(a);
    auto out = matmul(lhs, b);
    auto expected = matmul(contiguous(lhs), b);
    CHECK(allclose(out, expected, rtol, atol).item<bool>());
  };

  check(complex64, 1e-5f, 1e-5f);
  check(float16, 1e-2f, 1e-2f);
  check(bfloat16, 2e-2f, 2e-2f);
}

TEST_CASE("test segmented_mm in bfloat16 and float16 on the cpu") {
  // Each segment must match the matmul of its slices. The reference runs in
  // float32, through cblas rather than simd_gemm or BNNS. |a·b| <= 6 and
  // K <= 39 keep every sum an integer of magnitude <= 234, exact in bfloat16
  // and float16, so the results can be compared with array_equal.
  StreamOrDevice cpu = Device::cpu;
  int M = 18, N = 17, K = 39;
  std::vector<std::pair<int, int>> segs = {{0, 16}, {16, 39}, {5, 13}};
  auto segments = array({0, 16, 16, 39, 5, 13}, Shape{3, 2});
  auto av =
      subtract(remainder(arange(M * K, cpu), array(7), cpu), array(3), cpu);
  auto bv =
      subtract(remainder(arange(K * N, cpu), array(5), cpu), array(2), cpu);
  auto reference = [&](const array& x, const array& y, Dtype dtype) {
    auto p = matmul(astype(x, float32, cpu), astype(y, float32, cpu), cpu);
    return astype(p, dtype, cpu);
  };
  for (auto dtype : {bfloat16, float16}) {
    for (bool a_t : {false, true}) {
      for (bool b_t : {false, true}) {
        CAPTURE(dtype);
        CAPTURE(a_t);
        CAPTURE(b_t);
        auto x = a_t
            ? transpose(astype(reshape(av, {K, M}, cpu), dtype, cpu), cpu)
            : astype(reshape(av, {M, K}, cpu), dtype, cpu);
        auto y = b_t
            ? transpose(astype(reshape(bv, {N, K}, cpu), dtype, cpu), cpu)
            : astype(reshape(bv, {K, N}, cpu), dtype, cpu);
        std::vector<array> parts;
        for (auto [k0, k1] : segs) {
          parts.push_back(reference(
              slice(x, {0, k0}, {M, k1}, cpu),
              slice(y, {k0, 0}, {k1, N}, cpu),
              dtype));
        }
        CHECK(array_equal(matmul(x, y, cpu), reference(x, y, dtype), cpu)
                  .item<bool>());
        CHECK(array_equal(
                  segmented_mm(x, y, segments, cpu), stack(parts, cpu), cpu)
                  .item<bool>());
      }
    }
  }
}

TEST_CASE("test segmented_mm float16 and bfloat16") {
  auto a_f32 = reshape(arange(0, 4 * 12, float32), {4, 12}) * 0.1f;
  auto b_f32 = reshape(arange(0, 12 * 6, float32), {12, 6}) * 0.1f;
  auto segments = array({0u, 3u, 3u, 8u, 8u, 12u}, {3, 2}, uint32);

  for (auto dt : {float16, bfloat16}) {
    auto a = astype(a_f32, dt);
    auto b = astype(b_f32, dt);
    auto a_ref = astype(a, float32);
    auto b_ref = astype(b, float32);

    // Non-transposed
    auto expected = astype(segmented_mm(a_ref, b_ref, segments), float32);
    auto actual = astype(segmented_mm(a, b, segments), float32);
    CHECK(allclose(actual, expected, 1e-2, 1e-2).item<bool>());

    // Transposed a
    auto a_t = transpose(
        astype(reshape(arange(0, 12 * 4, float32), {12, 4}) * 0.1f, dt));
    auto a_t_ref = astype(a_t, float32);
    expected = astype(segmented_mm(a_t_ref, b_ref, segments), float32);
    actual = astype(segmented_mm(a_t, b, segments), float32);
    CHECK(allclose(actual, expected, 1e-2, 1e-2).item<bool>());

    // Transposed b
    auto b_t = transpose(
        astype(reshape(arange(0, 6 * 12, float32), {6, 12}) * 0.1f, dt));
    auto b_t_ref = astype(b_t, float32);
    expected = astype(segmented_mm(a_ref, b_t_ref, segments), float32);
    actual = astype(segmented_mm(a, b_t, segments), float32);
    CHECK(allclose(actual, expected, 1e-2, 1e-2).item<bool>());

    // Both transposed
    expected = astype(segmented_mm(a_t_ref, b_t_ref, segments), float32);
    actual = astype(segmented_mm(a_t, b_t, segments), float32);
    CHECK(allclose(actual, expected, 1e-2, 1e-2).item<bool>());
  }
}
