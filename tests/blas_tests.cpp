// Copyright © 2023 Apple Inc.

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
