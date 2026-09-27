// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

// Every |got - reference| is at most c * eps32 * sum_k |a_ik b_kj|. The
// reference and the magnitude are computed in float64 by MLX's own ops on the
// CPU. That makes an fp32 kernel's rounding bound rigorous for any summation
// order with c = K + 2, while an index or precision bug exceeds it.
bool within_sum_bound(
    const array& got,
    const array& a,
    const array& b,
    double c) {
  auto a64 = astype(a, float64);
  auto b64 = astype(b, float64);
  auto reference = matmul(a64, b64);
  auto magnitude = matmul(abs(a64), abs(b64));
  auto bound = multiply(
      array(c * std::numeric_limits<float>::epsilon(), float64), magnitude);
  return all(less_equal(abs(subtract(astype(got, float64), reference)), bound))
      .item<bool>();
}

// max |got - reference| / max |reference|, both widened to float32.
float relative_error(const array& got, const array& reference) {
  auto g = astype(got, float32);
  auto r = astype(reference, float32);
  return (max(abs(subtract(g, r))) / max(abs(r))).item<float>();
}

} // namespace

namespace {

array sdpa_reference(
    const array& q,
    const array& k,
    const array& v,
    float scale) {
  auto scores =
      multiply(matmul(q, transpose(k, {0, 1, 3, 2})), array(scale, q.dtype()));
  return matmul(softmax(scores, -1, /* precise = */ true), v);
}

} // namespace

TEST_CASE("highway sdpa declines shapes and types its kernel cannot hold") {
  // (query/key head dim, value head dim, dtype): unequal head dims, one above
  // the kernel's 256, and float64, which the kernel does not implement.
  struct Case {
    int dqk, dv;
    Dtype dtype;
  };
  for (auto c : std::vector<Case>{
           {192, 128, float32},
           {128, 192, float32},
           {320, 320, float32},
           {64, 64, float64}}) {
    CAPTURE(c.dqk);
    CAPTURE(c.dv);
    auto q = astype(
        random::normal({1, 4, 8, c.dqk}, float32, 0.0f, 1.0f, random::key(1)),
        c.dtype);
    auto k = astype(
        random::normal({1, 4, 16, c.dqk}, float32, 0.0f, 1.0f, random::key(2)),
        c.dtype);
    auto v = astype(
        random::normal({1, 4, 16, c.dv}, float32, 0.0f, 1.0f, random::key(3)),
        c.dtype);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.dqk));
    auto y = fast::scaled_dot_product_attention(q, k, v, scale);
    CHECK(relative_error(y, sdpa_reference(q, k, v, scale)) < 1e-5f);
  }
}

TEST_CASE("highway float64 elementwise math computes in double") {
  // Values whose float and double results differ in the double's low bits.
  std::vector<double> a = {
      1.1,
      2.2,
      3.3,
      10.5,
      123.456,
      0.01,
      7.77,
      1e5,
      2.5,
      9.75,
      0.3,
      42.0,
      5.5,
      8.25,
      1.9,
      6.6,
      3.7};
  std::vector<double> b = {
      0.7,
      1.3,
      -2.9,
      3.1,
      -0.45,
      2.2,
      1.7,
      -3.3,
      0.9,
      1.1,
      -1.7,
      2.6,
      0.35,
      -4.1,
      1.05,
      2.9,
      -0.8};
  const int n = static_cast<int>(a.size());
  auto x = array(a.data(), {n}, float64);
  auto y = array(b.data(), {n}, float64);
  auto logs = log(x);
  auto powers = power(x, y);
  auto angles = arctan2(x, y);
  auto remainders = remainder(x, y);
  for (int i = 0; i < n; ++i) {
    CAPTURE(a[i]);
    CAPTURE(b[i]);
    CHECK(slice(logs, {i}, {i + 1}).item<double>() == std::log(a[i]));
    CHECK(slice(powers, {i}, {i + 1}).item<double>() == std::pow(a[i], b[i]));
    CHECK(slice(angles, {i}, {i + 1}).item<double>() == std::atan2(a[i], b[i]));
    // base_simd's rule: std::remainder's result, moved into b's sign.
    double r = std::remainder(a[i], b[i]);
    if (r != 0 && (r < 0) != (b[i] < 0)) {
      r += b[i];
    }
    CHECK(slice(remainders, {i}, {i + 1}).item<double>() == r);
  }
}

TEST_CASE("highway integer power with a negative exponent") {
  // base_simd.h's rule (hwy/base, lines 249-267): a signed integer to a
  // negative power is 0, for every base, 1 included. #3019's facade looped
  // forever here.
  std::vector<int> bases = {
      2, 3, -2, 5, 7, 1, 4, 6, 2, -3, 1, 0, 9, 2, -1, 3, 5};
  std::vector<int> exponents = {
      -1, -2, -3, 2, 0, -1, 3, -5, 10, 3, 7, 0, 2, -8, 5, 4, 1};
  std::vector<int> expected = {
      0, 0, 0, 25, 1, 0, 64, 0, 1024, -27, 1, 1, 81, 0, -1, 81, 5};
  const int n = static_cast<int>(bases.size());
  auto y = power(
      array(bases.begin(), {n}, int32), array(exponents.begin(), {n}, int32));
  for (int i = 0; i < n; ++i) {
    CAPTURE(bases[i]);
    CAPTURE(exponents[i]);
    CHECK(slice(y, {i}, {i + 1}).item<int>() == expected[i]);
  }
}

TEST_CASE("highway bf16 sum does not depend on the thread count") {
  // 2^20 + 3 ones, then -2^20: the sum is exactly 3. Every thread's partial is
  // exact in float; rounded to bf16 before they are combined, the total is
  // never 3 for any pool of 2 to 64 threads (0 with 2, 4, 8, 16 or 32, 4096
  // with 6, -8192 with 12). A plain count of ones is no test: at many pool
  // sizes each partial happens to be exact in bf16. item<float> on a bf16 array
  // would read four bytes of a two-byte buffer, so widen first.
  auto x = concatenate(
      {ones({(1 << 20) + 3}, bfloat16), full({1}, -1048576.0f, bfloat16)});
  CHECK(astype(sum(x), float32).item<float>() == 3.0f);
}

TEST_CASE("highway facade fma rounds once on every target") {
  // MLX's erfinv computes t = fma(a, -a, 1). Where Highway's MulAdd rounds a *
  // a first, near |a| = 1 the result keeps few of its bits: 64 units in the
  // last place at 0.99983, where one rounding (the scalar code's std::fma)
  // keeps the worst over this range to about 2.1.
  const int n = 1001;
  std::vector<float> xs(n);
  for (int i = 0; i < n; ++i) {
    xs[i] = 0.999f + 0.000000999f * static_cast<float>(i); // 0.999 to 0.999999
  }
  auto y = erfinv(array(xs.data(), {n}, float32));
  double worst = 0;
  for (int i = 0; i < n; ++i) {
    // erf^-1 in double, by bisection on std::erf.
    double low = 0, high = 6;
    for (int k = 0; k < 80; ++k) {
      const double mid = (low + high) / 2;
      (std::erf(mid) < xs[i] ? low : high) = mid;
    }
    const double truth = (low + high) / 2;
    const float t32 = static_cast<float>(truth);
    const double unit = std::nextafter(t32, INFINITY) - t32;
    const double got = slice(y, {i}, {i + 1}).item<float>();
    REQUIRE(std::isfinite(got));
    worst = std::max(worst, std::abs(got - truth) / unit);
  }
  CAPTURE(worst);
  CHECK(worst <= 8.0);
}

TEST_CASE("highway float16 keeps inf and nan") {
  // Highway emulates float16 -> float32 where a target has no conversion
  // instruction, and reads exponent 31 as finite there: inf became 65536.
  const uint16_t pattern[4] = {
      0x7C00, 0xFC00, 0x7E00, 0x3C00}; // inf -inf nan 1
  std::vector<uint16_t> bits(64);
  for (int i = 0; i < 64; ++i) {
    bits[i] = pattern[i % 4];
  }
  auto x = view(array(bits.data(), {64}, uint16), float16);
  auto half_at = [](const array& a, int i) {
    return static_cast<float>(slice(a, {i}, {i + 1}).item<float16_t>());
  };
  auto bool_at = [](const array& a, int i) {
    return slice(a, {i}, {i + 1}).item<bool>();
  };
  auto difference = subtract(x, x); // NaN for inf, -inf and NaN
  auto sine = sin(x); // NaN for inf, -inf and NaN
  auto nan = isnan(x);
  auto inf = isinf(x);
  // The other direction, float32 -> float16: an overflow, and a NaN.
  auto overflow = multiply(full({64}, 65504.0f, float16), array(2.0f, float16));
  auto root = sqrt(full({64}, -1.0f, float16));
  for (int i = 0; i < 64; ++i) {
    CAPTURE(i);
    const bool special = i % 4 != 3;
    CHECK(std::isnan(half_at(difference, i)) == special);
    CHECK(std::isnan(half_at(sine, i)) == special);
    CHECK(bool_at(nan, i) == (i % 4 == 2));
    CHECK(bool_at(inf, i) == (i % 4 < 2));
    CHECK(half_at(overflow, i) == INFINITY);
    CHECK(std::isnan(half_at(root, i)));
  }
}
