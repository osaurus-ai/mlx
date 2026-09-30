// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <stdexcept>
#include <string>

#include "doctest/doctest.h"

#include "mlx/backend/cpu/highway_info.h"
#include "mlx/backend/cpu/scratch.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

// |got - a b| <= c * eps32 * sum_k |a_ik b_kj|, both computed in float64 by
// MLX's ops: rigorous for fp32 at c = K + 2, in any summation order.
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
  // base_simd.h's rule: a signed integer to a negative power is 0, for every
  // base, 1 included. #3019's facade looped forever here.
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
  // 2^20 + 3 ones, then -2^20: exactly 3. Partials rounded to bf16 gave 0 with
  // 8 threads, and -8192 with 9 or 12. Widen before item<float>.
  auto x = concatenate(
      {ones({(1 << 20) + 3}, bfloat16), full({1}, -1048576.0f, bfloat16)});
  CHECK(astype(sum(x), float32).item<float>() == 3.0f);
}

TEST_CASE("highway facade fma rounds once on every target") {
  // erfinv computes fma(a, -a, 1): rounding a * a first left 64 units in the
  // last place at 0.99983, where one rounding keeps the worst near 2.1.
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
  // Where Highway emulates float16 -> float32, it decodes exponent 31 as a
  // finite exponent: inf as 65536.
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

TEST_CASE("highway bf16 sin and cos of lanes past 2^23") {
  // A vector with a lane past 2^23 takes std::sin and std::cos lane by lane,
  // written through Simd's operator[]. Every vector here holds one.
  std::vector<float> xs(64);
  for (int i = 0; i < 32; ++i) {
    xs[2 * i] = std::ldexp(128.0f + 4 * i, 17); // 2^24 and up, exact in bf16
    xs[2 * i + 1] = 0.25f * (i + 1);
  }
  auto x = astype(array(xs.data(), {64}, float32), bfloat16);
  auto sine = astype(sin(x), float32);
  auto cosine = astype(cos(x), float32);
  for (int i = 0; i < 64; ++i) {
    CAPTURE(xs[i]);
    const float s = slice(sine, {i}, {i + 1}).item<float>();
    const float c = slice(cosine, {i}, {i + 1}).item<float>();
    CHECK(std::abs(s - std::sin(xs[i])) < 1.0f / 64);
    CHECK(std::abs(c - std::cos(xs[i])) < 1.0f / 64);
  }
}

TEST_CASE("highway floor, ceil, trunc and round keep the sign of zero") {
  // As in C, each result has its input's sign, zeros included. 16 values fill
  // whole vectors at every target, so no lane takes the scalar path.
  std::vector<float> xs;
  for (float v : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.5f, 3.75f}) {
    xs.push_back(-v);
    xs.push_back(v);
  }
  struct Rounding {
    std::string name;
    array (*mlx)(const array&);
    double (*c)(double);
  };
  const Rounding roundings[] = {
      {"floor",
       [](const array& a) { return floor(a); },
       [](double v) { return std::floor(v); }},
      {"ceil",
       [](const array& a) { return ceil(a); },
       [](double v) { return std::ceil(v); }},
      {"trunc",
       [](const array& a) { return trunc(a); },
       [](double v) { return std::trunc(v); }},
      {"round",
       [](const array& a) { return round(a); },
       [](double v) { return std::rint(v); }}};
  auto element = [](const array& a, int i) -> double {
    auto e = slice(a, {i}, {i + 1});
    if (a.dtype() == float16) {
      return static_cast<float>(e.item<float16_t>());
    } else if (a.dtype() == bfloat16) {
      return static_cast<float>(e.item<bfloat16_t>());
    } else if (a.dtype() == float32) {
      return e.item<float>();
    }
    return e.item<double>();
  };
  const int n = static_cast<int>(xs.size());
  for (auto dtype : {float32, float64, bfloat16, float16}) {
    auto x = array(xs.begin(), {n}, dtype);
    for (const auto& r : roundings) {
      auto y = r.mlx(x);
      for (int i = 0; i < n; ++i) {
        const double got = element(y, i);
        const double want = r.c(xs[i]);
        CAPTURE(r.name);
        CAPTURE(dtype);
        CAPTURE(xs[i]);
        CAPTURE(got);
        CAPTURE(want);
        CHECK(got == want);
        CHECK(std::signbit(got) == std::signbit(want));
      }
    }
  }
}

TEST_CASE(
    "highway maximum and minimum: NaN wins, and a tie of +0 and -0 returns b") {
  // As base_simd.h. 64 elements hold every pair in whole vectors. The NaNs
  // differ in sign, so a NaN result shows its operand (not in bf16: one NaN).
  const float a_values[4] = {0.0f, -0.0f, std::copysign(NAN, -1.0f), 1.0f};
  const float b_values[4] = {-0.0f, 0.0f, 1.0f, NAN};
  std::vector<float> as, bs;
  for (int i = 0; i < 64; ++i) {
    as.push_back(a_values[i % 4]);
    bs.push_back(b_values[(i / 4) % 4]);
  }
  auto reference = [](float a, float b, bool is_max) {
    if (std::isnan(b)) {
      return b;
    }
    if (std::isnan(a)) {
      return a;
    }
    return (is_max ? a > b : a < b) ? a : b;
  };
  auto at = [](const array& a, int i) {
    return slice(astype(a, float32), {i}, {i + 1}).item<float>();
  };
  for (auto dtype : {float32, float64, bfloat16, float16}) {
    auto a = array(as.begin(), {64}, dtype);
    auto b = array(bs.begin(), {64}, dtype);
    auto hi = maximum(a, b);
    auto lo = minimum(a, b);
    for (int i = 0; i < 64; ++i) {
      // One element takes base_simd.h's scalar path.
      auto a1 = slice(a, {i}, {i + 1});
      auto b1 = slice(b, {i}, {i + 1});
      for (bool is_max : {true, false}) {
        const float got = at(is_max ? hi : lo, i);
        const float want = reference(as[i], bs[i], is_max);
        const float scalar = at(is_max ? maximum(a1, b1) : minimum(a1, b1), 0);
        CAPTURE(dtype);
        CAPTURE(is_max);
        CAPTURE(as[i]);
        CAPTURE(bs[i]);
        CAPTURE(got);
        CAPTURE(scalar);
        CHECK(std::isnan(got) == std::isnan(want));
        CHECK(std::signbit(got) == std::signbit(scalar));
        if (!std::isnan(want)) {
          CHECK(got == want);
          CHECK(std::signbit(got) == std::signbit(want));
        }
      }
    }
  }
  // Integers take Highway's Max and Min.
  std::vector<int32_t> xs(64), ys(64), hi_want(64), lo_want(64);
  for (int i = 0; i < 64; ++i) {
    xs[i] = i % 7 - 3;
    ys[i] = 2 - i % 5;
    hi_want[i] = std::max(xs[i], ys[i]);
    lo_want[i] = std::min(xs[i], ys[i]);
  }
  auto x = array(xs.begin(), {64}, int32);
  auto y = array(ys.begin(), {64}, int32);
  CHECK(array_equal(maximum(x, y), array(hi_want.begin(), {64}, int32))
            .item<bool>());
  CHECK(array_equal(minimum(x, y), array(lo_want.begin(), {64}, int32))
            .item<bool>());
}

TEST_CASE("highway max and min reductions return NaN from any position") {
  // 67 elements: every lane of whole vectors, and a scalar tail.
  for (auto dtype : {float32, float64, bfloat16, float16}) {
    for (int p = 0; p < 67; ++p) {
      std::vector<float> xs(67);
      for (int i = 0; i < 67; ++i) {
        xs[i] = 0.5f * (i % 9) - 2.0f;
      }
      xs[p] = NAN;
      auto x = array(xs.begin(), {67}, dtype);
      CAPTURE(dtype);
      CAPTURE(p);
      CHECK(std::isnan(astype(max(x), float32).item<float>()));
      CHECK(std::isnan(astype(min(x), float32).item<float>()));
    }
  }
}

TEST_CASE("highway fp32 matmul splits short inputs by columns, exactly") {
  namespace hi = mlx::core::cpu::highway_info;
  const int k = 256;
  // M * N * K >= 65536 for every m below. 1003 columns do not divide evenly
  // among 8 threads, so the last slice is short.
  const int n = 1003;
  for (int m : {1, 3, 8, 15}) {
    for (bool a_t : {false, true}) {
      for (bool b_t : {false, true}) {
        CAPTURE(m);
        CAPTURE(a_t);
        CAPTURE(b_t);
        auto a = random::normal(
            a_t ? Shape{k, m} : Shape{m, k},
            float32,
            0.0f,
            1.0f,
            random::key(m));
        auto b = random::normal(
            b_t ? Shape{n, k} : Shape{k, n},
            float32,
            0.0f,
            1.0f,
            random::key(100 + m));
        auto lhs = a_t ? transpose(a) : a;
        auto rhs = b_t ? transpose(b) : b;
        hi::reset_stats();
        auto y = matmul(lhs, rhs);
        CHECK(within_sum_bound(y, lhs, rhs, k + 2));
        if (cpu::ThreadPool::instance().max_threads() > 1) {
          CHECK(hi::sgemm_column_splits() > 0);
        }
      }
    }
  }
}

TEST_CASE("highway scratch past 64 MiB is freed after its operation") {
  // M >= 32 dequantizes W to float32 scratch. 9216 x 2048 floats are 75.5 MB on
  // their own, above the 64 MiB the lease keeps, whatever else the path stages.
  auto w = random::normal({9216, 2048}, float32, 0.0f, 0.02f, random::key(41));
  auto q = quantize(w, 64, 4);
  auto x = random::normal({64, 2048}, float32, 0.0f, 1.0f, random::key(42));
  const size_t before = cpu::scratch_retained_bytes();
  CHECK(
      sum(quantized_matmul(x, q[0], q[1], q[2], true, 64, 4)).item<float>() !=
      0.0f);
  const size_t after_large = cpu::scratch_retained_bytes();
  CHECK(after_large <= before);

  // A small one keeps its buffer for the next call.
  auto ws = random::normal({64, 256}, float32, 0.0f, 1.0f, random::key(43));
  auto qs = quantize(ws, 64, 4);
  auto xs = random::normal({32, 256}, float32, 0.0f, 1.0f, random::key(44));
  CHECK(
      sum(quantized_matmul(xs, qs[0], qs[1], qs[2], true, 64, 4))
          .item<float>() != 0.0f);
  CHECK(cpu::scratch_retained_bytes() > after_large);
}

TEST_CASE("highway scratch leases do not nest on one thread") {
  std::vector<float> buffer;
  {
    cpu::ScratchLease first(buffer, 16);
    // A second lease could reallocate the buffer under the first.
    CHECK_THROWS_AS(cpu::ScratchLease(buffer, 1 << 20), std::logic_error);
    CHECK(buffer.size() == 16);
  }
  CHECK_NOTHROW(cpu::ScratchLease(buffer, 16));
}
