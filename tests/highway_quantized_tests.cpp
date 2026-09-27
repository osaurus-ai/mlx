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

// The same bound for an affine quantized matmul against its dequantized
// weights. The kernels compute s * sum(x * q) + b * sum(x) per group, so their
// rounding scales with sum |x| * (|s| * q + |b|), which exceeds sum |x * w|
// where s * q and b nearly cancel. dequantize() with |s| and |b| gives
// |s| * q + |b|.
bool within_quantized_bound(
    const array& got,
    const array& x,
    const std::vector<array>& q,
    int group_size,
    int bits,
    double c) {
  auto x64 = astype(x, float64);
  auto w64 = astype(dequantize(q[0], q[1], q[2], group_size, bits), float64);
  auto w_abs =
      astype(dequantize(q[0], abs(q[1]), abs(q[2]), group_size, bits), float64);
  auto reference = matmul(x64, transpose(w64));
  auto magnitude = matmul(abs(x64), transpose(w_abs));
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

TEST_CASE("highway affine 3, 5 and 6-bit matmul at every row count") {
  // #3019 sent every transposed call with M >= 32 to a dequantize-then-SGEMM
  // path that reads 32 / bits values per word: wrong for byte-packed widths.
  auto w = random::normal({256, 512}, float32, 0.0f, 1.0f, random::key(5));
  for (int bits : {3, 5, 6}) {
    for (int group_size : {32, 64, 128}) {
      auto q = quantize(w, group_size, bits);
      for (int rows : {1, 31, 32, 33, 64}) {
        CAPTURE(bits);
        CAPTURE(group_size);
        CAPTURE(rows);
        auto x =
            random::normal({rows, 512}, float32, 0.0f, 1.0f, random::key(rows));
        auto y = quantized_matmul(x, q[0], q[1], q[2], true, group_size, bits);
        CHECK(within_quantized_bound(y, x, q, group_size, bits, 512 + 2));
      }
    }
  }
}

TEST_CASE("highway gather_qmm merges experts without the byte-packing bug") {
  // _bs_qmm_dispatch_typed merges consecutive rows of one expert into a single
  // call with M * batch rows, which reaches the M >= 32 path.
  auto w = random::normal({2, 64, 256}, float32, 0.0f, 1.0f, random::key(11));
  auto q = quantize(w, 64, 6);
  auto w_hat = dequantize(q[0], q[1], q[2], 64, 6);
  auto x = random::normal({40, 1, 256}, float32, 0.0f, 1.0f, random::key(12));
  auto experts = zeros({40}, uint32);
  auto y = gather_qmm(x, q[0], q[1], q[2], std::nullopt, experts, true, 64, 6);
  auto reference = matmul(x, transpose(take(w_hat, experts, 0), {0, 2, 1}));
  CHECK(relative_error(y, reference) < 1e-5f);
}

namespace {

// One group of `values` through qqmm against a quantized identity, which
// returns the activations exactly as QQMatmul quantized and dequantized them.
// The identity quantizes exactly: mxfp4 stores 1 as 4 with scale 2^-2, mxfp8 as
// 256 with scale 2^-8.
std::vector<float> through_qqmm(
    const std::vector<float>& values,
    const std::string& mode,
    int bits) {
  const int k = static_cast<int>(values.size());
  auto x = array(values.data(), {1, k}, float32);
  auto identity = eye(k, float32);
  auto qw = quantize(identity, 32, bits, mode);
  auto y = qqmm(x, qw[0], qw[1], 32, bits, mode);
  std::vector<float> out(k);
  for (int i = 0; i < k; ++i) {
    out[i] = slice(y, {0, i}, {1, i + 1}).item<float>();
  }
  return out;
}

} // namespace

TEST_CASE("highway qqmm mxfp4 rounds half to even on the e2m1 grid") {
  // amax 6.0 gives an exact E8M0 scale of 1, so each value is rounded alone.
  // Ties go to the even mantissa, as upstream's to_fp4_e2m1 does.
  std::vector<float> values = {
      0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 6.0f};
  values.resize(32, 0.0f);
  auto out = through_qqmm(values, "mxfp4", 4);
  std::vector<float> expected = {
      0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 4.0f, 4.0f, 6.0f};
  for (int i = 0; i < 8; ++i) {
    CAPTURE(values[i]);
    CHECK(out[i] == expected[i]);
  }
}

TEST_CASE("highway qqmm mxfp8 rounds the block scale up") {
  // amax / 448 = 1.3: a nearest E8M0 scale of 1 would saturate the block's
  // largest element to 448. Rounding the scale up to 2 keeps it: 582.4 / 2 =
  // 291.2, which E4M3 rounds to 288, so the element returns as 576.
  std::vector<float> values(32, 1.0f);
  values[0] = 582.4f;
  auto out = through_qqmm(values, "mxfp8", 8);
  CHECK(out[0] == 576.0f);
}
