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
