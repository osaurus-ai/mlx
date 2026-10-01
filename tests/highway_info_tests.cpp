// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include "doctest/doctest.h"

#include "mlx/backend/cpu/highway_info.h"
#include "mlx/backend/cpu/precision.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace hi = mlx::core::cpu::highway_info;

namespace {
std::vector<int64_t> bits_of(int64_t set) {
  std::vector<int64_t> out;
  for (int i = 0; i < 63; ++i) {
    if (set & (int64_t{1} << i)) {
      out.push_back(int64_t{1} << i);
    }
  }
  return out;
}

// max |got - reference| / max |reference|, both widened to float32.
float relative_error(const array& got, const array& reference) {
  auto g = astype(got, float32);
  auto r = astype(reference, float32);
  return (max(abs(subtract(g, r))) / max(abs(r))).item<float>();
}
} // namespace

TEST_CASE("highway info names what was compiled and what the CPU supports") {
  hi::set_targets_for_test(0);
  REQUIRE(hi::compiled_targets() != 0);
  REQUIRE(hi::supported_targets() != 0);
  for (auto t : bits_of(hi::compiled_targets())) {
    CHECK(std::string(hi::target_name(t)).size() > 0);
  }
}

TEST_CASE("highway rms_norm records the target that ran, on every target") {
  hi::set_targets_for_test(0);
  auto runnable = hi::compiled_targets() & hi::supported_targets();
  auto x = random::normal({4, 512}, float32, 0.0f, 1.0f, random::key(31));
  auto weight = ones({512}, float32);
  for (auto target : bits_of(runnable)) {
    CAPTURE(
        std::string(
            hi::target_name(target))); // doctest prints a char* as an address
    hi::set_targets_for_test(target);
    hi::reset_stats();
    CHECK(sum(fast::rms_norm(x, weight, 1e-5f)).item<float>() != 0.0f);
    auto s = hi::stats(hi::Family::RmsNorm);
    CHECK(s.executed == target);
    CHECK(s.highway_calls > 0);
    CHECK(s.fallback_calls == 0);
  }
  hi::set_targets_for_test(0);
}

TEST_CASE("highway int8 family records only when the int8 path ran") {
  // QmmTInt8 is dispatched for every affine 4/8-bit call below 32 rows; it runs
  // the int8 kernels only when the switch is on.
  auto w = random::normal({64, 256}, float32, 0.0f, 1.0f, random::key(33));
  auto x = random::normal({1, 256}, float32, 0.0f, 1.0f, random::key(34));
  auto q = quantize(w, 64, 4);
  hi::reset_stats();
  CHECK(
      sum(quantized_matmul(x, q[0], q[1], q[2], true, 64, 4)).item<float>() !=
      0.0f);
  CHECK(hi::stats(hi::Family::QmmAffineInt8).highway_calls == 0);
  {
    cpu::detail::QuantizedInt8Scope on(true);
    CHECK(
        sum(quantized_matmul(x, q[0], q[1], q[2], true, 64, 4)).item<float>() !=
        0.0f);
  }
  CHECK(hi::stats(hi::Family::QmmAffineInt8).highway_calls > 0);
}

TEST_CASE("highway norms and rope compose float64 through the fallback") {
  // #3019's kernels implement float32, float16 and bfloat16; float64 threw.
  auto x = random::normal({2, 4, 3, 64}, float32, 0.0f, 1.0f, random::key(32));
  auto x64 = astype(x, float64);
  auto w64 = ones({64}, float64);
  hi::reset_stats();
  auto rms = fast::rms_norm(x64, w64, 1e-5f);
  auto layer = fast::layer_norm(x64, w64, zeros({64}, float64), 1e-5f);
  auto rope = fast::rope(x64, 64, false, 10000.0f, 1.0f, 3);
  CHECK(
      relative_error(rms, fast::rms_norm(x, ones({64}, float32), 1e-5f)) <
      1e-5f);
  CHECK(
      relative_error(
          layer,
          fast::layer_norm(
              x, ones({64}, float32), zeros({64}, float32), 1e-5f)) < 1e-5f);
  CHECK(
      relative_error(rope, fast::rope(x, 64, false, 10000.0f, 1.0f, 3)) <
      1e-5f);
  for (auto family :
       {hi::Family::RmsNorm, hi::Family::LayerNorm, hi::Family::Rope}) {
    CHECK(hi::stats(family).fallback_calls > 0);
  }
}

TEST_CASE("highway compiles no SVE target") {
  // The dispatched kernels are not vector-length agnostic yet.
  for (auto t : bits_of(hi::compiled_targets())) {
    const std::string name = hi::target_name(t);
    CAPTURE(name);
    CHECK_FALSE(name.starts_with("SVE"));
  }
}
