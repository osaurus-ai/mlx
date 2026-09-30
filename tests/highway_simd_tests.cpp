// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include "doctest/doctest.h"

#include "mlx/backend/cpu/simd/simd.h"

using namespace mlx::core;

namespace {

// A lane's bits in hexadecimal; a float as hexfloat, subnormals included.
template <typename T>
std::string text(T value) {
  std::ostringstream out;
  if constexpr (std::is_floating_point_v<T>) {
    out << std::hexfloat << value;
  } else {
    out << "0x" << std::hex << +static_cast<std::make_unsigned_t<T>>(value);
  }
  return out.str();
}

// Vector r holds values[(r + i) % 8] in lane i, so each value meets each lane.
template <typename T, int N>
void check_each_lane(const std::string& type, const T (&values)[8]) {
  for (int r = 0; r < 8; ++r) {
    T lanes[N];
    for (int i = 0; i < N; ++i) {
      lanes[i] = values[(r + i) % 8];
    }
    const simd::Simd<bool, N> got(simd::load<T, N>(lanes));
    for (int i = 0; i < N; ++i) {
      CAPTURE(type);
      CAPTURE(N);
      CAPTURE(i);
      CAPTURE(text(lanes[i]));
      CHECK(got[i] == static_cast<bool>(lanes[i]));
    }
  }
}

} // namespace

TEST_CASE("highway Simd<bool> of an integer vector reads each lane as bool") {
  const uint8_t u8[8] = {0x01, 0x80, 0x7F, 0x00, 0x02, 0x40, 0xFF, 0x10};
  // Every width of Simd<bool, N> the facade defines; FromFP8 uses 4.
  check_each_lane<uint8_t, 2>("uint8_t", u8);
  check_each_lane<uint8_t, 4>("uint8_t", u8);
  check_each_lane<uint8_t, simd::max_size<uint8_t>>("uint8_t", u8);
  check_each_lane<uint8_t, 16>("uint8_t", u8);
  const uint32_t u32[8] = {
      1, 0x80000000, 0x7FFFFFFF, 0, 2, 0x40000000, 0xFFFFFFFF, 0x10};
  check_each_lane<uint32_t, simd::max_size<uint32_t>>("uint32_t", u32);
  const int32_t i32[8] = {1, INT32_MIN, INT32_MAX, 0, 2, 0x40000000, -1, 0x10};
  check_each_lane<int32_t, simd::max_size<int32_t>>("int32_t", i32);
}

TEST_CASE("highway Simd<bool> of a float vector reads each lane as bool") {
  // As in C++, -0.0 is false and NaN is true.
  using F = std::numeric_limits<float>;
  const float f32[8] = {
      1.0f,
      -0.0f,
      F::quiet_NaN(),
      0.0f,
      2.0f,
      -1.0f,
      F::infinity(),
      F::denorm_min()};
  check_each_lane<float, simd::max_size<float>>("float", f32);
  using D = std::numeric_limits<double>;
  const double f64[8] = {
      1.0,
      -0.0,
      D::quiet_NaN(),
      0.0,
      2.0,
      -1.0,
      D::infinity(),
      D::denorm_min()};
  check_each_lane<double, simd::max_size<double>>("double", f64);
}
