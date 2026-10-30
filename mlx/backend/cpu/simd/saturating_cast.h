#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

#include "mlx/types/half_types.h"

namespace mlx::core::simd {

// A float to integer conversion is undefined in C++ when the value does not
// fit, and for NaN. The result differs between arm64 and x86 and between
// integer widths.
template <typename To, typename From>
inline constexpr bool is_float_to_int_cast_v = is_floating_point_v<From> &&
    std::is_integral_v<To> && !std::is_same_v<To, bool>;

// std::fmin and std::fmax are one instruction each on arm64 and in clang, but
// calls into libm in GCC on x86.
#if defined(__aarch64__) || defined(__clang__)
inline constexpr bool inline_fmin_fmax = true;
#else
inline constexpr bool inline_fmin_fmax = false;
#endif

// 2**n in F. Doubling is exact, so no rounding mode is involved.
template <typename F>
constexpr F exact_pow2(int n) {
  F r = 1;
  for (int i = 0; i < n; ++i) {
    r *= 2;
  }
  return r;
}

// Same as static_cast<To>(v), but a float (float16, bfloat16, float32,
// float64) converted to an integer type other than bool is defined for every
// value: NaN gives 0, a value that is too large or too small gives the
// maximum or the minimum of To, and other values are truncated toward zero.
template <typename To, typename From>
constexpr To saturating_cast(From v) {
  if constexpr (is_float_to_int_cast_v<To, From>) {
    using F = std::conditional_t<
        std::is_same_v<From, float16_t> || std::is_same_v<From, bfloat16_t>,
        float,
        From>;
    F x = static_cast<F>(v);
    if constexpr (
        inline_fmin_fmax &&
        std::numeric_limits<To>::digits <= std::numeric_limits<F>::digits) {
      // The bounds of To are exact in F. Clamp (fmin and fmax return the other
      // operand for NaN), convert, and give 0 for NaN.
      constexpr F max_f = static_cast<F>(std::numeric_limits<To>::max());
      constexpr F min_f = static_cast<F>(std::numeric_limits<To>::lowest());
      To r = static_cast<To>(std::fmax(std::fmin(x, max_f), min_f));
      return x == x ? r : To(0);
    } else {
      // Powers of two are exact in F. The maximum of To is not (INT32_MAX
      // rounds to 2**31 in float).
      constexpr F hi = exact_pow2<F>(std::numeric_limits<To>::digits);
      constexpr F lo = std::is_signed_v<To> ? -hi : F(0);
      // No branches, so that loops can vectorize. NaN fails all comparisons.
      bool in_range = (x > lo) & (x < hi);
      To r = static_cast<To>(in_range ? x : F(0));
      r = (x >= hi) ? std::numeric_limits<To>::max() : r;
      r = (x <= lo) ? std::numeric_limits<To>::lowest() : r;
      return r;
    }
  } else {
    return static_cast<To>(v);
  }
}

} // namespace mlx::core::simd
