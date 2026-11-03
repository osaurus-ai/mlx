#pragma once

// Required for using M_LN2 in MSVC.
#define _USE_MATH_DEFINES

#include <math.h>
#include <stdint.h>
#include <algorithm>
#include <complex>
#include <functional>

#ifdef _MSC_VER
#include <intrin.h> // For _BitScanReverse
#endif

#include "mlx/types/complex.h"
#include "mlx/types/half_types.h"

namespace mlx::core::simd {
template <typename T, int N>
struct Simd;

template <typename T>
static constexpr int max_size = 1;

template <typename T>
struct Simd<T, 1> {
  static constexpr int size = 1;
  T value;
  Simd() {}
  template <typename U>
  Simd(Simd<U, 1> v) : value(v.value) {}
  template <typename U>
  Simd(U v) : value(v) {}

  T operator[](int) const {
    return value;
  }

  T& operator[](int) {
    return value;
  }
};

template <typename T, int N>
Simd<T, N> load(const T* x) {
  return *(Simd<T, N>*)x;
}

template <typename T, int N>
void store(T* dst, Simd<T, N> x) {
  // Maintain invariant that bool is either 0 or 1 as
  // simd comparison ops set all bits in the result to 1
  if constexpr (std::is_same_v<T, bool> && N > 1) {
    x = x & 1;
  }
  *(Simd<T, N>*)dst = x;
}

template <typename T>
Simd<T, 1> rint(Simd<T, 1> in) {
  if constexpr (is_complex<T>) {
    return Simd<T, 1>{
        T{std::rint(in.value.real()), std::rint(in.value.imag())}};
  } else {
    return Simd<T, 1>{std::rint(in.value)};
  }
}

template <typename T>
Simd<T, 1> rsqrt(Simd<T, 1> in) {
  return T(1.0) / sqrt(in);
}

template <typename T>
Simd<T, 1> recip(Simd<T, 1> in) {
  return T(1.0) / in;
}

#define DEFAULT_UNARY(name, op)    \
  template <typename T>            \
  Simd<T, 1> name(Simd<T, 1> in) { \
    return op(in.value);           \
  }

DEFAULT_UNARY(operator!, std::logical_not{})
DEFAULT_UNARY(acos, std::acos)
DEFAULT_UNARY(acosh, std::acosh)
DEFAULT_UNARY(asin, std::asin)
DEFAULT_UNARY(asinh, std::asinh)
DEFAULT_UNARY(atan, std::atan)
DEFAULT_UNARY(atanh, std::atanh)
DEFAULT_UNARY(ceil, std::ceil)
DEFAULT_UNARY(conj, std::conj)
DEFAULT_UNARY(cosh, std::cosh)
DEFAULT_UNARY(expm1, std::expm1)
DEFAULT_UNARY(floor, std::floor)
DEFAULT_UNARY(log, std::log)
DEFAULT_UNARY(log10, std::log10)
DEFAULT_UNARY(sinh, std::sinh)
DEFAULT_UNARY(sqrt, std::sqrt)
DEFAULT_UNARY(tan, std::tan)
DEFAULT_UNARY(tanh, std::tanh)

template <typename T>
Simd<T, 1> log1p(Simd<T, 1> in) {
  if constexpr (is_complex<T>) {
    auto x = in.value.real();
    auto y = in.value.imag();
    auto zabs = std::abs(in.value);
    auto theta = std::atan2(y, x + 1);
    if (zabs < 0.5) {
      auto r = x * (2 + x) + y * y;
      if (r == 0) { // handle underflow
        return Simd<T, 1>{T{x, theta}};
      }
      return Simd<T, 1>{T{((decltype(x))(0.5)) * std::log1p(r), theta}};
    } else {
      auto z0 = std::hypot(x + 1, y);
      return Simd<T, 1>{T{std::log(z0), theta}};
    }
  } else {
    return Simd<T, 1>{std::log1p(in.value)};
  }
}

template <typename T>
Simd<T, 1> log2(Simd<T, 1> in) {
  if constexpr (is_complex<T>) {
    auto out = std::log(in.value);
    auto scale = decltype(out.real())(M_LN2);
    return Simd<T, 1>{T{out.real() / scale, out.imag() / scale}};
  } else {
    return Simd<T, 1>{std::log2(in.value)};
  }
}

template <typename T>
Simd<T, 1> operator~(Simd<T, 1> in) {
  return ~in.value;
}

template <typename T>
auto real(Simd<T, 1> in) -> Simd<decltype(std::real(in.value)), 1> {
  return std::real(in.value);
}
template <typename T>
auto imag(Simd<T, 1> in) -> Simd<decltype(std::imag(in.value)), 1> {
  return std::imag(in.value);
}
template <typename T>
Simd<bool, 1> isnan(Simd<T, 1> in) {
  return mlx::core::isnan(in.value);
}

#define DEFAULT_BINARY(OP)                                                 \
  template <typename T1, typename T2>                                      \
  auto operator OP(Simd<T1, 1> a, Simd<T2, 1> b)                           \
      ->Simd<decltype(a.value OP b.value), 1> {                            \
    return a.value OP b.value;                                             \
  }                                                                        \
  template <typename T1, typename T2>                                      \
  auto operator OP(T1 a, Simd<T2, 1> b)->Simd<decltype(a OP b.value), 1> { \
    return a OP b.value;                                                   \
  }                                                                        \
  template <typename T1, typename T2>                                      \
  auto operator OP(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value OP b), 1> { \
    return a.value OP b;                                                   \
  }

// Signed integer overflow is undefined behavior in C++, while mlx wraps integer
// add, subtract and multiply around. When the result type is a signed integer,
// apply op to the operands converted to the unsigned type of the same size and
// convert the result back (modular since C++20). The result type is the one to
// look at: operands of 8 and 16 bits are promoted to int first, and the product
// of two uint16 values can overflow int.
template <typename Op, typename T1, typename T2>
auto wrapping(Op op, T1 a, T2 b) {
  using R = decltype(op(a, b));
  if constexpr (std::is_integral_v<R> && std::is_signed_v<R>) {
    using U = std::make_unsigned_t<R>;
    return static_cast<R>(op(static_cast<U>(a), static_cast<U>(b)));
  } else {
    return op(a, b);
  }
}

#define WRAPPING_BINARY(OP)                                                   \
  template <typename T1, typename T2>                                         \
  auto operator OP(Simd<T1, 1> a, Simd<T2, 1> b)                              \
      ->Simd<decltype(a.value OP b.value), 1> {                               \
    return wrapping([](auto x, auto y) { return x OP y; }, a.value, b.value); \
  }                                                                           \
  template <typename T1, typename T2>                                         \
  auto operator OP(T1 a, Simd<T2, 1> b)->Simd<decltype(a OP b.value), 1> {    \
    return wrapping([](auto x, auto y) { return x OP y; }, a, b.value);       \
  }                                                                           \
  template <typename T1, typename T2>                                         \
  auto operator OP(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value OP b), 1> {    \
    return wrapping([](auto x, auto y) { return x OP y; }, a.value, b);       \
  }

WRAPPING_BINARY(+)
WRAPPING_BINARY(-)
WRAPPING_BINARY(*)
DEFAULT_BINARY(/)
DEFAULT_BINARY(<<)
DEFAULT_BINARY(>>)
DEFAULT_BINARY(|)
DEFAULT_BINARY(^)
DEFAULT_BINARY(&)
DEFAULT_BINARY(&&)
DEFAULT_BINARY(||)

// Negation and abs wrap around too: -MIN and abs(MIN) are MIN, as in two's
// complement. A signed result is computed as 0 - x in the unsigned type, like
// the arithmetic above.
template <typename T>
Simd<T, 1> operator-(Simd<T, 1> in) {
  if constexpr (std::is_integral_v<T>) {
    return wrapping([](auto x, auto y) { return x - y; }, T(0), in.value);
  } else {
    return std::negate{}(in.value);
  }
}

template <typename T>
Simd<T, 1> abs(Simd<T, 1> in) {
  if constexpr (std::is_unsigned_v<T>) {
    return in;
  } else if constexpr (std::is_integral_v<T>) {
    return in.value < 0 ? -in : in;
  } else {
    return std::abs(in.value);
  }
}

template <typename T>
Simd<T, 1> clz(Simd<T, 1> x_) {
#ifdef _MSC_VER
  // MSVC doesn't have __builtin_clz, use _BitScanReverse instead
  unsigned long index;
  if (_BitScanReverse(&index, static_cast<unsigned long>(x_.value))) {
    return static_cast<T>(31 - index);
  }
  return static_cast<T>(32); // All zeros case
#else
  return __builtin_clz(x_.value);
#endif
}

// Integer division by zero traps on x86 and returns 0 on arm64. Define the
// quotient as 0 on all platforms, which keeps a == (a / b) * b + (a % b).
template <typename T>
Simd<T, 1> divide(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  if constexpr (std::is_integral_v<T>) {
    if (b == 0) {
      return T(0);
    }
  }
  return a / b;
}

template <typename T>
Simd<T, 1> remainder(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  T r;
  if constexpr (std::is_integral_v<T>) {
    if (b == 0) {
      return a;
    }
    r = a % b;
  } else {
    r = std::remainder(a, b);
  }
  if constexpr (is_signed_v<T>) {
    if (r != 0 && (r < 0 != b < 0)) {
      r += b;
    }
  }
  return r;
}

template <typename T>
Simd<T, 1> maximum(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  if constexpr (!std::is_integral_v<T>) {
    if (mlx::core::isnan(a)) {
      return a;
    }
  }
  return (a > b) ? a : b;
}

template <typename T>
Simd<T, 1> minimum(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  if constexpr (!std::is_integral_v<T>) {
    if (mlx::core::isnan(a)) {
      return a;
    }
  }
  return (a < b) ? a : b;
}

template <typename T>
Simd<T, 1> pow(Simd<T, 1> a, Simd<T, 1> b) {
  T base = a.value;
  T exp = b.value;
  if constexpr (!std::is_integral_v<T>) {
    return std::pow(base, exp);
  } else {
    T res = 1;
    if constexpr (std::is_signed_v<T>) {
      if (exp < 0) {
        return 0;
      }
    }
    // The power is the exact one modulo 2^N: the products wrap around, and the
    // last squaring of the base, which the result does not need, may too.
    auto mul = [](auto x, auto y) { return x * y; };
    while (exp) {
      if (exp & 1) {
        res = wrapping(mul, res, base);
      }
      exp >>= 1;
      base = wrapping(mul, base, base);
    }
    return res;
  }
}

template <typename T>
Simd<T, 1> atan2(Simd<T, 1> a, Simd<T, 1> b) {
  return std::atan2(a.value, b.value);
}

#define DEFAULT_COMPARISONS(OP)                             \
  template <typename T1, typename T2>                       \
  Simd<bool, 1> operator OP(Simd<T1, 1> a, Simd<T2, 1> b) { \
    return a.value OP b.value;                              \
  }                                                         \
  template <typename T1, typename T2>                       \
  Simd<bool, 1> operator OP(T1 a, Simd<T2, 1> b) {          \
    return a OP b.value;                                    \
  }                                                         \
  template <typename T1, typename T2>                       \
  Simd<bool, 1> operator OP(Simd<T1, 1> a, T2 b) {          \
    return a.value OP b;                                    \
  }

DEFAULT_COMPARISONS(>)
DEFAULT_COMPARISONS(<)
DEFAULT_COMPARISONS(>=)
DEFAULT_COMPARISONS(<=)
DEFAULT_COMPARISONS(==)
DEFAULT_COMPARISONS(!=)

template <typename MaskT, typename T>
Simd<T, 1> select(Simd<MaskT, 1> mask, Simd<T, 1> x, Simd<T, 1> y) {
  return mask.value ? x.value : y.value;
}

template <typename T>
Simd<T, 1> clamp(Simd<T, 1> v, Simd<T, 1> min, Simd<T, 1> max) {
  return std::clamp(v.value, min.value, max.value);
}

template <typename T, typename U>
Simd<T, 1> fma(Simd<T, 1> x, Simd<T, 1> y, U z) {
  return std::fma(x.value, y.value, Simd<T, 1>(z).value);
}

// Reductions
#define DEFAULT_REDUCTION(name, type) \
  template <typename T>               \
  type name(Simd<T, 1> x) {           \
    return x.value;                   \
  }

DEFAULT_REDUCTION(max, T)
DEFAULT_REDUCTION(min, T)
DEFAULT_REDUCTION(sum, T)
DEFAULT_REDUCTION(prod, T)
DEFAULT_REDUCTION(any, bool)
DEFAULT_REDUCTION(all, bool)

} // namespace mlx::core::simd
