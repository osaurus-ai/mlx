// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/cpu/simd/type.h"
#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "mlx/types/complex.h"
#include "mlx/types/half_types.h"
#endif // MLX_USE_HIGHWAY_KERNELS

namespace mlx::core::simd {

template <typename To, typename From>
inline To bit_cast(const From& from) {
  static_assert(sizeof(To) == sizeof(From));
#if defined(_MSC_VER) && !defined(__clang__)
  return __builtin_bit_cast(To, from);
#else
  To to;
  __builtin_memcpy(&to, &from, sizeof(To));
  return to;
#endif
}

constexpr float inf = std::numeric_limits<float>::infinity();

/**
 * Compute exp(x) in an optimizer friendly way as follows:
 *
 * First change the problem to computing 2**y where y = x / ln(2).
 *
 * Now we will compute 2**y as 2**y1 * 2**y2 where y1 is the integer part
 * `ipart` and y2 is fractional part. For the integer part we perform bit
 * shifting and for the fractional part we use a polynomial approximation.
 *
 * The algorithm and constants of the polynomial taken from
 * https://github.com/akohlmey/fastermath/blob/master/src/exp.c which took them
 * from Cephes math library.
 *
 * Note: The implementation below is a general fast exp. There could be faster
 *       implementations for numbers strictly < 0.
 */
template <typename T, int N>
Simd<T, N> exp(Simd<T, N> in) {
  if constexpr (is_complex<T>) {
    return Simd<T, 1>{std::exp(in.value)};
  } else if constexpr (std::is_same_v<T, double>) {
    // The path below computes in float32.
    Simd<T, N> out;
    for (int i = 0; i < N; ++i) {
      out[i] = std::exp(in[i]);
    }
    return out;
  } else {
    Simd<float, N> x_init = in;
    auto x = x_init * 1.442695f; // multiply with log_2(e)
    Simd<float, N> ipart, fpart;
    ipart = floor(x + 0.5);
    fpart = x - ipart;

    x = 1.535336188319500e-4f;
    x = fma(x, fpart, 1.339887440266574e-3f);
    x = fma(x, fpart, 9.618437357674640e-3f);
    x = fma(x, fpart, 5.550332471162809e-2f);
    x = fma(x, fpart, 2.402264791363012e-1f);
    x = fma(x, fpart, 6.931472028550421e-1f);
    x = fma(x, fpart, 1.000000000000000f);

    // generate 2**ipart in the floating point representation using integer
    // bitshifting. The exponent field is ipart + 127. ipart is integral, so
    // adding 1.5 * 2**23 + 127 puts ipart + 127 in the low bits of the
    // mantissa, and the shift by 23 keeps only these bits. This is exact for
    // |ipart| < 2**22, other lanes are replaced below. Nothing converts a
    // float to an integer.
    Simd<float, N> magic = ipart + 12583039.0f;
    auto epart = bit_cast<Simd<uint32_t, N>>(magic) << 23;

    // Deal with NaN and Inf
    auto result =
        select(isnan(x_init), x_init, bit_cast<Simd<float, N>>(epart) * x);
    result = select(x_init > 88.0f, Simd<float, N>(inf), result);
    result = select(x_init < -88.0f, Simd<float, N>(0), result);
    return Simd<T, N>(result);
  }
}

/* Implementation from:
 * https://github.com/JishinMaster/simd_utils/blob/3c1433a86fb38edcc9b02039f3c9a65b16640976/neon_mathfun.h#L357
 * which originally came from the Cephes math library.
 */
template <bool Sine, typename T, int N>
Simd<T, N> sincos(Simd<T, N> in) {
  auto sign_mask_sin = in < 0;
  in = abs(in);
  Simd<float, N> x = in;

  // scale by 4/Pi
  auto y = x * 1.27323954473516f;

  // store the integer part of y in mm0. sincos_checked sends |x| > 2**23 to
  // libm, so y is in [0, 2**24) or NaN, and NaN lanes become 0.
  Simd<uint32_t, N> emm2 =
      convert_in_range<uint32_t>(select(isnan(y), Simd<float, N>(0), y));

  // j=(j+1) & (~1) (see the cephes sources)
  emm2 = emm2 + 1;
  emm2 = emm2 & ~1;

  y = emm2;

  // Get the polynom selection mask. There is one polynom for 0 <= x <= Pi/4
  // and another one for Pi/4<x<=Pi/2. Both branches will be computed.
  auto poly_mask = (emm2 & 2) != 0;

  // The magic pass: "Extended precision modular arithmetic"
  // x = ((x - y * DP1) - y * DP2) - y * DP3
  x = fma(y, Simd<float, N>(-0.78515625f), x);
  x = fma(y, Simd<float, N>(-2.4187564849853515625e-4f), x);
  x = fma(y, Simd<float, N>(-3.77489497744594108e-8f), x);

  sign_mask_sin = sign_mask_sin ^ ((emm2 & 4) != 0);
  auto sign_mask_cos = ((emm2 - 2) & 4) != 0;

  // Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
  // and the second polynom      (Pi/4 <= x <= 0) in y2
  auto z = x * x;

  auto y1 =
      fma(z, Simd<float, N>(2.443315711809948e-5f), -1.388731625493765e-3f);
  auto y2 = fma(z, Simd<float, N>(-1.9515295891e-4f), 8.3321608736e-3f);
  y1 = fma(y1, z, 4.166664568298827e-2f);
  y2 = fma(y2, z, -1.6666654611e-1f);
  y1 = y1 * z;
  y2 = y2 * z;
  y1 = y1 * z;
  y2 = fma(x, y2, x);
  y1 = fma(z, Simd<float, N>(-0.5f), y1);
  y1 = y1 + 1.0f;

  if constexpr (Sine) {
    auto ys = select(poly_mask, y1, y2);
    return select(sign_mask_sin, -ys, ys);
  } else {
    auto yc = select(poly_mask, y2, y1);
    return select(sign_mask_cos, yc, -yc);
  }
}

// sincos reduces the argument by rounding x * 4/pi into a uint32, which is
// only exact while that product fits in a float's mantissa. Past that the
// result degrades and then leaves [-1, 1] altogether, so send those arguments
// to libm instead. 2^23 keeps x * 4/pi under 2^24 with room to spare.
template <bool Sine, typename T, int N>
Simd<T, N> sincos_checked(Simd<T, N> x) {
  if constexpr (std::is_same_v<T, double>) {
    // The path below computes in float32.
    Simd<T, N> out;
    for (int i = 0; i < N; ++i) {
      out[i] = Sine ? std::sin(x[i]) : std::cos(x[i]);
    }
    return out;
  }
  Simd<float, N> xf = x;
  if (any(abs(xf) > Simd<float, N>(8388608.0f))) {
    Simd<T, N> out;
    for (int i = 0; i < N; ++i) {
      float v = xf[i];
      out[i] = static_cast<T>(Sine ? std::sin(v) : std::cos(v));
    }
    return out;
  }
  return sincos<Sine>(x);
}

template <typename T, int N>
Simd<T, N> sin(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::sin(x.value);
  } else {
    return sincos_checked<true>(x);
  }
}

template <typename T, int N>
Simd<T, N> cos(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::cos(x.value);
  } else {
    return sincos_checked<false>(x);
  }
}

template <typename T, int N>
Simd<T, N> erf(Simd<T, N> x) {
  if constexpr (std::is_same_v<T, double>) {
    // The path below computes in float32.
    Simd<T, N> out;
    for (int i = 0; i < N; ++i) {
      out[i] = std::erf(x[i]);
    }
    return out;
  } else {
    // The float erf of metal/kernels/erf.h (error at most 1 ulp). 1 - erfc(x)
    // loses the relative accuracy near 0, where erf(x) is about 1.128 x.
    Simd<float, N> a = x;
    auto t = abs(a);
    auto s = a * a;
    // |x| <= 0.927734375
    auto p = fma(Simd<float, N>(-5.96761703e-4f), s, 4.99119423e-3f);
    p = fma(p, s, -2.67681349e-2f);
    p = fma(p, s, 1.12819925e-1f);
    p = fma(p, s, -3.76125336e-1f);
    p = fma(p, s, 1.28379166e-1f);
    p = fma(p, a, a);
    // |x| > 0.927734375
    auto r = fma(Simd<float, N>(-1.72853470e-5f), t, 3.83197126e-4f);
    auto u = fma(Simd<float, N>(-3.88396438e-3f), t, 2.42546219e-2f);
    r = fma(r, s, u);
    r = fma(r, t, -1.06777877e-1f);
    r = fma(r, t, -6.34846687e-1f);
    r = fma(r, t, -1.28717512e-1f);
    r = fma(r, t, -t);
    r = -expm1(r);
    r = select(a < 0.0f, -r, r);
    return Simd<T, N>(select(t > 0.927734375f, r, p));
  }
}

template <typename T, int N>
#if defined(MLX_USE_HIGHWAY_KERNELS)
Simd<T, N> sigmoid(Simd<T, N> x) {
  // For float16/bfloat16 inputs, compute in float32 precision for accuracy
  // For float32/float64, compute in the same precision
  using ComputeT = std::conditional_t<
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      float,
      T>;
  Simd<ComputeT, N> in = x;
  auto y = recip(ComputeT(1.0f) + exp(abs(in)));
  return Simd<T, N>(select(in < Simd<ComputeT, N>{0}, y, ComputeT(1.0f) - y));
}

// Complex sigmoid: 1 / (1 + exp(-z))
template <int N>
Simd<complex64_t, N> sigmoid(Simd<complex64_t, N> x) {
  return Simd<complex64_t, N>(recip(Simd<float, N>(1.0f) + exp(-x)));
}

template <typename T, int N>
#endif // MLX_USE_HIGHWAY_KERNELS
Simd<T, N> erfinv(Simd<T, N> a_) {
#if !defined(MLX_USE_HIGHWAY_KERNELS)
  Simd<float, N> a = a_;
  auto t = fma(a, 0.0f - a, 1.0f);
#else
  // Use explicit ComputeT so half-type Simd (float16/bfloat16) promotes to
  // float32 and bare float literals don't cause template deduction ambiguity
  // in fma(). For float/double T, ComputeT == T so behavior is unchanged.
  using ComputeT = std::conditional_t<
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      float,
      T>;
  Simd<ComputeT, N> a = a_;
  auto t = fma(a, ComputeT(0.0f) - a, ComputeT(1.0f));
#endif // MLX_USE_HIGHWAY_KERNELS
  t = log(t);
  auto lhs = [](auto t) {
#if !defined(MLX_USE_HIGHWAY_KERNELS)
    Simd<float, N> p;
    p = 3.03697567e-10f; //  0x1.4deb44p-32
    p = fma(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26
    p = fma(p, t, 1.22150334e-6f); //  0x1.47e512p-20
    p = fma(p, t, 2.84108955e-5f); //  0x1.dca7dep-16
    p = fma(p, t, 3.93552968e-4f); //  0x1.9cab92p-12
    p = fma(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9
    p = fma(p, t, 4.83185798e-3f); //  0x1.3ca920p-8
    p = fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    return fma(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1
#else
    Simd<ComputeT, N> p;
    p = ComputeT(3.03697567e-10f); //  0x1.4deb44p-32
    p = fma(p, t, ComputeT(2.93243101e-8f)); //  0x1.f7c9aep-26
    p = fma(p, t, ComputeT(1.22150334e-6f)); //  0x1.47e512p-20
    p = fma(p, t, ComputeT(2.84108955e-5f)); //  0x1.dca7dep-16
    p = fma(p, t, ComputeT(3.93552968e-4f)); //  0x1.9cab92p-12
    p = fma(p, t, ComputeT(3.02698812e-3f)); //  0x1.8cc0dep-9
    p = fma(p, t, ComputeT(4.83185798e-3f)); //  0x1.3ca920p-8
    p = fma(p, t, ComputeT(-2.64646143e-1f)); // -0x1.0eff66p-2
    return fma(p, t, ComputeT(8.40016484e-1f)); //  0x1.ae16a4p-1
#endif // MLX_USE_HIGHWAY_KERNELS
  };
  auto rhs = [](auto t) {
#if !defined(MLX_USE_HIGHWAY_KERNELS)
    Simd<float, N> p;
    p = 5.43877832e-9f; //  0x1.75c000p-28
    p = fma(p, t, 1.43285448e-7f); //  0x1.33b402p-23
    p = fma(p, t, 1.22774793e-6f); //  0x1.499232p-20
    p = fma(p, t, 1.12963626e-7f); //  0x1.e52cd2p-24
    p = fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = fma(p, t, 2.31468678e-3f); //  0x1.2f6400p-9
    p = fma(p, t, 1.15392581e-2f); //  0x1.7a1e50p-7
    p = fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    return fma(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1
#else
    Simd<ComputeT, N> p;
    p = ComputeT(5.43877832e-9f); //  0x1.75c000p-28
    p = fma(p, t, ComputeT(1.43285448e-7f)); //  0x1.33b402p-23
    p = fma(p, t, ComputeT(1.22774793e-6f)); //  0x1.499232p-20
    p = fma(p, t, ComputeT(1.12963626e-7f)); //  0x1.e52cd2p-24
    p = fma(p, t, ComputeT(-5.61530760e-5f)); // -0x1.d70bd0p-15
    p = fma(p, t, ComputeT(-1.47697632e-4f)); // -0x1.35be90p-13
    p = fma(p, t, ComputeT(2.31468678e-3f)); //  0x1.2f6400p-9
    p = fma(p, t, ComputeT(1.15392581e-2f)); //  0x1.7a1e50p-7
    p = fma(p, t, ComputeT(-2.32015476e-1f)); // -0x1.db2aeep-3
    return fma(p, t, ComputeT(8.86226892e-1f)); //  0x1.c5bf88p-1
#endif // MLX_USE_HIGHWAY_KERNELS
  };
#if !defined(MLX_USE_HIGHWAY_KERNELS)
  auto thresh = 6.125f;
#else
  auto thresh = ComputeT(6.125f);
#endif // MLX_USE_HIGHWAY_KERNELS
  // Compute both branches and select if N > 1
  if constexpr (N == 1) {
    if ((abs(t) > thresh).value) { // maximum ulp error = 2.35793
#if !defined(MLX_USE_HIGHWAY_KERNELS)
      return a * lhs(t);
#else
      return Simd<T, N>(a * lhs(t));
#endif // MLX_USE_HIGHWAY_KERNELS
    } else { // maximum ulp error = 2.35002
#if !defined(MLX_USE_HIGHWAY_KERNELS)
      return a * rhs(t);
#else
      return Simd<T, N>(a * rhs(t));
#endif // MLX_USE_HIGHWAY_KERNELS
    }
  } else {
#if !defined(MLX_USE_HIGHWAY_KERNELS)
    return a * select(abs(t) > thresh, lhs(t), rhs(t));
#else
    return Simd<T, N>(a * select(abs(t) > thresh, lhs(t), rhs(t)));
#endif // MLX_USE_HIGHWAY_KERNELS
  }
}

} // namespace mlx::core::simd
