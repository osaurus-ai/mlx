// Copyright © 2023 Apple Inc.

#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>

#include "mlx/array.h"
#include "mlx/backend/cpu/encoder.h"

namespace mlx::core {

namespace {

// Converting a floating point value that does not fit in an integer type is
// undefined. arange converts start and start + step, and either can be out of
// range although every output fits: start + step for a range of one element,
// start for an empty range. Such a value gives the maximum or the minimum of
// T, and NaN (which the arange op rejects) gives 0.
template <typename T>
T arange_cast(double v) {
  if constexpr (std::is_integral_v<T>) {
    // 2^digits is exact in a double; the maximum of a 64-bit type is not.
    const double hi = std::ldexp(1.0, std::numeric_limits<T>::digits);
    const double lo = std::is_signed_v<T> ? -hi : 0.0;
    if (v >= hi) {
      return std::numeric_limits<T>::max();
    }
    if (v <= lo) {
      return std::numeric_limits<T>::lowest();
    }
    if (v != v) {
      return 0;
    }
  }
  return v;
}

// For integer types, op(a, b) wraps around: it is done in the unsigned type
// of the same size, as in simd::wrapping. The step and the running sum need
// not fit in T when every output does.
template <typename Op, typename T, typename S>
auto arange_wrapping(Op op, T a, S b) {
  if constexpr (std::is_integral_v<T>) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(op(static_cast<U>(a), static_cast<U>(b)));
  } else {
    return op(a, b);
  }
}

template <typename T>
void arange(T start, T next, array& out, size_t size, Stream stream) {
  auto ptr = out.data<T>();
  auto step_size = arange_wrapping(std::minus<>(), next, start);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([ptr, start, step_size, size]() mutable {
    if constexpr (is_floating_point_v<T>) {
      // A running sum in T loses precision. Compute start + i * step in
      // double. Element 0 is start, also when step is inf (0 * inf is NaN).
      double first = static_cast<double>(start);
      double step = static_cast<double>(step_size);
      if (size > 0) {
        ptr[0] = start;
      }
      for (size_t i = 1; i < size; ++i) {
        // Two statements, so that Clang does not make an FMA of them.
        double offset = i * step;
        ptr[i] = static_cast<T>(first + offset);
      }
    } else {
      for (size_t i = 0; i < size; ++i) {
        ptr[i] = start;
        start = arange_wrapping(std::plus<>(), start, step_size);
      }
    }
  });
}

} // namespace

} // namespace mlx::core
