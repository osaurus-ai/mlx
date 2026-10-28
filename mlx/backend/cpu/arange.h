// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cpu/encoder.h"

namespace mlx::core {

namespace {

template <typename T>
void arange(T start, T next, array& out, size_t size, Stream stream) {
  auto ptr = out.data<T>();
  auto step_size = next - start;
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
        start += step_size;
      }
    }
  });
}

} // namespace

} // namespace mlx::core
