// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include "mlx/backend/cpu/precision.h"

#include <atomic>
#include <cstdlib>

namespace mlx::core::cpu {

namespace detail {
bool parse_quantized_int8(const char* value) {
  return value != nullptr && std::atoi(value) != 0;
}
} // namespace detail

namespace {
std::atomic<bool>& int8_switch() {
  static std::atomic<bool> enabled{
      detail::parse_quantized_int8(std::getenv("MLX_CPU_QUANTIZED_INT8"))};
  return enabled;
}
} // namespace

bool quantized_int8() {
  return int8_switch().load(std::memory_order_relaxed);
}

void set_quantized_int8(bool enabled) {
  int8_switch().store(enabled, std::memory_order_relaxed);
}

bool quantized_int8_available() {
#if defined(MLX_USE_HIGHWAY_KERNELS)
  return true;
#else
  return false;
#endif
}

bool quantized_float32_accumulation() {
#if defined(MLX_USE_HIGHWAY_KERNELS)
  return true;
#else
  return false;
#endif
}

} // namespace mlx::core::cpu
