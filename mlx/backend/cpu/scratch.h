// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::cpu {

// Bytes of kernel scratch every thread keeps between operations, process-wide.
// MLX_API gives the static default visibility, so that a shared library and
// the code linked against it count in one place.
MLX_API inline std::atomic<size_t>& scratch_retained() {
  static std::atomic<size_t> bytes{0};
  return bytes;
}

MLX_API inline size_t scratch_retained_bytes() {
  return scratch_retained().load(std::memory_order_relaxed);
}

// A thread's scratch buffer, borrowed for one operation, which frees it if it
// grew past kKeepBytes. A thread holds one lease at a time.
class ScratchLease {
 public:
  static constexpr size_t kKeepBytes = size_t{64} << 20;

  ScratchLease(std::vector<float>& buffer, size_t floats) : buffer_(buffer) {
    // A second lease could reallocate the buffer under the first.
    if (leased_) {
      throw std::logic_error("ScratchLease: this thread already holds a lease");
    }
    const size_t before = buffer_.capacity();
    if (buffer_.size() < floats) {
      buffer_.resize(floats);
    }
    scratch_retained().fetch_add(
        (buffer_.capacity() - before) * sizeof(float),
        std::memory_order_relaxed);
    leased_ = true;
  }
  ScratchLease(const ScratchLease&) = delete;
  ScratchLease& operator=(const ScratchLease&) = delete;
  ~ScratchLease() {
    leased_ = false;
    const size_t bytes = buffer_.capacity() * sizeof(float);
    if (bytes > kKeepBytes) {
      std::vector<float>().swap(buffer_);
      scratch_retained().fetch_sub(bytes, std::memory_order_relaxed);
    }
  }
  float* data() {
    return buffer_.data();
  }

 private:
  static inline thread_local bool leased_ = false;
  std::vector<float>& buffer_;
};

} // namespace mlx::core::cpu
