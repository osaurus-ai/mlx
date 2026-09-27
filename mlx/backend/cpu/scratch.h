// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
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

// A thread's scratch buffer, borrowed for one operation. The buffer stays with
// the thread for reuse, unless the operation left it larger than kKeepBytes:
// then the lease frees it when the operation ends, so one large prefill does
// not hold memory outside MLX's allocator for the thread's lifetime. Leases on
// one buffer do not nest: a second could reallocate it under the first. The
// kernels take one per operation, at its top.
class ScratchLease {
 public:
  static constexpr size_t kKeepBytes = size_t{64} << 20;

  ScratchLease(std::vector<float>& buffer, size_t floats) : buffer_(buffer) {
    const size_t before = buffer_.capacity();
    if (buffer_.size() < floats) {
      buffer_.resize(floats);
    }
    scratch_retained().fetch_add(
        (buffer_.capacity() - before) * sizeof(float),
        std::memory_order_relaxed);
  }
  ScratchLease(const ScratchLease&) = delete;
  ScratchLease& operator=(const ScratchLease&) = delete;
  ~ScratchLease() {
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
  std::vector<float>& buffer_;
};

} // namespace mlx::core::cpu
