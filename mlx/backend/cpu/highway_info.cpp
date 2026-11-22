// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if defined(MLX_USE_HIGHWAY_KERNELS)

#include "mlx/backend/cpu/highway_info.h"

#include <atomic>

#include "hwy/highway.h"
#include "hwy/targets.h"

namespace mlx::core::cpu::highway_info {

namespace {
constexpr int kFamilies = static_cast<int>(Family::Count);
// A cache line per family: the kernels of different families run at once.
struct alignas(64) Counters {
  std::atomic<int64_t> executed{0};
  std::atomic<uint64_t> highway_calls{0};
  std::atomic<uint64_t> fallback_calls{0};
};
Counters counters[kFamilies];
std::atomic<uint64_t> column_splits{0};
} // namespace

// HWY_TARGETS is what foreach_target.h compiles in every dispatched file of
// this build: they share one set of compiler flags and defines.
int64_t compiled_targets() {
  return HWY_TARGETS;
}

int64_t supported_targets() {
  return hwy::SupportedTargets();
}

void set_targets_for_test(int64_t targets) {
  hwy::SetSupportedTargetsForTest(targets);
}

const char* target_name(int64_t target) {
  return hwy::TargetName(target);
}

FamilyStats stats(Family family) {
  auto& c = counters[static_cast<int>(family)];
  return {
      c.executed.load(std::memory_order_relaxed),
      c.highway_calls.load(std::memory_order_relaxed),
      c.fallback_calls.load(std::memory_order_relaxed)};
}

void reset_stats() {
  for (auto& c : counters) {
    c.executed.store(0, std::memory_order_relaxed);
    c.highway_calls.store(0, std::memory_order_relaxed);
    c.fallback_calls.store(0, std::memory_order_relaxed);
  }
  column_splits.store(0, std::memory_order_relaxed);
}

// Loads first: after the first call per target, both are reads.
void record(Family family, int64_t target) {
  auto& c = counters[static_cast<int>(family)];
  if ((c.executed.load(std::memory_order_relaxed) & target) != target) {
    c.executed.fetch_or(target, std::memory_order_relaxed);
  }
  if (c.highway_calls.load(std::memory_order_relaxed) == 0) {
    c.highway_calls.store(1, std::memory_order_relaxed);
  }
}

void record_fallback(Family family) {
  auto& c = counters[static_cast<int>(family)];
  if (c.fallback_calls.load(std::memory_order_relaxed) == 0) {
    c.fallback_calls.store(1, std::memory_order_relaxed);
  }
}

uint64_t sgemm_column_splits() {
  return column_splits.load(std::memory_order_relaxed);
}

void record_sgemm_column_split() {
  column_splits.fetch_add(1, std::memory_order_relaxed);
}

} // namespace mlx::core::cpu::highway_info

#endif // defined(MLX_USE_HIGHWAY_KERNELS)
