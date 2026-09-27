// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "mlx/api.h"

namespace mlx::core::cpu::highway_info {

// Kernel families with dynamically dispatched Highway kernels.
enum class Family : int {
  QmmAffineDequant = 0,
  QmmAffineInt8,
  QmmFp,
  RmsNorm,
  LayerNorm,
  Rope,
  Sdpa,
  Count
};

// Highway targets compiled into this build's dispatched kernels (HWY_* bits).
MLX_API int64_t compiled_targets();
// Highway targets this CPU supports, as dispatch sees them (after a test mask).
MLX_API int64_t supported_targets();
// Restricts dispatch to `targets`, which must be supported; 0 restores the
// CPU's own set. For tests: process-wide, and not safe while kernels run.
MLX_API void set_targets_for_test(int64_t targets);
// Highway's name for one target bit, such as "AVX2"; "Unknown" for a bit it
// does not know.
MLX_API const char* target_name(int64_t target);

// What a family's kernels did since the last reset_stats(). `executed` is the
// OR of every target whose kernel ran. `highway_calls` is 1 once a dispatched
// kernel ran, and `fallback_calls` once an entry point sent a call to code that
// is not dynamically dispatched: both count to 1 and stop, so that recording
// in a row loop writes shared memory only once.
struct FamilyStats {
  int64_t executed;
  uint64_t highway_calls;
  uint64_t fallback_calls;
};
MLX_API FamilyStats stats(Family family);
MLX_API void reset_stats();

MLX_API void record(Family family, int64_t target);
MLX_API void record_fallback(Family family);

// fp32 GEMMs split by columns across the pool (gemms/cblas.cpp), since the
// last reset_stats().
MLX_API uint64_t sgemm_column_splits();
MLX_API void record_sgemm_column_split();

} // namespace mlx::core::cpu::highway_info

// For the kernels. MLX_HIGHWAY_RECORD goes inside HWY_NAMESPACE code, where
// HWY_TARGET is the target that code was compiled for. Fully qualified,
// because the kernels' own namespaces hold an unrelated `highway`.
#define MLX_HIGHWAY_RECORD(family)        \
  ::mlx::core::cpu::highway_info::record( \
      ::mlx::core::cpu::highway_info::Family::family, HWY_TARGET)
#define MLX_HIGHWAY_RECORD_FALLBACK(family)        \
  ::mlx::core::cpu::highway_info::record_fallback( \
      ::mlx::core::cpu::highway_info::Family::family)
