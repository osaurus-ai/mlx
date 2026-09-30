// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mlx/api.h"
#include "mlx/backend/cpu/threading/base.h"

namespace mlx::core::cpu {

// How many threads the CPU pool runs, and why.
struct ThreadConfig {
  int threads;
  // "MLX_CPU_THREADS", "cgroup cpu.max" or "physical cores".
  const char* reason;
  // Set when MLX_CPU_THREADS is not a positive integer: the pool then uses its
  // default, and keeps the message here for a caller to report.
  std::string error;
};

// The pool's configuration, resolved once per process.
MLX_API const ThreadConfig& thread_config();

// Whether OpenBLAS's thread setter was found, and whether the pool pinned
// OpenBLAS to one thread (it does at every pool size).
MLX_API bool openblas_present();
MLX_API bool openblas_pinned();

namespace detail {
// A positive integer up to 4096, or nullopt; `error` is set for a value that is
// present but invalid, and left alone for nullptr.
MLX_API std::optional<int> parse_thread_count(
    const char* value,
    std::string& error);
// cgroup v2 cpu.max, "<quota> <period>" or "max <period>": floor(quota /
// period), at least 1. nullopt without a limit, or for text it cannot read.
MLX_API std::optional<int> parse_cpu_max(std::string_view contents);
// Distinct topology keys among `cpus`. A CPU with no topology counts as a core
// of its own.
using Topology = std::function<std::optional<std::pair<int, int>>(int cpu)>;
MLX_API int count_physical_cores(
    const std::vector<int>& cpus,
    const Topology& topology);
// The topology key of `cpu`, read under `root`: its package and the first
// CPU of its core's CPU list. nullopt where neither list can be read.
MLX_API std::optional<std::pair<int, int>> sysfs_topology(
    const std::string& root,
    int cpu);
// The tightest cgroup v2 cpu.max from the "0::" line's cgroup up to `root`;
// nullopt without that line, for a relative path, or without a limit.
MLX_API std::optional<int> cgroup_cpu_limit(
    const std::string& root,
    std::string_view self_cgroup);
// The override if valid; else the physical cores among `allowed_cpus`, capped
// by `cgroup_limit`.
MLX_API ThreadConfig resolve_thread_config(
    const char* override_value,
    const std::vector<int>& allowed_cpus,
    const Topology& topology,
    std::optional<int> cgroup_limit);

// For tests: a pool of `threads` threads, the calling thread included.
MLX_API std::unique_ptr<ThreadPoolBackend> make_thread_pool(int threads);
// For tests: the points where a pool worker calls the test hook.
enum class PoolTestPoint {
  AfterReady, // The worker has told the pool that it is ready.
  BeforeClaim, // The worker has seen a free slot, and not yet claimed it.
};
using PoolTestHook = void (*)(PoolTestPoint point, int worker);
// Process-wide. Null, the default, removes the hook.
MLX_API void set_pool_test_hook(PoolTestHook hook);
} // namespace detail

} // namespace mlx::core::cpu
