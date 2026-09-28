// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "mlx/backend/cpu/threading/common.h"
#include "mlx/backend/cpu/threading/config.h"

namespace d = mlx::core::cpu::detail;

TEST_CASE("highway MLX_CPU_THREADS takes a positive integer") {
  std::string error;
  CHECK(d::parse_thread_count("8", error) == 8);
  CHECK(error.empty());
  for (const char* bad : {"0", "-3", "eight", "", "4x", "99999"}) {
    CAPTURE(std::string(bad));
    std::string e;
    CHECK_FALSE(d::parse_thread_count(bad, e).has_value());
    CHECK_FALSE(e.empty());
  }
  std::string unset;
  CHECK_FALSE(d::parse_thread_count(nullptr, unset).has_value());
  CHECK(unset.empty());
}

TEST_CASE("highway cgroup cpu.max becomes whole CPUs, rounded down") {
  CHECK_FALSE(d::parse_cpu_max("max 100000").has_value());
  CHECK(d::parse_cpu_max("200000 100000") == 2);
  CHECK(d::parse_cpu_max("150000 100000") == 1);
  CHECK(d::parse_cpu_max("50000 100000") == 1);
  CHECK(d::parse_cpu_max("400000 100000\n") == 4);
  CHECK_FALSE(d::parse_cpu_max("garbage").has_value());
}

TEST_CASE(
    "highway cgroup limit: the tightest cpu.max on the path to the root") {
  namespace fs = std::filesystem;
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root =
      fs::temp_directory_path() / ("mlx-cgroup-" + std::to_string(stamp));
  fs::create_directories(root / "a" / "b");
  std::ofstream(root / "a" / "cpu.max") << "300000 100000\n";
  std::ofstream(root / "a" / "b" / "cpu.max") << "max 100000\n";
  CHECK(d::cgroup_cpu_limit(root.string(), "0::/a/b\n") == 3);
  CHECK_FALSE(d::cgroup_cpu_limit(root.string(), "0::/\n").has_value());
  // Not absolute, and cgroup v1 only: no limit, and no endless walk.
  CHECK_FALSE(d::cgroup_cpu_limit(root.string(), "0::a/b\n").has_value());
  CHECK_FALSE(d::cgroup_cpu_limit(root.string(), "12:cpu:/a\n").has_value());
  std::ofstream(root / "a" / "b" / "cpu.max") << "150000 100000\n";
  CHECK(d::cgroup_cpu_limit(root.string(), "0::/a/b\n") == 1);
  fs::remove_all(root);
}

TEST_CASE("highway physical cores are counted within the affinity mask") {
  // Two cores with two SMT siblings each: CPUs 0 and 1 share a core.
  d::Topology smt = [](int cpu) -> std::optional<std::pair<int, int>> {
    return std::make_pair(0, cpu / 2);
  };
  CHECK(d::count_physical_cores({0, 1, 2, 3}, smt) == 2);
  CHECK(d::count_physical_cores({0, 2}, smt) == 2);
  CHECK(d::count_physical_cores({0, 1}, smt) == 1);
  d::Topology unknown = [](int) -> std::optional<std::pair<int, int>> {
    return std::nullopt;
  };
  CHECK(d::count_physical_cores({0, 1, 2}, unknown) == 3);
}

TEST_CASE("highway pool size: the override, else the tighter limit") {
  d::Topology cores = [](int cpu) -> std::optional<std::pair<int, int>> {
    return std::make_pair(0, cpu);
  };
  std::vector<int> eight = {0, 1, 2, 3, 4, 5, 6, 7};
  auto overridden = d::resolve_thread_config("3", eight, cores, 2);
  CHECK(overridden.threads == 3);
  CHECK(std::string(overridden.reason) == "MLX_CPU_THREADS");
  auto quota = d::resolve_thread_config(nullptr, eight, cores, 2);
  CHECK(quota.threads == 2);
  CHECK(std::string(quota.reason) == "cgroup cpu.max");
  auto unlimited =
      d::resolve_thread_config(nullptr, eight, cores, std::nullopt);
  CHECK(unlimited.threads == 8);
  CHECK(std::string(unlimited.reason) == "physical cores");
  auto bad = d::resolve_thread_config("zero", eight, cores, std::nullopt);
  CHECK(bad.threads == 8);
  CHECK_FALSE(bad.error.empty());
}

TEST_CASE("highway pool rethrows a worker's exception and stays usable") {
  auto& pool = cpu::ThreadPool::instance();
  const int n = pool.max_threads();
  if (n < 2) {
    return; // with one thread, no task runs on a worker
  }
  CHECK_THROWS_WITH_AS(
      pool.parallel_for(
          n,
          [](int tid, int nth) {
            if (tid == nth - 1) {
              throw std::runtime_error("worker failed");
            }
          }),
      "worker failed",
      std::runtime_error);
  std::atomic<int> slots{0};
  pool.parallel_for(n, [&](int, int) { slots.fetch_add(1); });
  CHECK(slots.load() == n);
}

TEST_CASE("highway pool waits for every slot when slot 0 throws") {
  auto& pool = cpu::ThreadPool::instance();
  const int n = pool.max_threads();
  if (n < 2) {
    return;
  }
  // Slot 0 runs on the calling thread. If its exception unwinds at once, the
  // workers are still running the task, and `finished` is short.
  std::atomic<int> finished{0};
  CHECK_THROWS_AS(
      pool.parallel_for(
          n,
          [&](int tid, int) {
            if (tid == 0) {
              throw std::runtime_error("slot 0 failed");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            finished.fetch_add(1);
          }),
      std::runtime_error);
  CHECK(finished.load() == n - 1);
}

TEST_CASE("highway nested parallel_for runs its range inline") {
  auto& pool = cpu::ThreadPool::instance();
  const int n = std::max(2, pool.max_threads());
  const int outer_slots = std::min(n, pool.max_threads());
  std::vector<std::atomic<int>> hits(64);
  pool.parallel_for(n, [&](int, int) {
    pool.parallel_for(n, [&](int tid, int nth) {
      for (int i = tid; i < 64; i += nth) {
        hits[i].fetch_add(1);
      }
    });
  });
  for (auto& h : hits) {
    CHECK(h.load() == outer_slots);
  }
}

TEST_CASE("highway pool pins OpenBLAS at every pool size") {
  const int pool = cpu::ThreadPool::instance().max_threads();
  if (!cpu::openblas_present()) {
    MESSAGE("skipped: the linked BLAS has no openblas_set_num_threads");
    return;
  }
  CAPTURE(pool);
  CHECK(cpu::openblas_pinned());
#if defined(__linux__)
  // Ask OpenBLAS itself, not the flag stored next to the call.
  using GetThreads = int (*)();
  auto get = reinterpret_cast<GetThreads>(
      dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
  REQUIRE(get != nullptr);
  CHECK(get() == 1);
#endif
}

TEST_CASE("highway pool size agrees with the thread configuration") {
  const auto& config = cpu::thread_config();
  const int pool = cpu::ThreadPool::instance().max_threads();
  MESSAGE(
      "threads " << config.threads << " (" << std::string(config.reason)
                 << "), pool " << pool);
  // The pool clamps to MAX_WORKERS + 1 (thread_pool.h; 129 in #3019).
  CHECK(pool == std::min(config.threads, 129));
}
