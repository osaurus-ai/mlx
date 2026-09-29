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
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
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

namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::seconds;

// Runs `call` on a new thread and waits at most `limit` for it. If the call
// does not return, its thread stays: what the call uses must stay too.
bool returns_within(std::function<void()> call, seconds limit) {
  auto returned = std::make_shared<std::promise<void>>();
  auto future = returned->get_future();
  std::thread caller([call = std::move(call), returned] {
    call();
    returned->set_value();
  });
  if (future.wait_for(limit) != std::future_status::ready) {
    caller.detach();
    return false;
  }
  caller.join();
  return true;
}

// Holds a worker after it told the pool that it is ready, until the first
// call runs its slot 0.
struct LateWorker {
  std::mutex mtx;
  std::condition_variable cv;
  bool first_call = false;
  int held = 0;
  std::atomic<int> slots{0};
} late;

void hold_after_ready(d::PoolTestPoint point, int) {
  if (point != d::PoolTestPoint::AfterReady) {
    return;
  }
  std::unique_lock<std::mutex> lk(late.mtx);
  ++late.held;
  late.cv.wait_for(lk, seconds(2), [] { return late.first_call; });
}

// The slots of one call, as they ran. Static: a call that returns too early
// must not free what its slots still use.
struct SlotLog {
  void reset(int slots) {
    n = slots;
    for (auto& r : runs) {
      r = 0;
    }
    wrong = 0;
    wrong_tid = -1;
    wrong_nth = -1;
    finished = 0;
    at_return = -1;
  }
  void ran(int tid, int nth) {
    if (tid < 0 || tid >= n || nth != n) {
      wrong_tid = tid;
      wrong_nth = nth;
      wrong.fetch_add(1);
    } else {
      runs[tid].fetch_add(1);
    }
    finished.fetch_add(1);
  }
  int n = 0;
  std::atomic<int> runs[8];
  std::atomic<int> wrong{0};
  std::atomic<int> wrong_tid{-1};
  std::atomic<int> wrong_nth{-1};
  std::atomic<int> finished{0};
  int at_return = -1;
} first_log, second_log;

// Worker 2 gets no wake flag for either call below: it wakes from its sleep
// for the first call, and the hook holds its claim until the second call runs.
constexpr int kStaleWorker = 2;
struct StaleClaim {
  std::mutex mtx;
  std::condition_variable cv;
  // 0: off. 1: first call. 2: second call next. 3: worker 2 may go. 4: all.
  int phase = 0;
  bool held = false;
  Clock::time_point deadline;
} stale;

void set_phase(int phase) {
  std::lock_guard<std::mutex> lk(stale.mtx);
  stale.phase = phase;
  stale.cv.notify_all();
}

void hold_stale_claim(d::PoolTestPoint point, int worker) {
  if (point != d::PoolTestPoint::BeforeClaim) {
    return;
  }
  std::unique_lock<std::mutex> lk(stale.mtx);
  if (stale.phase == 1 && worker == kStaleWorker && !stale.held) {
    stale.held = true;
    stale.cv.notify_all();
    stale.cv.wait_for(
        lk, seconds(5), [] { return stale.phase == 0 || stale.phase >= 3; });
  } else if (stale.phase == 1 && worker != kStaleWorker) {
    // Worker 2 must see a free slot of the first call.
    stale.cv.wait_until(
        lk, stale.deadline, [] { return stale.held || stale.phase != 1; });
  } else if ((stale.phase == 2 || stale.phase == 3) && worker != kStaleWorker) {
    // The second call's other claims come after the held claim.
    stale.cv.wait_for(
        lk, seconds(5), [] { return stale.phase == 0 || stale.phase >= 4; });
  }
}

void check_log(const SlotLog& log, const std::string& call) {
  INFO("the " << call << " call, " << log.n << " slots");
  CHECK_MESSAGE(
      log.wrong.load() == 0,
      "slot " << log.wrong_tid.load() << " ran with nth "
              << log.wrong_nth.load());
  for (int t = 0; t < log.n; ++t) {
    CHECK_MESSAGE(
        log.runs[t].load() == 1,
        "slot " << t << " ran " << log.runs[t].load() << " times");
  }
  CHECK_MESSAGE(
      log.at_return == log.n,
      "parallel_for returned after " << log.at_return << " slots finished");
}

} // namespace

TEST_CASE(
    "highway pool worker delayed after announcing ready takes the first call") {
  {
    std::lock_guard<std::mutex> lk(late.mtx);
    late.first_call = false;
    late.held = 0;
  }
  late.slots = 0;
  d::set_pool_test_hook(hold_after_ready);
  auto pool = d::make_thread_pool(2); // One worker.
  auto* p = pool.get();
  const bool returned = returns_within(
      [p] {
        p->parallel_for(2, [](int tid, int) {
          if (tid == 0) {
            std::lock_guard<std::mutex> lk(late.mtx);
            late.first_call = true;
            late.cv.notify_all();
          }
          late.slots.fetch_add(1);
        });
      },
      seconds(5));
  d::set_pool_test_hook(nullptr);
  CHECK_MESSAGE(returned, "the first parallel_for did not return in 5 s");
  if (!returned) {
    (void)pool.release(); // Its call still waits for the worker.
    return;
  }
  CHECK(late.slots.load() == 2);
  std::lock_guard<std::mutex> lk(late.mtx);
  CHECK(late.held == 1);
}

TEST_CASE("highway pool claim delayed past its call does not run in the next") {
  const int calls[2][2] = {{3, 2}, {2, 3}};
  for (const auto& c : calls) {
    const int n1 = c[0];
    const int n2 = c[1];
    CAPTURE(n1);
    CAPTURE(n2);
    bool held = false;
    for (int attempt = 0; attempt < 5 && !held; ++attempt) {
      auto pool = d::make_thread_pool(4); // Workers 0, 1 and 2.
      std::this_thread::sleep_for(milliseconds(100)); // Let them all sleep.
      {
        std::lock_guard<std::mutex> lk(stale.mtx);
        stale.phase = 1;
        stale.held = false;
        stale.deadline = Clock::now() + seconds(1);
      }
      first_log.reset(n1);
      second_log.reset(n2);
      d::set_pool_test_hook(hold_stale_claim);
      auto* p = pool.get();
      bool returned = returns_within(
          [p, n1] {
            p->parallel_for(
                n1, [](int tid, int nth) { first_log.ran(tid, nth); });
            first_log.at_return = first_log.finished.load();
          },
          seconds(10));
      {
        std::lock_guard<std::mutex> lk(stale.mtx);
        held = stale.held;
      }
      if (returned && held) {
        set_phase(2);
        returned = returns_within(
            [p, n2] {
              p->parallel_for(n2, [](int tid, int nth) {
                if (tid == 0) {
                  set_phase(3); // The held claim goes first.
                  std::this_thread::sleep_for(milliseconds(50));
                  set_phase(4);
                } else {
                  std::this_thread::sleep_for(milliseconds(20));
                }
                second_log.ran(tid, nth);
              });
              second_log.at_return = second_log.finished.load();
            },
            seconds(10));
      }
      d::set_pool_test_hook(nullptr);
      set_phase(0);
      CHECK_MESSAGE(returned, "a parallel_for did not return in 10 s");
      if (!returned) {
        (void)pool.release(); // A call still waits in it.
        return;
      }
      check_log(first_log, "first");
      if (held) {
        check_log(second_log, "second");
      }
    }
    CHECK_MESSAGE(held, "no attempt held a claim of the first call");
  }
}
