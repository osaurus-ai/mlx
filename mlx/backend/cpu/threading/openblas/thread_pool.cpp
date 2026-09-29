// Copyright © 2026 Apple Inc.
#if defined(MLX_USE_HIGHWAY_KERNELS)

#include "mlx/backend/cpu/threading/openblas/thread_pool.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/backend/cpu/threading/config.h"

#include <algorithm>
#include <cstdlib>

// Spin-wait hint: reduces power consumption and avoids starving sibling
// hyperthreads during busy-wait loops.
#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_ARM64) || defined(_M_ARM)
#define MLX_SPIN_PAUSE() __yield()
#else
#define MLX_SPIN_PAUSE() _mm_pause()
#endif
#elif defined(__SSE2__)
#include <immintrin.h>
#define MLX_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define MLX_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#define MLX_SPIN_PAUSE() ((void)0)
#endif

// OpenBLAS thread coordination -- pin BLAS to single-threaded at startup when
// OpenBLAS is the linked BLAS implementation.
#ifdef _WIN32
#include <windows.h>
static void (*blas_set_threads)(int) = nullptr;
static void init_blas_funcs() {
  static bool done = false;
  if (done)
    return;
  done = true;
  HMODULE h = GetModuleHandleA("libopenblas.dll");
  if (!h)
    h = GetModuleHandleA("openblas.dll");
  if (h) {
    blas_set_threads =
        (void (*)(int))GetProcAddress(h, "openblas_set_num_threads");
  }
}
static void set_blas_threads(int n) {
  init_blas_funcs();
  if (blas_set_threads)
    blas_set_threads(n);
}
#elif defined(__unix__) || defined(__linux__)
#include <dlfcn.h>
static void (*blas_set_threads)(int) = nullptr;
static void init_blas_funcs() {
  static bool done = false;
  if (done)
    return;
  done = true;
  blas_set_threads =
      (void (*)(int))dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
}
static void set_blas_threads(int n) {
  init_blas_funcs();
  if (blas_set_threads)
    blas_set_threads(n);
}
#else
static void (*blas_set_threads)(int) = nullptr;
static void init_blas_funcs() {}
static void set_blas_threads(int n) {
  (void)n;
}
#endif

static std::atomic<bool>& openblas_pinned_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}

namespace mlx::core::cpu {

namespace {
int get_default_threads() {
  return thread_config().threads;
}

// Set while this thread runs a slot of some parallel_for.
thread_local bool in_parallel_for = false;

std::atomic<detail::PoolTestHook> test_hook{nullptr};

// Calls the test hook if a test set one. Otherwise costs one relaxed load.
inline void test_point(detail::PoolTestPoint point, int worker) {
  if (auto hook = test_hook.load(std::memory_order_relaxed)) {
    hook(point, worker);
  }
}
} // namespace

void CPUThreadPool::run_slot(int slot, int nth) {
  in_parallel_for = true;
  try {
    (*task_ptr_)(slot, nth);
  } catch (...) {
    std::lock_guard<std::mutex> lk(error_mtx_);
    if (!first_error_) {
      first_error_ = std::current_exception();
    }
  }
  in_parallel_for = false;
}

CPUThreadPool::CPUThreadPool() : CPUThreadPool(get_default_threads()) {}

CPUThreadPool::CPUThreadPool(int threads)
    : max_threads_(std::clamp(threads, 1, MAX_WORKERS + 1)) {
  // Spawn max_threads_ - 1 workers. The main thread takes slot 0 in
  // parallel_for, so we only need (max_threads_ - 1) workers for the
  // remaining slots. This saves one thread of spin overhead.
  int n_workers = max_threads_ - 1;
  workers_.reserve(n_workers);
  for (int i = 0; i < n_workers; i++) {
    workers_.emplace_back([this, i] { worker_loop(i); });
  }
  // Wait for all workers to be initialized before allowing parallel_for.
  // This prevents a race where a late-starting worker misses the first
  // task notification and never sees the condition become true.
  while (ready_.load(std::memory_order_acquire) < n_workers) {
    MLX_SPIN_PAUSE();
  }
  // Look OpenBLAS up at any pool size, so that openblas_present() can answer.
  init_blas_funcs();
  // Pin OpenBLAS to one thread at every pool size, 1 included: cblas.cpp splits
  // SGEMMs across the pool, and MLX_CPU_THREADS is the one setting.
  set_blas_threads(1);
  openblas_pinned_flag().store(blas_set_threads != nullptr);
}

CPUThreadPool::~CPUThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    stop_ = true;
  }
  // Update per-worker flags and gen_ to wake all workers.
  uint64_t new_gen = gen_.fetch_add(1, std::memory_order_release) + 1;
  int n_workers = static_cast<int>(workers_.size());
  for (int i = 0; i < n_workers; i++) {
    worker_slots_[i].wake_gen.store(new_gen, std::memory_order_release);
  }
  cv_.notify_all();
  for (auto& w : workers_) {
    w.join();
  }
}

// Workers spin briefly on their private cache-line flag before falling back
// to cv_.wait. The spin window covers the typical inter-parallel_for gap
// during token generation (~30-50us), keeping workers ready for immediate
// dispatch without OS wakeup latency (~10-50us for futex).
static constexpr int WORKER_SPIN_COUNT = 32768; // ~160us at ~5ns/iter

void CPUThreadPool::claim_slot(int worker_id, uint64_t gen) {
  uint64_t c = claim_.load(std::memory_order_acquire);
  while ((c >> 32) == (gen & 0xffffffff)) {
    int nth = static_cast<int>((c >> 16) & 0xffff);
    int slot = static_cast<int>(c & 0xffff);
    if (slot >= nth) {
      return;
    }
    test_point(detail::PoolTestPoint::BeforeClaim, worker_id);
    if (claim_.compare_exchange_weak(
            c, c + 1, std::memory_order_acquire, std::memory_order_acquire)) {
      run_slot(slot, nth);
      done_.fetch_add(1, std::memory_order_acq_rel);
      return;
    }
  }
}

void CPUThreadPool::worker_loop(int worker_id) {
  // Read gen_ before this worker is ready: a call can start just after that.
  uint64_t my_gen = gen_.load(std::memory_order_acquire);
  // Signal that this worker is ready and waiting for tasks.
  ready_.fetch_add(1, std::memory_order_release);
  test_point(detail::PoolTestPoint::AfterReady, worker_id);

  while (true) {
    // Phase 1: Spin on per-worker flag (private cache line, no contention).
    // The main thread writes to each worker's flag after setting up the task.
    // The acquire load on wake_gen provides happens-before for all writes
    // the main thread did before the release store (task_ptr_, claim_, done_),
    // so we can access them without the mutex.
    bool woken_by_spin = false;
    for (int i = 0; i < WORKER_SPIN_COUNT; i++) {
      uint64_t wake =
          worker_slots_[worker_id].wake_gen.load(std::memory_order_acquire);
      if (wake > my_gen) {
        my_gen = wake;
        woken_by_spin = true;
        break;
      }
      MLX_SPIN_PAUSE();
    }

    if (woken_by_spin) {
      // Fast path: skip mutex entirely.
      if (stop_)
        return;
      claim_slot(worker_id, my_gen);
      continue;
    }

    // Phase 2: cv_.wait fallback for long idle periods.
    // Workers that exhaust the spin count sleep here until cv_.notify_one.
    {
      std::unique_lock<std::mutex> lk(mtx_);
      // Announce sleeping INSIDE mutex -- the main thread reads
      // sleeping_count_ under the same mutex to decide how many
      // cv_.notify_one calls to make. This gives an exact count.
      sleeping_count_.fetch_add(1, std::memory_order_relaxed);
      cv_.wait(lk, [&] {
        return gen_.load(std::memory_order_acquire) != my_gen || stop_;
      });
      sleeping_count_.fetch_sub(1, std::memory_order_relaxed);
      if (stop_)
        return;
      my_gen = gen_.load(std::memory_order_acquire);
      // Release mutex immediately -- task state is visible via gen_ acquire
      // (happens-before from parallel_for's gen_.fetch_add release).
      lk.unlock();
      claim_slot(worker_id, my_gen);
    }
  }
}

void CPUThreadPool::parallel_for(
    int n_threads,
    std::function<void(int tid, int nth)> f) {
  n_threads = std::min(std::max(n_threads, 1), max_threads_);
  if (n_threads == 1) {
    f(0, 1);
    return;
  }

  // Nested: run every slot inline, since the workers are busy with the outer
  // call and dispatch_mtx_ is not recursive. Callers size state by n_threads.
  if (in_parallel_for) {
    for (int slot = 0; slot < n_threads; ++slot) {
      f(slot, n_threads);
    }
    return;
  }

  // Serialize concurrent parallel_for calls from different CPU streams.
  // All task state (task_ptr_, claim_, done_, etc.) is shared, so
  // concurrent calls would corrupt each other. The second caller blocks
  // until the first completes -- this is correct because the workers are
  // shared and can only process one task at a time anyway.
  std::lock_guard<std::mutex> dispatch_lk(dispatch_mtx_);
  first_error_ = nullptr;

  int needed_workers = n_threads - 1;
  int n_workers = static_cast<int>(workers_.size());

  // -- NORMAL DISPATCH PATH -------------------------------------------
  // Used when workers are spinning on wake_gen or sleeping in cv_.wait.
  // Requires mutex for gen_ update (cv_ lost-wakeup safety) and
  // per-worker wake_gen writes.

  // Set up the task. The release store to claim_ below publishes it.
  task_ptr_ = &f;
  done_.store(0, std::memory_order_relaxed);

  int wake_count = std::min(needed_workers, n_workers);

  // Increment generation under the mutex to prevent lost-wakeup race.
  // Without the mutex, a worker between cv_.wait's predicate check and
  // the actual wait() call could miss the notify. The mutex serializes
  // gen_ update with the workers' predicate-to-wait transition.
  //
  uint64_t new_gen;
  int n_sleeping;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    n_sleeping = sleeping_count_.load(std::memory_order_relaxed);
    new_gen = gen_.fetch_add(1, std::memory_order_release) + 1;
    // Slot 0 is the caller's, so the workers start at slot 1.
    claim_.store(
        (new_gen << 32) | (static_cast<uint64_t>(n_threads) << 16) | 1,
        std::memory_order_release);
  }

  // Write per-worker wake flags -- spinning workers see these immediately.
  // Writing wake_gen only to needed workers avoids cache-line invalidations
  // on idle workers' slots.
  for (int i = 0; i < wake_count; i++) {
    worker_slots_[i].wake_gen.store(new_gen, std::memory_order_release);
  }

  // Wake sleeping workers. Using notify_all instead of Nxnotify_one
  // because a fast worker can finish its task, exhaust the spin loop,
  // and re-enter cv_.wait before all notify_one calls are sent -- causing
  // it to "absorb" a notification meant for another worker (lost wakeup).
  // notify_all is a single futex(FUTEX_WAKE, INT_MAX) syscall on Linux,
  // actually cheaper than 31x futex(FUTEX_WAKE, 1). Workers whose
  // my_gen already matches gen_ will re-check the predicate and go back
  // to sleep immediately (no spurious task execution).
  if (n_sleeping > 0) {
    cv_.notify_all();
  }

  // Main thread executes slot 0
  run_slot(0, n_threads);
  done_.fetch_add(1, std::memory_order_acq_rel);

  // Wait for workers -- spin then yield
  for (int i = 0; done_.load(std::memory_order_acquire) < n_threads; i++) {
    if (i < 4096) {
      MLX_SPIN_PAUSE();
    } else {
      std::this_thread::yield();
    }
  }

  if (first_error_) {
    std::exception_ptr error = first_error_;
    first_error_ = nullptr;
    std::rethrow_exception(error);
  }
}

int CPUThreadPool::max_threads() const {
  return max_threads_;
}

std::unique_ptr<ThreadPoolBackend> create_thread_pool_backend() {
  return std::make_unique<CPUThreadPool>();
}

namespace detail {
std::unique_ptr<ThreadPoolBackend> make_thread_pool(int threads) {
  return std::make_unique<CPUThreadPool>(threads);
}

void set_pool_test_hook(PoolTestHook hook) {
  test_hook.store(hook, std::memory_order_relaxed);
}
} // namespace detail

bool openblas_present() {
  // The pool's constructor runs init_blas_funcs, whose one-time guard is not
  // synchronised: let it happen there, once.
  ThreadPool::instance();
  return blas_set_threads != nullptr;
}

bool openblas_pinned() {
  ThreadPool::instance(); // the pool pins at construction
  return openblas_pinned_flag().load();
}

} // namespace mlx::core::cpu
#endif // defined(MLX_USE_HIGHWAY_KERNELS)
