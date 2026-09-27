// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if defined(MLX_USE_HIGHWAY_KERNELS)

#include "mlx/backend/cpu/threading/config.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

// Physical core detection -- AVX2/FMA workloads get no benefit from SMT
// (hyperthreads share the same SIMD execution units, L1/L2 cache, and memory
// bandwidth). Using physical core count avoids over-subscription and reduces
// atomic/mutex contention in the thread pool. Benchmarked: 16 physical cores
// is +5-7% faster than logical core count for quantized LLM inference.
#if defined(_WIN32)
#include <windows.h>
static int get_physical_cores() {
  DWORD len = 0;
  GetLogicalProcessorInformation(nullptr, &len);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    return 0;
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
      len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  if (!GetLogicalProcessorInformation(buf.data(), &len))
    return 0;
  int cores = 0;
  for (auto& info : buf) {
    if (info.Relationship == RelationProcessorCore)
      cores++;
  }
  return cores;
}
#endif

namespace mlx::core::cpu {

namespace detail {

std::optional<int> parse_thread_count(const char* value, std::string& error) {
  if (value == nullptr) {
    return std::nullopt;
  }
  const std::string text(value);
  char* end = nullptr;
  errno = 0;
  const long n = std::strtol(text.c_str(), &end, 10);
  if (text.empty() || *end != '\0' || errno != 0 || n <= 0 || n > 4096) {
    error = "MLX_CPU_THREADS must be a positive integer up to 4096, got \"" +
        text + "\"";
    return std::nullopt;
  }
  return static_cast<int>(n);
}

std::optional<int> parse_cpu_max(std::string_view contents) {
  std::istringstream in{std::string(contents)};
  std::string quota;
  long long period = 0;
  if (!(in >> quota >> period) || period <= 0 || quota == "max") {
    return std::nullopt;
  }
  char* end = nullptr;
  const long long q = std::strtoll(quota.c_str(), &end, 10);
  if (*end != '\0' || q <= 0) {
    return std::nullopt;
  }
  return static_cast<int>(std::max<long long>(1, q / period));
}

int count_physical_cores(
    const std::vector<int>& cpus,
    const Topology& topology) {
  std::set<std::pair<int, int>> cores;
  int unknown = 0;
  for (int cpu : cpus) {
    if (auto t = topology(cpu)) {
      cores.insert(*t);
    } else {
      ++unknown;
    }
  }
  return static_cast<int>(cores.size()) + unknown;
}

std::optional<int> cgroup_cpu_limit(
    const std::string& root,
    std::string_view self_cgroup) {
  std::string path;
  std::istringstream lines{std::string(self_cgroup)};
  for (std::string line; std::getline(lines, line);) {
    if (line.rfind("0::", 0) == 0) {
      path = line.substr(3);
      break;
    }
  }
  if (path.empty() || path[0] != '/') {
    return std::nullopt; // the walk below needs an absolute path to end
  }
  std::optional<int> limit;
  std::string dir = root + (path == "/" ? std::string() : path);
  while (true) {
    std::ifstream max_file(dir + "/cpu.max");
    std::string contents;
    if (max_file && std::getline(max_file, contents)) {
      if (auto n = parse_cpu_max(contents)) {
        limit = limit ? std::min(*limit, *n) : *n;
      }
    }
    if (dir.size() <= root.size()) {
      break;
    }
    dir = dir.substr(0, dir.find_last_of('/'));
  }
  return limit;
}

ThreadConfig resolve_thread_config(
    const char* override_value,
    const std::vector<int>& allowed_cpus,
    const Topology& topology,
    std::optional<int> cgroup_limit) {
  ThreadConfig config{1, "physical cores", {}};
  if (auto n = parse_thread_count(override_value, config.error)) {
    config.threads = *n;
    config.reason = "MLX_CPU_THREADS";
    return config;
  }
  config.threads = std::max(1, count_physical_cores(allowed_cpus, topology));
  if (cgroup_limit && *cgroup_limit < config.threads) {
    config.threads = *cgroup_limit;
    config.reason = "cgroup cpu.max";
  }
  return config;
}

} // namespace detail

namespace {

std::vector<int> allowed_cpus() {
  std::vector<int> cpus;
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int i = 0; i < CPU_SETSIZE; ++i) {
      if (CPU_ISSET(i, &set)) {
        cpus.push_back(i);
      }
    }
  }
#endif
  if (cpus.empty()) {
    const int n = std::max(1u, std::thread::hardware_concurrency());
    for (int i = 0; i < n; ++i) {
      cpus.push_back(i);
    }
  }
  return cpus;
}

std::optional<std::pair<int, int>> sysfs_topology(int cpu) {
  const std::string base =
      "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
  std::ifstream core_file(base + "core_id");
  int core = 0;
  int package = 0;
  if (!(core_file >> core)) {
    return std::nullopt;
  }
  std::ifstream package_file(base + "physical_package_id");
  package_file >> package;
  return std::make_pair(package, core);
}

std::string read_file(const char* path) {
  std::ifstream in(path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

} // namespace

const ThreadConfig& thread_config() {
  static const ThreadConfig config = [] {
    ThreadConfig resolved = detail::resolve_thread_config(
        std::getenv("MLX_CPU_THREADS"),
        allowed_cpus(),
        sysfs_topology,
        detail::cgroup_cpu_limit(
            "/sys/fs/cgroup", read_file("/proc/self/cgroup")));
#if defined(_WIN32)
    // No sysfs topology on Windows: count its cores as #3019 did.
    if (std::string_view(resolved.reason) == "physical cores") {
      if (const int cores = get_physical_cores(); cores > 0) {
        resolved.threads = cores;
      }
    }
#endif
    return resolved;
  }();
  return config;
}

} // namespace mlx::core::cpu

#endif // defined(MLX_USE_HIGHWAY_KERNELS)
