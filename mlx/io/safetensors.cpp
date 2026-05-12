// Copyright © 2023 Apple Inc.
//
#include <json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_set>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "mlx/allocator.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/io.h"
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

using json = nlohmann::json;

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
#define ST_F8_E4M3 "F8_E4M3"

// Note: Complex numbers aren't in the spec yet so this could change -
// https://github.com/huggingface/safetensors/issues/389
#define ST_C64 "C64"

namespace mlx::core {

std::string dtype_to_safetensor_str(Dtype t) {
  switch (t) {
    case float32:
      return ST_F32;
    case bfloat16:
      return ST_BF16;
    case float16:
      return ST_F16;
    case int64:
      return ST_I64;
    case int32:
      return ST_I32;
    case int16:
      return ST_I16;
    case int8:
      return ST_I8;
    case uint64:
      return ST_U64;
    case uint32:
      return ST_U32;
    case uint16:
      return ST_U16;
    case uint8:
      return ST_U8;
    case bool_:
      return ST_BOOL;
    case complex64:
      return ST_C64;
    default:
      throw std::runtime_error("[save_safetensors] received invalid dtype.");
  }
}

Dtype dtype_from_safetensor_str(std::string_view str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else if (str == ST_C64) {
    return complex64;
  } else if (str == ST_F8_E4M3) {
    return uint8;
  } else {
    throw std::runtime_error(
        "[safetensor] unsupported dtype " + std::string(str));
  }
}

#ifndef _WIN32
namespace {

bool env_truthy(const char* key) {
  auto raw = std::getenv(key);
  if (!raw) {
    return false;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value == "1" || value == "true" || value == "on" ||
      value == "yes";
}

bool mmap_safetensors_enabled() {
  return env_truthy("MLX_SAFETENSORS_MMAP") ||
      env_truthy("VMLINUX_MMAP_SAFETENSORS");
}

struct MmapShard {
  void* base{nullptr};
  size_t size{0};
  std::string path;

  MmapShard(void* base, size_t size, std::string path)
      : base(base), size(size), path(std::move(path)) {}

  MmapShard(const MmapShard&) = delete;
  MmapShard& operator=(const MmapShard&) = delete;

  ~MmapShard() {
    if (base && base != MAP_FAILED && size > 0) {
      munmap(base, size);
    }
  }
};

struct ParsedRoutedName {
  int32_t layer;
  int32_t expert;
  bool stacked;
};

std::optional<ParsedRoutedName> match_routed_name(const std::string& name) {
  static const std::string vl_prefix = R"((?:(?:model|language_model)\.)*)";
  static const std::vector<std::regex> per_expert = {
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate|up|down)_proj\.(?:weight|tq_packed|tq_norms)$)"),
      std::regex(
          R"(^layers\.(\d+)\.ffn\.experts\.(\d+)\.(?:w[123]|(?:gate|up|down)_proj)\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          R"(^layers\.(\d+)\.ffn\.experts\.(\d+)\.(?:w[123]|(?:gate|up|down)_proj)\.weight$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.weight$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.weight$)")};
  static const std::vector<std::regex> stacked = {
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.weight$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.tq_packed$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.weight$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.block_sparse_moe\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.fc[12]\.(?:weight|tq_packed|tq_norms)$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.(?:gate|up|down)_proj\.weight$)"),
      std::regex(
          R"(^layers\.(\d+)\.ffn\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$)")};

  std::smatch match;
  for (const auto& regex : per_expert) {
    if (std::regex_match(name, match, regex) && match.size() >= 3) {
      return ParsedRoutedName{
          static_cast<int32_t>(std::stoi(match[1].str())),
          static_cast<int32_t>(std::stoi(match[2].str())),
          false};
    }
  }
  for (const auto& regex : stacked) {
    if (std::regex_match(name, match, regex) && match.size() >= 2) {
      return ParsedRoutedName{
          static_cast<int32_t>(std::stoi(match[1].str())),
          0,
          true};
    }
  }
  return std::nullopt;
}

struct MmapTensorRegion {
  std::weak_ptr<MmapShard> shard;
  int32_t layer;
  int32_t expert;
  size_t offset;
  size_t length;
};

struct LiveMmapTensorRegion {
  std::shared_ptr<MmapShard> shard;
  int32_t layer;
  int32_t expert;
  size_t offset;
  size_t length;
};

class SafetensorsMmapRegistry {
 public:
  static SafetensorsMmapRegistry& instance() {
    static SafetensorsMmapRegistry registry;
    return registry;
  }

  void register_region(
      const std::shared_ptr<MmapShard>& shard,
      int32_t layer,
      int32_t expert,
      size_t offset,
      size_t length) {
    std::lock_guard<std::mutex> lock(mutex_);
    regions_.push_back(MmapTensorRegion{
        std::weak_ptr<MmapShard>(shard), layer, expert, offset, length});
  }

  int64_t advise_routed(int32_t advice, int32_t cold_pct) {
    auto regions = live_regions();
    if (regions.empty()) {
      return 0;
    }

    if (advice != 0) {
      return advise_regions(regions, advice);
    }

    cold_pct = std::max<int32_t>(0, std::min<int32_t>(100, cold_pct));
    if (cold_pct == 0) {
      return 0;
    }
    if (cold_pct == 100) {
      return advise_regions(regions, advice);
    }

    std::sort(regions.begin(), regions.end(), [](const auto& lhs, const auto& rhs) {
      return std::tie(lhs.layer, lhs.expert, lhs.offset, lhs.length) <
          std::tie(rhs.layer, rhs.expert, rhs.offset, rhs.length);
    });

    std::unordered_map<int32_t, std::vector<int32_t>> experts_by_layer;
    for (const auto& region : regions) {
      auto& experts = experts_by_layer[region.layer];
      if (std::find(experts.begin(), experts.end(), region.expert) ==
          experts.end()) {
        experts.push_back(region.expert);
      }
    }
    std::unordered_set<int64_t> cold_pairs;
    for (auto& [layer, experts] : experts_by_layer) {
      std::sort(experts.begin(), experts.end());
      const auto hot_pct = 100 - cold_pct;
      const auto hot_count = static_cast<size_t>(
          (static_cast<int64_t>(experts.size()) * hot_pct + 99) / 100);
      for (size_t i = hot_count; i < experts.size(); ++i) {
        cold_pairs.insert(pair_key(layer, experts[i]));
      }
    }

    std::vector<LiveMmapTensorRegion> selected;
    selected.reserve(regions.size());
    for (const auto& region : regions) {
      if (cold_pairs.contains(pair_key(region.layer, region.expert))) {
        selected.push_back(region);
      }
    }
    return advise_regions(selected, advice);
  }

  int64_t advise_experts(
      int32_t advice,
      const int32_t* layers,
      const int32_t* experts,
      int64_t count) {
    if (count <= 0 || layers == nullptr || experts == nullptr) {
      return 0;
    }
    std::unordered_set<int64_t> requested;
    requested.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      requested.insert(pair_key(layers[i], experts[i]));
    }

    std::vector<LiveMmapTensorRegion> selected;
    for (auto& region : live_regions()) {
      if (requested.contains(pair_key(region.layer, region.expert))) {
        selected.push_back(std::move(region));
      }
    }
    return advise_regions(selected, advice);
  }

 private:
  static int64_t pair_key(int32_t layer, int32_t expert) {
    return (static_cast<int64_t>(layer) << 32) ^
        static_cast<uint32_t>(expert);
  }

  std::vector<LiveMmapTensorRegion> live_regions() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<LiveMmapTensorRegion> live;
    live.reserve(regions_.size());
    auto write = regions_.begin();
    for (auto read = regions_.begin(); read != regions_.end(); ++read) {
      if (auto shard = read->shard.lock()) {
        live.push_back(LiveMmapTensorRegion{
            std::move(shard),
            read->layer,
            read->expert,
            read->offset,
            read->length});
        *write++ = *read;
      }
    }
    regions_.erase(write, regions_.end());
    return live;
  }

  static int64_t advise_regions(
      const std::vector<LiveMmapTensorRegion>& regions,
      int32_t advice) {
    int64_t advised = 0;
    for (const auto& region : regions) {
      advised += advise_region(region, advice);
    }
    return advised;
  }

  static int64_t advise_region(const LiveMmapTensorRegion& region, int32_t advice) {
    if (!region.shard || region.length == 0 || region.offset >= region.shard->size) {
      return 0;
    }
    const auto clamped_length =
        std::min(region.length, region.shard->size - region.offset);
    const auto page = static_cast<uintptr_t>(getpagesize());
    const auto base = reinterpret_cast<uintptr_t>(region.shard->base);
    const auto start = base + region.offset;
    const auto end = start + clamped_length;
    const auto aligned_start = start & ~(page - 1);
    const auto aligned_end = (end + page - 1) & ~(page - 1);
    if (aligned_end <= aligned_start) {
      return 0;
    }
    const auto os_advice = advice == 1 ? MADV_WILLNEED : MADV_DONTNEED;
    if (madvise(
            reinterpret_cast<void*>(aligned_start),
            aligned_end - aligned_start,
            os_advice) != 0) {
      return 0;
    }
    return static_cast<int64_t>(clamped_length);
  }

  std::mutex mutex_;
  std::vector<MmapTensorRegion> regions_;
};

std::optional<SafetensorsLoad> load_safetensors_mmap(
    const std::string& file,
    StreamOrDevice) {
  int fd = open(file.c_str(), O_RDONLY | O_BINARY);
  if (fd < 0) {
    return std::nullopt;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    return std::nullopt;
  }
  const auto file_size = static_cast<size_t>(st.st_size);
  void* raw = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (raw == MAP_FAILED) {
    return std::nullopt;
  }

  auto unmap_on_failure = [&]() {
    if (raw && raw != MAP_FAILED) {
      munmap(raw, file_size);
      raw = nullptr;
    }
  };

  if (file_size < 8) {
    unmap_on_failure();
    return std::nullopt;
  }

  const auto* base = static_cast<const char*>(raw);
  uint64_t json_header_length = 0;
  std::memcpy(&json_header_length, base, sizeof(json_header_length));
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  if (json_header_length == 0 ||
      json_header_length >= kMaxJsonHeaderLength ||
      8 + json_header_length > file_size) {
    unmap_on_failure();
    return std::nullopt;
  }

  json metadata;
  try {
    metadata = json::parse(base + 8, base + 8 + json_header_length);
  } catch (...) {
    unmap_on_failure();
    return std::nullopt;
  }
  if (!metadata.is_object()) {
    unmap_on_failure();
    return std::nullopt;
  }

  auto buffer = allocator::make_buffer(raw, file_size);
  if (buffer.ptr() == nullptr) {
    unmap_on_failure();
    return std::nullopt;
  }

  auto shard = std::make_shared<MmapShard>(raw, file_size, file);
  raw = nullptr;
  array base_array(
      buffer,
      Shape{1},
      uint8,
      [shard](allocator::Buffer buffer) {
        allocator::release(buffer);
      });

  const size_t data_start = static_cast<size_t>(json_header_length) + 8;
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }

    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    if (data_offsets.size() != 2 || data_offsets[1] < data_offsets[0]) {
      unmap_on_failure();
      return std::nullopt;
    }

    Dtype type = dtype_from_safetensor_str(dtype);
    const auto tensor_offset = data_start + data_offsets[0];
    const auto tensor_length = data_offsets[1] - data_offsets[0];
    if (tensor_offset > file_size || tensor_length > file_size - tensor_offset) {
      return std::nullopt;
    }
    if (tensor_offset % size_of(type) != 0) {
      return std::nullopt;
    }

    array tensor(
        allocator::Buffer(nullptr),
        shape,
        type,
        [](allocator::Buffer) {});
    tensor.copy_shared_buffer(
        base_array,
        tensor.strides(),
        tensor.flags(),
        tensor.size(),
        static_cast<int64_t>(tensor_offset / size_of(type)));
    res.insert({item.key(), tensor});

    if (auto routed = match_routed_name(item.key())) {
      if (routed->stacked && !shape.empty() && shape[0] > 1) {
        const auto experts = static_cast<size_t>(shape[0]);
        if (experts > 0 && tensor_length % experts == 0) {
          const auto per_expert = tensor_length / experts;
          for (size_t expert = 0; expert < experts; ++expert) {
            SafetensorsMmapRegistry::instance().register_region(
                shard,
                routed->layer,
                static_cast<int32_t>(expert),
                tensor_offset + expert * per_expert,
                per_expert);
          }
        } else {
          SafetensorsMmapRegistry::instance().register_region(
              shard,
              routed->layer,
              routed->expert,
              tensor_offset,
              tensor_length);
        }
      } else {
        SafetensorsMmapRegistry::instance().register_region(
            shard,
            routed->layer,
            routed->expert,
            tensor_offset,
            tensor_length);
      }
    }
  }
  return SafetensorsLoad{std::move(res), std::move(metadata_map)};
}

} // namespace
#endif

int64_t safetensors_mmap_advise_routed(int32_t advice, int32_t cold_pct) {
#ifndef _WIN32
  return SafetensorsMmapRegistry::instance().advise_routed(advice, cold_pct);
#else
  return 0;
#endif
}

int64_t safetensors_mmap_advise_experts(
    int32_t advice,
    const int32_t* layers,
    const int32_t* experts,
    int64_t count) {
#ifndef _WIN32
  return SafetensorsMmapRegistry::instance().advise_experts(
      advice, layers, experts, count);
#else
  return 0;
#endif
}

/** Load array from reader in safetensor format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error(
        "[load_safetensors] Failed to open " + in_stream->label());
  }

  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);

  uint64_t jsonHeaderLength = 0;
  // This is the same limit as in the original Rust Safetensors code.
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json header length " + in_stream->label());
  }
  // Load the json metadata
  auto rawJson = std::make_unique<char[]>(jsonHeaderLength);
  in_stream->read(rawJson.get(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.get(), rawJson.get() + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json metadata " + in_stream->label());
  }
  size_t offset = jsonHeaderLength + 8;
  // Load the arrays using metadata
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    res.insert(
        {item.key(),
         array(
             shape,
             type,
             std::make_shared<Load>(
                 stream, in_stream, offset + data_offsets.at(0), false),
             std::vector<array>{})});
  }
  return {res, metadata_map};
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
#ifndef _WIN32
  if (mmap_safetensors_enabled()) {
    if (auto loaded = load_safetensors_mmap(file, s)) {
      return *std::move(loaded);
    }
  }
#endif
  return load_safetensors(std::make_shared<io::ParallelFileReader>(file), s);
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error(
        "[save_safetensors] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  json _metadata;
  for (auto& [key, value] : metadata) {
    _metadata[key] = value;
  }
  parent["__metadata__"] = _metadata;

  {
    std::vector<array> to_eval;
    to_eval.reserve(a.size());
    for (auto& p : a) {
      p.second = contiguous(p.second);
      to_eval.push_back(p.second);
    }
    eval(std::move(to_eval));
  }

  size_t offset = 0;
  for (auto& [key, arr] : a) {
    if (arr.nbytes() == 0) {
      throw std::invalid_argument(
          "[save_safetensors] cannot serialize an empty array key: " + key);
    }

    json child;
    child["dtype"] = dtype_to_safetensor_str(arr.dtype());
    child["shape"] = arr.shape();
    child["data_offsets"] = std::vector<size_t>{offset, offset + arr.nbytes()};
    parent[key] = child;
    offset += arr.nbytes();
  }

  auto header = parent.dump();
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    out_stream->write(arr.data<char>(), arr.nbytes());
  }
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensors(
      std::make_shared<io::FileWriter>(std::move(file)), a, metadata);
}

} // namespace mlx::core
