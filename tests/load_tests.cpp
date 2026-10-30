// Copyright © 2023 Apple Inc.

#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name).string();
}

TEST_CASE("test save_safetensors") {
  std::string file_path = get_temp_file("test_arr.safetensors");
  auto map = std::unordered_map<std::string, array>();
  map.insert({"test", array({1.0, 2.0, 3.0, 4.0})});
  map.insert({"test2", ones({2, 2})});
  auto _metadata = std::unordered_map<std::string, std::string>();
  _metadata.insert({"test", "test"});
  _metadata.insert({"test2", "test2"});
  save_safetensors(file_path, map, _metadata);
  auto [dict, metadata] = load_safetensors(file_path);

  CHECK_EQ(metadata, _metadata);

  CHECK_EQ(dict.size(), 2);
  CHECK_EQ(dict.count("test"), 1);
  CHECK_EQ(dict.count("test2"), 1);
  array test = dict.at("test");
  CHECK_EQ(test.dtype(), float32);
  CHECK_EQ(test.shape(), Shape{4});
  CHECK(array_equal(test, array({1.0, 2.0, 3.0, 4.0})).item<bool>());
  array test2 = dict.at("test2");
  CHECK_EQ(test2.dtype(), float32);
  CHECK_EQ(test2.shape(), Shape{2, 2});
  CHECK(array_equal(test2, ones({2, 2})).item<bool>());
}

// Helper to write a raw safetensors file from a JSON header and data buffer
void write_raw_safetensors(
    const std::string& path,
    const std::string& json_header,
    const std::vector<char>& data) {
  std::ofstream out(path, std::ios::binary);
  uint64_t header_len = json_header.size();
  out.write(reinterpret_cast<const char*>(&header_len), 8);
  out.write(json_header.data(), json_header.size());
  out.write(data.data(), data.size());
}

TEST_CASE("test safetensors file boundary validation") {
  // Test that loading a safetensors file where data_offsets extend beyond the
  // actual file size throws an error instead of reading out-of-bounds memory.

  SUBCASE("data_offsets beyond file boundary") {
    std::string file_path = get_temp_file("test_oob_safetensors.safetensors");

    // Create a header claiming a 4MB tensor but only provide 4 bytes of data
    std::string json_header =
        R"({"tensor":{"dtype":"F32","shape":[1000,1000],"data_offsets":[0,4000000]}})";
    std::vector<char> data(4, 0); // Only 4 bytes of actual data

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }

  SUBCASE("data_offsets begin > end") {
    std::string file_path = get_temp_file("test_reversed_offsets.safetensors");

    std::string json_header =
        R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[100,0]}})";
    std::vector<char> data(200, 0);

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }

  SUBCASE("valid file still loads correctly") {
    std::string file_path = get_temp_file("test_valid_safetensors.safetensors");
    auto map = std::unordered_map<std::string, array>();
    map.insert({"test", array({1.0, 2.0, 3.0, 4.0})});
    save_safetensors(file_path, map);
    auto [dict, metadata] = load_safetensors(file_path);

    CHECK_EQ(dict.size(), 1);
    CHECK_EQ(dict.count("test"), 1);
    array test = dict.at("test");
    CHECK(array_equal(test, array({1.0, 2.0, 3.0, 4.0})).item<bool>());
  }

  SUBCASE("mismatched data_offsets") {
    std::string file_path = get_temp_file("test_bad_offsets.safetensors");
    std::string json_header =
        R"({"t":{"dtype":"F32","shape":[10,10],"data_offsets":[0,4]}})";
    std::vector<char> data(400, 0);

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }

  SUBCASE("bad data_offsets count") {
    std::string file_path = get_temp_file("test_bad_offsets_count.safetensors");
    std::string json_header =
        R"({"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}})";
    std::vector<char> data(4, 0);

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }
}

TEST_CASE("test gguf") {
  std::string file_path = get_temp_file("test_arr.gguf");
  using dict = std::unordered_map<std::string, array>;
  dict original_weights = {
      {"test", array({1.0f, 2.0f, 3.0f, 4.0f})},
      {"test2", reshape(arange(6), {3, 2})}};

  {
    // Check saving loading just arrays, no metadata
    save_gguf(file_path, original_weights);
    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 0);
    CHECK_EQ(loaded_weights.size(), 2);
    CHECK_EQ(loaded_weights.count("test"), 1);
    CHECK_EQ(loaded_weights.count("test2"), 1);
    for (auto [k, v] : loaded_weights) {
      CHECK(array_equal(v, original_weights.at(k)).item<bool>());
    }
  }

  // Test saving and loading string metadata
  std::unordered_map<std::string, GGUFMetaData> original_metadata;
  original_metadata.insert({"test_str", "my string"});

  save_gguf(file_path, original_weights, original_metadata);
  auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
  CHECK_EQ(loaded_metadata.size(), 1);
  CHECK_EQ(loaded_metadata.count("test_str"), 1);
  CHECK_EQ(std::get<std::string>(loaded_metadata.at("test_str")), "my string");

  CHECK_EQ(loaded_weights.size(), 2);
  CHECK_EQ(loaded_weights.count("test"), 1);
  CHECK_EQ(loaded_weights.count("test2"), 1);
  for (auto [k, v] : loaded_weights) {
    CHECK(array_equal(v, original_weights.at(k)).item<bool>());
  }

  std::vector<Dtype> unsupported_types = {
      bool_, uint8, uint32, uint64, int64, bfloat16, complex64};
  for (auto t : unsupported_types) {
    dict to_save = {{"test", astype(arange(5), t)}};
    CHECK_THROWS(save_gguf(file_path, to_save, original_metadata));
  }

  std::vector<Dtype> supported_types = {int8, int32, float16, float32};
  for (auto t : supported_types) {
    auto arr = astype(arange(5), t);
    dict to_save = {{"test", arr}};
    save_gguf(file_path, to_save, original_metadata);
    const auto& [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK(array_equal(loaded_weights.at("test"), arr).item<bool>());
  }
}

// Writes a one-tensor GGUF (name "t", ndim 1, dim 4, type F32) whose tensor
// data offset field is set verbatim to `tensor_data_offset`. Writes
// `data_bytes` bytes of tensor data, defaulting to the full four floats.
void write_raw_gguf(
    const std::string& path,
    uint64_t tensor_data_offset,
    size_t data_bytes = 4 * sizeof(float)) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&out](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  auto u64 = [&out](uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), 8);
  };
  out.write("GGUF", 4);
  u32(3); // version
  u64(1); // tensor_count
  u64(0); // metadata_kv_count
  u64(1); // tensor name length
  out.write("t", 1);
  u32(1); // ndim
  u64(4); // dim[0]
  u32(0); // GGUF_TYPE_F32
  u64(tensor_data_offset);
  while (out.tellp() % 32 != 0) { // default GGUF alignment
    out.put(0);
  }
  std::vector<char> data(data_bytes, 0);
  out.write(data.data(), data.size());
}

TEST_CASE("test gguf tensor data offset validation") {
  // A crafted tensor data offset must be rejected rather than turned into a
  // pointer outside the mapping. See ml-explore/mlx#4136.
  SUBCASE("valid offset loads") {
    std::string file_path = get_temp_file("test_gguf_offset_ok.gguf");
    write_raw_gguf(file_path, 0);
    auto [weights, metadata] = load_gguf(file_path);
    CHECK_EQ(weights.size(), 1);
    CHECK(array_equal(weights.at("t"), zeros({4}, float32)).item<bool>());
  }

  SUBCASE("offset past the end of the file") {
    std::string file_path = get_temp_file("test_gguf_offset_past_end.gguf");
    write_raw_gguf(file_path, 1ull << 20);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("offset far past the end of the file") {
    std::string file_path = get_temp_file("test_gguf_offset_far_past.gguf");
    write_raw_gguf(file_path, 1ull << 40);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("offset that overflows the data section base") {
    // Wraps back to an in-mapping address, so an end-pointer-only check would
    // silently read the wrong bytes instead of reading out of bounds.
    std::string file_path = get_temp_file("test_gguf_offset_wrap.gguf");
    write_raw_gguf(file_path, ~0ull - 8);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("tensor extends past the end of the file") {
    // In-range offset, but the data is truncated: only the extent check
    // catches this.
    std::string file_path = get_temp_file("test_gguf_truncated.gguf");
    write_raw_gguf(file_path, 0, 4 * sizeof(float) - 1);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("tensor starts inside the file but ends past it") {
    // A small offset, so the start is in range and only the extent decides.
    // Pins the extent check to the tensor's own start rather than to the
    // start of the data section.
    std::string file_path = get_temp_file("test_gguf_partial_overrun.gguf");
    write_raw_gguf(file_path, 8);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("offset just past the end of the file") {
    // Only a few bytes past the end rather than far outside it, so the
    // resulting pointer is still in the mapped page and reads succeed
    // silently. Pins the offset bound to the file size exactly.
    std::string file_path = get_temp_file("test_gguf_offset_just_past.gguf");
    write_raw_gguf(file_path, 20);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }
}

// Writes a one-tensor GGUF whose tensor header carries `dims` and `type`
// verbatim, so a test can drive the dimension product independently of the
// amount of tensor data in the file. The name ends in ".weight" because the
// quantized loader derives the scales and biases names from it.
void write_raw_gguf_dims(
    const std::string& path,
    const std::vector<uint64_t>& dims,
    uint32_t type,
    size_t data_bytes) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&out](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  auto u64 = [&out](uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), 8);
  };
  out.write("GGUF", 4);
  u32(3); // version
  u64(1); // tensor_count
  u64(0); // metadata_kv_count
  u64(8); // tensor name length
  out.write("w.weight", 8);
  u32(dims.size());
  for (auto dim : dims) {
    u64(dim);
  }
  u32(type);
  u64(0); // tensor data offset
  while (out.tellp() % 32 != 0) { // default GGUF alignment
    out.put(0);
  }
  std::vector<char> data(data_bytes, 0);
  out.write(data.data(), data.size());
}

TEST_CASE("test gguf tensor dimension validation") {
  // The element count exists in two widths: gguflib keeps the 64 bit product in
  // tensor.num_weights, and get_shape narrows each dimension to a 32 bit
  // ShapeElem. Only the byte size is checked against the file, so dimensions
  // that make the two disagree size a buffer from one and index it with the
  // other. See ml-explore/mlx#4244.
  const uint32_t q8_0 = 8;

  SUBCASE("valid quantized tensor loads") {
    std::string file_path = get_temp_file("test_gguf_dims_ok.gguf");
    // One block of 32 weights: 2 bytes of scale plus 32 bytes of weights.
    write_raw_gguf_dims(file_path, {32}, q8_0, 34);
    auto [weights, metadata] = load_gguf(file_path);
    CHECK_EQ(weights.at("w.weight").shape(), Shape{8});
    CHECK_EQ(weights.at("w.scales").shape(), Shape{1});
    CHECK_EQ(weights.at("w.biases").shape(), Shape{1});
  }

  SUBCASE("dimension past the ShapeElem range") {
    // Narrows to a shape of 32 that no longer describes the file.
    std::string file_path = get_temp_file("test_gguf_dim_narrowed.gguf");
    write_raw_gguf_dims(file_path, {(1ull << 32) + 32}, q8_0, 34);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("dimension that narrows to a negative shape") {
    std::string file_path = get_temp_file("test_gguf_dim_negative.gguf");
    write_raw_gguf_dims(file_path, {1ull << 31}, q8_0, 34);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("dimension product that wraps to a small byte size") {
    // 96 * 384307168202282326 wraps to 64, so the tensor needs only 68 bytes of
    // data and passes the byte size check, while the narrowed shape
    // {1431655766, 96} still describes 2^37 elements. That drives the scales
    // element count past INT32_MAX, which is what truncates the allocation size
    // while the extractor still loops over the full count.
    std::string file_path = get_temp_file("test_gguf_dim_wrap.gguf");
    write_raw_gguf_dims(file_path, {96, 384307168202282326ull}, q8_0, 68);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("dimension product that wraps to zero") {
    // Wraps num_weights to exactly 0, so the tensor claims no data at all.
    std::string file_path = get_temp_file("test_gguf_dim_wrap_zero.gguf");
    write_raw_gguf_dims(
        file_path,
        {(1ull << 32) + (1ull << 25),
         (1ull << 32) + (1ull << 15),
         (1ull << 32) + (1ull << 24)},
        q8_0,
        0);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("tensor without dimensions") {
    // gguf_load_quantized indexes the last shape element unconditionally.
    std::string file_path = get_temp_file("test_gguf_no_dims.gguf");
    write_raw_gguf_dims(file_path, {}, q8_0, 0);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }
}

// Writes a metadata-only GGUF (no tensors) whose metadata KV section is
// `kv_section` verbatim, so a caller can encode values whose lengths exceed the
// file to exercise check_metadata_value_in_file(). `kv_count` must match the
// number of KV pairs encoded in `kv_section`.
void write_raw_gguf_metadata(
    const std::string& path,
    uint64_t kv_count,
    const std::vector<char>& kv_section) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&out](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  auto u64 = [&out](uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), 8);
  };
  out.write("GGUF", 4);
  u32(3); // version
  u64(0); // tensor_count
  u64(kv_count); // metadata_kv_count
  out.write(kv_section.data(), kv_section.size());
}

TEST_CASE("test gguf metadata value validation") {
  // A STRING/ARRAY metadata value claiming a length larger than the file must
  // be rejected rather than read past the end of the mapping. See PR #4212.

  auto append_string_kv = [](std::vector<char>& b,
                             const std::string& key,
                             uint64_t claimed_len,
                             bool write_payload) {
    auto put = [&](const void* p, size_t n) {
      b.insert(
          b.end(),
          static_cast<const char*>(p),
          static_cast<const char*>(p) + n);
    };
    uint64_t klen = key.size();
    put(&klen, 8);
    put(key.data(), key.size());
    uint32_t vt = 8; // GGUF_VALUE_TYPE_STRING
    put(&vt, 4);
    put(&claimed_len, 8);
    if (write_payload) {
      b.insert(b.end(), claimed_len, '\0');
    }
  };

  auto append_array_kv = [](std::vector<char>& b,
                            const std::string& key,
                            uint32_t elt_type,
                            uint64_t claimed_len) {
    auto put = [&](const void* p, size_t n) {
      b.insert(
          b.end(),
          static_cast<const char*>(p),
          static_cast<const char*>(p) + n);
    };
    uint64_t klen = key.size();
    put(&klen, 8);
    put(key.data(), key.size());
    uint32_t vt = 9; // GGUF_VALUE_TYPE_ARRAY
    put(&vt, 4);
    put(&elt_type, 4);
    put(&claimed_len, 8);
  };

  SUBCASE("valid empty and small strings load") {
    std::vector<char> kv;
    append_string_kv(kv, "empty", 0, false);
    append_string_kv(kv, "small", 5, true);
    std::string file_path = get_temp_file("test_gguf_meta_ok.gguf");
    write_raw_gguf_metadata(file_path, 2, kv);
    auto [weights, metadata] = load_gguf(file_path);
    CHECK(weights.empty());
    CHECK(std::get<std::string>(metadata.at("empty")) == "");
    CHECK(std::get<std::string>(metadata.at("small")) == std::string(5, '\0'));
  }

  SUBCASE("string length extends past the end of the file") {
    // Claims 100 bytes of payload, none of which are present.
    std::vector<char> kv;
    append_string_kv(kv, "s", 100, false);
    std::string file_path = get_temp_file("test_gguf_meta_str_past.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("string length far past the end of the file") {
    std::vector<char> kv;
    append_string_kv(kv, "s", 1ull << 40, false);
    std::string file_path = get_temp_file("test_gguf_meta_str_far.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("fixed-size array length extends past the end of the file") {
    // GGUF_VALUE_TYPE_UINT8 = 0; claims 2^40 elements, none present.
    std::vector<char> kv;
    append_array_kv(kv, "a", 0, 1ull << 40);
    std::string file_path = get_temp_file("test_gguf_meta_arr_past.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("string array element length extends past the end of the file") {
    // GGUF_VALUE_TYPE_STRING = 8; two elements, neither present.
    std::vector<char> kv;
    append_array_kv(kv, "a", 8, 2);
    std::string file_path = get_temp_file("test_gguf_meta_strarr_past.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }
}

TEST_CASE("test gguf metadata") {
  std::string file_path = get_temp_file("test_arr.gguf");
  using dict = std::unordered_map<std::string, array>;
  dict original_weights = {
      {"test", array({1.0f, 2.0f, 3.0f, 4.0f})},
      {"test2", reshape(arange(6), {3, 2})}};

  // Scalar array
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    original_metadata.insert({"test_arr", array(1.0)});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    auto arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK_EQ(arr.item<float>(), 1.0f);
  }

  // 1D Array
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    auto arr = array({1.0, 2.0});
    original_metadata.insert({"test_arr", arr});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    auto loaded_arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK(array_equal(arr, loaded_arr).item<bool>());

    // Preserves dims
    arr = array({1.0});
    original_metadata["test_arr"] = arr;
    save_gguf(file_path, original_weights, original_metadata);

    std::tie(loaded_weights, loaded_metadata) = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    loaded_arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK(array_equal(arr, loaded_arr).item<bool>());
  }

  // 1D int64 array with negative values
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    auto arr = array({static_cast<int64_t>(-1)}, int64);
    original_metadata.insert({"test_arr", arr});
    save_gguf(file_path, original_weights, original_metadata);
    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK(array_equal(arr, std::get<array>(loaded_metadata.at("test_arr")))
              .item<bool>());
  }

  // > 1D array throws
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    original_metadata.insert({"test_arr", array({1.0}, {1, 1})});
    CHECK_THROWS(save_gguf(file_path, original_weights, original_metadata));
  }

  // empty array throws
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    original_metadata.insert({"test_arr", array({})});
    CHECK_THROWS(save_gguf(file_path, original_weights, original_metadata));
  }

  // vector of string
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    std::vector<std::string> data = {"data1", "data2", "data1234"};
    original_metadata.insert({"meta", data});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("meta"), 1);
    auto& strs = std::get<std::vector<std::string>>(loaded_metadata["meta"]);
    CHECK_EQ(strs.size(), 3);
    for (int i = 0; i < strs.size(); ++i) {
      CHECK_EQ(strs[i], data[i]);
    }
  }

  // vector of string, string, scalar, and array
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    std::vector<std::string> data = {"data1", "data2", "data1234"};
    original_metadata.insert({"meta1", data});
    original_metadata.insert({"meta2", array(2.5)});
    original_metadata.insert({"meta3", array({1, 2, 3})});
    original_metadata.insert({"meta4", "last"});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 4);
    auto& strs = std::get<std::vector<std::string>>(loaded_metadata["meta1"]);
    CHECK_EQ(strs.size(), 3);
    for (int i = 0; i < strs.size(); ++i) {
      CHECK_EQ(strs[i], data[i]);
    }
    auto& arr = std::get<array>(loaded_metadata["meta2"]);
    CHECK_EQ(arr.item<float>(), 2.5);

    arr = std::get<array>(loaded_metadata["meta3"]);
    CHECK(array_equal(arr, array({1, 2, 3})).item<bool>());

    auto& str = std::get<std::string>(loaded_metadata["meta4"]);
    CHECK_EQ(str, "last");
  }
}

namespace {

// The GGUF_VALUE_TYPE_* codes of gguflib.h.
constexpr uint32_t kGGUFUint8 = 0;
constexpr uint32_t kGGUFInt8 = 1;
constexpr uint32_t kGGUFUint16 = 2;
constexpr uint32_t kGGUFInt16 = 3;
constexpr uint32_t kGGUFUint32 = 4;
constexpr uint32_t kGGUFInt32 = 5;
constexpr uint32_t kGGUFFloat32 = 6;
constexpr uint32_t kGGUFBool = 7;
constexpr uint32_t kGGUFString = 8;
constexpr uint32_t kGGUFArray = 9;
constexpr uint32_t kGGUFUint64 = 10;
constexpr uint32_t kGGUFInt64 = 11;
constexpr uint32_t kGGUFFloat64 = 12;

// Writes a GGUF file with metadata and no tensors. begin_at() pads the key to
// put the value at a chosen offset.
class RawGGUFMetadata {
 public:
  static constexpr size_t kHeaderSize = 24; // magic, version, two counts

  // Pads `key` with underscores so that the value starts `residue` bytes past
  // a multiple of 8. Returns the padded key.
  std::string begin_at(std::string key, uint32_t value_type, size_t residue) {
    size_t value_offset = kHeaderSize + bytes_.size() + 8 + key.size() + 4;
    key.append((residue + 8 - value_offset % 8) % 8, '_');
    begin(key, value_type);
    return key;
  }

  void begin(const std::string& key, uint32_t value_type) {
    put<uint64_t>(key.size());
    bytes_.insert(bytes_.end(), key.begin(), key.end());
    put<uint32_t>(value_type);
    value_offsets_.push_back(kHeaderSize + bytes_.size());
  }

  template <typename T>
  void put(T v) {
    const char* p = reinterpret_cast<const char*>(&v);
    bytes_.insert(bytes_.end(), p, p + sizeof(T));
  }

  void put_string(const std::string& s) {
    put<uint64_t>(s.size());
    bytes_.insert(bytes_.end(), s.begin(), s.end());
  }

  template <typename T>
  void put_array(uint32_t element_type, std::initializer_list<T> elements) {
    put<uint32_t>(element_type);
    put<uint64_t>(elements.size());
    for (auto e : elements) {
      put<T>(e);
    }
  }

  void put_string_array(const std::vector<std::string>& elements) {
    put<uint32_t>(kGGUFString);
    put<uint64_t>(elements.size());
    for (const auto& e : elements) {
      put_string(e);
    }
  }

  // True if no value starts at a multiple of 8, the alignment of gguf_value.
  bool all_values_misaligned() const {
    for (auto offset : value_offsets_) {
      if (offset % 8 == 0) {
        return false;
      }
    }
    return !value_offsets_.empty();
  }

  void write(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    auto u32 = [&out](uint32_t v) {
      out.write(reinterpret_cast<const char*>(&v), 4);
    };
    auto u64 = [&out](uint64_t v) {
      out.write(reinterpret_cast<const char*>(&v), 8);
    };
    out.write("GGUF", 4);
    u32(3); // version
    u64(0); // tensor_count
    u64(value_offsets_.size()); // metadata_kv_count
    out.write(bytes_.data(), bytes_.size());
  }

 private:
  std::vector<char> bytes_;
  std::vector<size_t> value_offsets_;
};

using MetadataMap = std::unordered_map<std::string, GGUFMetaData>;

// The scalar `key` loaded as a 0-d array of `dtype` that holds `expected`.
template <typename T>
void check_gguf_scalar(
    const MetadataMap& metadata,
    const std::string& key,
    Dtype dtype,
    T expected) {
  INFO(key);
  auto it = metadata.find(key);
  REQUIRE(it != metadata.end());
  auto arr = std::get_if<array>(&it->second);
  REQUIRE(arr != nullptr);
  CHECK((arr->dtype() == dtype));
  CHECK(arr->ndim() == 0);
  CHECK(arr->size() == 1);
  CHECK(arr->data<T>()[0] == expected);
}

// The array `key` loaded as a 1-d array of `dtype` that holds `expected`.
template <typename T>
void check_gguf_array(
    const MetadataMap& metadata,
    const std::string& key,
    Dtype dtype,
    std::initializer_list<T> expected) {
  INFO(key);
  auto it = metadata.find(key);
  REQUIRE(it != metadata.end());
  auto arr = std::get_if<array>(&it->second);
  REQUIRE(arr != nullptr);
  CHECK((arr->dtype() == dtype));
  CHECK(arr->ndim() == 1);
  REQUIRE(arr->size() == expected.size());
  CHECK(arr->shape(0) == static_cast<int>(expected.size()));
  size_t i = 0;
  for (auto e : expected) {
    INFO("element " << i);
    CHECK(arr->data<T>()[i] == e);
    i++;
  }
}

} // namespace

TEST_CASE("test gguf metadata at unaligned offsets") {
  // GGUF does not align its values. The loader must not assume an alignment
  // for any value type. On arm64 and x86-64 a misaligned read works, so only
  // the undefined behavior sanitizer shows the error.

  const std::string nul_str("a\0b\xc3\xa9", 5); // embedded NUL, UTF-8

  SUBCASE("every value type, scalar and array, in one file") {
    RawGGUFMetadata f;
    auto s_u8 = f.begin_at("s_u8", kGGUFUint8, 1);
    f.put<uint8_t>(200);
    auto a_u8 = f.begin_at("a_u8", kGGUFArray, 3);
    f.put_array<uint8_t>(kGGUFUint8, {0, 1, 127, 128, 255});
    auto s_i8 = f.begin_at("s_i8", kGGUFInt8, 5);
    f.put<int8_t>(-100);
    auto a_i8 = f.begin_at("a_i8", kGGUFArray, 7);
    f.put_array<int8_t>(kGGUFInt8, {-128, -1, 0, 1, 127});
    auto s_u16 = f.begin_at("s_u16", kGGUFUint16, 1);
    f.put<uint16_t>(65000);
    auto a_u16 = f.begin_at("a_u16", kGGUFArray, 3);
    f.put_array<uint16_t>(kGGUFUint16, {0, 1, 255, 256, 65535});
    auto s_i16 = f.begin_at("s_i16", kGGUFInt16, 5);
    f.put<int16_t>(-30000);
    auto a_i16 = f.begin_at("a_i16", kGGUFArray, 7);
    f.put_array<int16_t>(kGGUFInt16, {-32768, -1, 0, 1, 32767});
    auto s_u32 = f.begin_at("s_u32", kGGUFUint32, 1);
    f.put<uint32_t>(4000000000u);
    auto a_u32 = f.begin_at("a_u32", kGGUFArray, 3);
    f.put_array<uint32_t>(
        kGGUFUint32, {0u, 1u, 65536u, 0xDEADBEEFu, 0xFFFFFFFFu});
    auto s_i32 = f.begin_at("s_i32", kGGUFInt32, 5);
    f.put<int32_t>(-2000000000);
    auto a_i32 = f.begin_at("a_i32", kGGUFArray, 7);
    f.put_array<int32_t>(kGGUFInt32, {INT32_MIN, -1, 0, 1, INT32_MAX});
    auto s_u64 = f.begin_at("s_u64", kGGUFUint64, 1);
    f.put<uint64_t>(0xFEDCBA9876543210ull);
    auto a_u64 = f.begin_at("a_u64", kGGUFArray, 3);
    f.put_array<uint64_t>(
        kGGUFUint64,
        {0ull, 1ull, 1ull << 32, 0xFEDCBA9876543210ull, UINT64_MAX});
    auto s_i64 = f.begin_at("s_i64", kGGUFInt64, 5);
    f.put<int64_t>(-0x0123456789ABCDEFll);
    auto a_i64 = f.begin_at("a_i64", kGGUFArray, 7);
    f.put_array<int64_t>(kGGUFInt64, {INT64_MIN, -1, 0, 1, INT64_MAX});
    auto s_f32 = f.begin_at("s_f32", kGGUFFloat32, 1);
    f.put<float>(-1.25f);
    auto a_f32 = f.begin_at("a_f32", kGGUFArray, 3);
    f.put_array<float>(
        kGGUFFloat32,
        {-1.5f, 0.0f, 2.5f, 1.0e-30f, std::numeric_limits<float>::max()});
    auto s_bool = f.begin_at("s_bool", kGGUFBool, 5);
    f.put<uint8_t>(1);
    auto a_bool = f.begin_at("a_bool", kGGUFArray, 7);
    f.put_array<uint8_t>(kGGUFBool, {1, 0, 1, 1, 0});
    // Bytes other than 0 and 1 are not valid bools: they load as true.
    auto s_bool2 = f.begin_at("s_bool2", kGGUFBool, 1);
    f.put<uint8_t>(2);
    auto a_bool2 = f.begin_at("a_bool2", kGGUFArray, 3);
    f.put_array<uint8_t>(kGGUFBool, {2, 0, 255, 1});
    auto s_str = f.begin_at("s_str", kGGUFString, 1);
    f.put_string(nul_str);
    auto a_str = f.begin_at("a_str", kGGUFArray, 3);
    f.put_string_array(
        {"", "a", "bc", nul_str, "defghijklmnopqrstuvwxyz", "z"});
    // Empty arrays: the elements start at the very end of the value.
    auto e_u8 = f.begin_at("e_u8", kGGUFArray, 5);
    f.put_array<uint8_t>(kGGUFUint8, {});
    auto e_i32 = f.begin_at("e_i32", kGGUFArray, 7);
    f.put_array<int32_t>(kGGUFInt32, {});
    auto e_str = f.begin_at("e_str", kGGUFArray, 1);
    f.put_string_array({});
    // The last pair is a string, whose end is the end of the file.
    auto last = f.begin_at("last", kGGUFString, 3);
    f.put_string("last");
    REQUIRE(f.all_values_misaligned());

    std::string path = get_temp_file("test_gguf_unaligned_all.gguf");
    f.write(path);
    auto [weights, metadata] = load_gguf(path);
    CHECK(weights.empty());
    CHECK(metadata.size() == 28);

    check_gguf_scalar<uint8_t>(metadata, s_u8, uint8, 200);
    check_gguf_array<uint8_t>(metadata, a_u8, uint8, {0, 1, 127, 128, 255});
    check_gguf_scalar<int8_t>(metadata, s_i8, int8, -100);
    check_gguf_array<int8_t>(metadata, a_i8, int8, {-128, -1, 0, 1, 127});
    check_gguf_scalar<uint16_t>(metadata, s_u16, uint16, 65000);
    check_gguf_array<uint16_t>(
        metadata, a_u16, uint16, {0, 1, 255, 256, 65535});
    check_gguf_scalar<int16_t>(metadata, s_i16, int16, -30000);
    check_gguf_array<int16_t>(
        metadata, a_i16, int16, {-32768, -1, 0, 1, 32767});
    check_gguf_scalar<uint32_t>(metadata, s_u32, uint32, 4000000000u);
    check_gguf_array<uint32_t>(
        metadata, a_u32, uint32, {0u, 1u, 65536u, 0xDEADBEEFu, 0xFFFFFFFFu});
    check_gguf_scalar<int32_t>(metadata, s_i32, int32, -2000000000);
    check_gguf_array<int32_t>(
        metadata, a_i32, int32, {INT32_MIN, -1, 0, 1, INT32_MAX});
    check_gguf_scalar<uint64_t>(metadata, s_u64, uint64, 0xFEDCBA9876543210ull);
    check_gguf_array<uint64_t>(
        metadata,
        a_u64,
        uint64,
        {0ull, 1ull, 1ull << 32, 0xFEDCBA9876543210ull, UINT64_MAX});
    check_gguf_scalar<int64_t>(metadata, s_i64, int64, -0x0123456789ABCDEFll);
    check_gguf_array<int64_t>(
        metadata, a_i64, int64, {INT64_MIN, -1, 0, 1, INT64_MAX});
    check_gguf_scalar<float>(metadata, s_f32, float32, -1.25f);
    check_gguf_array<float>(
        metadata,
        a_f32,
        float32,
        {-1.5f, 0.0f, 2.5f, 1.0e-30f, std::numeric_limits<float>::max()});
    check_gguf_scalar<bool>(metadata, s_bool, bool_, true);
    check_gguf_array<bool>(
        metadata, a_bool, bool_, {true, false, true, true, false});
    check_gguf_scalar<uint8_t>(metadata, s_bool2, bool_, 1);
    check_gguf_array<uint8_t>(metadata, a_bool2, bool_, {1, 0, 1, 1});

    CHECK(std::get<std::string>(metadata.at(s_str)) == nul_str);
    CHECK(
        std::get<std::vector<std::string>>(metadata.at(a_str)) ==
        std::vector<std::string>{
            "", "a", "bc", nul_str, "defghijklmnopqrstuvwxyz", "z"});
    check_gguf_array<uint8_t>(metadata, e_u8, uint8, {});
    check_gguf_array<int32_t>(metadata, e_i32, int32, {});
    CHECK(std::get<std::vector<std::string>>(metadata.at(e_str)).empty());
    CHECK(std::get<std::string>(metadata.at(last)) == "last");
  }

  // FLOAT64 values are the last pair here: the loader advances its offset by
  // the size of the float32 array it builds, not by the 8 bytes in the file.
  SUBCASE("float64 scalar") {
    RawGGUFMetadata f;
    auto lead = f.begin_at("lead", kGGUFUint8, 1);
    f.put<uint8_t>(7);
    auto s_f64 = f.begin_at("s_f64", kGGUFFloat64, 3);
    f.put<double>(0.1);
    REQUIRE(f.all_values_misaligned());

    std::string path = get_temp_file("test_gguf_unaligned_f64.gguf");
    f.write(path);
    auto [weights, metadata] = load_gguf(path);
    CHECK(metadata.size() == 2);
    check_gguf_scalar<uint8_t>(metadata, lead, uint8, 7);
    // FLOAT64 metadata is loaded as float32.
    check_gguf_scalar<float>(metadata, s_f64, float32, 0.1f);
  }

  SUBCASE("float64 array") {
    RawGGUFMetadata f;
    auto lead = f.begin_at("lead", kGGUFString, 5);
    f.put_string("x");
    auto a_f64 = f.begin_at("a_f64", kGGUFArray, 7);
    f.put_array<double>(
        kGGUFFloat64,
        {0.5,
         -2.25,
         1024.0,
         0.1,
         static_cast<double>(std::numeric_limits<float>::max())});
    REQUIRE(f.all_values_misaligned());

    std::string path = get_temp_file("test_gguf_unaligned_f64a.gguf");
    f.write(path);
    auto [weights, metadata] = load_gguf(path);
    CHECK(metadata.size() == 2);
    CHECK(std::get<std::string>(metadata.at(lead)) == "x");
    check_gguf_array<float>(
        metadata,
        a_f64,
        float32,
        {0.5f, -2.25f, 1024.0f, 0.1f, std::numeric_limits<float>::max()});
  }

  SUBCASE("nested arrays and unknown element types are rejected") {
    {
      RawGGUFMetadata f;
      f.begin_at("nested", kGGUFArray, 1);
      f.put<uint32_t>(kGGUFArray); // element type: array
      f.put<uint64_t>(0);
      REQUIRE(f.all_values_misaligned());
      std::string path = get_temp_file("test_gguf_unaligned_nested.gguf");
      f.write(path);
      CHECK_THROWS_AS(load_gguf(path), std::invalid_argument);
    }
    {
      RawGGUFMetadata f;
      f.begin_at("unknown", kGGUFArray, 3);
      f.put<uint32_t>(99); // element type: none
      f.put<uint64_t>(0);
      REQUIRE(f.all_values_misaligned());
      std::string path = get_temp_file("test_gguf_unaligned_unknown.gguf");
      f.write(path);
      CHECK_THROWS_AS(load_gguf(path), std::runtime_error);
    }
  }

  SUBCASE("save_gguf writes strings at unaligned offsets byte for byte") {
    auto read_all = [](const std::string& path) {
      std::ifstream in(path, std::ios::binary);
      return std::string(
          std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    std::vector<std::string> strs = {"a", "bcd", "", nul_str, "efghi"};

    RawGGUFMetadata expected_array;
    expected_array.begin("k", kGGUFArray);
    expected_array.put_string_array(strs);
    std::string expected_array_path =
        get_temp_file("test_gguf_unaligned_expected_a.gguf");
    expected_array.write(expected_array_path);

    std::string array_path = get_temp_file("test_gguf_unaligned_saved_a.gguf");
    MetadataMap array_metadata;
    array_metadata.insert({"k", strs});
    save_gguf(
        array_path, std::unordered_map<std::string, array>(), array_metadata);
    CHECK(read_all(array_path) == read_all(expected_array_path));

    RawGGUFMetadata expected_string;
    expected_string.begin("k", kGGUFString);
    expected_string.put_string(nul_str);
    std::string expected_string_path =
        get_temp_file("test_gguf_unaligned_expected_s.gguf");
    expected_string.write(expected_string_path);

    std::string string_path = get_temp_file("test_gguf_unaligned_saved_s.gguf");
    MetadataMap string_metadata;
    string_metadata.insert({"k", nul_str});
    save_gguf(
        string_path, std::unordered_map<std::string, array>(), string_metadata);
    CHECK(read_all(string_path) == read_all(expected_string_path));

    auto [weights, metadata] = load_gguf(array_path);
    CHECK(std::get<std::vector<std::string>>(metadata.at("k")) == strs);
  }
}

TEST_CASE("test single array serialization") {
  // Basic test
  {
    auto a = random::uniform(-5.f, 5.f, {2, 5, 12}, float32);

    std::string file_path = get_temp_file("test_arr.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  // Other shapes
  {
    auto a = random::uniform(
        -5.f,
        5.f,
        {
            1,
        },
        float32);

    std::string file_path = get_temp_file("test_arr_0.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  {
    auto a = random::uniform(
        -5.f,
        5.f,
        {
            46,
        },
        float32);

    std::string file_path = get_temp_file("test_arr_1.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  {
    auto a = random::uniform(-5.f, 5.f, {5, 2, 1, 3, 4}, float32);

    std::string file_path = get_temp_file("test_arr_2.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }
}

#ifndef _WIN32
namespace {
struct NamedMmapFixture {
  std::string path;
  size_t page = static_cast<size_t>(getpagesize());
  NamedMmapFixture() {
    auto pattern = get_temp_file("mlx-named-mmap-XXXXXX");
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int fd = mkstemp(name.data());
    if (fd < 0) {
      throw std::runtime_error("temporary mmap fixture creation failed");
    }
    path = name.data();
    const int result = ftruncate(fd, static_cast<off_t>(page * 4));
    close(fd);
    if (result != 0) {
      std::filesystem::remove(path);
      throw std::runtime_error("temporary mmap fixture sizing failed");
    }
  }
  ~NamedMmapFixture() {
    std::filesystem::remove(path);
  }
};
} // namespace

TEST_CASE("named mmap banks register expert ranges and release weak owners") {
  NamedMmapFixture file;
  constexpr int32_t layer = 1900000000;
  const auto prefix = "model.layers.1900000000.mlp.switch_mlp.gate_proj.tq2_";
  const auto baseline = safetensors_mmap_tracked_buffer_bytes();
  for (bool packed : {true, false}) {
    std::optional<array> retained;
    {
      const Shape shape = packed ? Shape{2, static_cast<int>(file.page / 4), 1}
                                 : Shape{2, static_cast<int>(file.page / 2)};
      // Zero-copy adoption is currently implemented only by MetalAllocator.
      // Other backends must reject the region without retaining a registry
      // entry; exercise that contract rather than assuming Metal on Linux CI.
      if (!metal::is_available()) {
        CHECK_THROWS_WITH_AS(
            mmap_file_region_named(
                file.path,
                file.page,
                2 * file.page,
                shape,
                packed ? uint32 : float16,
                std::string(prefix) + (packed ? "packed" : "scales")),
            ("[mmap_file_region] make_buffer failed: " + file.path).c_str(),
            std::runtime_error);
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
        CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 0);
        continue;
      }
      auto bank = mmap_file_region_named(
          file.path,
          file.page,
          2 * file.page,
          shape,
          packed ? uint32 : float16,
          std::string(prefix) + (packed ? "packed" : "scales"));
      CHECK_EQ(bank.shape(), shape);
      CHECK_EQ(
          safetensors_mmap_tracked_buffer_bytes(), baseline + 2 * file.page);
      int32_t selected_layer = layer;
      int32_t expert = 1;
      CHECK_EQ(
          safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1),
          file.page);
      CHECK_EQ(
          safetensors_mmap_advise_experts(0, &selected_layer, &expert, 1),
          file.page);
      expert = 2;
      CHECK_EQ(
          safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1), 0);
      CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 2 * file.page);
      retained = bank;
    }
    CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline + 2 * file.page);
    retained.reset();
    CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
    CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 0);
  }
}

TEST_CASE("wrapped routed named mmap preserves expert advice and weak ownership") {
  NamedMmapFixture file;
  constexpr int32_t layer = 1900000003;
  const auto prefix = "language_model.model.layers.1900000003.mlp.switch_mlp.";
  const auto baseline = safetensors_mmap_tracked_buffer_bytes();
  for (const auto* role : {"gate", "up", "down"}) {
    for (bool packed : {true, false}) {
      const Shape shape = packed ? Shape{2, static_cast<int>(file.page / 4), 1}
                                 : Shape{2, static_cast<int>(file.page / 2)};
      const auto name = std::string(prefix) + role + "_proj.tq2_" +
          (packed ? "packed" : "scales");
      if (!metal::is_available()) {
        CHECK_THROWS_WITH_AS(
            mmap_file_region_named(
                file.path, file.page, 2 * file.page, shape,
                packed ? uint32 : float16, name),
            ("[mmap_file_region] make_buffer failed: " + file.path).c_str(),
            std::runtime_error);
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
        CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 0);
        continue;
      }
      std::optional<array> retained;
      {
        auto bank = mmap_file_region_named(
            file.path, file.page, 2 * file.page, shape,
            packed ? uint32 : float16, name);
        CHECK_EQ(bank.shape(), shape);
        CHECK_EQ(bank.dtype(), packed ? uint32 : float16);
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline + 2 * file.page);
        int32_t selected_layer = layer, expert = 1;
        CHECK_EQ(safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1), file.page);
        CHECK_EQ(safetensors_mmap_advise_experts(0, &selected_layer, &expert, 1), file.page);
        expert = 2;
        CHECK_EQ(safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1), 0);
        CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 2 * file.page);
        retained = bank;
      }
      CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline + 2 * file.page);
      retained.reset();
      CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
      CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 0);
    }
  }
  for (const auto* name : {
           "language_model.layers.0.mlp.switch_mlp.gate_proj.tq2_packed",
           "language_model.language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq2_packed",
           "model.language_model.model.layers.0.mlp.switch_mlp.up_proj.tq2_packed",
           "language_model.model.model.layers.0.mlp.switch_mlp.down_proj.tq2_packed",
           "vision_model.model.layers.0.mlp.switch_mlp.gate_proj.tq2_packed",
           "language_modelx.model.layers.0.mlp.switch_mlp.gate_proj.tq2_packed",
           "language_model.model.layers.00.mlp.switch_mlp.gate_proj.tq2_packed",
           "language_model.model.layers.-1.mlp.switch_mlp.gate_proj.tq2_packed",
           "language_model.model.layers.2147483648.mlp.switch_mlp.gate_proj.tq2_packed",
           "language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq2_packed.extra"}) {
    CHECK_THROWS_AS(
        mmap_file_region_named(file.path, file.page, 2 * file.page,
                              {2, static_cast<int>(file.page / 4), 1}, uint32, name),
        std::invalid_argument);
  }
  CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
}

TEST_CASE("LFM named mmap admits canonical feed_forward expert banks") {
  NamedMmapFixture file;
  constexpr int32_t layer = 1900000002;
  const auto baseline = safetensors_mmap_tracked_buffer_bytes();
  for (const auto* role : {"gate", "up", "down"}) {
    for (bool packed : {true, false}) {
      const Shape shape = packed ? Shape{2, static_cast<int>(file.page / 4), 1}
                                 : Shape{2, static_cast<int>(file.page / 2)};
      const auto name = "model.layers.1900000002.feed_forward.switch_mlp." +
          std::string(role) + "_proj.tq2_" + (packed ? "packed" : "scales");
      if (!metal::is_available()) {
        CHECK_THROWS_WITH_AS(
            mmap_file_region_named(file.path, file.page, 2 * file.page, shape,
                                   packed ? uint32 : float16, name),
            ("[mmap_file_region] make_buffer failed: " + file.path).c_str(),
            std::runtime_error);
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
        continue;
      }
      {
        auto bank = mmap_file_region_named(file.path, file.page, 2 * file.page,
                                          shape, packed ? uint32 : float16, name);
        CHECK_EQ(bank.shape(), shape);
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline + 2 * file.page);
        int32_t selected_layer = layer, expert = 1;
        CHECK_EQ(safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1), file.page);
        CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 2 * file.page);
      }
      CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
      CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 0);
    }
  }
  for (const auto* name : {
           "model.layers.00.feed_forward.switch_mlp.gate_proj.tq2_packed",
           "model.layers.2147483648.feed_forward.switch_mlp.gate_proj.tq2_packed",
           "model.layers.0.feed_forward.gate_proj.tq2_packed",
           "model.layers.0.feed_forward.switch_mlp.router_proj.tq2_packed",
           "model.layers.0.feed_forward.switch_mlp.gate_proj.weight"}) {
    CHECK_THROWS_AS(mmap_file_region_named(file.path, file.page, 2 * file.page,
                                          {2, static_cast<int>(file.page / 4), 1}, uint32, name),
                    std::invalid_argument);
  }
  CHECK_THROWS_AS(mmap_file_region_named(file.path, file.page, 2 * file.page,
                                        {2, static_cast<int>(file.page / 4), 1}, float32,
                                        "model.layers.0.feed_forward.switch_mlp.gate_proj.tq2_packed"),
                  std::invalid_argument);
  CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
}

TEST_CASE("Nemotron-H named mmap admits canonical relu2 fc1/fc2 expert banks") {
  NamedMmapFixture file;
  constexpr int32_t layer = 1900000003;
  const auto baseline = safetensors_mmap_tracked_buffer_bytes();
  for (const auto* role : {"fc1", "fc2"}) {
    for (bool packed : {true, false}) {
      const Shape shape = packed ? Shape{2, static_cast<int>(file.page / 4), 1}
                                 : Shape{2, static_cast<int>(file.page / 2)};
      const auto name = "backbone.layers.1900000003.mixer.switch_mlp." +
          std::string(role) + ".tq2_" + (packed ? "packed" : "scales");
      if (!metal::is_available()) {
        CHECK_THROWS_WITH_AS(
            mmap_file_region_named(file.path, file.page, 2 * file.page, shape,
                                   packed ? uint32 : float16, name),
            ("[mmap_file_region] make_buffer failed: " + file.path).c_str(),
            std::runtime_error);
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
        continue;
      }
      {
        auto bank = mmap_file_region_named(file.path, file.page, 2 * file.page,
                                          shape, packed ? uint32 : float16, name);
        CHECK_EQ(bank.shape(), shape);
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline + 2 * file.page);
        int32_t selected_layer = layer, expert = 1;
        CHECK_EQ(safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1), file.page);
        CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 2 * file.page);
      }
      CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
      CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 0);
    }
  }
  for (const auto* name : {
           "backbone.layers.00.mixer.switch_mlp.fc1.tq2_packed",
           "backbone.layers.2147483648.mixer.switch_mlp.fc1.tq2_packed",
           "backbone.layers.0.mixer.fc1.tq2_packed",
           "backbone.layers.0.mixer.switch_mlp.fc3.tq2_packed",
           "backbone.layers.0.mixer.switch_mlp.gate_proj.tq2_packed",
           "language_model.backbone.layers.0.mixer.switch_mlp.fc1.tq2_packed",
           "backbone.layers.0.mixer.switch_mlp.fc1.weight"}) {
    CHECK_THROWS_AS(mmap_file_region_named(file.path, file.page, 2 * file.page,
                                          {2, static_cast<int>(file.page / 4), 1}, uint32, name),
                    std::invalid_argument);
  }
  CHECK_THROWS_AS(mmap_file_region_named(file.path, file.page, 2 * file.page,
                                        {2, static_cast<int>(file.page / 4), 1}, float32,
                                        "backbone.layers.0.mixer.switch_mlp.fc1.tq2_packed"),
                  std::invalid_argument);
  CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
}

TEST_CASE("dense named mmap keeps one-expert ownership and layer advice") {
  NamedMmapFixture file;
  constexpr int32_t layer = 1900000001;
  const auto baseline = safetensors_mmap_tracked_buffer_bytes();
  for (const auto* prefix :
       {"model.layers.1900000001.mlp.",
        "language_model.model.layers.1900000001.mlp."}) {
    for (const auto* role : {"gate_proj", "up_proj", "down_proj"}) {
      for (bool packed : {true, false}) {
        const Shape shape = packed
            ? Shape{1, static_cast<int>(file.page / 4), 1}
            : Shape{1, static_cast<int>(file.page / 2)};
        const auto name = std::string(prefix) + role + ".tq2_" +
            (packed ? "packed" : "scales");
        if (!metal::is_available()) {
          CHECK_THROWS_WITH_AS(
              mmap_file_region_named(
                  file.path,
                  file.page,
                  file.page,
                  shape,
                  packed ? uint32 : float16,
                  name),
              ("[mmap_file_region] make_buffer failed: " + file.path).c_str(),
              std::runtime_error);
          CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
          continue;
        }
        {
          auto bank = mmap_file_region_named(
              file.path,
              file.page,
              file.page,
              shape,
              packed ? uint32 : float16,
              name);
          CHECK_EQ(bank.shape(), shape);
          CHECK_EQ(
              safetensors_mmap_tracked_buffer_bytes(), baseline + file.page);
          CHECK_EQ(safetensors_mmap_advise_layer(1, layer), file.page);
          int32_t selected_layer = layer;
          int32_t expert = 0;
          CHECK_EQ(
              safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1),
              file.page);
          expert = 1;
          CHECK_EQ(
              safetensors_mmap_advise_experts(1, &selected_layer, &expert, 1),
              0);
        }
        CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
        CHECK_EQ(safetensors_mmap_advise_layer(1, layer), 0);
      }
    }
  }
  const Shape one{1, static_cast<int>(file.page / 4), 1};
  for (const auto& name :
       {"language_model.layers.0.mlp.gate_proj.tq2_packed",
        "vision_model.model.layers.0.mlp.up_proj.tq2_packed",
        "model.layers.0.mlp.experts.down_proj.tq2_packed",
        "model.layers.0.mlp.other_proj.tq2_packed",
        "model.layers.0.mlp.gate_proj.tq2_weight",
        "language_model.model.layers.00.mlp.gate_proj.tq2_packed",
        "language_model.model.layers.2147483648.mlp.up_proj.tq2_packed",
        "model.layers.00.mlp.down_proj.tq2_packed",
        "model.layers.2147483648.mlp.down_proj.tq2_packed"}) {
    CHECK_THROWS_AS(
        mmap_file_region_named(file.path, 0, file.page, one, uint32, name),
        std::invalid_argument);
  }
  for (const auto* prefix :
       {"model.layers.0.mlp.", "language_model.model.layers.0.mlp."}) {
    for (const auto* role : {"gate_proj", "up_proj", "down_proj"}) {
      CHECK_THROWS_AS(
          mmap_file_region_named(
              file.path,
              0,
              2 * file.page,
              {2, static_cast<int>(file.page / 4), 1},
              uint32,
              std::string(prefix) + role + ".tq2_packed"),
          std::invalid_argument);
    }
  }
  CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
}

TEST_CASE("named mmap rejects invalid contract before registering") {
  NamedMmapFixture file;
  const auto baseline = safetensors_mmap_tracked_buffer_bytes();
  const std::string name =
      "model.layers.1900000000.mlp.switch_mlp.up_proj.tq2_packed";
  const Shape valid{2, static_cast<int>(file.page / 4), 1};
  for (
      const std::string& invalid :
      {std::string(""),
       std::string("model.layers.0.mlp.gate.weight"),
       std::string("model.layers.00.mlp.switch_mlp.up_proj.tq2_packed"),
       std::string("model.layers.2147483648.mlp.switch_mlp.up_proj.tq2_packed"),
       std::string(
           "model.layers.99999999999999999999999.mlp.switch_mlp.up_proj.tq2_packed")}) {
    CHECK_THROWS(mmap_file_region_named(
        file.path, 0, 2 * file.page, valid, uint32, invalid));
  }
  CHECK_THROWS(mmap_file_region_named(
      file.path, 0, 2 * file.page, valid, float16, name));
  CHECK_THROWS(mmap_file_region_named(
      file.path, 0, 2 * file.page, {2, 1}, uint32, name));
  CHECK_THROWS(mmap_file_region_named(
      file.path, 0, 2 * file.page, {0, 1, 1}, uint32, name));
  CHECK_THROWS(mmap_file_region_named(
      file.path, 0, 2 * file.page, {-1, 1, 1}, uint32, name));
  CHECK_THROWS(mmap_file_region_named(
      file.path,
      0,
      2 * file.page,
      {2147483647, 2147483647, 2147483647},
      uint32,
      name));
  CHECK_THROWS(mmap_file_region_named(
      file.path, 3 * file.page, 2 * file.page, valid, uint32, name));
  CHECK_THROWS(
      mmap_file_region_named(file.path, 1, 2 * file.page, valid, uint32, name));
  CHECK_THROWS(mmap_file_region_named(
      file.path, 0, 2 * file.page - 1, valid, uint32, name));
  CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
}

TEST_CASE("unnamed mmap regions retain their unregistered behavior") {
  NamedMmapFixture file;
  const auto baseline = safetensors_mmap_tracked_buffer_bytes();
  if (!metal::is_available()) {
    CHECK_THROWS_WITH_AS(
        mmap_file_region(
            file.path,
            file.page,
            file.page,
            {static_cast<int>(file.page / 4)},
            uint32),
        ("[mmap_file_region] make_buffer failed: " + file.path).c_str(),
        std::runtime_error);
    CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
    return;
  }
  auto bank = mmap_file_region(
      file.path,
      file.page,
      file.page,
      {static_cast<int>(file.page / 4)},
      uint32);
  CHECK_EQ(bank.nbytes(), file.page);
  CHECK_EQ(safetensors_mmap_tracked_buffer_bytes(), baseline);
}
#endif
