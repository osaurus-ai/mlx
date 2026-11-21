// Copyright © 2023 Apple Inc.
#include <cassert>
#include <climits>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test array basics") {
  // Scalar
  array x(1.0);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.ndim(), 0);
  CHECK_EQ(x.shape(), Shape{});
  CHECK_THROWS_AS(x.shape(0), std::out_of_range);
  CHECK_THROWS_AS(x.shape(-1), std::out_of_range);
  CHECK_EQ(x.strides(), Strides{});
  CHECK_EQ(x.itemsize(), sizeof(float));
  CHECK_EQ(x.nbytes(), sizeof(float));
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.item<float>(), 1.0);

  // Scalar with specified type
  x = array(1, float32);
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.item<float>(), 1.0);

  // Scalar with specified type
  x = array(1, bool_);
  CHECK_EQ(x.dtype(), bool_);
  CHECK_EQ(x.itemsize(), sizeof(bool));
  CHECK_EQ(x.nbytes(), sizeof(bool));
  CHECK_EQ(x.item<bool>(), true);

  // Check shaped arrays
  x = array({1.0});
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.ndim(), 1);
  CHECK_EQ(x.shape(), Shape{1});
  CHECK_EQ(x.shape(0), 1);
  CHECK_EQ(x.shape(-1), 1);
  CHECK_THROWS_AS(x.shape(1), std::out_of_range);
  CHECK_THROWS_AS(x.shape(-2), std::out_of_range);
  CHECK_EQ(x.strides(), Strides{1});
  CHECK_EQ(x.item<float>(), 1.0);

  // Check empty array
  x = array({});
  CHECK_EQ(x.size(), 0);
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.itemsize(), sizeof(float));
  CHECK_EQ(x.nbytes(), 0);
  CHECK_THROWS_AS(x.item<float>(), std::invalid_argument);

  x = array({1.0, 1.0});
  CHECK_EQ(x.size(), 2);
  CHECK_EQ(x.shape(), Shape{2});
  CHECK_EQ(x.itemsize(), sizeof(float));
  CHECK_EQ(x.nbytes(), x.itemsize() * x.size());

  // Accessing item in non-scalar array throws
  CHECK_THROWS_AS(x.item<float>(), std::invalid_argument);

  x = array({1.0, 1.0, 1.0}, {1, 3});
  CHECK_EQ(x.size(), 3);
  CHECK_EQ(x.shape(), Shape{1, 3});
  CHECK_EQ(x.strides(), Strides{3, 1});

  // Test wrong size/shapes throw:
  CHECK_THROWS_AS(array({1.0, 1.0, 1.0}, {4}), std::invalid_argument);
  CHECK_THROWS_AS(array({1.0, 1.0, 1.0}, {1, 4}), std::invalid_argument);
  CHECK_THROWS_AS(array({1.0, 1.0, 1.0}, {1, 2}), std::invalid_argument);

  // Test array ids work as expected
  x = array(1.0);
  auto y = x;
  CHECK_EQ(y.id(), x.id());
  array z(2.0);
  CHECK_NE(z.id(), x.id());
  z = x;
  CHECK_EQ(z.id(), x.id());

  // Array creation from pointer
  float data[] = {0.0, 1.0, 2.0, 3.0};
  x = array(data, {4});
  CHECK_EQ(x.dtype(), float32);
  CHECK(array_equal(x, array({0.0, 1.0, 2.0, 3.0})).item<bool>());

  // Array creation from vectors
  {
    std::vector<int> data = {0, 1, 2, 3};
    x = array(data.begin(), {4});
    CHECK_EQ(x.dtype(), int32);
    CHECK(array_equal(x, array({0, 1, 2, 3})).item<bool>());
  }

  {
    std::vector<bool> data = {false, true, false, true};
    x = array(data.begin(), {4});
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({false, true, false, true})).item<bool>());
  }

  // Regression: vector<bool>::reference to fp16/bf16 stored raw bits
  {
    std::vector<bool> data = {true, false, true};
    auto bf = array(data.begin(), {3}, bfloat16);
    CHECK(array_equal(bf, array({1.0f, 0.0f, 1.0f}, bfloat16)).item<bool>());

    auto fp = array(data.begin(), {3}, float16);
    CHECK(array_equal(fp, array({1.0f, 0.0f, 1.0f}, float16)).item<bool>());
  }
}

TEST_CASE("test array types") {
#define basic_dtype_test(T, mlx_type) \
  T val = 42;                         \
  array x(val);                       \
  CHECK_EQ(x.dtype(), mlx_type);      \
  CHECK_EQ(x.item<T>(), val);         \
  x = array({val, val});              \
  CHECK_EQ(x.dtype(), mlx_type);

  // bool_
  {
    array x(true);
    CHECK_EQ(x.dtype(), bool_);
    CHECK_EQ(x.item<bool>(), true);

    x = array({true, false});
    CHECK_EQ(x.dtype(), bool_);

    x = array({true, false}, float32);
    CHECK_EQ(x.dtype(), float32);
    CHECK(array_equal(x, array({1.0f, 0.0f})).item<bool>());
  }

  // uint8
  {
    basic_dtype_test(uint8_t, uint8);
  }

  // uint16
  {
    basic_dtype_test(uint16_t, uint16);
  }

  // uint32
  {
    basic_dtype_test(uint32_t, uint32);
  }

  // uint64
  {
    basic_dtype_test(uint64_t, uint64);
  }

  // int8
  {
    basic_dtype_test(int8_t, int8);
  }

  // int16
  {
    basic_dtype_test(int16_t, int16);
  }

  // int32
  {
    basic_dtype_test(int32_t, int32);
  }

  // int64
  {
    basic_dtype_test(int64_t, int64);
  }

  // float16
  {
    basic_dtype_test(float16_t, float16);
  }

  // float32
  {
    basic_dtype_test(float, float32);
  }

  // bfloat16
  {
    basic_dtype_test(bfloat16_t, bfloat16);
  }

#undef basic_dtype_test

  // uint32
  {
    uint32_t val = UINT_MAX;
    array x(val);
    CHECK_EQ(x.dtype(), uint32);
    CHECK_EQ(x.item<uint32_t>(), val);

    x = array({1u, 2u});
    CHECK_EQ(x.dtype(), uint32);
  }

  // int32
  {
    array x(-1);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(x.item<int>(), -1);

    x = array({-1, 2});
    CHECK_EQ(x.dtype(), int32);

    std::vector<int> data{0, 1, 2};
    x = array(data.data(), {static_cast<int>(data.size())}, bool_);
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({false, true, true})).item<bool>());
  }

  // int64
  {
    int64_t val = static_cast<int64_t>(INT_MIN) - 1;
    array x(val);
    CHECK_EQ(x.dtype(), int64);
    CHECK_EQ(x.item<int64_t>(), val);

    x = array({val, val});
    CHECK_EQ(x.dtype(), int64);
  }

  // float32
  {
    array x(3.14f);
    CHECK_EQ(x.dtype(), float32);
    CHECK_EQ(x.item<float>(), 3.14f);

    x = array(1.25);
    CHECK_EQ(x.dtype(), float32);
    CHECK_EQ(x.item<float>(), 1.25f);

    x = array({1.0f, 2.0f});
    CHECK_EQ(x.dtype(), float32);

    x = array({1.0, 2.0});
    CHECK_EQ(x.dtype(), float32);

    std::vector<double> data{1.0, 2.0, 4.0};
    x = array(data.data(), {static_cast<int>(data.size())});
    CHECK_EQ(x.dtype(), float32);
    CHECK(array_equal(x, array({1.0f, 2.0f, 4.0f})).item<bool>());
  }

  // complex64
  {
    CHECK_EQ(sizeof(complex64_t), sizeof(std::complex<float>));

    complex64_t v = {1.0f, 1.0f};
    array x(v);
    CHECK_EQ(x.dtype(), complex64);
    CHECK_EQ(x.item<complex64_t>(), v);

    array y(std::complex<float>{1.0f, 1.0f});
    CHECK_EQ(x.dtype(), complex64);
    CHECK_EQ(x.item<complex64_t>(), v);
  }
}

TEST_CASE("test array metadata") {
  array x(1.0f);
  CHECK_EQ(x.data_size(), 1);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = array({1.0f}, {1, 1, 1});
  CHECK_EQ(x.data_size(), 1);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = array({1.0f, 1.0f}, {1, 2});
  CHECK_EQ(x.data_size(), 2);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = zeros({1, 1, 4});
  eval(x);
  CHECK_EQ(x.data_size(), 4);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = zeros({2, 4});
  eval(x);
  CHECK_EQ(x.data_size(), 8);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, false);

  x = array(1.0f);
  auto y = broadcast_to(x, {1, 1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = broadcast_to(x, {2, 8, 10});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  y = broadcast_to(x, {1, 0});
  eval(y);
  CHECK_EQ(y.data_size(), 0);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = broadcast_to(zeros({4, 2, 1}), {4, 2, 0});
  eval(y);
  CHECK_EQ(y.data_size(), 0);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array(1.0f);
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({1, 1, 1});
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({1, 1, 1});
  y = transpose(x, {0, 1, 2});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({1, 1, 1});
  y = transpose(x, {1, 2, 0});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({4, 1});
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 4);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({2, 3, 4});
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 24);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = transpose(x, {0, 2, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 24);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  y = transpose(transpose(x, {0, 2, 1}), {0, 2, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 24);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = array(1.0f);
  y = reshape(x, {1, 1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({2, 4});
  y = reshape(x, {8});
  eval(y);
  CHECK_EQ(y.data_size(), 8);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = reshape(x, {8, 1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 8);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = reshape(x, {1, 8, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 8);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({12});
  y = reshape(x, {2, 3, 2});
  eval(y);
  CHECK_EQ(y.data_size(), 12);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = array(1.0f);
  y = slice(x, {}, {});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f});
  y = slice(x, {-10}, {10}, {10});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 3}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 3);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 3}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 3);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {0, 3}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 0);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 2);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 2}, {2, 3});
  eval(y);
  CHECK_EQ(y.shape(), Shape{1, 1});
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({0.0f, 1.0f, 2.0f, 3.0f}, {1, 4});
  y = slice(x, {0, 0}, {1, 4}, {1, 2});
  eval(y);
  CHECK_EQ(y.shape(), Shape{1, 2});
  CHECK_EQ(y.flags().contiguous, false);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = broadcast_to(array(1.0f), {4, 10});
  y = slice(x, {0, 0}, {4, 10}, {2, 2});
  eval(y);
  CHECK_EQ(y.shape(), Shape{2, 5});
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = broadcast_to(array({1.0f, 2.0f}), {4, 2});
  y = slice(x, {0, 0}, {1, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 2);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = slice(x, {1, 0}, {2, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 2);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2});
  y = slice(x, {0, 0}, {2, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 4);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, false);

  y = slice(transpose(x), {0, 0}, {2, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 4);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({2, 4});
  auto out = split(x, 2);
  eval(out);
  for (auto y : out) {
    CHECK_EQ(y.data_size(), 4);
    CHECK_EQ(y.flags().contiguous, true);
    CHECK_EQ(y.flags().row_contiguous, true);
    CHECK_EQ(y.flags().col_contiguous, true);
  }
  out = split(x, 4, 1);
  eval(out);
  for (auto y : out) {
    CHECK_EQ(y.flags().contiguous, false);
    CHECK_EQ(y.flags().row_contiguous, false);
    CHECK_EQ(y.flags().col_contiguous, false);
  }
}

TEST_CASE("test array iteration") {
  // Dim 0 arrays
  auto arr = array(1);
  CHECK_THROWS(arr.begin());

  // Iterated arrays are read only
  CHECK(std::is_const_v<decltype(*arr.begin())>);

  arr = array({1, 2, 3, 4, 5});
  int i = 0;
  for (auto a : arr) {
    i++;
    CHECK_EQ(a.item<int>(), i);
  }
  CHECK_EQ(i, 5);

  arr = array({1, 2, 3, 4}, {2, 2});
  CHECK(array_equal(*arr.begin(), array({1, 2})).item<bool>());
  CHECK(array_equal(*(arr.begin() + 1), array({3, 4})).item<bool>());
  CHECK_EQ(arr.begin() + 2, arr.end());
}

TEST_CASE("test array shared buffer") {
  Shape shape = {2, 2};
  auto n_elem = shape[0] * shape[1];

  allocator::Buffer buf_b = allocator::malloc(n_elem * sizeof(float));
  void* buf_b_ptr = buf_b.raw_ptr();
  float* float_buf_b = (float*)buf_b_ptr;

  for (int i = 0; i < n_elem; i++) {
    float_buf_b[i] = 2.;
  }

  CHECK_EQ(float_buf_b[0], ((float*)buf_b_ptr)[0]);

  auto deleter = [float_buf_b](allocator::Buffer buf) {
    CHECK_EQ(float_buf_b, (float*)buf.raw_ptr());
    CHECK_EQ(float_buf_b[0], ((float*)buf.raw_ptr())[0]);
    allocator::free(buf);
  };

  array a = ones(shape, float32);
  array b = array(buf_b, shape, float32, deleter);

  eval(a + b);
}

TEST_CASE("test make empty array") {
  auto a = array({});
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), float32);

  a = array({}, int32);
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), int32);

  a = array({}, float32);
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), float32);

  a = array({}, bool_);
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), bool_);
}

TEST_CASE("test make array from user buffer") {
  int size = 4096;
  std::vector<int> buffer(size, 0);

  int count = 0;
  auto deleter = [&count, data = buffer.data()](void* ptr) {
    // make sure pointer is correct
    if (ptr == data) {
      count++;
    }
  };

  {
    auto a = array(buffer.data(), Shape{size}, int32, deleter);
    if (metal::is_available()) {
      CHECK_EQ(buffer.data(), a.data<int>());
    }
    auto b = a + array(1);
    eval(b);
    auto expected = ones({4096});
    CHECK(array_equal(b, expected).item<bool>());
  }
  // deleter should always get called
  CHECK_EQ(count, 1);
}

TEST_CASE("test negative indexing for shape/strides") {
  // 2D array: shape = {2, 3}
  std::vector<float> data(6, 1.0f);
  array a(data.begin(), Shape{2, 3});

  // Valid negative indexing
  CHECK_EQ(a.shape(-1), a.shape(1));
  CHECK_EQ(a.shape(-2), a.shape(0));
  CHECK_EQ(a.shape(-1), 3);
  CHECK_EQ(a.shape(-2), 2);

  CHECK_EQ(a.strides(-1), a.strides(1));
  CHECK_EQ(a.strides(-2), a.strides(0));
  CHECK_EQ(a.strides(-1), 1);
  CHECK_EQ(a.strides(-2), 3);

  // Invalid: too negative
  CHECK_THROWS_AS(a.shape(-3), std::out_of_range);
  CHECK_THROWS_AS(a.strides(-3), std::out_of_range);

  // Invalid: too positive
  CHECK_THROWS_AS(a.shape(2), std::out_of_range);
  CHECK_THROWS_AS(a.strides(2), std::out_of_range);
}

// https://github.com/ml-explore/mlx/pull/1590
TEST_CASE("test siblings circular references without eval") {
  std::weak_ptr<array::Data> tracker;
  auto fun = [&]() {
    array key({1, 2});
    auto splits = split(key, 2);
    {
      // Set fake data as a tracker for ArrayDesc's lifetime.
      splits[0].set_data(allocator::malloc(0));
      tracker = splits[0].data_shared_ptr();
    }
    auto a = reshape(splits[0], {});
    auto b = reshape(splits[1], {});
    return b;
  };
  fun();
  CHECK(tracker.expired());
}

// https://github.com/ml-explore/mlx/pull/4453
TEST_CASE("test siblings circular references released by assignment") {
  std::weak_ptr<array::Data> tracker;
  auto fun = [&]() {
    array sink({3, 4});
    array key({1, 2});
    auto splits = split(key, 2);
    {
      // Set fake data as a tracker for ArrayDesc's lifetime.
      splits[0].set_data(allocator::malloc(0));
      tracker = splits[0].data_shared_ptr();
    }
    // Drop both outputs by assigning over them rather than destroying
    // them. The siblings hold each other, so unless assignment runs the
    // same check the destructor does, nothing ever releases them.
    splits[0] = sink; // copy assignment
    splits[1] = array({5, 6}); // move assignment
  };
  fun();
  CHECK(tracker.expired());
}

TEST_CASE("test siblings circular references released by overwrite") {
  std::weak_ptr<array::Data> tracker;
  auto fun = [&]() {
    array key({1, 2});
    auto splits = split(key, 2);
    {
      // Set fake data as a tracker for ArrayDesc's lifetime.
      splits[0].set_data(allocator::malloc(0));
      tracker = splits[0].data_shared_ptr();
    }
    // Leave one output as the last reference, then replace it in place.
    // This is the path __setitem__ takes, with no compile involved.
    array last = splits[0];
    splits.clear();
    last.overwrite_descriptor(array({7, 8}));
  };
  fun();
  CHECK(tracker.expired());
}

TEST_CASE("test assigning a sibling over the last reference keeps the cycle") {
  array key({1, 2});
  auto splits = split(key, 2);
  // Leave one output as the last outside reference to the pair, then assign
  // its sibling over it. The sibling is reachable only through the array
  // being replaced, so it has to be held before the old descriptor is
  // checked: checked first, the pair looks unreferenced, its cycle is broken
  // and the sibling list the new value lives in is freed before it is read.
  array last = splits[0];
  splits.clear();
  last = last.siblings()[0];
  REQUIRE_EQ(last.siblings().size(), 1);
  CHECK(array_equal(last, array({2})).item<bool>());
}

namespace {

template <typename T>
struct InitRow {
  double in;
  T out;
};

// Input and result for one integer type: NaN gives 0, values out of range
// saturate, others truncate. Inputs are the bounds and their float neighbours.
struct InitRows {
  // clang-format off
  std::vector<InitRow<int8_t>> int8_rows = {
      {NAN, 0}, {INFINITY, 127}, {-INFINITY, -128}, {0.0, 0}, {-0.0, 0},
      {0.5, 0}, {-0.5, 0}, {1.0, 1}, {-1.0, -1}, {1.5, 1}, {-1.5, -1},
      {300.5, 127}, {-300.5, -128}, {1e+20, 127}, {-1e+20, -128},
      {65504.0, 127}, {-65504.0, -128}, {3.4028234663852886e+38, 127},
      {-3.4028234663852886e+38, -128}, {3.3895313892515355e+38, 127},
      {-3.3895313892515355e+38, -128}, {1.7976931348623157e+308, 127},
      {-1.7976931348623157e+308, -128}, {5e-324, 0}, {-5e-324, 0},
      {5.960464477539063e-08, 0}, {-5.960464477539063e-08, 0}, {127.9375, 127},
      {128.0, 127}, {128.125, 127}, {-128.125, -128}, {-128.0, -128},
      {-127.9375, -127}, {126.9375, 126}, {127.0625, 127}, {127.0, 127},
      {127.5, 127}, {129.0, 127}, {-129.0, -128}, {-127.5, -127}, {126.5, 126},
      {127.99999237060547, 127}, {128.00001525878906, 127},
      {-128.00001525878906, -128}, {-127.99999237060547, -127},
      {126.99999237060547, 126}, {127.00000762939453, 127},
      {127.99999999999999, 127}, {128.00000000000003, 127},
      {-128.00000000000003, -128}, {-127.99999999999999, -127},
      {126.99999999999999, 126}, {127.00000000000001, 127}, {-128.5, -128},
  };
  std::vector<InitRow<uint8_t>> uint8_rows = {
      {NAN, 0}, {INFINITY, 255}, {-INFINITY, 0}, {0.0, 0}, {-0.0, 0}, {0.5, 0},
      {-0.5, 0}, {1.0, 1}, {-1.0, 0}, {1.5, 1}, {-1.5, 0}, {300.5, 255},
      {-300.5, 0}, {1e+20, 255}, {-1e+20, 0}, {65504.0, 255}, {-65504.0, 0},
      {3.4028234663852886e+38, 255}, {-3.4028234663852886e+38, 0},
      {3.3895313892515355e+38, 255}, {-3.3895313892515355e+38, 0},
      {1.7976931348623157e+308, 255}, {-1.7976931348623157e+308, 0},
      {5e-324, 0}, {-5e-324, 0}, {5.960464477539063e-08, 0},
      {-5.960464477539063e-08, 0}, {255.875, 255}, {256.0, 255}, {256.25, 255},
      {254.875, 254}, {255.125, 255}, {255.0, 255}, {258.0, 255}, {254.0, 254},
      {255.99998474121094, 255}, {256.0000305175781, 255},
      {254.99998474121094, 254}, {255.00001525878906, 255},
      {255.99999999999997, 255}, {256.00000000000006, 255},
      {254.99999999999997, 254}, {255.00000000000003, 255},
      {-0.9999999999999999, 0}, {-0.9999999403953552, 0}, {-2.0, 0},
      {-1e+30, 0}, {255.5, 255}, {254.5, 254},
  };
  std::vector<InitRow<int16_t>> int16_rows = {
      {NAN, 0}, {INFINITY, 32767}, {-INFINITY, -32768}, {0.0, 0}, {-0.0, 0},
      {0.5, 0}, {-0.5, 0}, {1.0, 1}, {-1.0, -1}, {1.5, 1}, {-1.5, -1},
      {300.5, 300}, {-300.5, -300}, {1e+20, 32767}, {-1e+20, -32768},
      {65504.0, 32767}, {-65504.0, -32768}, {3.4028234663852886e+38, 32767},
      {-3.4028234663852886e+38, -32768}, {3.3895313892515355e+38, 32767},
      {-3.3895313892515355e+38, -32768}, {1.7976931348623157e+308, 32767},
      {-1.7976931348623157e+308, -32768}, {5e-324, 0}, {-5e-324, 0},
      {5.960464477539063e-08, 0}, {-5.960464477539063e-08, 0}, {32752.0, 32752},
      {32768.0, 32767}, {32800.0, 32767}, {-32800.0, -32768},
      {-32768.0, -32768}, {-32752.0, -32752}, {32640.0, 32640},
      {33024.0, 32767}, {-33024.0, -32768}, {-32640.0, -32640},
      {32767.998046875, 32767}, {32768.00390625, 32767},
      {-32768.00390625, -32768}, {-32767.998046875, -32767},
      {32766.998046875, 32766}, {32767.001953125, 32767}, {32767.0, 32767},
      {32767.999999999996, 32767}, {32768.00000000001, 32767},
      {-32768.00000000001, -32768}, {-32767.999999999996, -32767},
      {32766.999999999996, 32766}, {32767.000000000004, 32767},
      {-32768.5, -32768}, {-32769.0, -32768}, {-32767.5, -32767},
      {32767.5, 32767}, {32766.5, 32766},
  };
  std::vector<InitRow<uint16_t>> uint16_rows = {
      {NAN, 0}, {INFINITY, 65535}, {-INFINITY, 0}, {0.0, 0}, {-0.0, 0},
      {0.5, 0}, {-0.5, 0}, {1.0, 1}, {-1.0, 0}, {1.5, 1}, {-1.5, 0},
      {300.5, 300}, {-300.5, 0}, {1e+20, 65535}, {-1e+20, 0}, {65504.0, 65504},
      {-65504.0, 0}, {3.4028234663852886e+38, 65535},
      {-3.4028234663852886e+38, 0}, {3.3895313892515355e+38, 65535},
      {-3.3895313892515355e+38, 0}, {1.7976931348623157e+308, 65535},
      {-1.7976931348623157e+308, 0}, {5e-324, 0}, {-5e-324, 0},
      {5.960464477539063e-08, 0}, {-5.960464477539063e-08, 0}, {65280.0, 65280},
      {65536.0, 65535}, {66048.0, 65535}, {65535.99609375, 65535},
      {65536.0078125, 65535}, {65534.99609375, 65534}, {65535.00390625, 65535},
      {65535.0, 65535}, {65535.99999999999, 65535}, {65536.00000000001, 65535},
      {65534.99999999999, 65534}, {65535.00000000001, 65535},
      {-0.9999999999999999, 0}, {-0.9999999403953552, 0}, {-2.0, 0},
      {-1e+30, 0}, {65535.5, 65535}, {65534.5, 65534},
  };
  std::vector<InitRow<int32_t>> int32_rows = {
      {NAN, 0}, {INFINITY, 2147483647}, {-INFINITY, INT32_MIN}, {0.0, 0},
      {-0.0, 0}, {0.5, 0}, {-0.5, 0}, {1.0, 1}, {-1.0, -1}, {1.5, 1},
      {-1.5, -1}, {300.5, 300}, {-300.5, -300}, {1e+20, 2147483647},
      {-1e+20, INT32_MIN}, {65504.0, 65504}, {-65504.0, -65504},
      {3.4028234663852886e+38, 2147483647},
      {-3.4028234663852886e+38, INT32_MIN},
      {3.3895313892515355e+38, 2147483647},
      {-3.3895313892515355e+38, INT32_MIN},
      {1.7976931348623157e+308, 2147483647},
      {-1.7976931348623157e+308, INT32_MIN}, {5e-324, 0}, {-5e-324, 0},
      {5.960464477539063e-08, 0}, {-5.960464477539063e-08, 0},
      {2139095040.0, 2139095040}, {2147483648.0, 2147483647},
      {2164260864.0, 2147483647}, {-2164260864.0, INT32_MIN},
      {-2147483648.0, INT32_MIN}, {-2139095040.0, -2139095040},
      {2147483520.0, 2147483520}, {2147483904.0, 2147483647},
      {-2147483904.0, INT32_MIN}, {-2147483520.0, -2147483520},
      {2147483647.9999998, 2147483647}, {2147483648.0000005, 2147483647},
      {-2147483648.0000005, INT32_MIN}, {-2147483647.9999998, -2147483647},
      {2147483646.9999998, 2147483646}, {2147483647.0000002, 2147483647},
      {2147483647.0, 2147483647}, {-2147483648.5, INT32_MIN},
      {-2147483649.0, INT32_MIN}, {-2147483647.5, -2147483647},
      {2147483647.5, 2147483647}, {2147483646.5, 2147483646},
  };
  std::vector<InitRow<uint32_t>> uint32_rows = {
      {NAN, 0}, {INFINITY, 4294967295}, {-INFINITY, 0}, {0.0, 0}, {-0.0, 0},
      {0.5, 0}, {-0.5, 0}, {1.0, 1}, {-1.0, 0}, {1.5, 1}, {-1.5, 0},
      {300.5, 300}, {-300.5, 0}, {1e+20, 4294967295}, {-1e+20, 0},
      {65504.0, 65504}, {-65504.0, 0}, {3.4028234663852886e+38, 4294967295},
      {-3.4028234663852886e+38, 0}, {3.3895313892515355e+38, 4294967295},
      {-3.3895313892515355e+38, 0}, {1.7976931348623157e+308, 4294967295},
      {-1.7976931348623157e+308, 0}, {5e-324, 0}, {-5e-324, 0},
      {5.960464477539063e-08, 0}, {-5.960464477539063e-08, 0},
      {4278190080.0, 4278190080}, {4294967296.0, 4294967295},
      {4328521728.0, 4294967295}, {4294967040.0, 4294967040},
      {4294967808.0, 4294967295}, {4294967295.9999995, 4294967295},
      {4294967296.000001, 4294967295}, {4294967294.9999995, 4294967294},
      {4294967295.0000005, 4294967295}, {4294967295.0, 4294967295},
      {-0.9999999999999999, 0}, {-0.9999999403953552, 0}, {-2.0, 0},
      {-1e+30, 0}, {4294967295.5, 4294967295}, {4294967294.5, 4294967294},
  };
  std::vector<InitRow<int64_t>> int64_rows = {
      {NAN, 0}, {INFINITY, INT64_MAX}, {-INFINITY, INT64_MIN}, {0.0, 0},
      {-0.0, 0}, {0.5, 0}, {-0.5, 0}, {1.0, 1}, {-1.0, -1}, {1.5, 1},
      {-1.5, -1}, {300.5, 300}, {-300.5, -300}, {1e+20, INT64_MAX},
      {-1e+20, INT64_MIN}, {65504.0, 65504}, {-65504.0, -65504},
      {3.4028234663852886e+38, INT64_MAX}, {-3.4028234663852886e+38, INT64_MIN},
      {3.3895313892515355e+38, INT64_MAX}, {-3.3895313892515355e+38, INT64_MIN},
      {1.7976931348623157e+308, INT64_MAX},
      {-1.7976931348623157e+308, INT64_MIN}, {5e-324, 0}, {-5e-324, 0},
      {5.960464477539063e-08, 0}, {-5.960464477539063e-08, 0},
      {9187343239835811840.0, 9187343239835811840LL},
      {9223372036854775808.0, INT64_MAX}, {9295429630892703744.0, INT64_MAX},
      {-9295429630892703744.0, INT64_MIN}, {-9223372036854775808.0, INT64_MIN},
      {-9187343239835811840.0, -9187343239835811840LL},
      {9223371487098961920.0, 9223371487098961920LL},
      {9223373136366403584.0, INT64_MAX}, {-9223373136366403584.0, INT64_MIN},
      {-9223371487098961920.0, -9223371487098961920LL},
      {9223372036854774784.0, 9223372036854774784LL},
      {9223372036854777856.0, INT64_MAX}, {-9223372036854777856.0, INT64_MIN},
      {-9223372036854774784.0, -9223372036854774784LL},
  };
  std::vector<InitRow<uint64_t>> uint64_rows = {
      {NAN, 0}, {INFINITY, UINT64_MAX}, {-INFINITY, 0}, {0.0, 0}, {-0.0, 0},
      {0.5, 0}, {-0.5, 0}, {1.0, 1}, {-1.0, 0}, {1.5, 1}, {-1.5, 0},
      {300.5, 300}, {-300.5, 0}, {1e+20, UINT64_MAX}, {-1e+20, 0},
      {65504.0, 65504}, {-65504.0, 0}, {3.4028234663852886e+38, UINT64_MAX},
      {-3.4028234663852886e+38, 0}, {3.3895313892515355e+38, UINT64_MAX},
      {-3.3895313892515355e+38, 0}, {1.7976931348623157e+308, UINT64_MAX},
      {-1.7976931348623157e+308, 0}, {5e-324, 0}, {-5e-324, 0},
      {5.960464477539063e-08, 0}, {-5.960464477539063e-08, 0},
      {18374686479671623680.0, 18374686479671623680ULL},
      {18446744073709551616.0, UINT64_MAX},
      {18590859261785407488.0, UINT64_MAX},
      {18446742974197923840.0, 18446742974197923840ULL},
      {18446746272732807168.0, UINT64_MAX},
      {18446744073709549568.0, 18446744073709549568ULL},
      {18446744073709555712.0, UINT64_MAX}, {-0.9999999999999999, 0},
      {-0.9999999403953552, 0}, {-2.0, 0}, {-1e+30, 0},
  };
  // clang-format on
};

template <typename F>
void for_each_int_type(F f) {
  static const InitRows r;
  f(r.int8_rows);
  f(r.uint8_rows);
  f(r.int16_rows);
  f(r.uint16_rows);
  f(r.int32_rows);
  f(r.uint32_rows);
  f(r.int64_rows);
  f(r.uint64_rows);
}

template <typename S>
Dtype dtype_of() {
  return std::is_same_v<S, double> ? float64 : TypeToDtype<S>();
}

template <typename S>
double max_finite_of() {
  if constexpr (std::is_same_v<S, float16_t>) {
    return 65504.0;
  } else if constexpr (std::is_same_v<S, bfloat16_t>) {
    return 3.3895313892515355e38;
  } else {
    return std::numeric_limits<S>::max();
  }
}

// The rows whose input S holds exactly, as S values, repeated to at least m
// values and to a multiple of 4.
template <typename S, typename T>
std::pair<std::vector<S>, std::vector<T>> rows_as(
    const std::vector<InitRow<T>>& rows,
    int m) {
  std::vector<S> in;
  std::vector<T> out;
  for (const auto& r : rows) {
    if (std::isfinite(r.in) && std::abs(r.in) > max_finite_of<S>()) {
      continue;
    }
    S s = static_cast<S>(r.in);
    double back = static_cast<double>(s);
    if (std::isnan(r.in) ? std::isnan(back) : back == r.in) {
      in.push_back(s);
      out.push_back(r.out);
    }
  }
  int n = (std::max<int>(in.size(), m) + 3) / 4 * 4;
  for (int i = in.size(); i < n; ++i) {
    in.push_back(in[i % in.size()]);
    out.push_back(out[i % out.size()]);
  }
  return {in, out};
}

// The imaginary parts that complex64 inputs cycle through
const float init_imag[] = {0.0f, 2.5f, -1e30f, NAN, INFINITY};

template <typename T>
std::pair<std::vector<complex64_t>, std::vector<T>> complex_rows(
    const std::vector<InitRow<T>>& rows,
    int m) {
  auto [re, out] = rows_as<float>(rows, m);
  std::vector<complex64_t> in;
  for (size_t i = 0; i < re.size(); ++i) {
    in.push_back(complex64_t{re[i], init_imag[i % 5]});
  }
  return {in, out};
}

std::string dtype_name(Dtype dtype) {
  std::ostringstream os;
  os << dtype;
  return os.str();
}

bool equal_arrays(const array& a, const array& b) {
  return array_equal(a, b).item<bool>();
}

// Build arrays of type T from S values with each constructor that converts.
template <typename S, typename T>
void check_init_to_int(const std::vector<S>& x, const std::vector<T>& e) {
  auto dtype = TypeToDtype<T>();
  int n = x.size();
  array expected(e.begin(), {n}, dtype);
  CHECK(equal_arrays(array(x.data(), {n}, dtype), expected));
  CHECK(equal_arrays(array(x.cbegin(), {n}, dtype), expected));
  bool prefixes_ok = true;
  for (int m : {1, 3, 7, 17, 33}) {
    prefixes_ok &=
        equal_arrays(array(x.data(), {m}, dtype), slice(expected, {0}, {m}));
  }
  CHECK(prefixes_ok);
  bool scalars_ok = true;
  bool lists_ok = true;
  for (int i = 0; i < n; ++i) {
    scalars_ok &= array(x[i], dtype).item<T>() == e[i];
    lists_ok &= array({x[i]}, dtype).item<T>() == e[i];
  }
  for (int i = 0; i < n; i += 4) {
    auto want = slice(expected, {i}, {i + 4});
    lists_ok &=
        equal_arrays(array({x[i], x[i + 1], x[i + 2], x[i + 3]}, dtype), want);
    lists_ok &= equal_arrays(
        array({x[i], x[i + 1], x[i + 2], x[i + 3]}, {2, 2}, dtype),
        reshape(want, {2, 2}));
  }
  CHECK(scalars_ok);
  CHECK(lists_ok);
}

// array({x}, dtype) gives what astype from the source dtype gives.
template <typename S, typename T>
void check_init_matches_astype(const std::vector<S>& x) {
  auto dtype = TypeToDtype<T>();
  int n = x.size();
  bool ok = true;
  for (auto v : x) {
    ok &= equal_arrays(
        array({v}, dtype),
        astype(array({v}, dtype_of<S>()), dtype, Device::cpu));
  }
  ok &= equal_arrays(
      array(x.begin(), {n}, dtype),
      astype(array(x.begin(), {n}, dtype_of<S>()), dtype, Device::cpu));
  CHECK(ok);
}

} // namespace

TEST_CASE("test array construction from float to integer saturates") {
  // A float to integer conversion is undefined in C++ for NaN and for values
  // that do not fit. Building an array converts like astype.
  for_each_int_type([](const auto& rows) {
    using T = std::decay_t<decltype(rows[0].out)>;
    INFO("to " << dtype_name(TypeToDtype<T>()));
    {
      INFO("from float16");
      auto [x, e] = rows_as<float16_t>(rows, 40);
      check_init_to_int(x, e);
    }
    {
      INFO("from bfloat16");
      auto [x, e] = rows_as<bfloat16_t>(rows, 40);
      check_init_to_int(x, e);
    }
    {
      INFO("from float32");
      auto [x, e] = rows_as<float>(rows, 40);
      check_init_to_int(x, e);
    }
    {
      INFO("from float64");
      auto [x, e] = rows_as<double>(rows, 40);
      check_init_to_int(x, e);
    }
  });
}

TEST_CASE(
    "test array construction from complex64 to integer uses the real part") {
  // Like astype, the real part converts and the imaginary part is ignored.
  for_each_int_type([](const auto& rows) {
    using T = std::decay_t<decltype(rows[0].out)>;
    INFO("to " << dtype_name(TypeToDtype<T>()));
    auto [x, e] = complex_rows(rows, 40);
    check_init_to_int(x, e);
    bool ok = true;
    for (size_t i = 0; i < x.size(); ++i) {
      std::complex<float> v = x[i];
      ok &= array(v, TypeToDtype<T>()).template item<T>() == e[i];
    }
    CHECK(ok);
  });
}

TEST_CASE("test array construction from float to integer matches astype") {
  for_each_int_type([](const auto& rows) {
    using T = std::decay_t<decltype(rows[0].out)>;
    INFO("to " << dtype_name(TypeToDtype<T>()));
    check_init_matches_astype<float16_t, T>(rows_as<float16_t>(rows, 0).first);
    check_init_matches_astype<bfloat16_t, T>(
        rows_as<bfloat16_t>(rows, 0).first);
    check_init_matches_astype<float, T>(rows_as<float>(rows, 0).first);
    check_init_matches_astype<double, T>(rows_as<double>(rows, 0).first);
    check_init_matches_astype<complex64_t, T>(complex_rows(rows, 0).first);
  });
}
