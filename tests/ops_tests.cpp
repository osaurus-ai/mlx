// Copyright © 2023-2024 Apple Inc.

// Required for using M_PI_2 in MSVC.
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "doctest/doctest.h"

#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/simd/base_simd.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

#if defined(MLX_USE_HIGHWAY_KERNELS)
// In highway_simd_tests.cpp: simd.h declares a global bfloat16_t on arm64.
int highway_float_lanes();
float highway_reduce_sum(const float* lanes);
#endif

TEST_CASE("test copy") {
  array x(1.0);
  auto y = copy(x);
  CHECK_EQ(y.shape(), Shape{});
  CHECK_NE(y.id(), x.id());
  CHECK_EQ(y.item<float>(), 1.0f);

  x = array({1, 2}, {2, 1});
  y = copy(x);
  CHECK_EQ(y.shape(), Shape{2, 1});
  CHECK_EQ(y.dtype(), int32);
  CHECK_NE(y.id(), x.id());
  CHECK(array_equal(y, x).item<bool>());
}

TEST_CASE("test reshape") {
  array x(1.0);
  CHECK_EQ(reshape(x, {}).shape(), Shape{});
  CHECK_THROWS_AS(reshape(x, {2}), std::invalid_argument);
  auto y = reshape(x, {1, 1, 1});
  CHECK_EQ(y.shape(), Shape{1, 1, 1});
  y = reshape(x, {-1, 1, 1});
  CHECK_EQ(y.shape(), Shape{1, 1, 1});
  y = reshape(x, {1, 1, -1});
  CHECK_EQ(y.shape(), Shape{1, 1, 1});
  CHECK_THROWS_AS(reshape(x, {1, -1, -1}), std::invalid_argument);
  CHECK_THROWS_AS(reshape(x, {2, -1}), std::invalid_argument);

  x = zeros({2, 2, 2});
  y = reshape(x, {8});
  CHECK_EQ(y.shape(), Shape{8});
  CHECK_THROWS_AS(reshape(x, {7}), std::invalid_argument);
  y = reshape(x, {-1});
  CHECK_EQ(y.shape(), Shape{8});
  y = reshape(x, {-1, 2});
  CHECK_EQ(y.shape(), Shape{4, 2});
  CHECK_THROWS_AS(reshape(x, {-1, 7}), std::invalid_argument);

  // Works with empty array
  x = array({});
  y = reshape(x, {0, 0, 0});
  CHECK_EQ(y.shape(), Shape{0, 0, 0});
  y.eval();
  CHECK_EQ(y.size(), 0);
  CHECK_THROWS_AS(reshape(x, {}), std::invalid_argument);
  CHECK_THROWS_AS(reshape(x, {1}), std::invalid_argument);
  y = reshape(x, {1, 5, 0});
  CHECK_EQ(y.shape(), Shape{1, 5, 0});

  // Check that reshaping a transposed array doesn't result in a copy
  x = reshape(arange(64), {2, 4, 8});
  x.eval();
  CHECK_EQ(x.strides()[0], 32);
  CHECK_EQ(x.strides()[1], 8);
  CHECK_EQ(x.strides()[2], 1);
  y = reshape(transpose(x, {0, 2, 1}), {2, 4, 2, 4});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 2);
  CHECK_EQ(y.strides()[2], 1);
  CHECK_EQ(y.strides()[3], 8);
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Split transposed (2, 8, 4) -> (2, 8, 2, 2)
  y = reshape(transpose(x, {0, 2, 1}), {2, 8, 2, 2});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 1);
  CHECK_EQ(y.strides()[2], 16);
  CHECK_EQ(y.strides()[3], 8);
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Split transposed (2, 8, 4) -> (2, 8, 2, 1, 2)
  y = reshape(transpose(x, {0, 2, 1}), {2, 8, 2, 1, 2});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 1);
  CHECK_EQ(y.strides()[2], 16);
  // y.strides()[3] can be anything since y.shape()[3] == 1
  CHECK_EQ(y.strides()[4], 8);
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Split transposed (2, 8, 4) -> (2, 8, 2, 1, 2, 1)
  y = reshape(transpose(x, {0, 2, 1}), {2, 8, 2, 1, 2, 1});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 1);
  CHECK_EQ(y.strides()[2], 16);
  // y.strides()[3] can be anything since y.shape()[3] == 1
  CHECK_EQ(y.strides()[4], 8);
  // y.strides()[5] can be anything since y.shape()[5] == 1
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Check contiguity preservation
  x = ones({10, 10});
  eval(x);
  CHECK(x.flags().row_contiguous);
  CHECK(!x.flags().col_contiguous);
  y = reshape(x, {2, 5, 10});
  eval(y);
  CHECK(y.flags().row_contiguous);
  CHECK(!y.flags().col_contiguous);
  y = reshape(x, {10, 1, 10, 1});
  eval(y);
  CHECK(y.flags().row_contiguous);
  CHECK(!y.flags().col_contiguous);
  x = transpose(x, {1, 0});
  eval(x);
  CHECK(!x.flags().row_contiguous);
  CHECK(x.flags().col_contiguous);
  y = reshape(x, {2, 5, 10});
  eval(y);
  // Strides (5, 1, 10) are neither row nor col major
  CHECK(!y.flags().row_contiguous);
  CHECK(!y.flags().col_contiguous);
  y = reshape(x, {2, 50});
  eval(y);
  CHECK(y.flags().row_contiguous);
  CHECK(!y.flags().col_contiguous);
  y = reshape(x, {10, 1, 10, 1});
  eval(y);
  CHECK(!y.flags().row_contiguous);
  CHECK(y.flags().col_contiguous);
}

TEST_CASE("test flatten") {
  array x = zeros({2, 3, 4});
  CHECK_EQ(flatten(x).shape(), Shape({2 * 3 * 4}));

  CHECK_EQ(flatten(x, 1, 1).shape(), Shape({2, 3, 4}));
  CHECK_EQ(flatten(x, 1, 2).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, 1, 3).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, 1, -1).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, -2, -1).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, -3, -1).shape(), Shape({2 * 3 * 4}));
  CHECK_EQ(flatten(x, -4, -1).shape(), Shape({2 * 3 * 4}));

  // Check start > end throws
  CHECK_THROWS(flatten(x, 2, 1));

  // Check start >= ndim throws
  CHECK_THROWS(flatten(x, 5, 6));

  // Check end < 0 throws
  CHECK_THROWS(flatten(x, -5, -4));

  // Check scalar flattens to 1D
  x = array(1);
  CHECK_EQ(flatten(x, -3, -1).shape(), Shape({1}));
  CHECK_EQ(flatten(x, 0, 0).shape(), Shape({1}));
}

TEST_CASE("test unflatten") {
  array x = array(1);
  CHECK_THROWS(unflatten(x, 0, {1, 1}));

  x = array({1});
  auto out = unflatten(x, 0, {1, 1});
  CHECK_EQ(out.shape(), Shape({1, 1}));
  CHECK_THROWS(unflatten(x, 1, {1, 1}));
  CHECK_THROWS(unflatten(x, 0, {-1, -1}));
  CHECK_THROWS(unflatten(x, 0, {-1, 2}));
  CHECK_THROWS(unflatten(x, 0, {}));

  x = zeros({4, 8});
  out = unflatten(x, 1, {2, 2, 2});
  CHECK_EQ(out.shape(), Shape({4, 2, 2, 2}));
}

TEST_CASE("test squeeze and expand") {
  array x = zeros({2, 1, 2, 1, 2, 1});
  CHECK_EQ(squeeze(x).shape(), Shape{2, 2, 2});
  CHECK_EQ(squeeze(x, {1, 3, 5}).shape(), Shape{2, 2, 2});
  CHECK_EQ(squeeze(x, {-1, -3, -5}).shape(), Shape{2, 2, 2});
  CHECK_EQ(squeeze(x, 1).shape(), Shape{2, 2, 1, 2, 1});
  CHECK_EQ(squeeze(x, -1).shape(), Shape{2, 1, 2, 1, 2});

  CHECK_THROWS(squeeze(x, 0));
  CHECK_THROWS(squeeze(x, 2));
  CHECK_THROWS(squeeze(x, {1, 3, 1}));
  CHECK_THROWS(squeeze(x, {1, 3, -3}));

  // Out of bounds negative axes must throw and not wrap around
  x = zeros({1, 1, 1});
  CHECK_THROWS(squeeze(x, std::vector<int>{-4}));
  CHECK_THROWS(squeeze(x, {-5, 0}));
  CHECK_THROWS(squeeze(x, {0, 4}));

  x = zeros({2, 2});
  CHECK_EQ(expand_dims(x, 0).shape(), Shape{1, 2, 2});
  CHECK_EQ(expand_dims(x, -1).shape(), Shape{2, 2, 1});
  CHECK_EQ(expand_dims(x, 1).shape(), Shape{2, 1, 2});
  CHECK_EQ(expand_dims(x, {0, 1, 2}).shape(), Shape{1, 1, 1, 2, 2});
  CHECK_EQ(
      expand_dims(x, {0, 1, 2, 5, 6, 7}).shape(),
      Shape{1, 1, 1, 2, 2, 1, 1, 1});

  CHECK_THROWS(expand_dims(x, 3));
  CHECK_THROWS(expand_dims(x, -4));
  CHECK_THROWS(expand_dims(x, {0, 1, 0}));
  CHECK_THROWS(expand_dims(x, {0, 1, -4}));

  // Negative axes are resolved against the output shape and sorted
  CHECK_EQ(expand_dims(x, {3, -4}).shape(), Shape{1, 2, 2, 1});
  CHECK_EQ(expand_dims(x, {-1, -4}).shape(), Shape{1, 2, 2, 1});

  // Out of bounds negative axes must throw and not wrap around
  CHECK_THROWS(expand_dims(x, {-5, -4}));
  CHECK_THROWS(expand_dims(x, {-6, 0}));

  x = zeros({});
  CHECK_EQ(expand_dims(x, {-2, -1}).shape(), Shape{1, 1});
  CHECK_THROWS(expand_dims(x, {-3, -2}));
}

TEST_CASE("test slice") {
  array x = array(3);
  auto out = slice(x, {}, {});
  CHECK_EQ(out.item<int>(), 3);
  CHECK_THROWS_AS(slice(x, {1}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(slice(x, {}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(slice(x, {0}, {}), std::invalid_argument);

  x = array({3});
  out = slice(x, {0}, {1});
  CHECK_EQ(out.item<int>(), 3);
  out = slice(x, {-1}, {1});
  CHECK_EQ(out.item<int>(), 3);

  out = slice(x, {-3}, {10});
  CHECK_EQ(out.item<int>(), 3);

  out = slice(x, {1}, {0});
  eval(out);
  CHECK_EQ(out.shape(), Shape{0});

  out = slice(x, {0}, {1}, {1});
  CHECK_EQ(out.item<int>(), 3);

  out = slice(x, {0}, {1}, {10});
  CHECK_EQ(out.item<int>(), 3);

  x = array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 4});
  out = slice(x, {0, 0}, {2, 2});
  CHECK(array_equal(out, array({0, 1, 4, 5}, {2, 2})).item<bool>());

  out = slice(x, {0, 0}, {0, 2});
  CHECK(array_equal(out, reshape(array({}), {0, 2})).item<bool>());

  out = slice(x, {0, 2}, {2, 3});
  CHECK(array_equal(out, array({2, 6}, {2, 1})).item<bool>());

  out = slice(x, {0, 0}, {2, 4}, {1, 2});
  CHECK(array_equal(out, array({0, 2, 4, 6}, {2, 2})).item<bool>());

  // Check contiguity preservation
  x = ones({10, 10});
  eval(x);
  CHECK(x.flags().row_contiguous);
  CHECK(!x.flags().col_contiguous);
  out = slice(x, {0, 0}, {10, 5});
  eval(out);
  CHECK(!out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);
  out = slice(x, {0, 0}, {5, 10});
  eval(out);
  CHECK(out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);
  x = transpose(x, {1, 0});
  eval(x);
  CHECK(!x.flags().row_contiguous);
  CHECK(x.flags().col_contiguous);
  out = slice(x, {0, 0}, {10, 5});
  eval(out);
  CHECK(!out.flags().row_contiguous);
  CHECK(out.flags().col_contiguous);
  out = slice(x, {0, 0}, {5, 10});
  eval(out);
  CHECK(!out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);

  x = ones({6, 4, 10});
  out = slice(x, {0, 0, 0}, {6, 4, 10}, {2, 1, 2});
  eval(out);
  CHECK(!out.flags().contiguous);
  CHECK(!out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);

  // Check data size correctness
  x = ones({4});
  out = slice(x, {0}, {2});
  eval(out);
  CHECK_EQ(out.data_size(), 2);

  out = slice(x, {2}, {4});
  eval(out);
  CHECK_EQ(out.data_size(), 2);

  out = slice(x, {0}, {4}, {2});
  eval(out);
  CHECK_EQ(out.data_size(), 3);

  x = ones({4, 4});
  out = slice(x, {0, 0}, {2, 4});
  eval(out);
  CHECK_EQ(out.data_size(), 8);

  out = slice(x, {0, 0}, {1, 2});
  eval(out);
  CHECK_EQ(out.data_size(), 2);

  out = slice(x, {0, 1}, {4, 4});
  eval(out);
  CHECK_EQ(out.data_size(), 15);

  out = slice(x, {1, 2}, {3, 4});
  eval(out);
  CHECK_EQ(out.data_size(), 6);

  x = ones({4, 4, 4});
  out = slice(x, {0, 0, 0}, {4, 2, 2});
  eval(out);
  CHECK_EQ(out.data_size(), 54);

  x = ones({4, 4, 4});
  out = slice(x, {2, 2, 2}, {3, 3, 3});
  eval(out);
  CHECK_EQ(out.data_size(), 1);

  x = ones({4, 4, 4});
  out = slice(x, {2, 2, 2}, {3, 4, 3});
  eval(out);
  CHECK_EQ(out.data_size(), 5);

  x = ones({8});
  out = slice(x, {7}, {-9}, {-1});
  eval(out);
  CHECK_EQ(out.data_size(), 8);

  out = slice(x, {7}, {-9}, {-1});
  eval(out);
  CHECK_EQ(out.data_size(), 8);

  x = ones({4, 2});
  out = slice(x, {3, 0}, {-5, 2}, {-1, 1});
  eval(out);
  CHECK_EQ(out.data_size(), 8);

  x = as_strided(square(arange(16, float32)), {3, 2, 2}, {1, 1, 5}, 0);
  out = slice(x, {0, 0, 0}, {2, 2, 2});
  eval(out);
  CHECK_EQ(out.data_size(), 8);
  CHECK_FALSE(out.flags().contiguous);
  CHECK_EQ(sum(out).item<float>(), 152.0f);

  x = reshape(arange(6), {2, 3});
  x = slice(x, {1, 0}, {-3, 3}, {-1, 1});
  out = slice(x, {1, 0}, {-3, 3}, {-1, 1});
  eval(out);
  CHECK(out.flags().contiguous);
  CHECK(out.flags().row_contiguous);
  CHECK(array_equal(reshape(transpose(out), {-1}), array({0, 3, 1, 4, 2, 5}))
            .item<bool>());
}

TEST_CASE("test slice update") {
  array x = array({0., 0., 0., 0., 0., 0., 0., 0.}, {8}, float32);
  array y = array(
      {
          1.,
          2.,
          3.,
          4.,
      },
      {4},
      float32);

  auto out = slice_update(x, y, {2}, {6}, {1});
  CHECK(array_equal(slice(out, {2}, {6}, {1}), y).item<bool>());

  out = slice_update(x, y, {5}, {1}, {-1});
  CHECK(array_equal(slice(out, {5}, {1}, {-1}), y).item<bool>());

  x = reshape(x, {2, 4});
  out = slice_update(x, y, {0, 0}, {2, 4}, {1, 1});
  out = reshape(out, {8});
  CHECK(array_equal(slice(out, {0}, {4}, {1}), y).item<bool>());
  CHECK(array_equal(slice(out, {4}, {8}, {1}), y).item<bool>());
}

TEST_CASE("test slice update add") {
  // Basic slice update add
  auto x = zeros({8}, float32);
  auto y = ones({4}, float32);
  auto out = slice_update_add(x, y, {2}, {6}, {1});
  auto expected = array({0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  CHECK(array_equal(out, expected).item<bool>());

  // Overlapping slice update add
  x = zeros({8}, float32);
  y = ones({4}, float32);
  out = slice_update_add(x, y, {2}, {6}, {1});
  out = slice_update_add(out, y, {4}, {8}, {1});
  expected = array({0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f});
  CHECK(array_equal(out, expected).item<bool>());

  // Slice update add with stride
  x = zeros({10}, float32);
  y = ones({3}, float32);
  out = slice_update_add(x, y, {1}, {7}, {2});
  expected =
      array({0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  CHECK(array_equal(out, expected).item<bool>());

  // 2D slice update add
  x = zeros({4, 4}, float32);
  y = ones({2, 2}, float32);
  out = slice_update_add(x, y, {1, 1}, {3, 3}, {1, 1});
  expected = reshape(
      array(
          {0.0f,
           0.0f,
           0.0f,
           0.0f,
           0.0f,
           1.0f,
           1.0f,
           0.0f,
           0.0f,
           1.0f,
           1.0f,
           0.0f,
           0.0f,
           0.0f,
           0.0f,
           0.0f},
          {4, 4}),
      {4, 4});
  CHECK(array_equal(out, expected).item<bool>());

  // Overlapping 2D slice update add
  x = zeros({4, 4}, float32);
  y = ones({2, 2}, float32);
  out = slice_update_add(x, y, {0, 0}, {2, 2}, {1, 1});
  out = slice_update_add(out, y, {1, 1}, {3, 3}, {1, 1});
  expected = reshape(
      array(
          {1.0f,
           1.0f,
           0.0f,
           0.0f,
           1.0f,
           2.0f,
           1.0f,
           0.0f,
           0.0f,
           1.0f,
           1.0f,
           0.0f,
           0.0f,
           0.0f,
           0.0f,
           0.0f},
          {4, 4}),
      {4, 4});
  CHECK(array_equal(out, expected).item<bool>());

  // Slice update add with different dtypes
  x = zeros({4}, int32);
  y = ones({2}, int32);
  out = slice_update_add(x, y, {1}, {3}, {1});
  expected = array({0, 1, 1, 0});
  CHECK(array_equal(out, expected).item<bool>());

  // Empty slice update add
  x = arange(4, float32);
  y = array({});
  out = slice_update_add(x, y, {0}, {0}, {1});
  CHECK(array_equal(out, x).item<bool>());

  // Full array slice update add
  x = ones({4}, float32);
  y = full({4}, 2.0f, float32);
  out = slice_update_add(x, y, {0}, {4}, {1});
  expected = array({3.0f, 3.0f, 3.0f, 3.0f});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test dynamic slice") {
  auto src = reshape(arange(6), {2, 3});
  CHECK_THROWS(slice(src, array({1, 0, 0}), {0, 0, 0}, {1, 1}));
  CHECK_THROWS(slice(src, array({1, 0}), {0}, {1, 1}));
  CHECK_THROWS(slice(src, array({1}), {3}, {1, 1}));
  CHECK_THROWS(slice(src, array({1, 0}), {0, 0}, {1, 1}));

  CHECK_THROWS(slice(src, array({1}), {0}, {2, 4}));
  CHECK_THROWS(slice(src, array({1.0f}, float32), {0}, {1, 1}));

  auto out = slice(src, array({1}), {0}, {1, 2});
  auto expected = array({3, 4}, {1, 2});
  CHECK(array_equal(out, expected).item<bool>());

  out = slice(src, array({1, 1}), {0, 1}, {1, 2});
  expected = array({4, 5}, {1, 2});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test dynamic slice update") {
  auto src = zeros({2, 3}, int32);
  auto upd = ones({1, 2}, int32);
  CHECK_THROWS(slice_update(src, upd, array({1, 0, 0}), {0, 0, 0}));
  CHECK_THROWS(slice_update(src, upd, array({1, 0}), {0}));
  CHECK_THROWS(slice_update(src, upd, array({1}), {3}));
  CHECK_THROWS(slice_update(src, upd, array({1, 0}), {0, 0}));

  upd = ones({4}, int32);
  CHECK_THROWS(slice_update(src, upd, array({1}), {0}));
  upd = ones({1, 4}, int32);
  CHECK_THROWS(slice_update(src, upd, array({1}), {0}));
  CHECK_THROWS(slice_update(src, upd, array({1.0f}, float32), {0}));

  upd = ones({1, 2}, int32);
  auto out = slice_update(src, upd, array({1}), {0});
  auto expected = reshape(array({0, 0, 0, 1, 1, 0}), {2, 3});
  CHECK(array_equal(out, expected).item<bool>());

  upd = ones({1, 2}, int32);
  out = slice_update(src, upd, array({1, 1}), {0, 1});
  expected = reshape(array({0, 0, 0, 0, 1, 1}), {2, 3});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test split") {
  array x = array(1);
  CHECK_THROWS(split(x, 0));

  // Regression: non-scalar split with num_splits <= 0
  CHECK_THROWS(split(array({0, 1, 2, 3, 4, 5}), 0));
  CHECK_THROWS(split(array({0, 1, 2, 3, 4, 5}), -1));

  x = array({3});
  CHECK_EQ(split(x, 1)[0].item<int>(), 3);

  x = array({0, 1, 2});
  CHECK_THROWS(split(x, 3, 1));
  CHECK_THROWS(split(x, 3, -2));

  auto out = split(x, 3, 0);
  CHECK_EQ(out.size(), 3);

  out = split(x, 3, -1);
  CHECK_EQ(out.size(), 3);
  for (auto i = 0; i < 3; ++i) {
    CHECK_EQ(out[i].shape(), Shape{1});
    CHECK_EQ(out[i].dtype(), int32);
    CHECK_EQ(out[i].item<int>(), i);
  }

  x = array({0, 1, 2, 3, 4, 5}, {2, 3});
  out = split(x, 2);
  CHECK(array_equal(out[0], array({0, 1, 2}, {1, 3})).item<bool>());
  CHECK(array_equal(out[1], array({3, 4, 5}, {1, 3})).item<bool>());
  out = split(x, 3, 1);
  CHECK(array_equal(out[0], array({0, 3}, {2, 1})).item<bool>());
  CHECK(array_equal(out[1], array({1, 4}, {2, 1})).item<bool>());
  CHECK(array_equal(out[2], array({2, 5}, {2, 1})).item<bool>());

  x = zeros({8, 12});
  out = split(x, 2);
  CHECK_EQ(out.size(), 2);
  CHECK_EQ(out[0].shape(), Shape{4, 12});
  CHECK_EQ(out[1].shape(), Shape{4, 12});
  out = split(x, 3, 1);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].shape(), Shape{8, 4});
  CHECK_EQ(out[1].shape(), Shape{8, 4});
  CHECK_EQ(out[2].shape(), Shape{8, 4});

  out = split(x, Shape{});
  CHECK_EQ(out.size(), 1);
  CHECK_EQ(out[0].shape(), x.shape());

  out = split(x, {3, 7});
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].shape(), Shape{3, 12});
  CHECK_EQ(out[1].shape(), Shape{4, 12});
  CHECK_EQ(out[2].shape(), Shape{1, 12});

  out = split(x, Shape{20});
  CHECK_EQ(out.size(), 2);
  CHECK_EQ(out[0].shape(), Shape{8, 12});
  CHECK_EQ(out[1].shape(), Shape{0, 12});

  // Negative indices
  out = split(x, Shape{-5});
  CHECK_EQ(out[0].shape(), Shape{3, 12});
  CHECK_EQ(out[1].shape(), Shape{5, 12});

  // Different axis
  out = split(x, {2, 8}, 1);
  CHECK_EQ(out[0].shape(), Shape{8, 2});
  CHECK_EQ(out[1].shape(), Shape{8, 6});
  CHECK_EQ(out[2].shape(), Shape{8, 4});

  // Out of order indices
  x = arange(5);
  out = split(x, {2, 1, 2});
  CHECK(array_equal(out[0], array({0, 1})).item<bool>());
  CHECK(array_equal(out[1], array({})).item<bool>());
  CHECK(array_equal(out[2], array({1})).item<bool>());
  CHECK(array_equal(out[3], array({2, 3, 4})).item<bool>());

  // data_size is the distance between the first and the last element, so a
  // split that leaves holes reports more than the output has elements.
  x = ones({4, 4});
  out = split(x, 2, 1);
  eval(out);
  CHECK_EQ(out[0].data_size(), 14);
  CHECK_EQ(out[1].data_size(), 14);

  // Splitting the first axis leaves no holes.
  out = split(x, 2, 0);
  eval(out);
  CHECK_EQ(out[0].data_size(), 8);
  CHECK_EQ(out[1].data_size(), 8);

  x = ones({2, 3, 4});
  out = split(x, Shape{2}, 2);
  eval(out);
  CHECK_EQ(out[0].data_size(), 22);
  CHECK_EQ(out[1].data_size(), 22);

  // A split output and the matching slice are the same view.
  x = ones({1, 2, 8, 12});
  out = split(x, Shape{8}, 3);
  auto sliced = slice(x, {0, 0, 0, 8}, {1, 2, 8, 12});
  eval(out);
  eval(sliced);
  CHECK_EQ(out[1].data_size(), sliced.data_size());
  CHECK_EQ(out[1].data_size(), 184);

  // (4, 4) array, but slice keeps the strides at (8, 1)
  x = slice(reshape(arange(32), {4, 8}), {0, 0}, {4, 4});
  // out[0] and out[1] are (4, 2)
  out = split(x, Shape{2}, 1);
  // same out out[1]
  auto holey = slice(x, {0, 2}, {4, 4});
  eval(out);
  eval(holey);
  CHECK_EQ(out[0].data_size(), 26);
  CHECK_EQ(out[1].data_size(), 26);
  CHECK_EQ(out[1].data_size(), holey.data_size());
  CHECK_EQ(out[0].strides(), x.strides());
  CHECK_EQ(out[1].strides(), holey.strides());
  CHECK(array_equal(out[1], holey).item<bool>());

  // A negative stride travels backwards, so it bounds the span from below.
  // strides (-4, 1)
  x = flip(reshape(arange(24), {6, 4}), 0);
  out = split(x, Shape{2}, 0);
  eval(out);
  // no holes
  CHECK_EQ(out[0].data_size(), 8);
  CHECK_EQ(out[1].data_size(), 16);
  CHECK_EQ(out[0].strides(), x.strides());
  CHECK_EQ(out[1].strides(), x.strides());
  CHECK(array_equal(out[0], slice(x, {0, 0}, {2, 4})).item<bool>());
  CHECK(array_equal(out[1], slice(x, {2, 0}, {6, 4})).item<bool>());

  // A zero-length axis empties every output. Splitting at 0 cannot get here,
  // split() sends that to slice().
  x = ones({0, 4});
  out = split(x, Shape{2}, 1);
  eval(out);
  CHECK_EQ(out[0].data_size(), 0);
  CHECK_EQ(out[1].data_size(), 0);
}

TEST_CASE("test flip") {
  array x = array({1, 2, 3, 4});
  CHECK(array_equal(flip(x), array({4, 3, 2, 1})).item<bool>());

  x = array({0, 1, 2, 3, 4, 5}, {2, 3});
  CHECK(
      array_equal(flip(x, 0), array({3, 4, 5, 0, 1, 2}, {2, 3})).item<bool>());
  CHECK(
      array_equal(flip(x, 1), array({2, 1, 0, 5, 4, 3}, {2, 3})).item<bool>());
  CHECK(
      array_equal(flip(x, -1), array({2, 1, 0, 5, 4, 3}, {2, 3})).item<bool>());
  // No axes -> flip all.
  CHECK(array_equal(flip(x), array({5, 4, 3, 2, 1, 0}, {2, 3})).item<bool>());
  CHECK(array_equal(
            flip(x, std::vector<int>{0, 1}), array({5, 4, 3, 2, 1, 0}, {2, 3}))
            .item<bool>());

  CHECK_THROWS(flip(x, 2));
}

TEST_CASE("test unstack") {
  array x = array({0, 1, 2, 3, 4, 5}, {3, 2});
  auto out = unstack(x);
  CHECK_EQ(out.size(), 3);
  CHECK(array_equal(out[0], array({0, 1})).item<bool>());
  CHECK(array_equal(out[1], array({2, 3})).item<bool>());
  CHECK(array_equal(out[2], array({4, 5})).item<bool>());
  CHECK_EQ(out[0].shape(), Shape{2});

  out = unstack(x, 1);
  CHECK_EQ(out.size(), 2);
  CHECK(array_equal(out[0], array({0, 2, 4})).item<bool>());
  CHECK(array_equal(out[1], array({1, 3, 5})).item<bool>());

  // stack is the inverse of unstack.
  CHECK(array_equal(stack(unstack(x, 1), 1), x).item<bool>());

  CHECK_THROWS(unstack(x, 2));
}

TEST_CASE("test swap and move axes") {
  // Test swapaxes
  array a(0.0);
  CHECK_THROWS(swapaxes(a, 0, 0));

  a = zeros({2});
  CHECK_THROWS(swapaxes(a, 0, 1));
  CHECK_EQ(swapaxes(a, 0, 0).shape(), Shape{2});
  CHECK_EQ(swapaxes(a, -1, -1).shape(), Shape{2});

  a = zeros({2, 3, 4});
  CHECK_THROWS(swapaxes(a, 0, -4));
  CHECK_THROWS(swapaxes(a, 0, 3));
  CHECK_THROWS(swapaxes(a, 3, 0));
  CHECK_THROWS(swapaxes(a, -4, 0));
  CHECK_EQ(swapaxes(a, 0, 2).shape(), Shape{4, 3, 2});
  CHECK_EQ(swapaxes(a, 0, 1).shape(), Shape{3, 2, 4});
  CHECK_EQ(swapaxes(a, 0, -1).shape(), Shape{4, 3, 2});
  CHECK_EQ(swapaxes(a, -2, 2).shape(), Shape{2, 4, 3});

  // Test moveaxis
  a = array(0.0);
  CHECK_THROWS(moveaxis(a, 0, 0));

  a = zeros({2});
  CHECK_THROWS(moveaxis(a, 0, 1));
  CHECK_EQ(moveaxis(a, 0, 0).shape(), Shape{2});
  CHECK_EQ(moveaxis(a, -1, -1).shape(), Shape{2});

  a = zeros({2, 3, 4});
  CHECK_THROWS(moveaxis(a, 0, -4));
  CHECK_THROWS(moveaxis(a, 0, 3));
  CHECK_THROWS(moveaxis(a, 3, 0));
  CHECK_THROWS(moveaxis(a, -4, 0));
  CHECK_EQ(moveaxis(a, 0, 2).shape(), Shape{3, 4, 2});
  CHECK_EQ(moveaxis(a, 0, 1).shape(), Shape{3, 2, 4});
  CHECK_EQ(moveaxis(a, 0, -1).shape(), Shape{3, 4, 2});
  CHECK_EQ(moveaxis(a, -2, 2).shape(), Shape{2, 4, 3});
}

TEST_CASE("test transpose") {
  array x(1);
  auto y = transpose(x);
  CHECK_EQ(y.shape(), Shape{});
  CHECK_EQ(y.item<int>(), 1);
  CHECK_THROWS_AS(transpose(x, {0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {1}), std::invalid_argument);

  x = array({1}, {1});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{1});
  CHECK_EQ(y.item<int>(), 1);

  // Negative indices
  y = transpose(x, {-1});
  CHECK_EQ(y.shape(), Shape{1});
  CHECK_EQ(y.item<int>(), 1);

  CHECK_THROWS_AS(transpose(x, {1}), std::out_of_range);
  CHECK_THROWS_AS(transpose(x, {0, 0}), std::invalid_argument);

  // Works with empty array
  x = array({});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{0});
  y.eval();
  CHECK_EQ(y.size(), 0);

  x = array({1, 2, 3, 4, 5, 6}, {2, 3});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{3, 2});
  y = transpose(x, {-1, 0});
  CHECK_EQ(y.shape(), Shape{3, 2});
  y = transpose(x, {-1, -2});
  CHECK_EQ(y.shape(), Shape{3, 2});
  y.eval();
  CHECK(array_equal(y, array({1, 4, 2, 5, 3, 6}, {3, 2})).item<bool>());
  y = transpose(x, {0, 1});
  CHECK_EQ(y.shape(), Shape{2, 3});
  CHECK(array_equal(y, x).item<bool>());
  y = transpose(x, {0, -1});
  CHECK_EQ(y.shape(), Shape{2, 3});
  CHECK(array_equal(y, x).item<bool>());

  CHECK_THROWS_AS(transpose(x, {}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0, 0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0, 1, 1}), std::invalid_argument);

  x = array({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{2, 3, 2});
  auto expected = array({1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12}, {2, 3, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = transpose(x, {0, 1, 2});
  CHECK_EQ(y.shape(), Shape{2, 3, 2});
  CHECK(array_equal(y, x).item<bool>());
  y = transpose(x, {1, 0, 2});
  CHECK_EQ(y.shape(), Shape{3, 2, 2});
  expected = array({1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}, {3, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());
  y = transpose(x, {0, 2, 1});
  CHECK_EQ(y.shape(), Shape{2, 2, 3});
  expected = array({1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12}, {2, 2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  // Check reshaping a transposed array
  x = array({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2});
  x = reshape(transpose(x), {2, 2, 2});
  expected = array({0, 2, 4, 6, 1, 3, 5, 7}, {2, 2, 2});
  CHECK(array_equal(x, expected).item<bool>());

  // Check maintaining contiguous status
  x = array({0, 1, 2, 3, 4, 5, 6, 7}, {1, 4, 1, 2});
  CHECK(x.flags().row_contiguous);
  x = transpose(x, {2, 1, 0, 3});
  eval(x);
  CHECK(x.flags().row_contiguous);
}

TEST_CASE("test comparison ops") {
  // Empty array
  {
    array x({});
    array y({});
    auto z = x == y;
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.shape(), Shape{0});
  }

  // Basic cases
  {
    array x(1.0);
    array y(1.0);
    CHECK(equal(x, y).item<bool>());
    CHECK((x == y).item<bool>());
    CHECK((x == 1.0f).item<bool>());
    CHECK((1.0f == y).item<bool>());

    CHECK(!(x != y).item<bool>());
    CHECK(!not_equal(x, y).item<bool>());
    CHECK(!(1.0f != y).item<bool>());
    CHECK(!(x != 1.0f).item<bool>());

    CHECK(array_equal(x, y).item<bool>());

    x = array(0.0);
    CHECK(!equal(x, y).item<bool>());
    CHECK(!array_equal(x, y).item<bool>());
    CHECK(not_equal(x, y).item<bool>());
  }

  // Greater and less
  {
    array x(1.0);
    array y(0.0);
    CHECK(greater(x, y).item<bool>());
    CHECK((x > 0.0f).item<bool>());
    CHECK((1.0f > y).item<bool>());
    CHECK(greater_equal(x, y).item<bool>());
    CHECK((1.0f >= y).item<bool>());
    CHECK(!(x > 1.0f).item<bool>());
    CHECK((x >= 1.0f).item<bool>());

    CHECK(less(y, x).item<bool>());
    CHECK((y < 1.0).item<bool>());
    CHECK((y <= 1.0f).item<bool>());
    CHECK(!(x < 1.0).item<bool>());
    CHECK((x <= 1.0f).item<bool>());
  }

  // Check array_equal works
  {
    auto x = zeros({5, 5});
    auto y = zeros({5, 5});
    CHECK(array_equal(x, y).item<bool>());

    x = zeros({1, 1});
    CHECK(!array_equal(x, y).item<bool>());

    x = ones({5, 5});
    CHECK(!array_equal(x, y).item<bool>());

    x = array({0.0f, 1.0f, NAN});
    y = array({0.0f, 1.0f, NAN});
    CHECK(!array_equal(x, y).item<bool>());
    CHECK(array_equal(x, y, true).item<bool>());
  }

  // Check other types
  {
    auto x = zeros({5, 5}, int32);
    auto y = zeros({5, 5}, int32);
    CHECK(array_equal(x, y).item<bool>());

    x = ones({5, 5}, bool_);
    y = ones({5, 5}, bool_);
    CHECK(array_equal(x, y).item<bool>());
  }

  // Check type promotion
  {
    array x(1.0f);
    array y(1);
    CHECK_EQ(equal(x, y).item<bool>(), true);

    x = array(true, bool_);
    CHECK_EQ(equal(x, y).item<bool>(), true);
  }

  // Broadcasting works
  {
    auto x = zeros({1, 2});
    auto y = zeros({2, 1});
    auto z = equal(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.shape(), Shape{2, 2});
    auto expected = array({true, true, true, true}, {2, 2});
    CHECK(array_equal(z, expected).item<bool>());

    x = array({1.0, 2.0}, {1, 2});
    y = array({1.0, 2.0}, {2, 1});
    z = equal(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.shape(), Shape{2, 2});
    expected = array({true, false, false, true}, {2, 2});
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({false, true, false, false}, {2, 2});
    z = greater(x, y);
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({true, true, false, true}, {2, 2});
    z = greater_equal(x, y);
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({false, false, true, false}, {2, 2});
    z = less(x, y);
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({true, false, true, true}, {2, 2});
    z = less_equal(x, y);
    CHECK(array_equal(z, expected).item<bool>());
  }
}

TEST_CASE("test float16 sigmoid mixed lanes") {
  const float values[] = {-12, -4, -1, 0, 1, 4, 12, -2};
  for (int size : {1, 7, 8, 9, 15, 16, 17}) {
    for (int offset = 0; offset < 8; ++offset) {
      std::vector<float> input(size);
      for (int i = 0; i < size; ++i) {
        input[i] = values[(i + offset) % 8];
      }
      auto result = sigmoid(array(input.begin(), {size}, float16), Device::cpu);
      result.eval();
      for (int i = 0; i < size; ++i) {
        CAPTURE(size);
        CAPTURE(offset);
        CAPTURE(i);
        auto expected = 1.0 / (1.0 + std::exp(-double(input[i])));
        CHECK_EQ(
            static_cast<float>(result.data<float16_t>()[i]),
            doctest::Approx(expected).epsilon(0.002));
      }
    }
  }
}

TEST_CASE("test is nan") {
  array x(1.0f);
  CHECK_FALSE(isnan(x).item<bool>());

  array y(NAN);
  CHECK(isnan(y).item<bool>());

  array z = identity(7);
  CHECK_FALSE(all(isnan(z)).item<bool>());

  array w = array({1.0f, NAN, 2.0f});
  CHECK_FALSE(all(isnan(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isnan(a).item<bool>());

  array b(1.0f, float16);
  CHECK_FALSE(isnan(b).item<bool>());

  array c(NAN, bfloat16);
  CHECK(isnan(c).item<bool>());

  array d(NAN, float16);
  CHECK(isnan(d).item<bool>());
}

TEST_CASE("test is inf") {
  array x(1.0f);
  CHECK_FALSE(isinf(x).item<bool>());

  auto inf = std::numeric_limits<float>::infinity();

  array y(inf);
  CHECK(isinf(y).item<bool>());

  auto neginf = -std::numeric_limits<float>::infinity();
  CHECK(isinf(array(neginf)).item<bool>());

  array z = identity(7);
  CHECK_FALSE(any(isinf(z)).item<bool>());

  array w = array({1.0f, inf, 2.0f});
  CHECK(array_equal(array({false, true, false}), isinf(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isinf(a).item<bool>());

  array b(1.0f, float16);
  CHECK_FALSE(isinf(b).item<bool>());

  array c(inf, bfloat16);
  CHECK(isinf(c).item<bool>());

  array d(inf, float16);
  CHECK(isinf(d).item<bool>());
}

TEST_CASE("test all close") {
  array x(1.0f);
  array y(1.0f);
  CHECK(allclose(x, y).item<bool>());

  y = array(1.1f);
  CHECK_FALSE(allclose(x, y).item<bool>());
  CHECK(allclose(x, y, 0.1).item<bool>());
  CHECK_FALSE(allclose(x, y, 0.01).item<bool>());
  CHECK(allclose(x, y, 0.01, 0.1).item<bool>());
}

TEST_CASE("test is close") {
  {
    array a({1.0, std::numeric_limits<float>::infinity()});
    array b({1.0, std::numeric_limits<float>::infinity()});
    CHECK(array_equal(isclose(a, b), array({true, true})).item<bool>());
  }
  {
    array a({1.0, -std::numeric_limits<float>::infinity()});
    array b({1.0, -std::numeric_limits<float>::infinity()});
    CHECK(array_equal(isclose(a, b), array({true, true})).item<bool>());
  }
  {
    array a({1.0, std::numeric_limits<float>::infinity()});
    array b({1.0, -std::numeric_limits<float>::infinity()});
    CHECK(array_equal(isclose(a, b), array({true, false})).item<bool>());
  }
  {
    array a({1.0, std::nan("1"), std::nan("1")});
    array b({1.0, std::nan("1"), 2.0});
    CHECK(array_equal(isclose(a, b), array({true, false, false})).item<bool>());
  }
  {
    array a({1.0, std::nan("1"), std::nan("1")});
    array b({1.0, std::nan("1"), 2.0});
    CHECK(
        array_equal(isclose(a, b, 1e-5, 1e-8, true), array({true, true, false}))
            .item<bool>());
  }
}

TEST_CASE("test reduction ops") {
  // Check shapes and throws correctly
  {
    auto x = array(1);
    auto out = sum(x);
    CHECK_EQ(out.ndim(), 0);
    CHECK_THROWS_AS(sum(x, 0), std::out_of_range);
    CHECK_THROWS_AS(sum(x, -1), std::out_of_range);
    out = sum(x, std::vector<int>{});
    CHECK_EQ(out.shape(), Shape{});
    CHECK_EQ(out.size(), 1);

    x = array({});
    out = sum(x);
    CHECK_EQ(out.shape(), Shape{});
    CHECK_EQ(out.size(), 1);
    out = sum(x, true);
    CHECK_EQ(out.shape(), Shape{1});
    out = sum(x, std::vector<int>{});
    CHECK_EQ(out.shape(), x.shape());

    x = zeros({2});
    out = sum(x);
    CHECK_EQ(out.ndim(), 0);
    out = sum(x, -1);
    CHECK_EQ(out.ndim(), 0);
    out = sum(x, -1, true);
    CHECK_EQ(out.ndim(), 1);
    CHECK_EQ(out.shape(), Shape{1});

    CHECK_THROWS_AS(sum(x, 1), std::out_of_range);
    CHECK_THROWS_AS(sum(x, -2), std::out_of_range);
    CHECK_THROWS_AS(sum(x, {0, 0}), std::invalid_argument);
    CHECK_THROWS_AS(sum(x, {-1, 0}), std::invalid_argument);

    x = zeros({2, 3, 4});
    out = sum(x, {0, 2});
    CHECK_EQ(out.shape(), Shape{3});
    out = sum(x, std::vector<int>{});
    CHECK_EQ(out.shape(), x.shape());

    out = sum(x, {0, -1});
    CHECK_EQ(out.shape(), Shape{3});

    out = sum(x, {0, -1}, true);
    CHECK_EQ(out.shape(), Shape{1, 3, 1});

    out = sum(x, true);
    CHECK_EQ(out.shape(), Shape{1, 1, 1});

    out = sum(x);
    CHECK_EQ(out.shape(), Shape{});

    CHECK_THROWS_AS(sum(x, 3), std::out_of_range);
    CHECK_THROWS_AS(sum(x, -4), std::out_of_range);
    CHECK_THROWS_AS(sum(x, {0, 1, -2}), std::invalid_argument);
  }

  // Test sum
  {
    auto x = array({});
    CHECK_EQ(sum(x).item<float>(), 0.0f);

    x = array({1, 2, 3});
    CHECK_EQ(sum(x).item<int>(), 6);
    CHECK(array_equal(sum(x, std::vector<int>{}), x).item<bool>());

    x = ones({2, 3});
    CHECK(array_equal(sum(x, 1), full({2}, 3.0f)).item<bool>());
    CHECK(array_equal(sum(x, 0), full({3}, 2.0f)).item<bool>());
    CHECK_EQ(sum(x, {0, 1}).item<float>(), 6.0f);

    x = ones({2, 3, 4});
    CHECK(array_equal(sum(x, 0), full({3, 4}, 2.0f)).item<bool>());
    CHECK(array_equal(sum(x, 1), full({2, 4}, 3.0f)).item<bool>());
    CHECK(array_equal(sum(x, 2), full({2, 3}, 4.0f)).item<bool>());
    CHECK(array_equal(sum(x, {0, 1}), full({4}, 6.0f)).item<bool>());
    CHECK(array_equal(sum(x, {0, 2}), full({3}, 8.0f)).item<bool>());
    CHECK(array_equal(sum(x, {1, 2}), full({2}, 12.0f)).item<bool>());

    // Output for bool gets higher precision
    x = array({true, true, true});
    CHECK_EQ(sum(x).item<int32_t>(), 3);

    x = array(2.0f);
    x = broadcast_to(x, {2, 2, 2});
    CHECK_EQ(sum(x).item<float>(), 16.0f);

    // Tests with non-uniform results after reduction
    x = array({1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}, {2, 3});
    CHECK(array_equal(sum(x, 0), full({3}, 3.0f)).item<bool>());
    CHECK(array_equal(sum(x, 1), array({3.0f, 6.0f}, {2})).item<bool>());
  }

  // Test unsigned sum
  {
    const int num_elems = 1000;

    auto x = astype(full({num_elems}, 255), uint8);
    CHECK_EQ(sum(x, Device::cpu).item<uint32_t>(), 255 * num_elems);

    x = astype(full({num_elems}, 65535), uint16);
    CHECK_EQ(sum(x, Device::cpu).item<uint32_t>(), 65535 * num_elems);

    x = full({3, 3, 3}, 10000, uint32);
    CHECK_EQ(sum(x, Device::cpu).item<uint32_t>(), 270000);

    x = full({3, 3, 3}, 10000, uint64);
    CHECK_EQ(sum(x, Device::cpu).item<uint64_t>(), 270000);
  }

  // Test prod
  {
    auto x = array({});
    CHECK_EQ(prod(x).item<float>(), 1.0f);

    x = array({2, 2, 2});
    CHECK_EQ(prod(x).item<int>(), 8);
    CHECK(array_equal(prod(x, std::vector<int>{}), x).item<bool>());

    x = full({2, 3}, 2.0f);
    CHECK(array_equal(prod(x, 1), full({2}, 8.0f)).item<bool>());
    CHECK(array_equal(prod(x, 0), full({3}, 4.0f)).item<bool>());
    CHECK_EQ(prod(x, {0, 1}).item<float>(), 64.0f);

    x = full({2, 3, 4}, 2.0f);
    CHECK(array_equal(prod(x, 0), full({3, 4}, 4.0f)).item<bool>());
    CHECK(array_equal(prod(x, 1), full({2, 4}, 8.0f)).item<bool>());
    CHECK(array_equal(prod(x, 2), full({2, 3}, 16.0f)).item<bool>());
    CHECK(array_equal(prod(x, {0, 1}), full({4}, 64.0f)).item<bool>());
    CHECK(array_equal(prod(x, {0, 2}), full({3}, 256.0f)).item<bool>());
    CHECK(array_equal(prod(x, {1, 2}), full({2}, 4096.0f)).item<bool>());

    // Tests with non-uniform results after reduction
    x = array({1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}, {2, 3});
    CHECK(array_equal(prod(x, 0), full({3}, 2.0f)).item<bool>());
    CHECK(array_equal(prod(x, 1), array({1.0f, 8.0f}, {2})).item<bool>());

    x = array({true, true, true, false, true, false}, {2, 3});
    CHECK(array_equal(prod(x, 0), array({false, true, false})).item<bool>());
    CHECK(array_equal(prod(x, 1), array({true, false})).item<bool>());
  }

  // Test unsigned prod
  {
    auto x = array({255, 255}, {2}, uint8);
    CHECK_EQ(prod(x, Device::cpu).item<uint32_t>(), 65025);

    x = array({65535, 2}, {2}, uint16);
    CHECK_EQ(prod(x, Device::cpu).item<uint32_t>(), 131070);

    x = array({100000, 2}, {2}, uint32);
    CHECK_EQ(prod(x, Device::cpu).item<uint32_t>(), 200000);

    x = array({100000, 2}, {2}, uint64);
    CHECK_EQ(prod(x, Device::cpu).item<uint64_t>(), 200000);
  }

  // Test all
  {
    auto x = array({});
    CHECK_EQ(all(x).item<bool>(), true);

    x = array({2, 2, 2});
    CHECK_EQ(all(x).item<bool>(), true);
    auto out = all(x, std::vector<int>{});
    CHECK(array_equal(out, array({true, true, true})).item<bool>());

    x = array({0, 2, 2});
    CHECK_EQ(all(x).item<bool>(), false);

    x = array({true, true, true, false, true, false}, {2, 3});
    CHECK(array_equal(all(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(all(x, 0), array({false, true, false})).item<bool>());
  }

  // Test any
  {
    auto x = array({});
    CHECK_EQ(any(x).item<bool>(), false);

    x = array({0, 0, 0});
    CHECK_EQ(any(x).item<bool>(), false);

    x = array({0, 2, 0});
    CHECK_EQ(any(x).item<bool>(), true);
    auto out = any(x, std::vector<int>{});
    CHECK(array_equal(out, array({false, true, false})).item<bool>());

    x = array({true, false, true, false, false, false}, {2, 3});
    CHECK(array_equal(any(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(any(x, 0), array({true, false, true})).item<bool>());
  }

  // Test max and min
  {
    auto x = array({});
    CHECK_THROWS(max(x));
    CHECK_THROWS(min(x));

    x = array({1.0f, 2.0f, 3.0f});
    CHECK_EQ(max(x).item<float>(), 3.0f);
    CHECK_EQ(min(x).item<float>(), 1.0f);

    x = array({-2.0f, -1.0f});
    CHECK_EQ(max(x).item<float>(), -1.0f);
    CHECK_EQ(min(x).item<float>(), -2.0f);

    constexpr float inf = std::numeric_limits<float>::infinity();
    x = array({inf});
    CHECK_EQ(min(x).item<float>(), inf);

    x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    CHECK(array_equal(max(x, 0), array({4.0f, 5.0f, 6.0f})).item<bool>());
    CHECK(array_equal(max(x, 1), array({3.0f, 6.0f})).item<bool>());
    CHECK(array_equal(min(x, 0), array({1.0f, 2.0f, 3.0f})).item<bool>());
    CHECK(array_equal(min(x, 1), array({1.0f, 4.0f})).item<bool>());

    x = array({1u, 2u, 3u});
    CHECK_EQ(max(x).item<uint32_t>(), 3u);
    CHECK_EQ(min(x).item<uint32_t>(), 1u);

    x = array({1u, 2u, 3u, 4u, 5u, 6u}, {2, 3});
    CHECK(array_equal(max(x, 0), array({4u, 5u, 6u})).item<bool>());
    CHECK(array_equal(max(x, 1), array({3u, 6u})).item<bool>());
    CHECK(array_equal(min(x, 0), array({1u, 2u, 3u})).item<bool>());
    CHECK(array_equal(min(x, 1), array({1u, 4u})).item<bool>());

    x = array({true, false, true, false, false, false}, {2, 3});
    CHECK(array_equal(max(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(max(x, 0), array({true, false, true})).item<bool>());

    x = array({true, true, true, false, true, false}, {2, 3});
    CHECK(array_equal(min(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(min(x, 0), array({false, true, false})).item<bool>());

    x = array({1.0f, NAN, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    CHECK(array_equal(max(x, 0), array({4.0f, NAN, 6.0f}), true).item<bool>());
    CHECK(array_equal(max(x, 1), array({NAN, 6.0f}), true).item<bool>());
  }

  // Test logsumexp
  {
    auto x = array({});
    CHECK_THROWS(logsumexp(x));

    constexpr float inf = std::numeric_limits<float>::infinity();

    x = array({-inf, -inf});
    CHECK_EQ(logsumexp(x).item<float>(), -inf);

    x = repeat(array(-inf), 5000);
    CHECK_EQ(logsumexp(x).item<float>(), -inf);

    x = array({0.0f, -inf});
    CHECK_EQ(logsumexp(x).item<float>(), 0.0f);

    x = array({0.0f, inf});
    CHECK_EQ(logsumexp(x).item<float>(), inf);

    x = reshape(arange(6, float32), {2, 3});

    std::vector<float> nums = {0.0f, 1.0f, 2.0f, 3.0f};
    x = array(nums.data(), {2, 2});
    auto y = logsumexp(x, {0, 1}, true);
    CHECK_EQ(y.shape(), Shape{1, 1});
    auto result = std::log(
        std::exp(nums[0]) + std::exp(nums[1]) + std::exp(nums[2]) +
        std::exp(nums[3]));
    CHECK(y.item<float>() == doctest::Approx(result));
    auto expected = array(
        {std::log(std::exp(nums[0]) + std::exp(nums[2])),
         std::log(std::exp(nums[1]) + std::exp(nums[3]))});
    CHECK(allclose(logsumexp(x, 0), expected).item<bool>());

    expected = array(
        {std::log(std::exp(nums[0]) + std::exp(nums[1])),
         std::log(std::exp(nums[2]) + std::exp(nums[3]))});
    CHECK(allclose(logsumexp(x, 1), expected).item<bool>());
  }

  // Test softmax
  {
    for (auto t : {float16, bfloat16, float32}) {
      const auto rtol = t == float32 ? 1e-5 : 1e-2;
      auto x = array({}, t);
      CHECK(array_equal(x, softmax(x)).item<bool>());

      // all zeros
      x = array({0., 0., 0., 0.}, t);
      auto y = array({0.25, 0.25, 0.25, 0.25}, t);
      CHECK(array_equal(y, softmax(x)).item<bool>());
      CHECK(array_equal(y, softmax(x, -1)).item<bool>());
      CHECK(array_equal(y, softmax(x, std::vector<int>{-1})).item<bool>());
      CHECK(array_equal(y, softmax(x, std::vector<int>{0})).item<bool>());

      auto ones = array(1.0f, t);
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());

      // all ones
      x = array({1., 1., 1., 1.}, t);
      CHECK(array_equal(y, softmax(x)).item<bool>());
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());

      // negative values
      x = array({-1., -2., -3., -4.}, t);
      y = array({0.643914, 0.236883, 0.0871443, 0.0320586}, t);
      CHECK(allclose(y, softmax(x), rtol).item<bool>());
      CHECK(allclose(ones, sum(softmax(x)), rtol).item<bool>());

      // positive and negative values
      x = array({1., 0., -1., 0.}, t);
      y = array({0.534447, 0.196612, 0.0723295, 0.196612}, t);
      CHECK(allclose(y, softmax(x), rtol).item<bool>());
      CHECK(allclose(ones, sum(softmax(x)), rtol).item<bool>());

      // large positive values
      x = array({1000., 1000., 1000.}, t);
      y = array({0.333333, 0.333333, 0.333333}, t);
      CHECK(allclose(y, softmax(x)).item<bool>());
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());

      // large negative values
      x = negative(x);
      CHECK(allclose(y, softmax(x)).item<bool>());
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());
    }
  }
}

TEST_CASE("test irregular binary ops") {
  // 1D strided
  {
    auto x = full({128}, 1.0f);
    auto y = full({64}, 1.0f);
    x = slice(x, {0}, {128}, {4});
    y = slice(y, {0}, {64}, {2});
    CHECK(array_equal(add(x, y), full({32}, 2.0f)).item<bool>());
  }

  // 2D broadcasts
  {
    auto x = full({32, 32}, 4.0f);
    auto y = full({32}, 4.0f);
    CHECK(array_equal(add(x, y), full({32, 32}, 8.0f)).item<bool>());
    y = reshape(y, {32, 1});
    CHECK(array_equal(add(x, y), full({32, 32}, 8.0f)).item<bool>());
    CHECK(array_equal(subtract(y, x), zeros({32, 32})).item<bool>());
  }
}

TEST_CASE("test arithmetic unary ops") {
  // Test negative
  {
    array x(1.0f);
    CHECK_EQ(negative(x).item<float>(), -1.0f);
    CHECK_EQ((-x).item<float>(), -1.0f);

    // works on empty array
    CHECK(array_equal(-array({}), array({})).item<bool>());

    // Throws on bool
    CHECK_THROWS(negative(array(true)));
  }

  // Test logical not
  {
    array x(false);
    CHECK_EQ(logical_not(x).item<bool>(), true);

    x = array(1.0f);
    auto y = logical_not(x);
    CHECK_EQ(y.dtype(), bool_);
    CHECK_EQ(y.item<bool>(), false);

    x = array(0);
    y = logical_not(x);
    CHECK_EQ(y.dtype(), bool_);
    CHECK_EQ(y.item<bool>(), true);
  }

  // Test logical and
  {
    array x(true);
    array y(true);
    CHECK_EQ(logical_and(x, y).item<bool>(), true);

    x = array(1.0f);
    y = array(1.0f);
    auto z = logical_and(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), true);

    x = array(0);
    y = array(1.0f);
    z = logical_and(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), false);
  }

  // Test logical or
  {
    array x(false);
    array y(false);
    CHECK_EQ(logical_or(x, y).item<bool>(), false);

    x = array(1.0f);
    y = array(1.0f);
    auto z = logical_or(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), true);

    x = array(0);
    y = array(1.0f);
    z = logical_or(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), true);
  }

  // Test abs
  {
    array x({-1.0f, 0.0f, 1.0f});
    CHECK(array_equal(abs(x), array({1.0f, 0.0f, 1.0f})).item<bool>());

    // works on empty array
    CHECK(array_equal(abs(array({})), array({})).item<bool>());

    // int32
    x = array({-1, 0, 1});
    CHECK(array_equal(abs(x), array({1, 0, 1})).item<bool>());

    // uint32
    x = array({1u, 0u, 1u});
    CHECK(array_equal(abs(x), array({1u, 0u, 1u})).item<bool>());

    // bool
    x = array({false, true});
    CHECK(array_equal(abs(x), array({false, true})).item<bool>());
  }

  // Test sign
  {
    array x({-1.0f, 0.0f, 1.0f});
    CHECK(array_equal(sign(x), x).item<bool>());

    // works on empty array
    CHECK(array_equal(sign(array({})), array({})).item<bool>());

    // int32
    x = array({-1, 0, 1});
    CHECK(array_equal(sign(x), x).item<bool>());

    // uint32
    x = array({1u, 0u, 1u});
    CHECK(array_equal(sign(x), x).item<bool>());

    // bool
    x = array({false, true});
    CHECK(array_equal(sign(x), x).item<bool>());

    // uint64
    array x_uint64(
        {uint64_t(0xa11cc311cb6acd70),
         uint64_t(0x7a375ac3ebb533f3),
         uint64_t(0x734969adf9d7190c),
         uint64_t(0xb400515a4f673424)});
    array expected(
        {uint64_t(0x0000000000000001),
         uint64_t(0x0000000000000001),
         uint64_t(0x0000000000000001),
         uint64_t(0x0000000000000001)});
    CHECK(array_equal(sign(x_uint64), expected).item<bool>());

    x_uint64 = array(
        {uint64_t(0xa11cc311cb6acd70),
         uint64_t(0x7a375ac3ebb533f3),
         uint64_t(0x734969adf9d7190c)});
    expected = array(
        {uint64_t(0x0000000000000001),
         uint64_t(0x0000000000000001),
         uint64_t(0x0000000000000001)});
    CHECK(array_equal(sign(x_uint64), expected).item<bool>());

    x_uint64 =
        array({uint64_t(0xa11cc311cb6acd70), uint64_t(0x7a375ac3ebb533f3)});
    expected =
        array({uint64_t(0x0000000000000001), uint64_t(0x0000000000000001)});
    CHECK(array_equal(sign(x_uint64), expected).item<bool>());

    x_uint64 = array({uint64_t(0xa11cc311cb6acd70)});
    expected = array({uint64_t(0x0000000000000001)});
    CHECK(array_equal(sign(x_uint64), expected).item<bool>());

    x_uint64 = array({uint64_t(0xffffffffffffffff)});
    expected = array({uint64_t(0x0000000000000001)});
    CHECK(array_equal(sign(x_uint64), expected).item<bool>());

    x_uint64 = array({uint64_t(0x0000000000000001)});
    expected = array({uint64_t(0x0000000000000001)});
    CHECK(array_equal(sign(x_uint64), expected).item<bool>());
  }

  constexpr float neginf = -std::numeric_limits<float>::infinity();

  // Test floor and ceil
  {
    array x(1.0f);
    CHECK_EQ(floor(x).item<float>(), 1.0f);
    CHECK_EQ(ceil(x).item<float>(), 1.0f);

    x = array(1.5f);
    CHECK_EQ(floor(x).item<float>(), 1.0f);
    CHECK_EQ(ceil(x).item<float>(), 2.0f);

    x = array(-1.5f);
    CHECK_EQ(floor(x).item<float>(), -2.0f);
    CHECK_EQ(ceil(x).item<float>(), -1.0f);

    x = array(neginf);
    CHECK_EQ(floor(x).item<float>(), neginf);
    CHECK_EQ(ceil(x).item<float>(), neginf);

    x = array(std::complex<float>(1.0f, 1.0f));
    CHECK_THROWS_AS(floor(x), std::invalid_argument);
    CHECK_THROWS_AS(ceil(x), std::invalid_argument);
  }

  // Test round
  {
    array x({0.5, -0.5, 1.5, -1.5, 2.3, 2.6});
    CHECK(array_equal(round(x), array({0, -0, 2, -2, 2, 3})).item<bool>());

    x = array({11, 222, 32});
    CHECK(array_equal(round(x, -1), array({10, 220, 30})).item<bool>());
  }

  // Test exponential
  {
    array x(0.0);
    CHECK_EQ(exp(x).item<float>(), 1.0);

    x = array(2.0);
    CHECK_EQ(exp(x).item<float>(), doctest::Approx(std::exp(2.0f)));

    CHECK(array_equal(exp(array({})), array({})).item<bool>());

    x = array(neginf);
    CHECK_EQ(exp(x).item<float>(), doctest::Approx(0.0f));

    // Integer input type
    x = array(2);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(exp(x).item<float>(), doctest::Approx(std::exp(2.0f)));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(allclose(exp(x), full({2, 2, 2}, std::exp(1.0f))).item<bool>());

    x = split(array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::exp(0.0f), std::exp(2.0f)}, {2, 1});
    CHECK(allclose(exp(x), expected).item<bool>());

    // Complex of -inf
    constexpr float inf = std::numeric_limits<float>::infinity();
    x = array(complex64_t{-inf, -inf});
    CHECK_EQ(exp(x).item<complex64_t>(), complex64_t{0, 0});
  }

  // Test expm1
  {
    array x(-1.0f);
    CHECK_EQ(expm1(x).item<float>(), doctest::Approx(std::expm1(-1.0f)));

    x = array(1.0f);
    CHECK_EQ(expm1(x).item<float>(), doctest::Approx(std::expm1(1.0f)));

    // Integer input type
    x = array(1);
    CHECK_EQ(expm1(x).dtype(), float32);
    CHECK_EQ(expm1(x).item<float>(), doctest::Approx(std::expm1(1.0f)));
  }

  // Test sine
  {
    array x(0.0);
    CHECK_EQ(sin(x).item<float>(), 0.0);

    x = array(M_PI_2);
    CHECK(sin(x).item<float>() == doctest::Approx(std::sin(M_PI_2)));

    CHECK(array_equal(sin(array({})), array({})).item<bool>());

    // Integer input type
    x = array(0);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(sin(x).item<float>(), std::sin(0.0f));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(allclose(sin(x), full({2, 2, 2}, std::sin(1.0f))).item<bool>());

    x = split(array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::sin(0.0f), std::sin(2.0f)}, {2, 1});
    CHECK(allclose(sin(x), expected).item<bool>());
  }

  // Test cos
  {
    array x(0.0);
    CHECK_EQ(cos(x).item<float>(), doctest::Approx(1.0));

    x = array(M_PI_2);
    CHECK(cos(x).item<float>() == doctest::Approx(std::cos(M_PI_2)));

    CHECK(array_equal(cos(array({})), array({})).item<bool>());

    // Integer input type
    x = array(0);
    CHECK_EQ(x.dtype(), int32);
    CHECK(cos(x).item<float>() == doctest::Approx(std::cos(0.0f)));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(allclose(cos(x), full({2, 2, 2}, std::cos(1.0f))).item<bool>());

    x = split(array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::cos(0.0f), std::cos(2.0f)}, {2, 1});
    CHECK(allclose(cos(x), expected).item<bool>());
  }

  // Test degrees
  {
    array x(0.0);
    CHECK_EQ(degrees(x).item<float>(), 0.0);

    x = array(M_PI_2);
    CHECK(degrees(x).item<float>() == doctest::Approx(90.0));

    CHECK(array_equal(degrees(array({})), array({})).item<bool>());

    // Integer input type
    x = array(0);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(degrees(x).item<float>(), 0.0);

    // Input is irregularly strided
    x = broadcast_to(array(M_PI_2), {2, 2, 2});
    CHECK(allclose(degrees(x), full({2, 2, 2}, 90.0)).item<bool>());

    float angles[] = {0.0f, M_PI_2, M_PI, 3.0f * M_PI_2};
    x = split(array(angles, {2, 2}), 2, 1)[0];
    auto expected = array({0.0f, 180.0f}, {2, 1});
    CHECK(allclose(degrees(x), expected).item<bool>());
  }

  // Test radians
  {
    array x(0.0);
    CHECK_EQ(radians(x).item<float>(), 0.0);

    x = array(90.0);
    CHECK(radians(x).item<float>() == doctest::Approx(M_PI_2));

    CHECK(array_equal(radians(array({})), array({})).item<bool>());

    // Integer input type
    x = array(90);
    CHECK_EQ(x.dtype(), int32);
    CHECK(radians(x).item<float>() == doctest::Approx(M_PI_2));

    // Input is irregularly strided
    x = broadcast_to(array(90.0f), {2, 2, 2});
    CHECK(allclose(radians(x), full({2, 2, 2}, M_PI_2)).item<bool>());

    x = split(array({0.0f, 90.0f, 180.0f, 270.0f}, {2, 2}), 2, 1)[0];
    float angles[] = {0.0f, M_PI};
    auto expected = array(angles, {2, 1});
    CHECK(allclose(radians(x), expected).item<bool>());
  }

  // Test log
  {
    array x(0.0);
    CHECK_EQ(log(x).item<float>(), neginf);

    x = array(1.0);
    CHECK_EQ(log(x).item<float>(), log(1.0f));

    // Integer input type
    x = array(1);
    CHECK_EQ(log(x).dtype(), float32);
    CHECK_EQ(log(x).item<float>(), log(1.0f));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(array_equal(log(x), full({2, 2, 2}, std::log(1.0f))).item<bool>());

    x = split(array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::log(1.0f), std::log(3.0f)}, {2, 1});
    CHECK(array_equal(log(x), expected).item<bool>());
  }

  // Test log2
  {
    array x(0.0);
    CHECK_EQ(log2(x).item<float>(), neginf);

    x = array(1.0);
    CHECK_EQ(log2(x).item<float>(), 0.0f);

    x = array(1024.0f);
    CHECK_EQ(log2(x).item<float>(), 10.0f);
  }

  // Test log10
  {
    array x(0.0);
    CHECK_EQ(log10(x).item<float>(), neginf);

    x = array(1.0);
    CHECK_EQ(log10(x).item<float>(), 0.0f);

    x = array(1000.0f);
    CHECK_EQ(log10(x).item<float>(), 3.0f);
  }

  // Test log1p
  {
    array x(-1.0f);
    CHECK_EQ(log1p(x).item<float>(), neginf);

    x = array(1.0f);
    CHECK_EQ(log1p(x).item<float>(), std::log1pf(1.0f));

    // Integer input type
    x = array(1);
    CHECK_EQ(log1p(x).dtype(), float32);
    CHECK_EQ(log1p(x).item<float>(), std::log1pf(1.0f));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(
        array_equal(log1p(x), full({2, 2, 2}, std::log1pf(1.0f))).item<bool>());

    x = split(array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::log1pf(1.0f), std::log1pf(3.0f)}, {2, 1});
    CHECK(array_equal(log1p(x), expected).item<bool>());
  }

  // Test sigmoid
  {
    array x(0.0);
    CHECK_EQ(sigmoid(x).item<float>(), 0.5f);

    // Integer input type
    x = array(0);
    CHECK_EQ(sigmoid(x).dtype(), float32);
    CHECK_EQ(sigmoid(x).item<float>(), 0.5f);

    constexpr auto inf = std::numeric_limits<float>::infinity();
    x = array(inf);
    CHECK_EQ(sigmoid(x).item<float>(), 1.0f);
    x = array(-inf);
    CHECK_EQ(sigmoid(x).item<float>(), 0.0f);
  }

  // Test square
  {
    array x(3.0);
    CHECK_EQ(square(x).item<float>(), 9.0);

    x = array(2);
    CHECK_EQ(square(x).item<int>(), 4);

    x = full({3, 3}, 2.0f);
    CHECK(array_equal(square(x), full({3, 3}, 4.0f)).item<bool>());
  }

  // Test sqrt and rsqrt
  {
    array x(4.0);
    CHECK_EQ(sqrt(x).item<float>(), 2.0);
    CHECK_EQ(rsqrt(x).item<float>(), 0.5);

    x = full({3, 3}, 9.0f);
    CHECK(array_equal(sqrt(x), full({3, 3}, 3.0f)).item<bool>());

    x = array(4, int32);
    CHECK_EQ(sqrt(x).item<float>(), 2.0f);
    CHECK_EQ(rsqrt(x).item<float>(), 0.5f);
  }

  // Test reciprocal
  {
    array x(8.0);
    CHECK_EQ(reciprocal(x).item<float>(), 0.125f);

    x = array(2);
    auto out = reciprocal(x);
    CHECK_EQ(out.dtype(), float32);
    CHECK_EQ(out.item<float>(), 0.5f);

    x = full({3, 3}, 2.0f);
    CHECK(array_equal(reciprocal(x), full({3, 3}, 0.5f)).item<bool>());
  }
}

TEST_CASE("test error functions") {
  constexpr float inf = std::numeric_limits<float>::infinity();
  array x(0.0f);
  CHECK_EQ(erf(x).item<float>(), 0.0f);
  x = array(inf);
  CHECK_EQ(erf(x).item<float>(), 1.0f);
  x = array(-inf);
  CHECK_EQ(erf(x).item<float>(), -1.0f);

  x = array(1, int32);
  CHECK_EQ(erf(x).dtype(), float32);

  x = array(0.0f);
  CHECK_EQ(erfinv(x).item<float>(), 0.0f);
  x = array(1.0f);
  CHECK_EQ(erfinv(x).item<float>(), inf);
  x = array(-1.0f);
  CHECK_EQ(erfinv(x).item<float>(), -inf);

  x = array(1, int32);
  CHECK_EQ(erfinv(x).dtype(), float32);

  x = array(2.0f);
  CHECK(std::isnan(erfinv(x).item<float>()));
  x = array(-2.0f);
  CHECK(std::isnan(erfinv(x).item<float>()));

  auto vals = {0.9f, 0.5f, 0.1f, -0.1f, -0.5f, -0.9f};
  // Expected values are generated from scipy's error function:
  //   python -c "import scipy.special as ss;
  //   vals = [0.9, 0.5, 0.1, -0.1, -0.5, -0.9];
  //   print([ss.erf(x) for x in vals])"
  {
    auto expected = {
        0.7969082124228322,
        0.5204998778130465,
        0.1124629160182849,
        -0.1124629160182849,
        -0.5204998778130465,
        -0.7969082124228322};
    for (int i = 0; i < vals.size(); ++i) {
      x = array(vals.begin()[i]);
      CHECK_EQ(erf(x).item<float>(), doctest::Approx(expected.begin()[i]));
    }
  }

  // Expected values are generated from scipy's inverse error function:
  //   python -c "import scipy.special as ss;
  //   vals = [0.9, 0.5, 0.1, -0.1, -0.5, -0.9];
  //   print([ss.erfinv(x) for x in vals])"
  {
    auto expected = {
        1.1630871536766738,
        0.4769362762044699,
        0.08885599049425778,
        -0.08885599049425769,
        -0.4769362762044699,
        -1.1630871536766743};
    for (int i = 0; i < vals.size(); ++i) {
      x = array(vals.begin()[i]);
      CHECK_EQ(erfinv(x).item<float>(), doctest::Approx(expected.begin()[i]));
    }
  }

  // float16_t
  {
    array x(0.0f, float16);
    auto out = erf(x);
    CHECK_EQ(out.dtype(), float16);
    CHECK_EQ(out.item<float16_t>(), 0.0f);

    out = erfinv(x);
    CHECK_EQ(out.dtype(), float16);
    CHECK_EQ(out.item<float16_t>(), 0.0f);
  }

  // bfloat
  {
    array x(0.0f, bfloat16);
    auto out = erf(x);
    CHECK_EQ(out.dtype(), bfloat16);
    CHECK_EQ(out.item<bfloat16_t>(), 0.0f);

    out = erfinv(x);
    CHECK_EQ(out.dtype(), bfloat16);
    CHECK_EQ(out.item<float16_t>(), 0.0f);
  }
}

TEST_CASE("test arithmetic binary ops") {
  array x(1.0);
  array y(1.0);
  auto z = add(x, y);
  CHECK_EQ(z.item<float>(), 2.0);
  z = x + y;
  CHECK_EQ(z.item<float>(), 2.0);
  z = add(z, x);
  CHECK_EQ(z.item<float>(), 3.0);
  z.eval(); // No-op
  CHECK_EQ(z.item<float>(), 3.0);

  // Chain a few adds:
  auto out = x;
  for (int i = 0; i < 10; ++i) {
    out = add(out, x);
  }
  CHECK_EQ(out.item<float>(), 11.0);

  // Works for different shapes
  x = array({1.0, 2.0, 3.0}, {1, 3});
  y = array({1.0, 2.0, 3.0}, {1, 3});
  z = add(x, y);
  CHECK_EQ(z.shape(), Shape{1, 3});
  auto eq = array_equal(z, array({2.0, 4.0, 6.0}, {1, 3}));
  CHECK(eq.item<bool>());

  // Works with scalars
  x = array({1.0, 2.0, 3.0}, {1, 3});
  y = x + 2.0;
  CHECK_EQ(y.dtype(), float32);
  eq = array_equal(y, array({3.0, 4.0, 5.0}, {1, 3}));
  CHECK(eq.item<bool>());
  y = 2.0 + x;
  CHECK_EQ(y.dtype(), float32);
  eq = array_equal(y, array({3.0, 4.0, 5.0}, {1, 3}));
  CHECK(eq.item<bool>());

  // Check type promotion
  y = 2 + x;
  CHECK_EQ(y.dtype(), float32);

  y = 2.0 + array({1, 2, 3});
  CHECK_EQ(y.dtype(), float32);
  CHECK(array_equal(y, array({3.0, 4.0, 5.0})).item<bool>());

  // Broadcasting works
  x = broadcast_to(array({1.0}), {10});
  y = broadcast_to(array({2.0}), {10});
  z = add(x, y);
  CHECK(array_equal(z, full({10}, 3.0)).item<bool>());

  x = array({1.0, 2.0}, {1, 2});
  y = array({1.0, 2.0}, {2, 1});
  z = add(x, y);
  CHECK_EQ(z.shape(), Shape{2, 2});
  eq = array_equal(z, array({2.0, 3.0, 3.0, 4.0}, {2, 2}));
  CHECK(eq.item<bool>());

  x = ones({3, 2, 1});
  z = x + 2.0;
  CHECK_EQ(z.shape(), Shape{3, 2, 1});
  eq = array_equal(z, array({3.0, 3.0, 3.0, 3.0, 3.0, 3.0}, {3, 2, 1}));
  CHECK(eq.item<bool>());

  // Works for empty arrays
  x = array({});
  y = array({});
  z = x + y;
  z.eval();
  CHECK_EQ(z.size(), 0);
  CHECK_EQ(z.shape(), Shape{0});

  // Check subtraction
  x = array({3, 2, 1});
  y = array({1, 1, 1});
  CHECK(array_equal(x - y, array({2, 1, 0})).item<bool>());

  // Check multiplication
  x = array({1, 2, 3});
  y = array({2, 2, 2});
  CHECK(array_equal(x * y, array({2, 4, 6})).item<bool>());

  // Check division
  x = array(1);
  y = array(1);
  CHECK_EQ(divide(x, y).item<float>(), 1.0f);

  x = array(1);
  y = array(0.5);
  CHECK_EQ(divide(x, y).item<float>(), 2.0f);

  x = array(1);
  y = array(4);
  CHECK_EQ(divide(x, y).item<float>(), 0.25f);

  x = array(true);
  y = array(true);
  CHECK_EQ(divide(x, y).item<float>(), 1.0f);

  x = array(false);
  y = array(true);
  CHECK_EQ(divide(x, y).item<float>(), 0.0f);

  x = array(true);
  y = array(false);
  CHECK(std::isinf(divide(x, y).item<float>()));

  x = array(false);
  y = array(false);
  CHECK(std::isnan(divide(x, y).item<float>()));

  // Integer division by zero gives quotient 0 and remainder a.
  if (default_device() == Device::cpu) {
    for (auto dt : {int8, int16, int32, int64, uint8, uint16, uint32, uint64}) {
      auto num = astype(array({7, 0, 5}, {3}), dt);
      auto den = zeros({3}, dt);
      CHECK(array_equal(floor_divide(num, den), zeros({3}, dt)).item<bool>());
      CHECK(array_equal(remainder(num, den), num).item<bool>());
    }
  }

  // Check maximum and minimum
  x = array(1.0f);
  y = array(0.0f);
  CHECK_EQ(maximum(x, y).item<float>(), 1.0f);
  CHECK_EQ(minimum(x, y).item<float>(), 0.0f);
  y = array(2.0f);
  CHECK_EQ(maximum(x, y).item<float>(), 2.0f);
  CHECK_EQ(minimum(x, y).item<float>(), 1.0f);

  // Check logaddexp
  x = array(0.0f);
  y = array(0.0f);
  CHECK_EQ(logaddexp(x, y).item<float>(), std::log(2.0f));

  x = array(0u);
  y = array(10000u);
  CHECK_EQ(logaddexp(x, y).item<float>(), 10000.0f);

  constexpr float inf = std::numeric_limits<float>::infinity();
  x = array(inf);
  y = array(3.0f);
  CHECK_EQ(logaddexp(x, y).item<float>(), inf);

  x = array(-inf);
  y = array(3.0f);
  CHECK_EQ(logaddexp(x, y).item<float>(), 3.0f);

  x = array(-inf);
  y = array(-inf);
  CHECK_EQ(logaddexp(x, y).item<float>(), -inf);

  x = array(inf);
  y = array(inf);
  CHECK_EQ(logaddexp(x, y).item<float>(), inf);

  x = array(-inf);
  y = array(inf);
  CHECK_EQ(logaddexp(x, y).item<float>(), inf);

  x = array(complex64_t{1, 1});
  y = array(complex64_t{-inf, -inf});
  CHECK_EQ(logaddexp(x, y).item<complex64_t>(), complex64_t{1, 1});
}

namespace {

template <typename T>
struct WrapRow {
  T a, b, sum, diff, prod;
};

// add, subtract and multiply of the rows, which are repeated `reps` times
template <typename T>
void check_wrapping(const std::vector<WrapRow<T>>& rows, int reps = 1) {
  std::vector<T> a, b, sum, diff, prod;
  for (int i = 0; i < reps; ++i) {
    for (const auto& r : rows) {
      a.push_back(r.a);
      b.push_back(r.b);
      sum.push_back(r.sum);
      diff.push_back(r.diff);
      prod.push_back(r.prod);
    }
  }
  Dtype dtype = TypeToDtype<T>();
  Shape shape{static_cast<int>(a.size())};
  auto make = [&](const std::vector<T>& v) {
    return array(v.begin(), shape, dtype);
  };
  auto equal = [](const array& p, const array& q) {
    return array_equal(p, q).item<bool>();
  };
  auto x = make(a);
  auto y = make(b);
  CHECK(equal(add(x, y), make(sum)));
  CHECK(equal(subtract(x, y), make(diff)));
  CHECK(equal(multiply(x, y), make(prod)));
}

} // namespace

TEST_CASE("test integer arithmetic wraps around") {
  // Integer add, subtract and multiply wrap around on overflow, as in numpy.
  // Signed overflow is undefined in C++, so the scalar kernels do the
  // arithmetic in unsigned types: this case fails under UBSan (USE_UBSAN=ON
  // with UBSAN_OPTIONS=halt_on_error=1) if they stop doing so. The arrays are
  // longer than the widest SIMD loop (max_size in simd/accelerate_simd.h), so
  // that the vector loop and the scalar tail both run where there is a vector
  // loop.
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();

  // Rows are a, b, a + b, a - b, a * b. 46341 * 46341 = 2^31 + 4633.
  check_wrapping<int32_t>({
      {imax, 1, imin, imax - 1, imax},
      {imin, -1, imax, imin + 1, imin},
      {imin, 1, imin + 1, imax, imin},
      {imax, imax, -2, 0, 1},
      {imin, imin, 0, 0, 0},
      {imax, imin, -1, -1, imin},
      {imax, 2, imin + 1, imax - 2, -2},
      {65536, 65536, 131072, 0, 0},
      {46341, 46341, 92682, 0, imin + 4633},
      {0, imin, imin, imin, 0},
      {-3, 7, 4, -10, -21},
  });

  // 3037000500 * 3037000500 = 2^63 + 145474192
  check_wrapping<int64_t>({
      {lmax, 1, lmin, lmax - 1, lmax},
      {lmin, -1, lmax, lmin + 1, lmin},
      {lmin, 1, lmin + 1, lmax, lmin},
      {lmax, lmax, -2, 0, 1},
      {lmin, lmin, 0, 0, 0},
      {lmax, lmin, -1, -1, lmin},
      {lmax, 2, lmin + 1, lmax - 2, -2},
      {4294967296, 4294967296, 8589934592, 0, 0},
      {3037000500, 3037000500, 6074001000, 0, lmin + 145474192},
      {0, lmin, lmin, lmin, 0},
      {-3, 7, 4, -10, -21},
  });

  // The operands of uint16 are promoted to int, where 65535 * 65535 overflows.
  check_wrapping<uint16_t>(
      {
          {65535, 65535, 65534, 0, 1},
          {65535, 65534, 65533, 1, 2},
          {256, 256, 512, 0, 0},
          {65535, 1, 0, 65534, 65535},
          {0, 1, 1, 65535, 0},
          {300, 300, 600, 0, 24464},
          {65535, 2, 1, 65533, 65534},
          {32768, 2, 32770, 32766, 0},
          {255, 255, 510, 0, 65025},
      },
      2);

  // A scalar against an array takes the broadcast kernels. Rows are v,
  // imax + v, v * imax, v - imax, imin - v.
  struct BroadcastRow {
    int32_t v, sum, prod, vdiff, sdiff;
  };
  std::vector<BroadcastRow> rows = {
      {1, imin, imax, imin + 2, imax},
      {2, imin + 1, -2, imin + 3, imax - 1},
      {imax, -2, 1, 0, 1},
      {imin, -1, imin, 1, 0},
      {-1, imax - 1, imin + 1, imin, imin + 1},
      {0, imax, 0, imin + 1, imin},
      {3, imin + 2, imax - 2, imin + 4, imax - 2},
      {imax - 1, -3, imin + 2, -1, 2},
      {imin + 1, 0, -1, 2, -1},
  };
  std::vector<int32_t> v, sum, prod, vdiff, sdiff;
  for (const auto& r : rows) {
    v.push_back(r.v);
    sum.push_back(r.sum);
    prod.push_back(r.prod);
    vdiff.push_back(r.vdiff);
    sdiff.push_back(r.sdiff);
  }
  Shape shape{static_cast<int>(rows.size())};
  auto make = [&](const std::vector<int32_t>& x) {
    return array(x.begin(), shape, int32);
  };
  auto equal = [](const array& p, const array& q) {
    return array_equal(p, q).item<bool>();
  };
  auto x = make(v);
  array smax(imax);
  array smin(imin);
  CHECK(equal(add(smax, x), make(sum)));
  CHECK(equal(multiply(x, smax), make(prod)));
  CHECK(equal(subtract(x, smax), make(vdiff)));
  CHECK(equal(subtract(smin, x), make(sdiff)));
}

namespace {

template <typename T>
struct UnaryRow {
  T x, neg, abs;
};

template <typename T>
struct PowRow {
  T base, exp, res;
};

bool same(const array& p, const array& q) {
  return array_equal(p, q).item<bool>();
}

// negative and abs of the rows
template <typename T>
void check_negative_abs(const std::vector<UnaryRow<T>>& rows) {
  std::vector<T> x, neg, abs_x;
  for (const auto& r : rows) {
    x.push_back(r.x);
    neg.push_back(r.neg);
    abs_x.push_back(r.abs);
  }
  Dtype dtype = TypeToDtype<T>();
  Shape shape{static_cast<int>(x.size())};
  auto make = [&](const std::vector<T>& v) {
    return array(v.begin(), shape, dtype);
  };
  CHECK(same(negative(make(x)), make(neg)));
  CHECK(same(abs(make(x)), make(abs_x)));
}

// base ^ exp of the rows
template <typename T>
void check_power(const std::vector<PowRow<T>>& rows) {
  std::vector<T> base, exp, res;
  for (const auto& r : rows) {
    base.push_back(r.base);
    exp.push_back(r.exp);
    res.push_back(r.res);
  }
  Dtype dtype = TypeToDtype<T>();
  Shape shape{static_cast<int>(base.size())};
  auto make = [&](const std::vector<T>& v) {
    return array(v.begin(), shape, dtype);
  };
  CHECK(same(power(make(base), make(exp)), make(res)));
}

} // namespace

TEST_CASE("test integer unary ops and pow wrap around") {
  // Integer negative, abs and pow wrap around, as in numpy: -MIN and abs(MIN)
  // are MIN, and a power is the exact one modulo 2^N (a negative exponent
  // still gives 0). Signed overflow is undefined in C++, so the scalar kernels
  // do the arithmetic in unsigned types: this case fails under UBSan
  // (USE_UBSAN=ON with UBSAN_OPTIONS=halt_on_error=1) if they stop doing so.
  // The arrays are longer than the widest SIMD loop (max_size in
  // simd/accelerate_simd.h), so that the vector loop and the scalar tail both
  // run where there is a vector loop. The expected values are exact integer
  // powers, reduced modulo 2^N. Rows are x, -x, abs(x) and base, exp, base^exp.
  const int8_t cmax = std::numeric_limits<int8_t>::max();
  const int8_t cmin = std::numeric_limits<int8_t>::min();
  const int16_t smax = std::numeric_limits<int16_t>::max();
  const int16_t smin = std::numeric_limits<int16_t>::min();
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();
  const uint8_t u8max = std::numeric_limits<uint8_t>::max();
  const uint16_t u16max = std::numeric_limits<uint16_t>::max();
  const uint32_t u32max = std::numeric_limits<uint32_t>::max();
  const uint64_t u64max = std::numeric_limits<uint64_t>::max();

  check_negative_abs<int8_t>({
      {cmin, cmin, cmin},
      {cmin + 1, cmax, cmax},
      {cmax, cmin + 1, cmax},
      {0, 0, 0},
      {1, -1, 1},
      {-1, 1, 1},
      {cmax - 1, cmin + 2, cmax - 1},
      {cmin + 2, cmax - 1, cmax - 1},
      {64, -64, 64},
      {-64, 64, 64},
      {100, -100, 100},
      {-100, 100, 100},
      {50, -50, 50},
      {-3, 3, 3},
      {2, -2, 2},
      {3, -3, 3},
      {-2, 2, 2},
      {77, -77, 77},
  });

  check_negative_abs<int16_t>({
      {smin, smin, smin},
      {smin + 1, smax, smax},
      {smax, smin + 1, smax},
      {0, 0, 0},
      {1, -1, 1},
      {-1, 1, 1},
      {2, -2, 2},
      {-2, 2, 2},
      {255, -255, 255},
      {-255, 255, 255},
      {256, -256, 256},
      {-256, 256, 256},
      {smax - 1, smin + 2, smax - 1},
      {smin + 2, smax - 1, smax - 1},
      {12345, -12345, 12345},
      {-12345, 12345, 12345},
      {1000, -1000, 1000},
      {7, -7, 7},
  });

  std::vector<UnaryRow<int32_t>> rows32 = {
      {imin, imin, imin},
      {imin + 1, imax, imax},
      {imax, imin + 1, imax},
      {0, 0, 0},
      {1, -1, 1},
      {-1, 1, 1},
      {2, -2, 2},
      {-2, 2, 2},
      {imax - 1, imin + 2, imax - 1},
      {imin + 2, imax - 1, imax - 1},
      {65536, -65536, 65536},
      {-65536, 65536, 65536},
      {123456789, -123456789, 123456789},
      {-123456789, 123456789, 123456789},
  };
  check_negative_abs(rows32);

  check_negative_abs<int64_t>({
      {lmin, lmin, lmin},
      {lmin + 1, lmax, lmax},
      {lmax, lmin + 1, lmax},
      {0, 0, 0},
      {1, -1, 1},
      {-1, 1, 1},
      {4294967296, -4294967296, 4294967296},
      {-4294967296, 4294967296, 4294967296},
      {lmax - 1, lmin + 2, lmax - 1},
      {lmin + 2, lmax - 1, lmax - 1},
      {3037000500, -3037000500, 3037000500},
      {-3037000500, 3037000500, 3037000500},
  });

  check_negative_abs<uint8_t>({
      {0, 0, 0},
      {1, u8max, 1},
      {2, u8max - 1, 2},
      {127, 129, 127},
      {128, 128, 128},
      {129, 127, 129},
      {u8max - 1, 2, u8max - 1},
      {u8max, 1, u8max},
      {3, u8max - 2, 3},
      {100, 156, 100},
      {200, 56, 200},
      {16, 240, 16},
      {15, 241, 15},
      {17, 239, 17},
      {64, 192, 64},
      {192, 64, 192},
      {77, 179, 77},
      {5, 251, 5},
  });

  check_negative_abs<uint16_t>({
      {0, 0, 0},
      {1, u16max, 1},
      {2, u16max - 1, 2},
      {32767, 32769, 32767},
      {32768, 32768, 32768},
      {32769, 32767, 32769},
      {u16max - 1, 2, u16max - 1},
      {u16max, 1, u16max},
      {3, u16max - 2, 3},
      {1000, 64536, 1000},
      {40000, 25536, 40000},
      {256, 65280, 256},
      {255, 65281, 255},
      {257, 65279, 257},
      {16384, 49152, 16384},
      {49152, 16384, 49152},
      {12345, 53191, 12345},
      {5, 65531, 5},
  });

  check_negative_abs<uint32_t>({
      {0, 0, 0},
      {1, u32max, 1},
      {2, u32max - 1, 2},
      {2147483647, 2147483649u, 2147483647},
      {2147483648u, 2147483648u, 2147483648u},
      {2147483649u, 2147483647, 2147483649u},
      {u32max - 1, 2, u32max - 1},
      {u32max, 1, u32max},
      {65536, 4294901760u, 65536},
      {123456789, 4171510507u, 123456789},
  });

  check_negative_abs<uint64_t>({
      {0, 0, 0},
      {1, u64max, 1},
      {2, u64max - 1, 2},
      {9223372036854775807, 9223372036854775809ULL, 9223372036854775807},
      {9223372036854775808ULL, 9223372036854775808ULL, 9223372036854775808ULL},
      {9223372036854775809ULL, 9223372036854775807, 9223372036854775809ULL},
      {u64max - 1, 2, u64max - 1},
      {u64max, 1, u64max},
      {4294967296, 18446744069414584320ULL, 4294967296},
      {3037000500, 18446744070672551116ULL, 3037000500},
  });

  check_power<int8_t>({
      {3, 5, -13},     {3, 4, 81},      {2, 6, 64},   {2, 7, cmin},
      {2, 8, 0},       {cmin, 1, cmin}, {cmin, 2, 0}, {cmax, 2, 1},
      {cmax, 3, cmax}, {-1, 7, -1},     {-1, 8, 1},   {5, 3, cmax - 2},
      {11, 2, 121},    {12, 2, -112},   {0, 0, 1},    {0, 9, 0},
      {-2, 7, cmin},   {-2, 8, 0},      {3, -1, 0},   {-5, 3, cmin + 3},
  });

  check_power<int16_t>({
      {3, 11, -19461}, {181, 2, 32761}, {182, 2, -32412}, {smin, 2, 0},
      {smin, 1, smin}, {2, 14, 16384},  {2, 15, smin},    {2, 16, 0},
      {smax, 2, 1},    {-1, 15, -1},    {-1, 16, 1},      {5, 7, 12589},
      {7, 5, 16807},   {0, 0, 1},       {0, 3, 0},        {1, 100, 1},
      {-2, 15, smin},  {-3, -1, 0},     {smin, smin, 0},
  });

  check_power<int32_t>({
      {3, 40, 689956897},
      {3, 21, 1870418611},
      {2, 30, 1073741824},
      {2, 31, imin},
      {2, 32, 0},
      {100000, 1, 100000},
      {100000, 2, 1410065408},
      {-2, 31, imin},
      {-2, 32, 0},
      {46341, 2, -2147479015},
      {-3, 21, -1870418611},
      {imax, 2, 1},
      {imax, imax, imax},
      {imin, 1, imin},
      {imin, 2, 0},
      {7, 0, 1},
      {0, 0, 1},
      {0, 5, 0},
      {-1, 31, -1},
      {3, -1, 0},
      {imin, -5, 0},
      {3, 2, 9},
  });

  check_power<int64_t>({
      {3, 41, -420491770248316829},
      {3, 40, -6289078614652622815},
      {2, 62, 4611686018427387904},
      {2, 63, lmin},
      {2, 64, 0},
      {3000000000, 1, 3000000000},
      {4000000000, 1, 4000000000},
      {3037000500, 2, -9223372036709301616},
      {-2, 63, lmin},
      {lmax, 2, 1},
      {lmax, lmax, lmax},
      {lmin, 1, lmin},
      {lmin, 2, 0},
      {10, 19, -8446744073709551616},
      {-1, 63, -1},
      {5, 0, 1},
      {3, -2, 0},
      {7, 3, 343},
  });

  check_power<uint8_t>({
      {u8max, 2, 1},
      {u8max, 3, u8max},
      {3, 7, 139},
      {2, 7, 128},
      {2, 8, 0},
      {16, 2, 0},
      {15, 2, 225},
      {17, 2, 33},
      {0, 0, 1},
      {1, 200, 1},
      {u8max - 1, 2, 4},
      {128, 2, 0},
      {7, 3, 87},
      {6, 4, 16},
      {9, 3, 217},
      {10, 3, 232},
      {11, 3, 51},
      {200, 2, 64},
  });

  check_power<uint16_t>({
      {u16max, 1, u16max},
      {u16max, 2, 1},
      {u16max, 3, u16max},
      {257, 2, 513},
      {3, 11, 46075},
      {255, 2, 65025},
      {2, 15, 32768},
      {2, 16, 0},
      {u16max - 1, 2, 4},
      {46341, 2, 4633},
      {46340, 2, 43024},
      {181, 2, 32761},
      {182, 3, 64792},
      {3, 20, 7057},
      {0, 0, 1},
      {1, u16max, 1},
      {u16max, u16max, u16max},
      {12345, 7, 52425},
      {300, 2, 24464},
  });

  check_power<uint32_t>({
      {u32max, 2, 1},
      {65537, 2, 131073},
      {65536, 2, 0},
      {3, 40, 689956897},
      {2, 31, 2147483648u},
      {2, 32, 0},
      {46341, 2, 2147488281u},
      {7, 0, 1},
      {5, 13, 1220703125},
      {u32max, u32max, u32max},
  });

  check_power<uint64_t>({
      {u64max, 2, 1},
      {4294967297, 2, 8589934593},
      {4294967296, 2, 0},
      {3, 41, 18026252303461234787ULL},
      {2, 63, 9223372036854775808ULL},
      {2, 64, 0},
      {3037000500, 2, 9223372037000250000ULL},
      {5, 0, 1},
      {7, 30, 1576789505350337489},
  });

  // A strided input and 0-d arrays take other kernels than a contiguous one.
  {
    std::vector<int32_t> padded, neg, abs_x;
    for (const auto& r : rows32) {
      padded.push_back(r.x);
      padded.push_back(0);
      neg.push_back(r.neg);
      abs_x.push_back(r.abs);
    }
    int n = static_cast<int>(rows32.size());
    array a(padded.begin(), {2 * n}, int32);
    auto s = slice(a, {0}, {2 * n}, {2});
    CHECK(same(negative(s), array(neg.begin(), {n}, int32)));
    CHECK(same(abs(s), array(abs_x.begin(), {n}, int32)));
    CHECK_EQ(negative(array(imin)).item<int32_t>(), imin);
    CHECK_EQ(abs(array(imin)).item<int32_t>(), imin);
  }

  // A scalar base or exponent against an array takes the broadcast kernels:
  // 3^e, and b^2.
  {
    std::vector<int32_t> e = {40, 21, 2, 0, 1, 31, 32, 3, 5};
    std::vector<int32_t> p = {
        689956897, 1870418611, 9, 1, 3, 1264544299, -501334399, 27, 243};
    std::vector<int32_t> b = {
        imin, imax, 46341, 65536, 65535, -46341, 0, 1, -1};
    std::vector<int32_t> sq = {
        0, 1, -2147479015, 0, -131071, -2147479015, 0, 1, 1};
    auto make = [](const std::vector<int32_t>& v) {
      return array(v.begin(), {static_cast<int>(v.size())}, int32);
    };
    CHECK(same(power(array(3), make(e)), make(p)));
    CHECK(same(power(make(b), array(2)), make(sq)));
  }
}

namespace {

array zero_like_dtype(const array& x) {
  return array(0, x.dtype());
}

// abs and negative compared with 0 and with the input, each in its own fused
// kernel (the five inputs hold the same values; a shared abs or negative would
// not be fused into the comparisons). A compiler that takes signed overflow
// as impossible may fold these comparisons (abs(x) < 0 is false, -x > 0 is
// x < 0, ...), which gives the wrong answer at MIN without any sanitizer.
std::vector<array> compare_abs_negative(const std::vector<array>& in) {
  return {
      less(abs(in[0]), zero_like_dtype(in[0])),
      greater_equal(abs(in[1]), zero_like_dtype(in[1])),
      greater(negative(in[2]), zero_like_dtype(in[2])),
      less(negative(in[3]), zero_like_dtype(in[3])),
      equal(negative(in[4]), in[4])};
}

} // namespace

TEST_CASE("test compiled integer unary ops wrap around") {
  // Scope switch to the cpu: compile() builds these kernels with the CPU JIT.
  StreamContext sc(Device::cpu);
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();
  std::vector<int32_t> v32 = {imin, imin + 1, -1, 0, 1, imax};
  std::vector<int64_t> v64 = {lmin, lmin + 1, -1, 0, 1, lmax};

  // Expected for x = MIN, MIN + 1, -1, 0, 1, MAX.
  std::vector<std::vector<bool>> expected = {
      {true, false, false, false, false, false}, // abs(x) < 0
      {false, true, true, true, true, true}, // abs(x) >= 0
      {false, true, true, false, false, false}, // -x > 0
      {true, false, false, false, true, true}, // -x < 0
      {true, false, false, true, false, false}, // -x == x
  };
  auto compiled = compile(compare_abs_negative);
  auto check = [&](const std::vector<array>& inputs) {
    auto out = compiled(inputs);
    REQUIRE_EQ(out.size(), 5);
    for (int i = 0; i < 5; ++i) {
      array want(expected[i].begin(), {6}, bool_);
      CHECK(same(out[i], want));
    }
  };
  auto fresh = [](const auto& v, Dtype dtype) {
    std::vector<array> inputs;
    for (int i = 0; i < 5; ++i) {
      inputs.push_back(array(v.begin(), {6}, dtype));
    }
    return inputs;
  };
  check(fresh(v32, int32));
  check(fresh(v64, int64));
}

// The cases below check the integer sums and products that do not go through
// the operators of simd::Simd: scan, scatter and the scalar operators of the
// reductions. Like the case above they pass without a sanitizer where the
// hardware wraps around, and fail under UBSan (USE_UBSAN=ON with
// UBSAN_OPTIONS=halt_on_error=1) if the arithmetic overflows a signed type.
namespace {

// a has this shape and these values (and its own dtype).
template <typename T>
bool equal_to(const array& a, const std::vector<T>& want, const Shape& shape) {
  auto w = array(want.begin(), shape, a.dtype());
  return a.shape() == shape && array_equal(a, w).item<bool>();
}

template <typename T>
bool equal_to(const array& a, const std::vector<T>& want) {
  return equal_to(a, want, {static_cast<int>(want.size())});
}

} // namespace

TEST_CASE("test cumsum and cumprod of integers wrap around") {
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();

  // imax + 1 = imin, imin + imax = -1, imax * 3 = imax - 2 and
  // (imax - 2) * imax = 3, the same with lmax.
  auto x = array({imax, 1, imax, 2}, int32);
  CHECK(equal_to<int32_t>(cumsum(x), {imax, imin, -1, 1}));
  CHECK(equal_to<int32_t>(cumsum(x, true), {1, imin + 2, imin + 1, 2}));
  CHECK(equal_to<int32_t>(cumsum(x, false, false), {0, imax, imin, -1}));
  CHECK(equal_to<int32_t>(cumsum(x, true, false), {imin + 2, imin + 1, 2, 0}));
  CHECK(
      equal_to<int32_t>(
          cumprod(array({imax, 3, imax, -1}, int32)), {imax, imax - 2, 3, -3}));

  auto y = array({lmax, int64_t(1), lmax, int64_t(2)}, int64);
  CHECK(equal_to<int64_t>(cumsum(y), {lmax, lmin, -1, 1}));
  CHECK(
      equal_to<int64_t>(
          cumprod(array({lmax, int64_t(3), lmax, int64_t(-1)}, int64)),
          {lmax, lmax - 2, 3, -3}));

  // Along the first axis of a 4 x 2 array the scan is strided. Reversed,
  // the products of column 0 are 2, 2 * imax = -2, -2 and -2 * imax = 2.
  auto z = array({imax, 1, 1, 1, imax, 1, 2, 1}, {4, 2}, int32);
  CHECK(
      equal_to<int32_t>(cumsum(z, 0), {imax, 1, imin, 2, -1, 3, 1, 4}, {4, 2}));
  CHECK(
      equal_to<int32_t>(
          cumprod(z, 0, true), {2, 1, -2, 1, -2, 1, 2, 1}, {4, 2}));

  // uint16 is promoted to int, where 65535 * 65535 overflows.
  CHECK(
      equal_to<uint16_t>(
          cumprod(array({65535, 65535, 3, 65535}, uint16)),
          {65535, 1, 3, 65533}));
}

TEST_CASE("test scatter add and prod of integers wrap around") {
  // Scope switch to the cpu: this tests the CPU scatter.
  StreamContext sc(Device::cpu);
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();
  auto idx = array({0, 2, 0}, int32);

  // a[0] = imax + 1 + imax = -1, a[2] = imin - 1 = imax.
  auto a = array({imax, 5, imin}, int32);
  auto u = array({1, -1, imax}, {3, 1}, int32);
  CHECK(equal_to<int32_t>(scatter_add(a, idx, u, 0), {-1, 5, imax}));
  // a[0] = imax * 3 * imax = 3, a[2] = 65536 * 65536 = 0.
  a = array({imax, 5, 65536}, int32);
  u = array({3, 65536, imax}, {3, 1}, int32);
  CHECK(equal_to<int32_t>(scatter_prod(a, idx, u, 0), {3, 5, 0}));

  auto b = array({lmax, int64_t(5), lmin}, int64);
  auto v = array({int64_t(1), int64_t(-1), lmax}, {3, 1}, int64);
  CHECK(equal_to<int64_t>(scatter_add(b, idx, v, 0), {-1, 5, lmax}));
  b = array({lmax, int64_t(5), int64_t(1) << 32}, int64);
  v = array({int64_t(3), int64_t(1) << 32, lmax}, {3, 1}, int64);
  CHECK(equal_to<int64_t>(scatter_prod(b, idx, v, 0), {3, 5, 0}));

  // uint16 is promoted to int: 65535 * 65535 * 3 = 3 (mod 2^16).
  auto c = scatter_prod(
      array({65535, 7}, uint16),
      array({0, 0}, int32),
      array({65535, 3}, {2, 1}, uint16),
      0);
  CHECK(equal_to<uint16_t>(c, {3, 7}));

  // scatter_add_axis: column 0 gets imax + 1 and imin - 1, column 1 gets
  // 2 + imax + imax.
  auto d = scatter_add_axis(
      array({imax, 1, imin, 2}, {2, 2}, int32),
      array({0, 1, 1, 1}, {2, 2}, int32),
      array({1, imax, -1, imax}, {2, 2}, int32),
      0);
  CHECK(equal_to<int32_t>(d, {imin, 1, imax, 0}, {2, 2}));
  auto e = scatter_add_axis(
      array({lmax, int64_t(0)}, int64),
      array({0, 0}, int32),
      array({int64_t(1), lmax}, int64),
      0);
  CHECK(equal_to<int64_t>(e, {-1, 0}));
}

TEST_CASE("test sum and prod of integers wrap around in every reduction") {
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();

  // Axes 0 and 2 of a 2 x 1 x 2 array are two contiguous segments, whose
  // totals are combined by the scalar operator.
  auto x = array({imax, 0, 1, 0}, {2, 1, 2}, int32);
  CHECK(equal_to<int32_t>(sum(x, std::vector<int>{0, 2}), {imin}));
  x = array({imax, 1, 3, 1}, {2, 1, 2}, int32);
  CHECK(equal_to<int32_t>(prod(x, std::vector<int>{0, 2}), {imax - 2}));

  // A broadcast input is reduced one element at a time by the scalar
  // operator: 3 * imax = imax - 2 and imax * imax = 1.
  CHECK(equal_to<int32_t>(sum(broadcast_to(array(imax), {3})), {imax - 2}, {}));
  CHECK(equal_to<int32_t>(prod(broadcast_to(array(imax), {2})), {1}, {}));
  CHECK(equal_to<int64_t>(sum(broadcast_to(array(lmax), {3})), {lmax - 2}, {}));
  CHECK(equal_to<int64_t>(prod(broadcast_to(array(lmax), {2})), {1}, {}));
  // uint16 is accumulated in int32: 65535 * 65535 overflows it.
  auto p = prod(broadcast_to(array(65535, uint16), {2}));
  CHECK_EQ(p.dtype(), uint32);
  CHECK(equal_to<uint32_t>(p, {4294836225u}, {}));

  // Longer than the widest SIMD loop (max_size in simd/accelerate_simd.h):
  // where there is a vector loop, the last element is added by the scalar
  // operator.
  std::vector<int32_t> v(9, 0);
  v[0] = imax;
  v[8] = 1;
  CHECK(equal_to<int32_t>(sum(array(v.begin(), {9}, int32)), {imin}, {}));
  std::fill(v.begin(), v.end(), 1);
  v[0] = imax;
  v[8] = 3;
  CHECK(equal_to<int32_t>(prod(array(v.begin(), {9}, int32)), {imax - 2}, {}));
  std::vector<int64_t> w(5, 0);
  w[0] = lmax;
  w[4] = 1;
  CHECK(equal_to<int64_t>(sum(array(w.begin(), {5}, int64)), {lmin}, {}));
}

TEST_CASE("test broadcast") {
  auto s = broadcast_shapes({1}, {1, 2});
  CHECK_EQ(s, Shape{1, 2});

  s = broadcast_shapes({1, 2}, {1});
  CHECK_EQ(s, Shape{1, 2});

  s = broadcast_shapes({2, 2}, {});
  CHECK_EQ(s, Shape{2, 2});

  s = broadcast_shapes({}, {1, 1});
  CHECK_EQ(s, Shape{1, 1});

  s = broadcast_shapes({1, 2, 1}, {2});
  CHECK_EQ(s, Shape{1, 2, 2});

  s = broadcast_shapes({2}, {1, 2, 1});
  CHECK_EQ(s, Shape{1, 2, 2});

  s = broadcast_shapes({2, 2, 2}, {1, 2, 1});
  CHECK_EQ(s, Shape{2, 2, 2});

  s = broadcast_shapes({2, 2, 2, 1}, {1, 2, 1});
  CHECK_EQ(s, Shape{2, 2, 2, 1});

  s = broadcast_shapes({0}, {0, 0});
  CHECK_EQ(s, Shape{0, 0});

  CHECK_EQ(broadcast_shapes({}, {0}), Shape{0});

  s = broadcast_shapes({5, 0}, {0, 5, 0});
  CHECK_EQ(s, Shape{0, 5, 0});

  CHECK_EQ(broadcast_shapes({}, {0}), Shape{0});
  CHECK_EQ(broadcast_shapes({1}, {0}), Shape{0});
  CHECK_EQ(broadcast_shapes({1}, {0}), Shape{0});
  CHECK_EQ(broadcast_shapes({1}, {0, 0}), Shape{0, 0});
  CHECK_EQ(broadcast_shapes({1, 1}, {0}), Shape{1, 0});
  CHECK_EQ(broadcast_shapes({1, 1}, {0, 0}), Shape{0, 0});
  CHECK_EQ(broadcast_shapes({2, 1}, {1, 0}), Shape{2, 0});
  CHECK_EQ(broadcast_shapes({2, 1}, {2, 0}), Shape{2, 0});
  CHECK_EQ(broadcast_shapes({2, 1}, {1, 2, 0}), Shape{1, 2, 0});
  CHECK_THROWS_AS(broadcast_shapes({2}, {0}), std::invalid_argument);
  CHECK_THROWS_AS(broadcast_shapes({2, 1}, {0, 0}), std::invalid_argument);

  CHECK_THROWS_AS(broadcast_shapes({3}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(broadcast_shapes({1, 3}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(broadcast_shapes({3}, {1, 2}), std::invalid_argument);
  CHECK_THROWS_AS(
      broadcast_shapes({1, 3, 2}, {1, 2, 2}), std::invalid_argument);

  auto x = full({1, 1}, 2.3f);
  CHECK_EQ(broadcast_to(x, {1, 1}).item<float>(), 2.3f);

  x = broadcast_to(x, {5, 1});
  CHECK_EQ(x.shape(), Shape{5, 1});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 0});

  CHECK_THROWS_AS(broadcast_to(x, {1, 5}), std::invalid_argument);
  x = broadcast_to(x, {5, 5});
  CHECK_EQ(x.shape(), Shape{5, 5});

  x = zeros({2, 1, 2});
  x = broadcast_to(x, {4, 2, 1, 2});
  CHECK_EQ(x.shape(), Shape{4, 2, 1, 2});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 2, 0, 1});

  // Broadcast on empty arrays works as expected
  x = array({});
  CHECK_THROWS_AS(broadcast_to(x, {1}), std::invalid_argument);

  // Broadcast to empty array works as expected
  x = array({1});
  auto y = broadcast_to(x, {0});
  eval(y);
  CHECK_EQ(y.size(), 0);
  CHECK_EQ(y.shape(), Shape{0});

  x = array({1, 2}, {2, 1});
  y = broadcast_to(x, {2, 0});
  eval(y);
  CHECK_EQ(y.size(), 0);
  CHECK_EQ(y.shape(), Shape{2, 0});

  // Check repeat application works
  x = zeros({2});
  x = broadcast_to(broadcast_to(x, {2, 2}), {2, 2});
  CHECK_EQ(x.shape(), Shape{2, 2});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 1});
  x = broadcast_to(broadcast_to(x, {2, 2}), {2, 2, 2});
  CHECK_EQ(x.shape(), Shape{2, 2, 2});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 0, 1});

  // Broadcast on transposed array works
  x = array({0, 1, 2, 3, 4, 5}, {2, 3});
  x = broadcast_to(transpose(x), {2, 3, 2});
  CHECK_EQ(x.shape(), Shape{2, 3, 2});
  y = broadcast_to(array({0, 3, 1, 4, 2, 5}, {3, 2}), {2, 3, 2});
  CHECK(array_equal(x, y).item<bool>());

  // Reshape on broadcasted array works
  x = array(1.0);
  x = broadcast_to(x, {2});
  x = reshape(x, {1, 2});
  CHECK(array_equal(x, ones({1, 2})).item<bool>());
}

TEST_CASE("test gather") {
  // Empty input, non-empty indices/slice
  CHECK_THROWS(gather(array({}), array({1}), 0, {1}));

  // More indices than dimensions
  CHECK_THROWS(gather(array(0), array({1}), 0, {1}));

  // Mismatch dimensions and indices
  CHECK_THROWS(gather(array({0}), {array({0})}, {0, 1}, {1}));
  CHECK_THROWS(gather(array({0}), array({0}), -1, {1}));

  // Repeat dimensions
  CHECK_THROWS(
      gather(array({0}, {1, 1}), {array({0}), array({0})}, {0, 0}, {1, 1}));

  // Slice sizes incorrect
  CHECK_THROWS(gather(array({0}), array({0}), 0, {2}));
  CHECK_THROWS(gather(array({0}), array({0}), 0, {0, 0}));
  CHECK_THROWS(gather(array({0}), array({0}), 0, {-1}));

  // Wrong index type
  CHECK_THROWS(gather(array({0}), array({0.0f}), 0, {0}));
  CHECK_THROWS(
      gather(array({0}, {1, 1}), {array({0}), array({0.0f})}, {0, 1}, {1, 1}));

  // Index arrays must be broadcastable
  CHECK_THROWS(gather(
      array({0}, {1, 1}),
      {array({0, 0, 0}, {3}), array({0, 0}, {2})},
      {0, 1},
      {1, 1}));

  // Basic test of correctness with 1D input
  auto x = arange(20);
  auto y = arange(10);
  auto out = gather(x, y, 0, {1});
  CHECK_EQ(out.shape(), Shape{10, 1});
  CHECK(array_equal(reshape(out, {-1}), y).item<bool>());

  out = gather(x, array({15}, uint32), 0, {1});
  CHECK_EQ(out.shape(), Shape{1, 1});
  CHECK_EQ(out.item<int32_t>(), 15);

  // No index gather works
  out = gather(x, {}, std::vector<int>{}, {10});
  CHECK_EQ(out.shape(), Shape{10});
  CHECK(array_equal(out, arange(10)).item<bool>());

  // Basic test of correctness with 2D input
  x = arange(128);
  x = reshape(x, {4, 32});
  y = array({0, 1}, uint32);
  out = gather(x, y, 0, {1, 32});
  CHECK_EQ(out.shape(), Shape{2, 1, 32});
  CHECK(array_equal(reshape(out, {64}), arange(64)).item<bool>());

  x = reshape(x, {64, 2});
  y = array({0}, uint32);
  out = gather(x, y, 0, {64, 1});
  CHECK_EQ(out.shape(), Shape{1, 64, 1});
  CHECK(array_equal(out, reshape(arange(0, 128, 2), {1, 64, 1})).item<bool>());

  // Basic test of correctness with 3D input
  x = arange(256);
  x = reshape(x, {8, 4, 8});
  y = array({0}, uint32);
  out = gather(x, y, 0, {8, 1, 1});
  CHECK_EQ(out.shape(), Shape{1, 8, 1, 1});
  CHECK(
      array_equal(out, reshape(arange(0, 256, 32), {1, 8, 1, 1})).item<bool>());

  x = broadcast_to(array({1, 2}), {20, 2});
  out = gather(x, array({5}), 0, {1, 1});
  CHECK_EQ(out.item<int>(), 1);
  out = gather(x, {array({5}), array({1})}, {0, 1}, {1, 1});
  CHECK_EQ(out.item<int>(), 2);
}

TEST_CASE("test take") {
  // Empty takes
  auto empty = astype(array({}), int32);
  auto z = take(array({1}), empty);
  CHECK_EQ(z.shape(), Shape{0});
  empty = reshape(empty, {1, 0, 1});
  z = take(array({1}), empty);
  CHECK_EQ(z.shape(), Shape{1, 0, 1});

  CHECK_THROWS(take(array({}), array(1)));

  z = take(array({}), empty);
  CHECK_EQ(z.size(), 0);

  // Take a single row
  auto x = reshape(arange(256), {8, 4, 8});
  z = take(x, array({0}, uint32), 0);
  CHECK_EQ(z.shape(), Shape{1, 4, 8});
  z = reshape(z, {32});
  CHECK(array_equal(z, arange(32)).item<bool>());

  z = take(x, array({1}, uint32), 0);
  z = reshape(z, {32});
  CHECK(array_equal(z, arange(32, 64)).item<bool>());

  // Take multiple rows
  x = arange(256);
  x = reshape(x, {8, 4, 8});
  z = take(x, array({0, 1}, uint32), 0);
  z = reshape(z, {64});
  CHECK(array_equal(z, arange(64)).item<bool>());

  // Take along middle axis
  x = reshape(arange(8), {2, 2, 2});
  z = take(x, array({0}), 1);
  CHECK(array_equal(z, array({0, 1, 4, 5}, {2, 1, 2})).item<bool>());

  // Irregular strides test
  auto a = array({1, 2, 3}, float32);
  auto indices = broadcast_to(array(0), {10});
  auto b = take(a, indices);
  CHECK(array_equal(b, ones({10})).item<bool>());

  // Take with 0 dim index
  z = take(array({0, 1, 2}), array(0));
  CHECK_EQ(z.item<int>(), 0);
  CHECK_EQ(z.ndim(), 0);

  // Check take with float indices crashes
  CHECK_THROWS(take(array({}), array({})));
  CHECK_THROWS(take(a, array({1.0, 2.0, 3.0})));

  // Check axis
  a = array({1, 2, 3, 4}, {2, 2});
  CHECK_THROWS(take(a, array({1}), -3));
  CHECK_THROWS(take(a, array({1}), 2));

  // Check negative indices
  a = array({1, 2, 3, 4}, {2, 2});
  CHECK_EQ(take(a, array({-1})).item<int>(), 4);
  CHECK(array_equal(take(a, array({1, -1})), array({2, 4})).item<bool>());
  CHECK(array_equal(take(a, array(-1), 0), array({3, 4})).item<bool>());

  // Check shapes
  a = zeros({2, 1, 1});
  auto out = take(a, array({1}), 0);
  CHECK(array_equal(out, zeros({1, 1, 1})).item<bool>());
  out = take(a, array({0}), 1);
  CHECK(array_equal(out, zeros({2, 1, 1})).item<bool>());
  out = take(a, array({0}), 1);
  CHECK(array_equal(out, zeros({2, 1, 1})).item<bool>());
  a = zeros({1, 2, 1});
  out = take(a, array({0}), 0);
  CHECK(array_equal(out, zeros({1, 2, 1})).item<bool>());
  out = take(a, array({0}), 1);
  CHECK(array_equal(out, zeros({1, 1, 1})).item<bool>());
  out = take(a, array({0, 1}), 1);
  CHECK(array_equal(out, zeros({1, 2, 1})).item<bool>());

  // Indices have wrong shape
  a = zeros({2, 3, 4});
  CHECK_THROWS(take(a, zeros({1, 3, 4}), 1));
  CHECK_THROWS(take(a, zeros({2, 3, 7}), 1));
  CHECK_THROWS(take(a, zeros({2, 3, 2}), 0));
}

TEST_CASE("test gather contiguity") {
  // Regression test for a CPU-backend bug where the gather "fast copy" path
  // copied a multi-dimensional slice from a column-contiguous source as a raw
  // (column-major) memory block, producing a transposed/wrong-stride result.
  // The bug only showed up on the CPU backend and is exercised by:
  //  - chained takes through a size-1 axis (which produce a col-contiguous
  //    intermediate), and
  //  - a direct take from a transposed (col-contiguous) source.

  // Chained gather through size-1 axes (issue repro).
  {
    auto u = reshape(array({1.0f, 2.0f}), {2, 1, 1});
    auto g = take(u, array({0, 1}, int32), 0, Device::cpu);
    g = take(g, array({0, 0, 0}, int32), 1, Device::cpu);
    g = take(g, array({0, 0, 0}, int32), 2, Device::cpu);
    CHECK_EQ(g.shape(), Shape{2, 3, 3});
    // Each batch must be uniform: batch 0 -> 1.0, batch 1 -> 2.0.
    auto expected = array(
        {1.0f,
         1.0f,
         1.0f,
         1.0f,
         1.0f,
         1.0f,
         1.0f,
         1.0f,
         1.0f,
         2.0f,
         2.0f,
         2.0f,
         2.0f,
         2.0f,
         2.0f,
         2.0f,
         2.0f,
         2.0f},
        {2, 3, 3});
    CHECK(array_equal(g, expected).item<bool>());
  }

  // Direct take from a column-contiguous source with a multi-dim slice.
  {
    auto base = astype(reshape(arange(24), {4, 3, 2}), int32);
    auto a = transpose(base, {2, 1, 0}); // [2, 3, 4], col-contiguous
    auto t = take(a, array({0, 1}, int32), 2, Device::cpu);
    CHECK_EQ(t.shape(), Shape{2, 3, 2});
    auto expected =
        array({0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11}, {2, 3, 2}, int32);
    CHECK(array_equal(t, expected).item<bool>());
  }
}

TEST_CASE("test take along axis") {
  // No zero dim arrays
  auto a = array(1);
  CHECK_THROWS(take_along_axis(a, array(0), 0));

  // Index and array size mismatches
  a = arange(5);
  CHECK_THROWS(take_along_axis(a, array({1}), 1));
  CHECK_THROWS(take_along_axis(a, array({1}, {1, 1}), 0));
  CHECK_THROWS(take_along_axis(a, array(1), -1));

  auto out = take_along_axis(a, array({1}), 0);
  CHECK_EQ(out.item<int>(), 1);
  out = take_along_axis(a, array({1}), -1);
  CHECK_EQ(out.item<int>(), 1);

  // Empty arrays
  a = reshape(array({}), {1, 0});
  CHECK_THROWS(take_along_axis(a, array({1}), 0));

  out = take_along_axis(a, reshape(array({1}), {1, 1}), 0);
  eval(out); // Make sure it runs
  CHECK_EQ(out.shape(), Shape{1, 0});

  auto inds = reshape(astype(array({}), int32), {1, 0});
  out = take_along_axis(a, inds, 0);
  eval(out); // Make sure it runs
  CHECK_EQ(out.shape(), Shape{1, 0});

  a = array({1, 2, 3, 4}, {2, 2});
  inds = array({0, 1}, {1, 2});
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({1, 4}, {1, 2})).item<bool>());

  inds = array({0, 1, 0, 1, 0, 0, 1, 0}, {4, 2}, int32);
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({1, 4, 1, 4, 1, 2, 3, 2}, {4, 2})).item<bool>());

  inds = array({0, 1}, {2, 1});
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({1, 4}, {2, 1})).item<bool>());

  // Broadcasting works
  inds = array({0}, {1, 1});
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({1, 2}, {1, 2})).item<bool>());
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({1, 3}, {2, 1})).item<bool>());

  inds = array({0, 1, 1, 0, 0, 1}, {2, 3}, int32);
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({1, 2, 2, 3, 3, 4}, {2, 3})).item<bool>());

  a = reshape(arange(8), {2, 2, 2});
  inds = array({0, 1, 0, 0, 1, 0, 0, 1}, {2, 2, 2});
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({0, 5, 2, 3, 4, 1, 2, 7}, {2, 2, 2}))
            .item<bool>());
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({0, 3, 0, 1, 6, 5, 4, 7}, {2, 2, 2}))
            .item<bool>());
  out = take_along_axis(a, inds, 2);
  CHECK(array_equal(out, array({0, 1, 2, 2, 5, 4, 6, 7}, {2, 2, 2}))
            .item<bool>());
}

TEST_CASE("test put along axis") {
  // No zero dim arrays
  auto a = array(1);
  auto v = array(1);
  CHECK_THROWS(put_along_axis(a, array(0), v, 0));

  // Index and array size mismatches
  a = arange(5);
  CHECK_THROWS(put_along_axis(a, array({1}), array({0}), 1));
  CHECK_THROWS(put_along_axis(a, array({1}, {1, 1}), array({0}), 0));
  CHECK_THROWS(put_along_axis(a, array(1), array(0), -1));

  auto expected = array({0, 0, 2, 3, 4});
  auto out = put_along_axis(a, array({1}), array({0}), 0);
  CHECK(array_equal(out, expected).item<bool>());

  // Empty arrays
  a = reshape(array({}), {1, 0});
  CHECK_THROWS(put_along_axis(a, array({1}), array({0}), 0));

  auto inds = reshape(astype(array({}), int32), {1, 0});
  out = take_along_axis(a, inds, 0);
  eval(out); // Make sure it runs
  CHECK_EQ(out.shape(), Shape{1, 0});

  a = array({1, 2, 3, 4}, {2, 2});
  inds = array({0, 1}, {1, 2});
  out = put_along_axis(a, inds, array({0}), 0);
  expected = array({0, 2, 3, 0}, {2, 2});
  CHECK(array_equal(out, expected).item<bool>());

  inds = array({0, 0, 1, 1}, {2, 2}, int32);
  auto values = array({2, 3, 4, 5}, {2, 2}, int32);
  out = put_along_axis(a, inds, values, 0);
  CHECK(array_equal(out, array({2, 3, 4, 5}, {2, 2})).item<bool>());

  inds = array({0, 1}, {2, 1});
  out = put_along_axis(a, inds, array({0}), 1);
  expected = array({0, 2, 3, 0}, {2, 2});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test scatter") {
  // More indices than dimensions
  CHECK_THROWS(scatter(array(0), array({1}), array(1), 0));

  // Mismatch dimensions and indices
  CHECK_THROWS(scatter(array({0}), {array({0})}, array({1}, {1, 1}), {0, 1}));
  CHECK_THROWS(scatter(array({0}), array({0}), array({1}, {1, 1}), -1));

  // Repeat dimensions
  CHECK_THROWS(scatter(
      array({0}, {1, 1}), {array({0}), array({0})}, array({1}), {0, 0}));

  // Update sizes incorrect
  CHECK_THROWS(scatter(array({0}), array({0}), array({0, 1}), 0));
  CHECK_THROWS(scatter(array({0}), array({0}), array({0, 1}, {2, 1}), 0));
  CHECK_THROWS(scatter(array({0}, {1}), array({0}), array({0, 1}, {1, 2}), 0));

  // Wrong index type
  CHECK_THROWS(scatter(array({0}), array({0.0f}), array({0}, {1, 1}), 0));
  CHECK_THROWS(scatter(
      array({0}, {1, 1}),
      {array({0}), array({0.0f})},
      array({1}, {1, 1, 1}),
      {0, 1}));

  // Index arrays must be broadcastable
  CHECK_THROWS(scatter(
      array({0}, {1, 1}),
      {array({0, 0, 0}, {3}), array({0, 0}, {2})},
      ones({3, 2, 1, 1}),
      {0, 1}));

  // Single element scatter
  auto in = zeros({4}, float32);
  auto inds = arange(2);
  auto updates = ones({2, 1}, float32);
  auto out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, array({1.0f, 1.0f, 0.0f, 0.0f})).item<bool>());

  // Single element scatter add
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = ones({3, 1}, float32);
  out = scatter_add(in, inds, updates, 0);
  CHECK(array_equal(out, array({3.0f, 1.0f, 1.0f, 2.0f})).item<bool>());

  // Single element scatter prod
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = full({3, 1}, 2.0f, float32);
  out = scatter_prod(in, inds, updates, 0);
  CHECK(array_equal(out, array({4.0f, 1.0f, 1.0f, 2.0f})).item<bool>());

  // Single element scatter max
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = array({1.0f, 6.0f, -2.0f}, {3, 1});
  out = scatter_max(in, inds, updates, 0);
  CHECK(array_equal(out, array({6.0f, 1.0f, 1.0f, 1.0f})).item<bool>());

  // Single element scatter min
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = array({1.0f, -6.0f, 2.0f}, {3, 1});
  out = scatter_min(in, inds, updates, 0);
  CHECK(array_equal(out, array({-6.0f, 1.0f, 1.0f, 1.0f})).item<bool>());

  // Empty scatter
  in = arange(4, float32);
  inds = astype(array({}), uint32);
  updates = reshape(array({}), {0, 1});
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, in).item<bool>());

  // Array scatters
  in = zeros({4, 4}, float32);
  inds = array({0, 1, 2, 3});
  updates = reshape(arange(16, float32), {4, 1, 4});
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, reshape(arange(16, float32), {4, 4})).item<bool>());

  // Array scatters with col contiguous updates
  in = zeros({4, 4}, float32);
  inds = array({0, 1, 2, 3});
  updates = transpose(reshape(arange(16, float32), {4, 1, 4}));
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, transpose(reshape(arange(16, float32), {4, 4})))
            .item<bool>());

  // Irregular strided index and reduce collision test
  in = zeros({10}, float32);
  inds = broadcast_to(array(3), {10});
  updates = ones({10, 1}, float32);
  out = scatter_add(in, inds, updates, 0);
  CHECK_EQ(take(out, array(3)).item<float>(), 10);

  // 1 element array with 0 dim index
  in = array({1}, int32);
  updates = array({2}, int32);
  out = scatter_max(in, array(0), updates, 0);
  CHECK_EQ(out.item<int>(), 2);

  // No index arrays or axes
  out = scatter_max(array(1), {}, array(2), std::vector<int>{});
  CHECK_EQ(out.item<int>(), 2);

  // Irregularly strided updates test
  in = ones({3, 3});
  updates = broadcast_to(array({2, 2, 2}), {1, 3, 3});
  inds = array({0});
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, ones({3, 3}) * 2).item<bool>());

  // Along different axis
  in = zeros({2, 3});
  updates = array({1, 2, 3, 4}, {2, 2, 1});
  inds = array({0, 2});
  out = scatter(in, inds, updates, 1);
  auto expected = array({1, 0, 3, 2, 0, 4}, {2, 3});
  CHECK(array_equal(out, expected).item<bool>());

  // Multiple index arrays
  in = zeros({2, 2});
  updates = array({1, 2}, {2, 1, 1});
  inds = array({0, 1});
  out = scatter(in, {inds, inds}, updates, {0, 1});
  CHECK(array_equal(out, array({1, 0, 0, 2}, {2, 2})).item<bool>());

  // Broadcasted indices
  in = zeros({2, 2});
  updates = array({5, 2, 9, 1}, {2, 2, 1, 1});
  auto inds0 = array({0, 1}, {2, 1});
  auto inds1 = array({0, 1}, {1, 2});
  out = scatter(in, {inds0, inds1}, updates, {0, 1});
  CHECK(array_equal(out, array({5, 2, 9, 1}, {2, 2})).item<bool>());

  // Brodacasted operand
  in = broadcast_to(array({0, 0}), {2, 2});
  updates = array({1, 1}, {2, 1, 1});
  inds = array({0, 1});
  out = scatter_add(in, inds, updates, 0);
  CHECK(array_equal(out, array({1, 0, 1, 0}, {2, 2})).item<bool>());

  // 1D scatter
  {
    auto dst = zeros({2, 4}, int32);
    auto src = reshape(array({1, 2, 3, 4}), {1, 1, 4});
    auto idx = array({1});
    auto expected = reshape(array({0, 0, 0, 0, 1, 2, 3, 4}), {2, 4});
    auto out = scatter(dst, idx, src, 0);
    CHECK(array_equal(out, expected).item<bool>());
  }

  // 1D indices with 2D update
  {
    auto dst = zeros({3, 4}, int32);
    auto indices = {array({1}), array({2})};
    auto axes = {0, 1};
    auto updates = reshape(array({1, 2, 3, 4}, int32), {1, 2, 2});
    auto out = scatter(dst, indices, updates, axes);
    auto expected =
        reshape(array({0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4}), {3, 4});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test masked_scatter") {
  // Wrong mask dtype
  CHECK_THROWS(masked_scatter(array({1, 2}), array({1, 2}), array({1, 2})));

  // Mask must be broadcastable to self array
  CHECK_THROWS(masked_scatter(
      array({1, 2, 3, 4}, {2, 2}),
      array({false, true, true, false}, {4, 1}),
      array({1, 2})));

  // 1D mask
  {
    auto self = zeros({4}, int32);
    auto mask = array({true, true, false, true});
    auto source = array({1, 2, 4});
    auto out = masked_scatter(self, mask, source);
    CHECK(array_equal(out, array({1, 2, 0, 4})).item<bool>());
  }

  // Empty mask
  {
    auto self = zeros({4}, int32);
    auto mask = array({false, false, false, false});
    auto source = array({1, 2, 4});
    auto out = masked_scatter(self, mask, source);
    CHECK(array_equal(out, self).item<bool>());
  }

  // Broadcasted mask
  {
    auto self = zeros({2, 2}, int32);
    auto mask = array({true, false});
    auto source = array({5, 6, 7, 8}, {2, 2});
    auto out = masked_scatter(self, mask, source);
    CHECK(array_equal(out, array({5, 6, 0, 0}, {2, 2})).item<bool>());
  }
}

TEST_CASE("test is positive infinity") {
  array x(1.0f);
  CHECK_FALSE(isposinf(x).item<bool>());

  array y(std::numeric_limits<float>::infinity());
  CHECK(isposinf(y).item<bool>());

  array z = identity(7);
  CHECK_FALSE(all(isposinf(z)).item<bool>());

  array w = array({1.0f, std::numeric_limits<float>::infinity(), 2.0f});
  CHECK_FALSE(all(isposinf(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isposinf(a).item<bool>());

  array b(std::numeric_limits<float>::infinity(), float16);
  CHECK(isposinf(b).item<bool>());

  array c(std::numeric_limits<float>::infinity(), bfloat16);
  CHECK(isposinf(c).item<bool>());
}

TEST_CASE("test is negative infinity") {
  array x(1.0f);
  CHECK_FALSE(isneginf(x).item<bool>());

  array y(-std::numeric_limits<float>::infinity());
  CHECK(isneginf(y).item<bool>());

  array z = identity(7);
  CHECK_FALSE(all(isneginf(z)).item<bool>());

  array w = array({1.0f, -std::numeric_limits<float>::infinity(), 2.0f});
  CHECK_FALSE(all(isneginf(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isneginf(a).item<bool>());

  array b(-std::numeric_limits<float>::infinity(), float16);
  CHECK(isneginf(b).item<bool>());

  array c(-std::numeric_limits<float>::infinity(), bfloat16);
  CHECK(isneginf(c).item<bool>());
}

TEST_CASE("test scatter types") {
  for (auto t : {bool_, uint8, uint16, int8, int16}) {
    auto in = zeros({4, 4}, t);
    auto inds = {arange(4), arange(4)};
    auto updates = ones({4, 1, 1}, t);
    auto out = scatter(in, inds, updates, {0, 1});
    auto expected =
        array({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, t);
    CHECK(array_equal(out, expected).item<bool>());
  }

  for (auto t : {float16, bfloat16}) {
    auto in = zeros({4, 4}, t);
    auto inds = {arange(4), arange(4)};
    auto updates = ones({4, 1, 1}, t);
    auto out = scatter(in, inds, updates, {0, 1});
    auto expected =
        array({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, t);
    CHECK(allclose(out, expected).item<bool>());
  }
}

TEST_CASE("test complex ops") {
  //  Creation ops
  {
    auto x = full({2, 2}, complex64_t{1, 1});
    CHECK_EQ(x.dtype(), complex64);
    std::initializer_list<complex64_t> expected = {
        {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    CHECK(array_equal(x, array(expected, {2, 2})).item<bool>());
  }

  // Unary ops
  {
    std::initializer_list<complex64_t> vals = {{0, 1}, {1, 0}, {1, 1}};
    auto x = array(vals);

    auto y = abs(x);
    CHECK_EQ(y.dtype(), float32);
    CHECK(array_equal(y, array({1.0f, 1.0f, std::sqrt(2.0f)})).item<bool>());

    y = negative(x);
    std::initializer_list<complex64_t> expected = {{0, -1}, {-1, 0}, {-1, -1}};
    CHECK(array_equal(y, array(expected)).item<bool>());

    y = exp(x);
    {
      std::initializer_list<complex64_t> expected = {
          {0.54030231, 0.84147098}, {2.71828183, 0.}, {1.46869394, 2.28735529}};
      CHECK(allclose(y, array(expected)).item<bool>());
    }

    y = sin(x);
    {
      std::initializer_list<complex64_t> expected = {
          {0., 1.17520119}, {0.84147098, 0.}, {1.29845758, 0.63496391}};
      CHECK(allclose(y, array(expected)).item<bool>());
    }

    y = cos(x);
    {
      std::initializer_list<complex64_t> expected = {
          {1.54308063, -0.}, {0.54030231, -0.}, {0.83373003, -0.98889771}};
      CHECK(allclose(y, array(expected)).item<bool>());
    }
  }

  // Binary ops
  {
    std::initializer_list<complex64_t> vals_x = {{0, 1}, {1, 0}, {1, 1}};
    auto x = array(vals_x);

    std::initializer_list<complex64_t> vals_y = {{2, 0}, {1, 1}, {0, 1}};
    auto y = array(vals_y);

    auto z = add(x, y);
    {
      std::initializer_list<complex64_t> expected = {{2, 1}, {2, 1}, {1, 2}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }

    z = subtract(x, y);
    {
      std::initializer_list<complex64_t> expected = {{-2, 1}, {0, -1}, {1, 0}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }

    z = multiply(x, y);
    {
      std::initializer_list<complex64_t> expected = {{0, 2}, {1, 1}, {-1, 1}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }

    z = maximum(x, y);
    {
      std::initializer_list<complex64_t> expected = {{2, 0}, {1, 1}, {1, 1}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }
  }

  // Reductions
  if (default_device() == Device::cpu) {
    std::initializer_list<complex64_t> vals = {{0, 0}, {1, 0}, {0, 1}};
    auto x = array(vals);
    CHECK_EQ(max(x).item<complex64_t>(), complex64_t{1, 0});
    CHECK_EQ(min(x).item<complex64_t>(), complex64_t{0, 0});
    CHECK_EQ(sum(x).item<complex64_t>(), complex64_t{1, 1});
    CHECK_EQ(prod(x).item<complex64_t>(), complex64_t{0, 0});
  }
}

namespace {

template <typename T>
struct ComplexCastRow {
  float re;
  T out;
};

std::string dtype_name(Dtype dtype) {
  std::ostringstream os;
  os << dtype;
  return os.str();
}

// Each row with several imaginary parts, which must not change the result.
// 256 values or more, so that a 4 x 4 x 4 x 4 slice is possible.
template <typename T>
std::pair<std::vector<complex64_t>, std::vector<T>> complex_cast_values(
    const std::vector<ComplexCastRow<T>>& rows) {
  std::vector<complex64_t> in;
  std::vector<T> out;
  for (float im : {0.0f, 2.5f, -1e30f, NAN, INFINITY}) {
    for (const auto& r : rows) {
      in.push_back(complex64_t{r.re, im});
      out.push_back(r.out);
    }
  }
  size_t m = in.size();
  size_t n = (m + 255) / 256 * 256;
  for (size_t i = m; i < n; ++i) {
    in.push_back(in[i % m]);
    out.push_back(out[i % m]);
  }
  return {in, out};
}

template <typename T>
void check_complex_to_int(const std::vector<ComplexCastRow<T>>& rows) {
  auto dtype = TypeToDtype<T>();
  INFO("from complex64 to " << dtype_name(dtype));
  auto equal = [](const array& a, const array& b) {
    return array_equal(a, b).item<bool>();
  };
  auto [in, out] = complex_cast_values(rows);
  int n = in.size();
  auto x = array(in.begin(), {n}, complex64);
  auto expected = array(out.begin(), {n}, dtype);

  // Contiguous, then strided with gaps in 1, 2 and 4 axes
  CHECK(equal(astype(x, dtype, Device::cpu), expected));
  CHECK(equal(
      astype(slice(x, {0}, {n}, {2}), dtype, Device::cpu),
      slice(expected, {0}, {n}, {2})));
  auto x2 = reshape(x, {n / 4, 4});
  auto e2 = reshape(expected, {n / 4, 4});
  CHECK(equal(
      astype(slice(x2, {0, 0}, {n / 4, 4}, {2, 2}), dtype, Device::cpu),
      slice(e2, {0, 0}, {n / 4, 4}, {2, 2})));
  auto x4 = reshape(x, {n / 64, 4, 4, 4});
  auto e4 = reshape(expected, {n / 64, 4, 4, 4});
  CHECK(equal(
      astype(
          slice(x4, {0, 0, 0, 0}, {n / 64, 4, 4, 4}, {2, 2, 2, 2}),
          dtype,
          Device::cpu),
      slice(e4, {0, 0, 0, 0}, {n / 64, 4, 4, 4}, {2, 2, 2, 2})));

  // 0-d, the Simd constructors, and the copy kernels that no public op
  // reaches with two types (CopyType::Scalar and a 0-d CopyType::General)
  auto stream = default_stream(Device::cpu);
  bool scalars_ok = true;
  bool simd_ok = true;
  bool fill_ok = true;
  bool zero_d_ok = true;
  for (int i = 0; i < n; ++i) {
    auto s = array(in[i]);
    scalars_ok &= astype(s, dtype, Device::cpu).template item<T>() == out[i];
    simd_ok &=
        simd::Simd<T, 1>(simd::Simd<complex64_t, 1>(in[i])).value == out[i];
    simd_ok &= simd::Simd<T, 1>(in[i]).value == out[i];
    eval(s);
    array filled(Shape{3}, dtype, nullptr, {});
    filled.set_data(allocator::malloc(filled.nbytes()));
    copy_cpu_inplace(s, filled, CopyType::Scalar, stream);
    array single(Shape{}, dtype, nullptr, {});
    single.set_data(allocator::malloc(single.nbytes()));
    copy_cpu_inplace(
        s,
        single,
        Shape{},
        Strides{},
        Strides{},
        0,
        0,
        CopyType::General,
        stream);
    synchronize(stream);
    fill_ok &= filled.data<T>()[0] == out[i] && filled.data<T>()[2] == out[i];
    zero_d_ok &= single.data<T>()[0] == out[i];
  }
  CHECK(scalars_ok);
  CHECK(simd_ok);
  CHECK(fill_ok);
  CHECK(zero_d_ok);
}

template <typename T>
std::vector<array> complex_cast_to(const std::vector<array>& inputs) {
  auto x = add(inputs[0], array(complex64_t{0.0f, 0.0f}), Device::cpu);
  return {astype(x, TypeToDtype<T>(), Device::cpu)};
}

template <typename T>
void check_compiled_complex_to_int(const std::vector<ComplexCastRow<T>>& rows) {
  auto dtype = TypeToDtype<T>();
  INFO("from complex64 to " << dtype_name(dtype));
  auto [in, out] = complex_cast_values(rows);
  int n = in.size();
  auto x = array(in.begin(), {n}, complex64);
  auto expected = array(out.begin(), {n}, dtype);
  auto equal = [](const array& a, const array& b) {
    return array_equal(a, b).item<bool>();
  };
  auto f = compile(complex_cast_to<T>);
  CHECK(equal(f({x})[0], expected));
  CHECK(equal(f({slice(x, {0}, {n}, {2})})[0], slice(expected, {0}, {n}, {2})));
}

// The input and the expected output for each integer type: the real part,
// truncated toward zero, the maximum or the minimum if it does not fit, and
// 0 for NaN, as on the GPU.
template <typename F>
void for_complex_cast_rows(F check) {
  std::vector<ComplexCastRow<int8_t>> int8_rows = {
      {NAN, 0},
      {INFINITY, 127},
      {-INFINITY, -128},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, -1},
      {300.5f, 127},
      {-300.5f, -128},
      {10000000000.0f, 127},
      {-10000000000.0f, -128},
      {3.4e+38f, 127},
      {-3.4e+38f, -128},
      {127.99999f, 127},
      {128.0f, 127},
      {128.00002f, 127},
      {127.0f, 127},
      {127.5f, 127},
      {-128.00002f, -128},
      {-128.0f, -128},
      {-127.99999f, -127},
      {-128.5f, -128},
      {-129.0f, -128},
  };
  std::vector<ComplexCastRow<uint8_t>> uint8_rows = {
      {NAN, 0},
      {INFINITY, 255},
      {-INFINITY, 0},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, 0},
      {300.5f, 255},
      {-300.5f, 0},
      {10000000000.0f, 255},
      {-10000000000.0f, 0},
      {3.4e+38f, 255},
      {-3.4e+38f, 0},
      {255.99998f, 255},
      {256.0f, 255},
      {256.00003f, 255},
      {255.0f, 255},
      {255.5f, 255},
      {-1.0f, 0},
      {-0.99999994f, 0},
      {-1e+30f, 0},
  };
  std::vector<ComplexCastRow<int16_t>> int16_rows = {
      {NAN, 0},
      {INFINITY, 32767},
      {-INFINITY, -32768},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, -1},
      {300.5f, 300},
      {-300.5f, -300},
      {10000000000.0f, 32767},
      {-10000000000.0f, -32768},
      {3.4e+38f, 32767},
      {-3.4e+38f, -32768},
      {32767.998f, 32767},
      {32768.0f, 32767},
      {32768.004f, 32767},
      {32767.0f, 32767},
      {32767.5f, 32767},
      {-32768.004f, -32768},
      {-32768.0f, -32768},
      {-32767.998f, -32767},
      {-32768.5f, -32768},
      {-32769.0f, -32768},
  };
  std::vector<ComplexCastRow<uint16_t>> uint16_rows = {
      {NAN, 0},
      {INFINITY, 65535},
      {-INFINITY, 0},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, 0},
      {300.5f, 300},
      {-300.5f, 0},
      {10000000000.0f, 65535},
      {-10000000000.0f, 0},
      {3.4e+38f, 65535},
      {-3.4e+38f, 0},
      {65535.996f, 65535},
      {65536.0f, 65535},
      {65536.01f, 65535},
      {65535.0f, 65535},
      {65535.5f, 65535},
      {-1.0f, 0},
      {-0.99999994f, 0},
      {-1e+30f, 0},
  };
  std::vector<ComplexCastRow<int32_t>> int32_rows = {
      {NAN, 0},
      {INFINITY, 2147483647},
      {-INFINITY, INT32_MIN},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, -1},
      {300.5f, 300},
      {-300.5f, -300},
      {10000000000.0f, 2147483647},
      {-10000000000.0f, INT32_MIN},
      {3.4e+38f, 2147483647},
      {-3.4e+38f, INT32_MIN},
      {2147483520.0f, 2147483520},
      {2147483648.0f, 2147483647},
      {2147483904.0f, 2147483647},
      {-2147483904.0f, INT32_MIN},
      {-2147483648.0f, INT32_MIN},
      {-2147483520.0f, -2147483520},
  };
  std::vector<ComplexCastRow<uint32_t>> uint32_rows = {
      {NAN, 0},
      {INFINITY, 4294967295},
      {-INFINITY, 0},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, 0},
      {300.5f, 300},
      {-300.5f, 0},
      {10000000000.0f, 4294967295},
      {-10000000000.0f, 0},
      {3.4e+38f, 4294967295},
      {-3.4e+38f, 0},
      {4294967040.0f, 4294967040},
      {4294967296.0f, 4294967295},
      {4294967808.0f, 4294967295},
      {-1.0f, 0},
      {-0.99999994f, 0},
      {-1e+30f, 0},
  };
  std::vector<ComplexCastRow<int64_t>> int64_rows = {
      {NAN, 0},
      {INFINITY, INT64_MAX},
      {-INFINITY, INT64_MIN},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, -1},
      {300.5f, 300},
      {-300.5f, -300},
      {10000000000.0f, 10000000000LL},
      {-10000000000.0f, -10000000000LL},
      {3.4e+38f, INT64_MAX},
      {-3.4e+38f, INT64_MIN},
      {9223371487098961920.0f, 9223371487098961920LL},
      {9223372036854775808.0f, INT64_MAX},
      {9223373136366403584.0f, INT64_MAX},
      {-9223373136366403584.0f, INT64_MIN},
      {-9223372036854775808.0f, INT64_MIN},
      {-9223371487098961920.0f, -9223371487098961920LL},
  };
  std::vector<ComplexCastRow<uint64_t>> uint64_rows = {
      {NAN, 0},
      {INFINITY, UINT64_MAX},
      {-INFINITY, 0},
      {0.0f, 0},
      {-0.0f, 0},
      {0.5f, 0},
      {-0.5f, 0},
      {1.5f, 1},
      {-1.5f, 0},
      {300.5f, 300},
      {-300.5f, 0},
      {10000000000.0f, 10000000000},
      {-10000000000.0f, 0},
      {3.4e+38f, UINT64_MAX},
      {-3.4e+38f, 0},
      {18446742974197923840.0f, 18446742974197923840ULL},
      {18446744073709551616.0f, UINT64_MAX},
      {18446746272732807168.0f, UINT64_MAX},
      {-1.0f, 0},
      {-0.99999994f, 0},
      {-1e+30f, 0},
  };
  check(int8_rows);
  check(uint8_rows);
  check(int16_rows);
  check(uint16_rows);
  check(int32_rows);
  check(uint32_rows);
  check(int64_rows);
  check(uint64_rows);
}

} // namespace

TEST_CASE("test astype from complex64 to integer uses the real part") {
  // A complex64 converted to an integer used to be a plain conversion of the
  // real part, which is undefined in C++ when the value does not fit.
  for_complex_cast_rows([](const auto& rows) { check_complex_to_int(rows); });
}

TEST_CASE("test compiled astype from complex64 to integer uses the real part") {
  for_complex_cast_rows(
      [](const auto& rows) { check_compiled_complex_to_int(rows); });
}

TEST_CASE("test as_strided op") {
  auto x = arange(10);
  auto y = as_strided(x, {3, 3}, {1, 1}, 0);
  auto expected = array({0, 1, 2, 1, 2, 3, 2, 3, 4}, {3, 3});
  eval(y);
  CHECK(array_equal(y, expected).item<bool>());
  CHECK_EQ(y.data_size(), 5);
  CHECK_FALSE(y.flags().contiguous);

  y = as_strided(x, {3, 3}, {0, 3}, 0);
  expected = array({0, 3, 6, 0, 3, 6, 0, 3, 6}, {3, 3});
  eval(y);
  CHECK(array_equal(y, expected).item<bool>());
  CHECK_EQ(y.data_size(), 7);
  CHECK_FALSE(y.flags().contiguous);

  x = arange(24);
  y = as_strided(x, {2, 3, 4}, {3, 1, 6}, 0);
  expected = array(
      {0, 6, 12, 18, 1, 7,  13, 19, 2, 8,  14, 20,
       3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23},
      {2, 3, 4});
  eval(y);
  CHECK(array_equal(y, expected).item<bool>());
  CHECK_EQ(y.data_size(), 24);
  CHECK(y.flags().contiguous);
  CHECK_FALSE(y.flags().row_contiguous);
  CHECK_FALSE(y.flags().col_contiguous);

  auto z = astype(y, float32);
  CHECK(array_equal(z, astype(expected, float32)).item<bool>());

  x = arange(10);
  y = as_strided(x, {10}, {-1}, 9);
  expected = array({9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, {10});
  eval(y);
  CHECK(array_equal(y, expected).item<bool>());
  CHECK_EQ(y.data_size(), 10);
  CHECK_FALSE(y.flags().contiguous);

  auto xf = square(arange(16, float32));
  y = as_strided(xf, {2, 2, 2}, {1, 1, 5}, 0);
  eval(y);
  CHECK_EQ(y.data_size(), 8);
  CHECK_FALSE(y.flags().contiguous);
  CHECK_EQ(sum(y).item<float>(), 152.0f);

  x = reshape(x, {2, 5}); // 0 1 2 3 ...
  x = transpose(x, {1, 0}); // 0 5 1 6 2 7 ...
  y = as_strided(x, {3, 3}, {2, 1}, 1);
  expected = array({5, 1, 6, 6, 2, 7, 7, 3, 8}, {3, 3});
  CHECK(array_equal(y, expected).item<bool>());

  x = slice(arange(12), {10}, {0}, {-2});
  y = as_strided(x, {2, 2}, {2, 1}, 1);
  expected = array({8, 6, 4, 2}, {2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  x = broadcast_to(array({1, 2, 3}), {2, 3});
  y = as_strided(x, {2, 2}, {2, 1}, 1);
  expected = array({2, 3, 1, 2}, {2, 2});
  CHECK(array_equal(y, expected).item<bool>());
}

TEST_CASE("test sort and scan on empty arrays") {
  // These must be evaluated, not only shape checked, to reach the kernel.
  auto check_empty = [](const array& r, Shape shape, Dtype dt) {
    auto out = r;
    eval(out);
    CHECK_EQ(out.shape(), shape);
    CHECK_EQ(out.dtype(), dt);
    CHECK_EQ(out.size(), 0);
  };

  for (auto dt : {float32, int32, uint32, bool_}) {
    auto x = zeros({0}, dt);
    check_empty(sort(x), Shape{0}, dt);
    check_empty(argsort(x), Shape{0}, uint32);
    check_empty(cumsum(x), Shape{0}, dt == bool_ ? int32 : dt);
    check_empty(cumprod(x), Shape{0}, dt);
    check_empty(cummax(x), Shape{0}, dt);
    check_empty(cummin(x), Shape{0}, dt);
  }

  // Empty along the sorted axis, non-empty elsewhere.
  auto x = zeros({3, 0}, float32);
  check_empty(sort(x, 1), Shape{3, 0}, float32);
  check_empty(argsort(x, 1), Shape{3, 0}, uint32);
  check_empty(cumsum(x, 1), Shape{3, 0}, float32);

  // Empty on another axis, so the sorted axis itself is non-empty.
  x = zeros({0, 3}, float32);
  check_empty(sort(x, 1), Shape{0, 3}, float32);
  check_empty(argsort(x, 1), Shape{0, 3}, uint32);
  check_empty(cumsum(x, 1), Shape{0, 3}, float32);

  // Non-contiguous empty input reaches the strided path.
  x = transpose(zeros({4, 0}, float32));
  check_empty(sort(x, 1), Shape{0, 4}, float32);
  check_empty(cumsum(x, 1), Shape{0, 4}, float32);
}

TEST_CASE("test scan op") {
  auto x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  auto y = cumsum(x, 1, false, true);
  auto expected = array({1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, false, false);
  expected = array({0.0f, 1.0f, 3.0f, 0.0f, 4.0f, 9.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, true, true);
  expected = array({6.0f, 5.0f, 3.0f, 15.0f, 11.0f, 6.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, true, false);
  expected = array({5.0f, 3.0f, 0.0f, 11.0f, 6.0f, 0.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});
  y = cumsum(x, 0, false, true);
  expected =
      array({1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, false, true);
  expected =
      array({1.0f, 2.0f, 4.0f, 6.0f, 5.0f, 6.0f, 12.0f, 14.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});
  y = cumsum(x, 0, true, true);
  expected =
      array({6.0f, 8.0f, 10.0f, 12.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, true, true);
  expected =
      array({4.0f, 6.0f, 3.0f, 4.0f, 12.0f, 14.0f, 7.0f, 8.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  x = reshape(x, {4, 2});
  y = cumsum(x, 0, false, false);
  expected = array({0.0f, 0.0f, 1.0f, 2.0f, 4.0f, 6.0f, 9.0f, 12.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 0, true, false);
  expected =
      array({15.0f, 18.0f, 12.0f, 14.0f, 7.0f, 8.0f, 0.0f, 0.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());

  // Check the vmap implementation
  auto fun = [](array x) { return cumsum(x, 0, false, true); };
  y = vmap(fun, 0, 0)(x);
  expected = array({1.0f, 3.0f, 3.0f, 7.0f, 5.0f, 11.0f, 7.0f, 15.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = vmap(fun, 1, 1)(x);
  expected = array({1.0f, 2.0f, 4.0f, 6.0f, 9.0f, 12.0f, 16.0f, 20.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());

  // Scanning an empty axis is a no-op
  x = zeros({2, 0});
  y = cumsum(x, 1);
  eval(y);
  CHECK_EQ(y.shape(), Shape{2, 0});
  y = cummax(x, 1);
  eval(y);
  CHECK_EQ(y.shape(), Shape{2, 0});
  x = zeros({0, 2});
  y = cumsum(x, 0);
  eval(y);
  CHECK_EQ(y.shape(), Shape{0, 2});
  y = cummin(x, 0);
  eval(y);
  CHECK_EQ(y.shape(), Shape{0, 2});
}

TEST_CASE("test sort op") {
  // Sorting an empty axis is a no-op
  auto x = zeros({2, 0});
  auto y = sort(x, 1);
  eval(y);
  CHECK_EQ(y.shape(), Shape{2, 0});
  y = argsort(x, 1);
  eval(y);
  CHECK_EQ(y.shape(), Shape{2, 0});
  x = zeros({0, 2});
  y = sort(x, 0);
  eval(y);
  CHECK_EQ(y.shape(), Shape{0, 2});
  y = argsort(x, 0);
  eval(y);
  CHECK_EQ(y.shape(), Shape{0, 2});
}

TEST_CASE("test pad") {
  auto x = zeros({1, 2, 3});
  CHECK_EQ(pad(x, 1).shape(), Shape{3, 4, 5});
  CHECK_EQ(pad(x, {0, 1}).shape(), Shape{2, 3, 4});
  CHECK_EQ(pad(x, {{1, 1}, {1, 2}, {3, 1}}).shape(), Shape{3, 5, 7});

  x = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  auto padded_x = pad(x, 1);
  auto expected = array(
      {0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       2.0f,
       0.0f,
       0.0f,
       3.0f,
       4.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f},
      {4, 4});
  CHECK(array_equal(padded_x, expected).item<bool>());

  // reflect padding (mirror without repeating the edge value)
  x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5});
  CHECK(array_equal(
            pad(x, {{2, 2}}, array(0.0f), "reflect"),
            array({3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f}, {9}))
            .item<bool>());
  CHECK(array_equal(
            pad(x, {{0, 3}}, array(0.0f), "reflect"),
            array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f}, {8}))
            .item<bool>());

  // symmetric padding (mirror repeating the edge value)
  CHECK(array_equal(
            pad(x, {{2, 2}}, array(0.0f), "symmetric"),
            array({2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 4.0f}, {9}))
            .item<bool>());
  CHECK(array_equal(
            pad(x, {{3, 0}}, array(0.0f), "symmetric"),
            array({3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {8}))
            .item<bool>());

  // multi-reflect: pad larger than the axis repeats the reflection (numpy
  // parity)
  x = array({1.0f, 2.0f, 3.0f}, {3});
  CHECK(array_equal(
            pad(x, {{5, 5}}, array(0.0f), "reflect"),
            array(
                {2.0f,
                 1.0f,
                 2.0f,
                 3.0f,
                 2.0f,
                 1.0f,
                 2.0f,
                 3.0f,
                 2.0f,
                 1.0f,
                 2.0f,
                 3.0f,
                 2.0f},
                {13}))
            .item<bool>());
}

TEST_CASE("test power") {
  CHECK_EQ(power(array(1), array(2)).item<int>(), 1);
  CHECK_EQ((power(array(-1), array(2))).item<int>(), 1);
  CHECK_EQ((power(array(-1), array(3))).item<int>(), -1);

  CHECK_EQ((power(array(true), array(false))).item<bool>(), true);
  CHECK_EQ((power(array(false), array(false))).item<bool>(), true);
  CHECK_EQ((power(array(true), array(true))).item<bool>(), true);
  CHECK_EQ((power(array(false), array(true))).item<bool>(), false);

  auto x = array(2.0f);
  CHECK_EQ(
      (power(x, array(0.5))).item<float>(),
      doctest::Approx(std::pow(2.0f, 0.5f)));
  CHECK_EQ(power(x, array(2.0f)).item<float>(), 4.0f);

  CHECK(std::isnan((power(array(-1.0f), array(0.5))).item<float>()));

  auto a = complex64_t{0.5, 0.5};
  auto b = complex64_t{0.5, 0.5};
  auto expected = std::pow(a, b);
  auto out = (power(array(a), array(b))).item<complex64_t>();
  CHECK(abs(out.real() - expected.real()) < 1e-7);
  CHECK(abs(out.imag() - expected.imag()) < 1e-7);

  a = complex64_t{-1.2, 0.1};
  b = complex64_t{2.2, 0.0};
  expected = std::pow(a, b);
  out = (power(array(a), array(b))).item<complex64_t>();
  CHECK(abs(out.real() - expected.real()) < 1e-6);
  CHECK(abs(out.imag() - expected.imag()) < 1e-6);
}

TEST_CASE("test where") {
  const float inf = std::numeric_limits<float>::infinity();

  array condition(true);
  array x(1.0f);
  array y(0.0f);
  auto out = where(condition, x, y);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.item<float>(), 1.0f);

  x = array({1, 2}, {2, 1});
  y = array({3, 4}, {1, 2});
  CHECK(array_equal(where(condition, x, y), broadcast_to(x, {2, 2}))
            .item<bool>());

  condition = array(false);
  CHECK(array_equal(where(condition, x, y), broadcast_to(y, {2, 2}))
            .item<bool>());

  condition = array({true, false});
  out = where(condition, x, y);
  auto expected = array({1, 4, 2, 4}, {2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  condition = array({true, false, false, true}, {2, 2});
  out = where(condition, x, y);
  expected = array({1, 4, 3, 2}, {2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  x = array(1);
  y = array(2);
  out = where(condition, x, y);
  expected = array({1, 2, 2, 1}, {2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  condition = array(true);
  x = array({1, 2, 3});
  y = array({3, 6, 13});
  CHECK(array_equal(where(condition, x, y), array({1, 2, 3})).item<bool>());

  condition = array(false);
  x = array({1, 2, 3});
  y = array({3, 6, 13});
  CHECK(array_equal(where(condition, x, y), array({3, 6, 13})).item<bool>());

  condition = array({1, 1, 0});
  x = array({1, 2, 3});
  y = array({11, 12, 13});
  CHECK(array_equal(where(condition, x, y), array({1, 2, 13})).item<bool>());

  condition = array({true, false}, {2, 1, 1});
  x = array({1, 2, 3, 4}, {2, 1, 2});
  y = array({11, 22, 33, 44}, {2, 2, 1});
  expected = array({1, 2, 1, 2, 33, 33, 44, 44}, {2, 2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  condition = array({true, false, false});
  x = array({inf, 2.0, 3.0});
  y = array({10.0, 20.0, -inf});
  CHECK(array_equal(where(condition, x, y), array({inf, 20.0, -inf}))
            .item<bool>());

  // 4-dim optimized case.
  condition = array({false});
  x = array({1, 2}, {2, 1, 1, 1});
  y = array({3, 4}, {1, 1, 2, 1});
  CHECK(array_equal(where(condition, x, y), array({3, 4, 3, 4}, {2, 1, 2, 1}))
            .item<bool>());

  // 5-dim optimized case.
  condition = array({true, false}, {2, 1, 1, 1, 1});
  x = array({1, 2, 3, 4}, {2, 1, 1, 1, 2});
  y = array({11, 22}, {1, 1, 2, 1, 1});
  CHECK(array_equal(
            where(condition, x, y),
            array({1, 2, 1, 2, 11, 11, 22, 22}, {2, 1, 2, 1, 2}))
            .item<bool>());
}

TEST_CASE("test stack") {
  auto x = array({});
  CHECK_EQ(stack({x}, 0).shape(), Shape{1, 0});
  CHECK_EQ(stack({x}, 1).shape(), Shape{0, 1});

  x = array({1, 2, 3}, {3});
  CHECK_EQ(stack({x}, 0).shape(), Shape{1, 3});
  CHECK_EQ(stack({x}, 1).shape(), Shape{3, 1});

  auto y = array({4, 5, 6}, {3});
  auto z = std::vector<array>{x, y};
  CHECK_EQ(stack(z).shape(), Shape{2, 3});
  CHECK_EQ(stack(z, 0).shape(), Shape{2, 3});
  CHECK_EQ(stack(z, 1).shape(), Shape{3, 2});
  CHECK_EQ(stack(z, -1).shape(), Shape{3, 2});
  CHECK_EQ(stack(z, -2).shape(), Shape{2, 3});

  CHECK_THROWS_MESSAGE(stack({}, 0), "No arrays provided for stacking");

  x = array({1, 2, 3}, {3}, float16);
  y = array({4, 5, 6}, {3}, int32);
  CHECK_EQ(stack({x, y}, 0).dtype(), float16);

  x = array({1, 2, 3}, {3}, int32);
  y = array({4, 5, 6, 7}, {4}, int32);
  CHECK_THROWS_MESSAGE(
      stack({x, y}, 0), "All arrays must have the same shape and dtype");
}

TEST_CASE("test full_like") {
  auto base_int = array({1, 2, 3}, {3}, int16);

  auto from_array_with_dtype = full_like(base_int, array(7.5f), float16);
  auto expected_float16 = array({7.5, 7.5, 7.5}, {3}, float16);
  CHECK_EQ(from_array_with_dtype.dtype(), float16);
  CHECK(array_equal(from_array_with_dtype, expected_float16).item<bool>());

  auto from_array_default_dtype = full_like(base_int, array(4.0f));
  auto expected_int16 = array({4, 4, 4}, {3}, int16);
  CHECK_EQ(from_array_default_dtype.dtype(), int16);
  CHECK(array_equal(from_array_default_dtype, expected_int16).item<bool>());

  auto from_scalar_with_dtype = full_like(base_int, 3.25f, float32);
  auto expected_float32 = array({3.25f, 3.25f, 3.25f}, {3}, float32);
  CHECK_EQ(from_scalar_with_dtype.dtype(), float32);
  CHECK(array_equal(from_scalar_with_dtype, expected_float32).item<bool>());

  auto base_float = array({1.0f, 2.0f}, {2}, float32);
  auto from_scalar_default_dtype = full_like(base_float, 2);
  auto expected_base_float = array({2.0f, 2.0f}, {2}, float32);
  CHECK_EQ(from_scalar_default_dtype.dtype(), float32);
  CHECK(
      array_equal(from_scalar_default_dtype, expected_base_float).item<bool>());
}

TEST_CASE("test eye") {
  auto eye_3 = eye(3);
  CHECK_EQ(eye_3.shape(), Shape{3, 3});
  auto expected_eye_3 =
      array({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {3, 3});
  CHECK(array_equal(eye_3, expected_eye_3).item<bool>());

  auto eye_3x2 = eye(3, 2);
  CHECK_EQ(eye_3x2.shape(), Shape{3, 2});
  auto expected_eye_3x2 = array({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {3, 2});
  CHECK(array_equal(eye_3x2, expected_eye_3x2).item<bool>());

  auto eye_0x0 = eye(0, 0);
  CHECK_EQ(eye_0x0.shape(), Shape{0, 0});
  CHECK_EQ(eye_0x0.size(), 0);
  CHECK_EQ(eye_0x0.dtype(), float32);
}

TEST_CASE("test tri") {
  auto _tri = tri(4, 4, 0, float32);
  CHECK_EQ(_tri.shape(), Shape{4, 4});
  auto expected_tri = array(
      {1.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       1.0f,
       0.0f,
       0.0f,
       1.0f,
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f,
       1.0f,
       1.0f},
      {4, 4});
  CHECK(array_equal(_tri, expected_tri).item<bool>());
}

TEST_CASE("test tril") {
  auto _tril = tril(full({4, 4}, 2.0f, float32), 0);
  CHECK_EQ(_tril.shape(), Shape{4, 4});
  auto expected_tri = array(
      {2.0f,
       0.0f,
       0.0f,
       0.0f,
       2.0f,
       2.0f,
       0.0f,
       0.0f,
       2.0f,
       2.0f,
       2.0f,
       0.0f,
       2.0f,
       2.0f,
       2.0f,
       2.0f},
      {4, 4});
  CHECK(array_equal(_tril, expected_tri).item<bool>());
}

TEST_CASE("test triu") {
  auto _triu = triu(full({4, 4}, 2.0f, float32), 0);
  CHECK_EQ(_triu.shape(), Shape{4, 4});
  auto expected_tri = array(
      {2.0f,
       2.0f,
       2.0f,
       2.0f,
       0.0f,
       2.0f,
       2.0f,
       2.0f,
       0.0f,
       0.0f,
       2.0f,
       2.0f,
       0.0f,
       0.0f,
       0.0f,
       2.0f},
      {4, 4});
  CHECK(array_equal(_triu, expected_tri).item<bool>());
}

TEST_CASE("test identity") {
  auto id_4 = identity(4);
  CHECK_EQ(id_4.shape(), Shape{4, 4});
  auto expected_id_4 = array(
      {1.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f},
      {4, 4});
  CHECK(array_equal(id_4, expected_id_4).item<bool>());
}

TEST_CASE("test eye with positive k offset") {
  auto eye_3_k1 = eye(3, 4, 1);
  CHECK_EQ(eye_3_k1.shape(), Shape{3, 4});
  auto expected_eye_3_k1 = array(
      {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
      {3, 4});
  CHECK(array_equal(eye_3_k1, expected_eye_3_k1).item<bool>());
}

TEST_CASE("test eye with negative k offset") {
  auto eye_4_k_minus1 = eye(4, 3, -1);
  CHECK_EQ(eye_4_k_minus1.shape(), Shape{4, 3});
  auto expected_eye_4_k_minus1 = array(
      {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
      {4, 3});
  CHECK(array_equal(eye_4_k_minus1, expected_eye_4_k_minus1).item<bool>());
}

TEST_CASE("test basic clipping") {
  array a({1.0f, 4.0f, 3.0f, 8.0f, 5.0f}, {5});
  array expected({2.0f, 4.0f, 3.0f, 6.0f, 5.0f}, {5});
  auto clipped = clip(a, array(2.0f), array(6.0f));
  CHECK(array_equal(clipped, expected).item<bool>());
}

TEST_CASE("test clipping with only min") {
  array a({-1.0f, 1.0f, 0.0f, 5.0f}, {4});
  array expected({0.0f, 1.0f, 0.0f, 5.0f}, {4});
  auto clipped = clip(a, array(0.0f), std::nullopt);
  CHECK(array_equal(clipped, expected).item<bool>());
}

TEST_CASE("test clipping with only max") {
  array a({2.0f, 3.0f, 4.0f, 5.0f}, {4});
  array expected({2.0f, 3.0f, 4.0f, 4.0f}, {4});
  auto clipped = clip(a, std::nullopt, array(4.0f));
  CHECK(array_equal(clipped, expected).item<bool>());
}

TEST_CASE("test linspace") {
  auto x = linspace(0, 10, 5);
  auto expected = array({0.0f, 2.5f, 5.0f, 7.5f, 10.0f}, {5});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(0, 10, 5, true, int32);
  expected = array({0, 2, 5, 7, 10}, {5});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(0, 1, 0);
  expected = array(std::initializer_list<float>{}, {0});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(0, 10, 5, false);
  expected = array({0.0f, 2.0f, 4.0f, 6.0f, 8.0f}, {5});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(0, 10, 5, false, int32);
  expected = array({0, 2, 4, 6, 8}, {5});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(1, 10, 1, false);
  expected = array({1.0f}, {1});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(0, 1, 0, false);
  expected = array(std::initializer_list<float>{}, {0});
  CHECK(array_equal(x, expected).item<bool>());
}

TEST_CASE("test quantize dequantize") {
  auto x1 = ones({128, 1});
  auto x2 = expand_dims(arange(0, 512, float32), 0);
  auto x = x1 * x2;

  for (int i = 2; i <= 8; i *= 2) {
    int el_per_int = 32 / i;
    auto res = quantize(x, 128, i);
    auto x_q = res[0];
    auto scales = res[1];
    auto biases = res[2];
    CHECK_EQ(x_q.shape(), Shape{128, 512 / el_per_int});
    CHECK_EQ(scales.shape(), Shape{128, 4});
    CHECK_EQ(biases.shape(), Shape{128, 4});

    auto x_hat = dequantize(x_q, scales, biases, 128, i);
    auto max_diff = max(abs(x - x_hat)).item<float>();
    CHECK(max_diff <= 127.0 / (1 << i));
  }
}

TEST_CASE("test affine quantized matmul accumulates in float32 on the cpu") {
  // As ml-explore/mlx#4803's test: x repeats [1, eps, -1, 0] and each block of
  // four weights along K is equal, so each block adds eps * w. A sum in the
  // input type loses eps * w next to w; in float32 every partial sum is exact.
  auto s = Device::cpu;
  constexpr int N = 128, K = 256;
  for (auto dtype : {bfloat16, float16}) {
    float eps = dtype == bfloat16 ? 1.0f / 256 : 1.0f / 2048;
    std::vector<float> xv;
    for (int i = 0; i < K / 4; i++) {
      xv.insert(xv.end(), {1.0f, eps, -1.0f, 0.0f});
    }
    auto x1 = array(xv.begin(), {K}, float32);
    auto x = astype(stack({x1, negative(x1, s)}, s), dtype, s);
    for (bool tr : {true, false}) {
      for (int gs : {32, 64, 128}) {
        for (int bits : {2, 4, 8}) {
          CAPTURE(dtype);
          CAPTURE(tr);
          CAPTURE(gs);
          CAPTURE(bits);
          int rows = tr ? N : K, cols = tr ? K : N, groups = cols / gs;
          // All bits set give the code 2**bits - 1, and the bias moves w to v.
          std::vector<int32_t> vv(rows * groups);
          for (int i = 0; i < rows; i++) {
            for (int j = 0; j < groups; j++) {
              vv[i * groups + j] = ((tr ? i : i / 4) + 3 * j) % 8 + 1;
            }
          }
          auto v = array(vv.begin(), {rows, groups}, int32);
          auto q = full(
              {rows, cols * bits / 32}, array(uint32_t(0xFFFFFFFF)), uint32, s);
          auto scales = ones(v.shape(), dtype, s);
          auto biases = astype(subtract(v, array((1 << bits) - 1), s), dtype, s);
          auto w = astype(
              dequantize(
                  q, scales, biases, gs, bits, "affine", std::nullopt,
                  std::nullopt, s),
              float32,
              s);
          auto y_ref = matmul(astype(x, float32, s), tr ? transpose(w, s) : w, s);
          auto y = quantized_matmul(x, q, scales, biases, tr, gs, bits, "affine", s);
          CHECK(array_equal(y, astype(y_ref, dtype, s), false, s).item<bool>());
        }
      }
    }
  }
}

TEST_CASE("test repeat") {
  auto data = array({13, 3, 16, 6, 14, 4, 15, 5, 11, 1, 12, 2}, {3, 2, 2});
  auto repeat_axis_0 = repeat(data, 2, 0);
  auto expected_axis_0 = array(
      {13, 3, 16, 6, 13, 3, 16, 6, 14, 4, 15, 5,
       14, 4, 15, 5, 11, 1, 12, 2, 11, 1, 12, 2},
      {6, 2, 2});

  auto repeat_axis_1 = repeat(data, 2, 1);
  auto expected_axis_1 = array(
      {13, 3, 13, 3, 16, 6, 16, 6, 14, 4, 14, 4,
       15, 5, 15, 5, 11, 1, 11, 1, 12, 2, 12, 2},
      {3, 4, 2});

  auto repeat_axis_2 = repeat(data, 2); // default axis == ndim - 1 == 2
  auto expected_axis_2 = array(
      {13, 13, 3, 3, 16, 16, 6, 6, 14, 14, 4, 4,
       15, 15, 5, 5, 11, 11, 1, 1, 12, 12, 2, 2},
      {24});

  // check output
  CHECK(array_equal(repeat_axis_0, expected_axis_0).item<bool>());
  CHECK(array_equal(repeat_axis_1, expected_axis_1).item<bool>());
  CHECK(array_equal(repeat_axis_2, expected_axis_2).item<bool>());

  auto data_2 = array({1, 3, 2}, {3});
  auto repeat_2 = repeat(data_2, 2, 0);
  auto expected_2 = array({1, 1, 3, 3, 2, 2}, {6});
  CHECK(array_equal(repeat_2, expected_2).item<bool>());

  auto data_3 = array({1, 2, 3, 4, 5, 4, 0, 1, 2}, {3, 3});
  auto repeat_3 = repeat(data_3, 2, 0);
  auto expected_3 =
      array({1, 2, 3, 1, 2, 3, 4, 5, 4, 4, 5, 4, 0, 1, 2, 0, 1, 2}, {6, 3});
  CHECK(array_equal(repeat_3, expected_3).item<bool>());

  // 0 repeats
  auto repeat_4 = repeat(data_3, 0, 0);
  auto expected_4 = array(std::initializer_list<int>{}, {0, 3});
  CHECK(array_equal(repeat_4, expected_4).item<bool>());

  repeat_4 = repeat(data_3, 0, 1);
  expected_4 = array(std::initializer_list<int>{}, {3, 0});
  CHECK(array_equal(repeat_4, expected_4).item<bool>());

  // negative repeats
  CHECK_THROWS_AS(repeat(data_3, -3, 0), std::invalid_argument);
}

TEST_CASE("tile") {
  auto x = array({1, 2, 3}, {3});
  auto y = tile(x, {2});
  auto expected = array({1, 2, 3, 1, 2, 3}, {6});
  CHECK(array_equal(y, expected).item<bool>());
  x = array({1, 2, 3, 4}, {2, 2});
  y = tile(x, {2});
  expected = array({1, 2, 1, 2, 3, 4, 3, 4}, {2, 4});
  CHECK(array_equal(y, expected).item<bool>());
  x = array({1, 2, 3, 4}, {2, 2});
  y = tile(x, {4, 1});
  expected = array({1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, {8, 2});
  CHECK(array_equal(y, expected).item<bool>());

  x = array({1, 2, 3, 4}, {2, 2});
  y = tile(x, {2, 2});
  expected = array({1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4}, {4, 4});
  CHECK(array_equal(y, expected).item<bool>());
  x = array({1, 2, 3}, {3});
  y = tile(x, {2, 2, 2});
  expected = array(
      {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3},
      {2, 2, 6});
  CHECK(array_equal(y, expected).item<bool>());
}

TEST_CASE("tensordot") {
  auto x = reshape(arange(60.), {3, 4, 5});
  auto y = reshape(arange(24.), {4, 3, 2});
  auto z = tensordot(x, y, {1, 0}, {0, 1});
  auto expected = array(
      {4400, 4730, 4532, 4874, 4664, 5018, 4796, 5162, 4928, 5306}, {5, 2});
  CHECK(array_equal(z, expected).item<bool>());
  x = reshape(arange(360.), {3, 4, 5, 6});
  y = reshape(arange(360.), {6, 4, 5, 3});
  CHECK_THROWS_AS(tensordot(x, y, {2, 1, 3}, {1, 2, 0}), std::invalid_argument);
  x = reshape(arange(60.), {3, 4, 5});
  y = reshape(arange(120.), {4, 5, 6});
  z = tensordot(x, y, 2);
  expected = array(
      {14820.,
       15010.,
       15200.,
       15390.,
       15580.,
       15770.,
       37620.,
       38210.,
       38800.,
       39390.,
       39980.,
       40570.,
       60420.,
       61410.,
       62400.,
       63390.,
       64380.,
       65370.},
      {3, 6});
  CHECK(array_equal(z, expected).item<bool>());
}

TEST_CASE("outer") {
  auto x = arange(1.0, 5.0);
  auto y = arange(1.0, 4.0);
  auto z = outer(x, y);
  auto expected = array(
      {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0}, {4, 3});
  CHECK(array_equal(z, expected).item<bool>());

  x = ones({5});
  y = linspace(-2., 2., 5);
  z = outer(x, y);
  expected = array(
      {-2., -1., 0.,  1.,  2., -2., -1., 0.,  1.,  2., -2., -1., 0.,
       1.,  2.,  -2., -1., 0., 1.,  2.,  -2., -1., 0., 1.,  2.},
      {5, 5});
  CHECK(array_equal(z, expected).item<bool>());
}

TEST_CASE("inner") {
  CHECK_THROWS_AS(
      inner(reshape(arange(5.), {1, 5}), reshape(arange(6.), {2, 3})),
      std::invalid_argument);
  auto x = array({1., 2., 3.});
  auto y = array({0., 1., 0.});
  auto z = inner(x, y);
  CHECK_EQ(z.item<float>(), 2.f);

  x = reshape(arange(24.), {2, 3, 4});
  y = arange(4.);
  z = inner(x, y);
  auto expected = array({14., 38., 62., 86., 110., 134.}, {2, 3});
  CHECK(array_equal(z, expected).item<bool>());

  x = reshape(arange(2.), {1, 1, 2});
  y = reshape(arange(6.), {3, 2});
  z = inner(x, y);
  expected = array({1., 3., 5.}, {1, 1, 3});
  CHECK(array_equal(z, expected).item<bool>());

  z = inner(eye(2), array(7.));
  expected = array({7., 0., 0., 7.}, {2, 2});
  CHECK(array_equal(z, expected).item<bool>());
}

TEST_CASE("test divmod") {
  auto x = array({1, 2, 3});
  auto y = array({1, 1, 1});
  auto out = divmod(x, y);
  CHECK(array_equal(out[0], array({1, 2, 3})).item<bool>());
  CHECK(array_equal(out[1], array({0, 0, 0})).item<bool>());

  x = array({5, 6, 7});
  y = array({2, 2, 2});
  out = divmod(x, y);
  CHECK(array_equal(out[0], array({2, 3, 3})).item<bool>());
  CHECK(array_equal(out[1], array({1, 0, 1})).item<bool>());

  // Siblings should be gone after evaling the graph
  CHECK(out[0].siblings().empty());
  CHECK(out[1].siblings().empty());

  x = array({5.0, 6.0, 7.0});
  y = array({2.0, 2.0, 2.0});
  out = divmod(x, y);
  CHECK(array_equal(out[0], array({2.0, 3.0, 3.0})).item<bool>());
  CHECK(array_equal(out[1], array({1.0, 0.0, 1.0})).item<bool>());

  x = array({1.0}, complex64);
  y = array({2.0}, complex64);
  CHECK_THROWS(divmod(x, y));

  // Check that we can eval on both outputs
  x = array({1.0});
  y = array({2.0});
  out = divmod(x, y);
  eval(out);
  CHECK_EQ(out[0].item<float>(), 0.0);
  CHECK_EQ(out[1].item<float>(), 1.0);

  // Check nested in the graph
  x = array({1.0});
  y = array({2.0});
  out = divmod(x, y);
  auto z = out[0] + out[1];
  CHECK_EQ(z.item<float>(), 1.0);

  // Check that we can still eval when one output goes out of scope
  std::vector<array> out_holder;
  {
    out_holder.push_back(divmod(x, y)[0]);
  }
  eval(out_holder);
  CHECK_EQ(out_holder[0].item<float>(), 0.0);

  // Check that we can still eval when the other output goes out of scope
  out_holder.clear();
  {
    out_holder.push_back(divmod(x, y)[1]);
  }
  eval(out_holder);
  CHECK_EQ(out_holder[0].item<float>(), 1.0);
}

TEST_CASE("test integer division of the minimum by -1") {
  // On x86 this division traps. The quotient is the minimum, remainder 0.
  // Eleven pairs, so that the SIMD lanes and the scalar tail both run.
  auto check = [](auto lo) {
    using T = decltype(lo);
    auto s = Device::cpu;
    auto a =
        array({lo, T(-7), T(7), T(7), T(-7), lo, T(5), T(-5), lo, T(-7), T(0)});
    auto b = array(
        {T(-1), T(2), T(0), T(-2), T(-2), T(1), T(3), T(3), T(-1), T(0), T(0)});
    auto q =
        array({lo, T(-4), T(0), T(-4), T(3), lo, T(1), T(-2), lo, T(-1), T(0)});
    auto r = array(
        {T(0), T(1), T(7), T(-1), T(-1), T(0), T(2), T(1), T(0), T(-7), T(0)});
    auto dtype = a.dtype();
    CAPTURE(dtype);
    auto fq = floor_divide(a, b, s);
    auto fr = remainder(a, b, s);
    CHECK(array_equal(fq, q, s).item<bool>());
    CHECK(array_equal(fr, r, s).item<bool>());

    // divmod gives floor_divide and remainder, also for a zero divisor.
    auto qr = divmod(a, b, s);
    CHECK(array_equal(qr[0], fq, s).item<bool>());
    CHECK(array_equal(qr[1], fr, s).item<bool>());
    CHECK(array_equal(qr[0], q, s).item<bool>());
    CHECK(array_equal(qr[1], r, s).item<bool>());
  };
  check(std::numeric_limits<int32_t>::min());
  check(std::numeric_limits<int64_t>::min());
}

namespace {

bool division_equal(const array& p, const array& q, StreamOrDevice s = {}) {
  return array_equal(p, q, s).item<bool>();
}

// One row is a, b, floor_divide(a, b) and remainder(a, b). A zero divisor
// keeps the guards of the CPU ops: floor_divide(a, 0) is -1 for a negative a
// and 0 otherwise, remainder(a, 0) is a. MIN // -1 wraps around to MIN, and
// MIN % -1 is 0. The expected values are exact: the floor quotient and the
// floor modulus, reduced modulo 2^N.
template <typename T>
struct DivRow {
  T a, b, q, r;
};

enum class DivOp { FloorDivide, Remainder, DivMod };

std::vector<array> apply_div(DivOp op, const array& a, const array& b) {
  auto s = Device::cpu;
  switch (op) {
    case DivOp::FloorDivide:
      return {floor_divide(a, b, s)};
    case DivOp::Remainder:
      return {remainder(a, b, s)};
    default:
      return divmod(a, b, s);
  }
}

// The outputs of the CPU op on the pairs (a, b), against the rows rs, for each
// of the broadcast paths: two arrays, a 0-d divisor, a 0-d dividend, stride-2
// slices and two 0-d arrays. The arrays are longer than the widest SIMD loop
// (max_size in simd/accelerate_simd.h), so that the vector loop and the scalar
// tail both run where there is a vector loop. A trap or a sanitizer report in
// the old code ends or marks the case before a value is compared.
template <typename T>
void check_division(DivOp op, const std::vector<DivRow<T>>& rows) {
  Dtype dtype = TypeToDtype<T>();
  auto make = [&](const std::vector<T>& v, const Shape& shape) {
    return array(v.begin(), shape, dtype);
  };
  auto check = [&](const array& a,
                   const array& b,
                   const std::vector<DivRow<T>>& rs,
                   const Shape& shape) {
    std::vector<T> q, r;
    for (const auto& x : rs) {
      q.push_back(x.q);
      r.push_back(x.r);
    }
    auto out = apply_div(op, a, b);
    if (op == DivOp::FloorDivide) {
      REQUIRE_EQ(out.size(), 1);
      CHECK(division_equal(out[0], make(q, shape), Device::cpu));
    } else if (op == DivOp::Remainder) {
      REQUIRE_EQ(out.size(), 1);
      CHECK(division_equal(out[0], make(r, shape), Device::cpu));
    } else {
      REQUIRE_EQ(out.size(), 2);
      CHECK(division_equal(out[0], make(q, shape), Device::cpu));
      CHECK(division_equal(out[1], make(r, shape), Device::cpu));
    }
  };
  auto column = [&](const std::vector<DivRow<T>>& rs, bool first) {
    std::vector<T> v;
    for (const auto& x : rs) {
      v.push_back(first ? x.a : x.b);
    }
    return v;
  };
  Shape shape{static_cast<int>(rows.size())};
  auto a = make(column(rows, true), shape);
  auto b = make(column(rows, false), shape);

  // Two arrays.
  check(a, b, rows, shape);

  // A 0-d divisor, for the divisors -1 and 0 (the unsigned types: 0 and 1).
  for (T d :
       {static_cast<T>(std::is_signed_v<T> ? -1 : 1), static_cast<T>(0)}) {
    std::vector<DivRow<T>> rs;
    for (const auto& x : rows) {
      if (x.b == d) {
        rs.push_back(x);
      }
    }
    if (!rs.empty()) {
      Shape s{static_cast<int>(rs.size())};
      check(make(column(rs, true), s), array(d, dtype), rs, s);
    }
  }

  // A 0-d dividend: the first dividend (MIN for the signed types) against
  // every divisor of its rows.
  {
    std::vector<DivRow<T>> rs;
    for (const auto& x : rows) {
      if (x.a == rows[0].a) {
        rs.push_back(x);
      }
    }
    Shape s{static_cast<int>(rs.size())};
    check(array(rows[0].a, dtype), make(column(rs, false), s), rs, s);
  }

  // Stride-2 slices of both operands, with a filler pair (1, 1) in between.
  {
    std::vector<T> fa, fb;
    for (const auto& x : rows) {
      fa.push_back(x.a);
      fa.push_back(1);
      fb.push_back(x.b);
      fb.push_back(1);
    }
    Shape big{static_cast<int>(fa.size())};
    Shape stop = big;
    check(
        slice(make(fa, big), {0}, stop, {2}, Device::cpu),
        slice(make(fb, big), {0}, stop, {2}, Device::cpu),
        rows,
        shape);
  }

  // Two 0-d arrays: the first rows.
  for (size_t i = 0; i < rows.size() && i < 4; ++i) {
    check(array(rows[i].a, dtype), array(rows[i].b, dtype), {rows[i]}, Shape{});
  }
}

std::vector<DivRow<int8_t>> rows_int8() {
  const int8_t cmax = std::numeric_limits<int8_t>::max();
  const int8_t cmin = std::numeric_limits<int8_t>::min();
  return {
      {cmin, -1, cmin, 0},
      {cmin, 0, -1, cmin},
      {cmin, 2, -64, 0},
      {cmin, -3, 42, -2},
      {cmin + 1, -1, cmax, 0},
      {cmin + 1, 0, -1, cmin + 1},
      {cmin + 1, 2, -64, 1},
      {cmin + 1, -3, 42, -1},
      {-7, -1, 7, 0},
      {-7, 0, -1, -7},
      {-7, 2, -4, 1},
      {-7, -3, 2, -1},
      {-1, -1, 1, 0},
      {-1, 0, -1, -1},
      {-1, 2, -1, 1},
      {-1, -3, 0, -1},
      {0, -1, 0, 0},
      {0, 0, 0, 0},
      {0, 2, 0, 0},
      {0, -3, 0, 0},
      {1, -1, -1, 0},
      {1, 0, 0, 1},
      {1, 2, 0, 1},
      {1, -3, -1, -2},
      {7, -1, -7, 0},
      {7, 0, 0, 7},
      {7, 2, 3, 1},
      {7, -3, -3, -2},
      {cmax, -1, cmin + 1, 0},
      {cmax, 0, 0, cmax},
      {cmax, 2, 63, 1},
      {cmax, -3, -43, -2},
  };
}

std::vector<DivRow<int16_t>> rows_int16() {
  const int16_t smax = std::numeric_limits<int16_t>::max();
  const int16_t smin = std::numeric_limits<int16_t>::min();
  return {
      {smin, -1, smin, 0},
      {smin, 0, -1, smin},
      {smin, 2, -16384, 0},
      {smin, -3, 10922, -2},
      {smin + 1, -1, smax, 0},
      {smin + 1, 0, -1, smin + 1},
      {smin + 1, 2, -16384, 1},
      {smin + 1, -3, 10922, -1},
      {-7, -1, 7, 0},
      {-7, 0, -1, -7},
      {-7, 2, -4, 1},
      {-7, -3, 2, -1},
      {-1, -1, 1, 0},
      {-1, 0, -1, -1},
      {-1, 2, -1, 1},
      {-1, -3, 0, -1},
      {0, -1, 0, 0},
      {0, 0, 0, 0},
      {0, 2, 0, 0},
      {0, -3, 0, 0},
      {1, -1, -1, 0},
      {1, 0, 0, 1},
      {1, 2, 0, 1},
      {1, -3, -1, -2},
      {7, -1, -7, 0},
      {7, 0, 0, 7},
      {7, 2, 3, 1},
      {7, -3, -3, -2},
      {smax, -1, smin + 1, 0},
      {smax, 0, 0, smax},
      {smax, 2, 16383, 1},
      {smax, -3, -10923, -2},
  };
}

std::vector<DivRow<int32_t>> rows_int32() {
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  return {
      {imin, -1, imin, 0},
      {imin, 0, -1, imin},
      {imin, 2, -1073741824, 0},
      {imin, -3, 715827882, -2},
      {imin + 1, -1, imax, 0},
      {imin + 1, 0, -1, imin + 1},
      {imin + 1, 2, -1073741824, 1},
      {imin + 1, -3, 715827882, -1},
      {-7, -1, 7, 0},
      {-7, 0, -1, -7},
      {-7, 2, -4, 1},
      {-7, -3, 2, -1},
      {-1, -1, 1, 0},
      {-1, 0, -1, -1},
      {-1, 2, -1, 1},
      {-1, -3, 0, -1},
      {0, -1, 0, 0},
      {0, 0, 0, 0},
      {0, 2, 0, 0},
      {0, -3, 0, 0},
      {1, -1, -1, 0},
      {1, 0, 0, 1},
      {1, 2, 0, 1},
      {1, -3, -1, -2},
      {7, -1, -7, 0},
      {7, 0, 0, 7},
      {7, 2, 3, 1},
      {7, -3, -3, -2},
      {imax, -1, imin + 1, 0},
      {imax, 0, 0, imax},
      {imax, 2, 1073741823, 1},
      {imax, -3, -715827883, -2},
  };
}

std::vector<DivRow<int64_t>> rows_int64() {
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();
  return {
      {lmin, -1, lmin, 0},
      {lmin, 0, -1, lmin},
      {lmin, 2, -4611686018427387904, 0},
      {lmin, -3, 3074457345618258602, -2},
      {lmin + 1, -1, lmax, 0},
      {lmin + 1, 0, -1, lmin + 1},
      {lmin + 1, 2, -4611686018427387904, 1},
      {lmin + 1, -3, 3074457345618258602, -1},
      {-7, -1, 7, 0},
      {-7, 0, -1, -7},
      {-7, 2, -4, 1},
      {-7, -3, 2, -1},
      {-1, -1, 1, 0},
      {-1, 0, -1, -1},
      {-1, 2, -1, 1},
      {-1, -3, 0, -1},
      {0, -1, 0, 0},
      {0, 0, 0, 0},
      {0, 2, 0, 0},
      {0, -3, 0, 0},
      {1, -1, -1, 0},
      {1, 0, 0, 1},
      {1, 2, 0, 1},
      {1, -3, -1, -2},
      {7, -1, -7, 0},
      {7, 0, 0, 7},
      {7, 2, 3, 1},
      {7, -3, -3, -2},
      {lmax, -1, lmin + 1, 0},
      {lmax, 0, 0, lmax},
      {lmax, 2, 4611686018427387903, 1},
      {lmax, -3, -3074457345618258603, -2},
  };
}

std::vector<DivRow<uint8_t>> rows_uint8() {
  const uint8_t u8max = std::numeric_limits<uint8_t>::max();
  return {
      {0, 0, 0, 0},         {0, 1, 0, 0},       {0, 2, 0, 0},
      {0, 7, 0, 0},         {1, 0, 0, 1},       {1, 1, 1, 0},
      {1, 2, 0, 1},         {1, 7, 0, 1},       {7, 0, 0, 7},
      {7, 1, 7, 0},         {7, 2, 3, 1},       {7, 7, 1, 0},
      {127, 0, 0, 127},     {127, 1, 127, 0},   {127, 2, 63, 1},
      {127, 7, 18, 1},      {128, 0, 0, 128},   {128, 1, 128, 0},
      {128, 2, 64, 0},      {128, 7, 18, 2},    {u8max, 0, 0, u8max},
      {u8max, 1, u8max, 0}, {u8max, 2, 127, 1}, {u8max, 7, 36, 3},
  };
}

std::vector<DivRow<uint16_t>> rows_uint16() {
  const uint16_t u16max = std::numeric_limits<uint16_t>::max();
  return {
      {0, 0, 0, 0},           {0, 1, 0, 0},          {0, 2, 0, 0},
      {0, 7, 0, 0},           {1, 0, 0, 1},          {1, 1, 1, 0},
      {1, 2, 0, 1},           {1, 7, 0, 1},          {7, 0, 0, 7},
      {7, 1, 7, 0},           {7, 2, 3, 1},          {7, 7, 1, 0},
      {32767, 0, 0, 32767},   {32767, 1, 32767, 0},  {32767, 2, 16383, 1},
      {32767, 7, 4681, 0},    {32768, 0, 0, 32768},  {32768, 1, 32768, 0},
      {32768, 2, 16384, 0},   {32768, 7, 4681, 1},   {u16max, 0, 0, u16max},
      {u16max, 1, u16max, 0}, {u16max, 2, 32767, 1}, {u16max, 7, 9362, 1},
  };
}

std::vector<DivRow<uint32_t>> rows_uint32() {
  const uint32_t u32max = std::numeric_limits<uint32_t>::max();
  return {
      {0, 0, 0, 0},
      {0, 1, 0, 0},
      {0, 2, 0, 0},
      {0, 7, 0, 0},
      {1, 0, 0, 1},
      {1, 1, 1, 0},
      {1, 2, 0, 1},
      {1, 7, 0, 1},
      {7, 0, 0, 7},
      {7, 1, 7, 0},
      {7, 2, 3, 1},
      {7, 7, 1, 0},
      {2147483647, 0, 0, 2147483647},
      {2147483647, 1, 2147483647, 0},
      {2147483647, 2, 1073741823, 1},
      {2147483647, 7, 306783378, 1},
      {2147483648u, 0, 0, 2147483648u},
      {2147483648u, 1, 2147483648u, 0},
      {2147483648u, 2, 1073741824, 0},
      {2147483648u, 7, 306783378, 2},
      {u32max, 0, 0, u32max},
      {u32max, 1, u32max, 0},
      {u32max, 2, 2147483647, 1},
      {u32max, 7, 613566756, 3},
  };
}

std::vector<DivRow<uint64_t>> rows_uint64() {
  const uint64_t u64max = std::numeric_limits<uint64_t>::max();
  return {
      {0, 0, 0, 0},
      {0, 1, 0, 0},
      {0, 2, 0, 0},
      {0, 7, 0, 0},
      {1, 0, 0, 1},
      {1, 1, 1, 0},
      {1, 2, 0, 1},
      {1, 7, 0, 1},
      {7, 0, 0, 7},
      {7, 1, 7, 0},
      {7, 2, 3, 1},
      {7, 7, 1, 0},
      {9223372036854775807, 0, 0, 9223372036854775807},
      {9223372036854775807, 1, 9223372036854775807, 0},
      {9223372036854775807, 2, 4611686018427387903, 1},
      {9223372036854775807, 7, 1317624576693539401, 0},
      {9223372036854775808ULL, 0, 0, 9223372036854775808ULL},
      {9223372036854775808ULL, 1, 9223372036854775808ULL, 0},
      {9223372036854775808ULL, 2, 4611686018427387904, 0},
      {9223372036854775808ULL, 7, 1317624576693539401, 1},
      {u64max, 0, 0, u64max},
      {u64max, 1, u64max, 0},
      {u64max, 2, 9223372036854775807, 1},
      {u64max, 7, 2635249153387078802, 1},
  };
}

std::vector<DivRow<bool>> rows_bool() {
  return {
      {false, false, false, false},
      {false, true, false, false},
      {true, false, false, true},
      {true, true, true, false},
  };
}

void check_division_tables(DivOp op) {
  check_division(op, rows_int8());
  check_division(op, rows_int16());
  check_division(op, rows_int32());
  check_division(op, rows_int64());
  check_division(op, rows_uint8());
  check_division(op, rows_uint16());
  check_division(op, rows_uint32());
  check_division(op, rows_uint64());
  check_division(op, rows_bool());
}

// floor_divide(x, -1) > 0, < 0 and == x, one chain per output and distinct
// inputs, so that each is fused into one kernel with the constant -1.
std::vector<array> compare_floor_divide_by_minus_one(
    const std::vector<array>& in) {
  auto zero = array(0, in[0].dtype());
  auto minus_one = array(-1, in[0].dtype());
  return {
      greater(floor_divide(in[0], minus_one), zero),
      less(floor_divide(in[1], minus_one), zero),
      equal(floor_divide(in[2], minus_one), in[2])};
}

} // namespace

TEST_CASE("test floor_divide at the minimum and at zero") {
  // floor_divide(MIN, -1) is MIN: the quotient wraps around, as in two's
  // complement. It is undefined in C++ and traps on x86. A zero divisor gives
  // 0, or -1 for a negative dividend, as before.
  check_division_tables(DivOp::FloorDivide);
}

TEST_CASE("test remainder at the minimum and at zero") {
  // remainder(MIN, -1) is 0, which traps on x86 when computed with %. A zero
  // divisor gives the dividend, as before.
  check_division_tables(DivOp::Remainder);
}

TEST_CASE("test divmod at the minimum and at zero") {
  // divmod gives floor_divide and remainder, also at (MIN, -1), where it
  // trapped on x86, and at a zero divisor, where it trapped on x86 and where
  // the compiler took the divisor of a bool to be true (divmod(true, false)
  // gave (true, false) in a Release build).
  check_division_tables(DivOp::DivMod);
}

TEST_CASE("test compiled floor_divide by a constant -1") {
  // The compiled kernel is built with the constant -1 as the divisor. The
  // compiler may not take MIN / -1 to be impossible: -x > 0 is not x < 0.
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();
  std::vector<int32_t> v32 = {imin, imin + 1, -1, 0, 1, imax};
  std::vector<int64_t> v64 = {lmin, lmin + 1, -1, 0, 1, lmax};

  // Expected for x = MIN, MIN + 1, -1, 0, 1, MAX: floor_divide(x, -1) is MIN,
  // MAX, 1, 0, -1, MIN + 1.
  std::vector<std::vector<bool>> expected = {
      {false, true, true, false, false, false}, // x // -1 > 0
      {true, false, false, false, true, true}, // x // -1 < 0
      {true, false, false, true, false, false}, // x // -1 == x
  };
  auto compiled = compile(compare_floor_divide_by_minus_one);
  auto check = [&](const std::vector<array>& inputs) {
    auto out = compiled(inputs);
    REQUIRE_EQ(out.size(), 3);
    for (int i = 0; i < 3; ++i) {
      array want(expected[i].begin(), {6}, bool_);
      CHECK(division_equal(out[i], want));
    }
  };
  auto fresh = [](const auto& v, Dtype dtype) {
    std::vector<array> inputs;
    for (int i = 0; i < 3; ++i) {
      inputs.push_back(array(v.begin(), {6}, dtype));
    }
    return inputs;
  };
  check(fresh(v32, int32));
  check(fresh(v64, int64));
}

TEST_CASE("test diagonal") {
  auto x = array({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2});
  auto out = diagonal(x);
  CHECK(array_equal(out, array({0, 3}, {2})).item<bool>());

  CHECK_THROWS_AS(diagonal(x, 1, 6, 0), std::out_of_range);
  CHECK_THROWS_AS(diagonal(x, 1, 0, -3), std::out_of_range);

  x = array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {3, 4});
  out = diagonal(x, 2, 1, 0);
  CHECK(array_equal(out, array({8}, {1})).item<bool>());

  out = diagonal(x, -1, 0, 1);
  CHECK(array_equal(out, array({4, 9}, {2})).item<bool>());

  out = diagonal(x, -5, 0, 1);
  eval(out);
  CHECK_EQ(out.shape(), Shape{0});

  x = array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {3, 2, 2});
  out = diagonal(x, 1, 0, 1);
  CHECK(array_equal(out, array({2, 3}, {2, 1})).item<bool>());

  out = diagonal(x, 0, 2, 0);
  CHECK(array_equal(out, array({0, 5, 2, 7}, {2, 2})).item<bool>());

  out = diagonal(x, 1, -1, 0);
  CHECK(array_equal(out, array({4, 9, 6, 11}, {2, 2})).item<bool>());

  x = reshape(arange(16), {2, 2, 2, 2});
  out = diagonal(x, 0, 0, 1);
  CHECK(array_equal(out, array({0, 12, 1, 13, 2, 14, 3, 15}, {2, 2, 2}))
            .item<bool>());

  CHECK_THROWS_AS(diagonal(x, 0, 1, 1), std::invalid_argument);

  x = array({0, 1}, {2});
  CHECK_THROWS_AS(diagonal(x, 0, 0, 1), std::invalid_argument);
}

TEST_CASE("test diag") {
  // To few or too many dimensions
  CHECK_THROWS(diag(array(0.0)));
  CHECK_THROWS(diag(array({0.0}, {1, 1, 1})));

  // Test with 1D array
  auto x = array({0, 1, 2, 3}, {4});
  auto out = diag(x, 0);
  CHECK(
      array_equal(
          out, array({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3}, {4, 4}))
          .item<bool>());

  out = diag(x, 1);
  CHECK(array_equal(
            out,
            array(
                {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0},
                {5, 5}))
            .item<bool>());

  out = diag(x, -1);
  CHECK(array_equal(
            out,
            array(
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0},
                {5, 5}))
            .item<bool>());

  // Test with 2D array
  x = array({0, 1, 2, 3, 4, 5, 6, 7, 8}, {3, 3});
  out = diag(x, 0);
  CHECK(array_equal(out, array({0, 4, 8}, {3})).item<bool>());

  out = diag(x, 1);
  CHECK(array_equal(out, array({1, 5}, {2})).item<bool>());

  out = diag(x, -1);
  CHECK(array_equal(out, array({3, 7}, {2})).item<bool>());
}

TEST_CASE("test issubdtype") {
  const auto cats = {
      complexfloating,
      floating,
      inexact,
      signedinteger,
      unsignedinteger,
      integer,
      number,
      generic};
  const auto types = {
      bool_,
      uint8,
      uint16,
      uint32,
      uint64,
      int8,
      int16,
      int32,
      int64,
      float16,
      float32,
      bfloat16,
      complex64};
  for (const auto& type : types) {
    CHECK(issubdtype(type, type));
    CHECK(issubdtype(type, generic));
    switch (kindof(type)) {
      case Dtype::Kind::b:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK_FALSE(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK_FALSE(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::u:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK_FALSE(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK(issubdtype(type, unsignedinteger));
        CHECK(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::i:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK_FALSE(issubdtype(type, inexact));
        CHECK(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::f:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK(issubdtype(type, floating));
        CHECK(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::c:
        CHECK(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::V:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK(issubdtype(type, floating));
        CHECK(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
    }
  }

  for (const auto& type : types) {
    CHECK(issubdtype(type, type));
    CHECK(issubdtype(type, generic));
    for (auto type1 : types) {
      CHECK_EQ(issubdtype(type, type1), type == type1);
    }
  }

  for (const auto& cat : cats) {
    CHECK(issubdtype(cat, cat));
    switch (cat) {
      case Dtype::Category::complexfloating:
        CHECK(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::floating:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK(issubdtype(cat, floating));
        CHECK(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::inexact:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::signedinteger:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::unsignedinteger:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK(issubdtype(cat, unsignedinteger));
        CHECK(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::integer:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::number:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::generic:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK_FALSE(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
    }
  }
}

TEST_CASE("test atleast_1d") {
  auto x = array(1);
  auto out = atleast_1d(x);
  CHECK_EQ(out.ndim(), 1);
  CHECK_EQ(out.shape(), Shape{1});

  x = array({1, 2, 3}, {3});
  out = atleast_1d(x);
  CHECK_EQ(out.ndim(), 1);
  CHECK_EQ(out.shape(), Shape{3});

  x = array({1, 2, 3}, {3, 1});
  out = atleast_1d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{3, 1});
}

TEST_CASE("test atleast_1d vector") {
  auto x = std::vector<array>{
      array(1), array({1, 2, 3}, {3}), array({1, 2, 3}, {3, 1})};
  auto out = atleast_1d(x);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].ndim(), 1);
  CHECK_EQ(out[0].shape(), Shape{1});
  CHECK_EQ(out[1].ndim(), 1);
  CHECK_EQ(out[1].shape(), Shape{3});
  CHECK_EQ(out[2].ndim(), 2);
  CHECK_EQ(out[2].shape(), Shape{3, 1});
}

TEST_CASE("test atleast_2d") {
  auto x = array(1);
  auto out = atleast_2d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{1, 1});

  x = array({1, 2, 3}, {3});
  out = atleast_2d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{1, 3});

  x = array({1, 2, 3}, {3, 1});
  out = atleast_2d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{3, 1});
}

TEST_CASE("test atleast_2d vector") {
  auto x = std::vector<array>{
      array(1), array({1, 2, 3}, {3}), array({1, 2, 3}, {3, 1})};
  auto out = atleast_2d(x);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].ndim(), 2);
  CHECK_EQ(out[0].shape(), Shape{1, 1});
  CHECK_EQ(out[1].ndim(), 2);
  CHECK_EQ(out[1].shape(), Shape{1, 3});
  CHECK_EQ(out[2].ndim(), 2);
  CHECK_EQ(out[2].shape(), Shape{3, 1});
}

TEST_CASE("test atleast_3d") {
  auto x = array(1);
  auto out = atleast_3d(x);
  CHECK_EQ(out.ndim(), 3);
  CHECK_EQ(out.shape(), Shape{1, 1, 1});

  x = array({1, 2, 3}, {3});
  out = atleast_3d(x);
  CHECK_EQ(out.ndim(), 3);
  CHECK_EQ(out.shape(), Shape{1, 3, 1});

  x = array({1, 2, 3}, {3, 1});
  out = atleast_3d(x);
  CHECK_EQ(out.ndim(), 3);
  CHECK_EQ(out.shape(), Shape{3, 1, 1});
}

TEST_CASE("test atleast_3d vector") {
  auto x = std::vector<array>{
      array(1), array({1, 2, 3}, {3}), array({1, 2, 3}, {3, 1})};
  auto out = atleast_3d(x);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].ndim(), 3);
  CHECK_EQ(out[0].shape(), Shape{1, 1, 1});
  CHECK_EQ(out[1].ndim(), 3);
  CHECK_EQ(out[1].shape(), Shape{1, 3, 1});
  CHECK_EQ(out[2].ndim(), 3);
  CHECK_EQ(out[2].shape(), Shape{3, 1, 1});
}

TEST_CASE("test topk") {
  auto x = reshape(arange(10), {2, 5});

  {
    auto y = topk(x, 1, 1);
    CHECK(array_equal(y, array({4, 9}, {2, 1})).item<bool>());
  }

  {
    auto y = topk(x, 2, 0);
    CHECK(array_equal(y, x).item<bool>());
  }

  {
    auto y = topk(x, 1, 0);
    CHECK(array_equal(y, array({5, 6, 7, 8, 9}, {1, 5})).item<bool>());
  }
}

TEST_CASE("test meshgrid") {
  // Test default
  auto x = array({1, 2, 3}, {3});
  auto in = std::vector<array>{x};
  auto out = meshgrid(in);
  CHECK(array_equal(out[0], x).item<bool>());

  // Test different lengths
  auto y = array({4, 5}, {2});
  in = std::vector<array>{x, y};
  out = meshgrid(in);
  auto expected_zero = array({1, 2, 3, 1, 2, 3}, {2, 3});
  auto expected_one = array({4, 4, 4, 5, 5, 5}, {2, 3});
  CHECK(array_equal(out[0], expected_zero).item<bool>());
  CHECK(array_equal(out[1], expected_one).item<bool>());

  // Test sparse true
  in = std::vector<array>{x, x};
  out = meshgrid(in, true);
  expected_zero = array({1, 2, 3}, {1, 3});
  expected_one = array({1, 2, 3}, {3, 1});
  CHECK(array_equal(out[0], expected_zero).item<bool>());
  CHECK(array_equal(out[1], expected_one).item<bool>());
}

TEST_CASE("test conv1d") {
  auto in = astype(
      array(
          {0.5488135,
           0.71518937,
           0.60276338,
           0.54488318,
           0.4236548,
           0.64589411},
          {1, 3, 2}),
      float16);

  int stride = 1;
  int padding = 1;

  {
    int groups = 1;
    auto wt = astype(
        array(
            {

                0.43758721, 0.891773,   0.96366276, 0.38344152,
                0.79172504, 0.52889492,

                0.56804456, 0.92559664, 0.07103606, 0.0871293,
                0.0202184,  0.83261985,

                0.77815675, 0.87001215, 0.97861834, 0.79915856,
                0.46147936, 0.78052918,

                0.11827443, 0.63992102, 0.14335329, 0.94466892,
                0.52184832, 0.41466194

            },
            {4, 3, 2}),
        float16);

    auto expected = array(
        {1.56836,
         0.567383,
         1.8125,
         1.29492,
         2.34375,
         1.61035,
         2.77539,
         1.61328,
         1.40527,
         0.933105,
         1.87402,
         1.09082},
        {1, 3, 4});

    auto out = conv1d(in, wt, stride, padding, /* dilation= */ 1, groups);
    CHECK(allclose(out, expected).item<bool>());
  }

  {
    int groups = 2;
    auto wt = array(
        {0.43758721,
         0.891773,
         0.96366276,

         0.38344152,
         0.79172504,
         0.52889492,

         0.56804456,
         0.92559664,
         0.07103606,

         0.0871293,
         0.0202184,
         0.83261985

        },
        {4, 3, 1});

    auto expected = array(
        {1.07007,
         0.753201,
         0.700818,
         0.468176,
         1.18568,
         0.91152,
         0.956607,
         0.611213,
         0.641404,
         0.566401,
         0.907472,
         0.0605397},
        {1, 3, 4});

    auto out = conv1d(in, wt, stride, padding, /* dilation= */ 1, groups);
    CHECK(allclose(out, expected).item<bool>());
  }
}

TEST_CASE("test conv2d") {
  auto in = array(
      {0.57429284,
       -0.21628855,
       -0.18673691,
       -0.3793517,

       0.3059678,
       -0.8137168,
       0.6168841,
       -0.26912728},
      {1, 2, 2, 2});

  std::pair<int, int> stride{1, 1};
  std::pair<int, int> padding{0, 0};

  {
    int groups = 1;

    auto wt = array(
        {0.3190391,   -0.24937038, 1.4621079,   -2.0601406,  -0.3224172,
         -0.38405436, 1.1337694,   -1.0998913,  -0.1724282,  -0.8778584,
         0.04221375,  0.58281523,  -1.1006192,  1.1447237,   0.9015907,
         0.50249434,  0.90085596,  -0.68372786, -0.12289023, -0.93576944,
         -0.26788807, 0.53035545,  -0.69166076, -0.39675352, -0.6871727,
         -0.84520566, -0.6712461,  -0.0126646,  -1.1173104,  0.2344157,
         1.6598022,   0.74204415},
        {4, 2, 2, 2});

    auto expected =
        array({1.9549234, -0.98542136, 0.2097499, 0.20991313}, {1, 1, 1, 4});
    auto out = conv2d(in, wt, stride, padding, /* dilation= */ {1, 1}, groups);
    CHECK(allclose(out, expected).item<bool>());
  }

  {
    int groups = 2;
    auto wt = array(
        {0.3190391,
         -0.24937038,

         1.46210794,
         -2.06014071,

         -0.3224172,
         -0.38405435,

         1.13376944,
         -1.09989127,

         -0.17242821,
         -0.87785842,

         0.04221375,
         0.58281521,

         -1.10061918,
         1.14472371,

         0.90159072,
         0.50249434},
        {4, 2, 2, 1});

    auto expected = array(
        {-0.59372161, -0.44505326, 0.17910982, -1.06507601}, {1, 1, 1, 4});

    auto out = conv2d(in, wt, stride, padding, /* dilation= */ {1, 1}, groups);
    CHECK(allclose(out, expected).item<bool>());
  }

  {
    in = array(
        {0.57429284,
         -0.21628855,
         -0.18673691,
         -0.3793517,

         0.3059678,
         -0.8137168,
         0.6168841,
         -0.26912728,

         0.57429284,
         -0.21628855,
         -0.18673691,
         -0.3793517,

         0.3059678,
         -0.8137168,
         0.6168841,
         -0.26912728},
        {2, 2, 2, 2});

    int groups = 2;
    auto wt = array(
        {0.3190391,
         -0.24937038,

         1.46210794,
         -2.06014071,

         -0.3224172,
         -0.38405435,

         1.13376944,
         -1.09989127,

         -0.17242821,
         -0.87785842,

         0.04221375,
         0.58281521,

         -1.10061918,
         1.14472371,

         0.90159072,
         0.50249434},
        {4, 2, 2, 1});

    auto expected = array(
        {-0.59372161, -0.44505326, 0.17910982, -1.06507601}, {1, 1, 1, 4});

    auto out = conv2d(in, wt, stride, padding, /* dilation= */ {1, 1}, groups);
    CHECK(allclose(out, expected).item<bool>());
  }
}

TEST_CASE("test trace") {
  auto in = eye(3);
  auto out = trace(in).item<float>();
  CHECK_EQ(out, 3.0);

  in = array({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3}, int32);
  auto out2 = trace(in).item<int>();
  CHECK_EQ(out2, 15);

  in = reshape(arange(8), {2, 2, 2});
  auto out3 = trace(in, 0, 0, 1);
  CHECK(array_equal(out3, array({6, 8}, {2})).item<bool>());

  auto out4 = trace(in, 0, 1, 2, float32);
  CHECK(array_equal(out4, array({3, 11}, {2})).item<bool>());
}

TEST_CASE("test view") {
  auto in = array(3);
  CHECK_THROWS(view(in, int64));

  in = array({1, 2, 3});
  CHECK_THROWS(view(in, int64));

  in = array({1, 2, 3, 4}, int64);
  auto out = view(in, int32);
  CHECK(array_equal(out, array({1, 0, 2, 0, 3, 0, 4, 0})).item<bool>());
}

TEST_CASE("test roll") {
  auto x = reshape(arange(10), {2, 5});

  auto y = roll(x, 2);
  CHECK(array_equal(y, array({8, 9, 0, 1, 2, 3, 4, 5, 6, 7}, {2, 5}))
            .item<bool>());

  y = roll(x, -2);
  CHECK(array_equal(y, array({2, 3, 4, 5, 6, 7, 8, 9, 0, 1}, {2, 5}))
            .item<bool>());

  y = roll(x, 2, 1);
  CHECK(array_equal(y, array({3, 4, 0, 1, 2, 8, 9, 5, 6, 7}, {2, 5}))
            .item<bool>());

  y = roll(x, -2, 1);
  CHECK(array_equal(y, array({2, 3, 4, 0, 1, 7, 8, 9, 5, 6}, {2, 5}))
            .item<bool>());

  y = roll(x, 2, {0, 0, 0});
  CHECK(array_equal(y, array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 5}))
            .item<bool>());

  y = roll(x, 1, {1, 1, 1});
  CHECK(array_equal(y, array({2, 3, 4, 0, 1, 7, 8, 9, 5, 6}, {2, 5}))
            .item<bool>());

  y = roll(x, {1, 2}, {0, 1});
  CHECK(array_equal(y, array({8, 9, 5, 6, 7, 3, 4, 0, 1, 2}, {2, 5}))
            .item<bool>());

  y = roll(array({}), 0, 0);
  CHECK(array_equal(y, array({})).item<bool>());
}

TEST_CASE("test contiguous") {
  auto x = array({1, 2, 3});
  x = contiguous(broadcast_to(x, {2, 2, 3}));
  eval(x);
  CHECK(x.flags().row_contiguous);
  CHECK_EQ(x.strides(), decltype(x.strides()){6, 3, 1});

  x = array({1, 2, 1, 2}, {2, 2});
  x = contiguous(transpose(x), true);
  eval(x);
  CHECK(x.flags().col_contiguous);
  CHECK_EQ(x.strides(), decltype(x.strides()){1, 2});
}

TEST_CASE("test bitwise shift operations") {
  std::vector<Dtype> dtypes = {
      int8, int16, int32, int64, uint8, uint16, uint32, uint64};

  for (const auto& dtype : dtypes) {
    array x = full({4}, 1, dtype);
    array y = full({4}, 2, dtype);

    auto left_shift_result = left_shift(x, y);
    CHECK_EQ(left_shift_result.dtype(), dtype);
    CHECK(array_equal(left_shift_result, array({4, 4, 4, 4}, dtype))
              .item<bool>());

    auto right_shift_result = right_shift(full({4}, 4, dtype), y);
    CHECK_EQ(right_shift_result.dtype(), dtype);
    CHECK(array_equal(right_shift_result, full({4}, 1, dtype)).item<bool>());
  }

  array x = array({127, -128}, int8);
  array y = array({1, 1}, int8);
  auto left_shift_result = left_shift(x, y);
  auto right_shift_result = right_shift(x, y);

  CHECK(array_equal(left_shift_result, array({-2, 0}, int8)).item<bool>());
  CHECK(array_equal(right_shift_result, array({63, -64}, int8)).item<bool>());

  array x_bool = full({4}, true, bool_);
  array y_bool = full({4}, true, bool_);
  auto left_shift_bool_result = left_shift(x_bool, y_bool);
  auto right_shift_bool_result = right_shift(x_bool, y_bool);

  CHECK_EQ(left_shift_bool_result.dtype(), uint8);
  CHECK(array_equal(left_shift_bool_result, full({4}, 2, uint8)).item<bool>());

  CHECK_EQ(right_shift_bool_result.dtype(), uint8);
  CHECK(array_equal(right_shift_bool_result, full({4}, 0, uint8)).item<bool>());
}

namespace {

bool shifts_equal(const array& p, const array& q) {
  return array_equal(p, q).item<bool>();
}

// x << c and x >> c for every x of `x` and every c of `c`. want_left and
// want_right hold the results, one row per x and one column per c. They follow
// numpy: a count in [0, width of the dtype) is the usual shift, any other
// count gives 0, or -1 for a right shift of a negative value. They were
// computed from Python integers and compared with numpy. The same grid goes
// through every kernel that has its own loop: contiguous, strided,
// transposed, broadcast, and a 0-d operand.
template <typename T>
void check_shifts(
    const std::vector<T>& x,
    const std::vector<T>& c,
    const std::vector<T>& want_left,
    const std::vector<T>& want_right) {
  const int nx = static_cast<int>(x.size());
  const int nc = static_cast<int>(c.size());
  const int n = nx * nc;
  if (want_left.size() != x.size() * c.size() ||
      want_right.size() != x.size() * c.size()) {
    FAIL("the table does not match its values and counts");
    return;
  }
  Dtype dtype = TypeToDtype<T>();
  auto vec = [&](const std::vector<T>& v) {
    return array(v.begin(), {static_cast<int>(v.size())}, dtype);
  };

  // every value against every count
  std::vector<T> xs, cs, xp, cp;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < nc; ++j) {
      xs.push_back(x[i]);
      cs.push_back(c[j]);
      // the same pairs at the even places of arrays that hold other values at
      // the odd places
      xp.push_back(x[i]);
      xp.push_back(x[0]);
      cp.push_back(c[j]);
      cp.push_back(c[0]);
    }
  }
  array X = vec(xs);
  array C = vec(cs);
  array L = vec(want_left);
  array R = vec(want_right);
  CHECK(shifts_equal(left_shift(X, C), L));
  CHECK(shifts_equal(right_shift(X, C), R));

  // strided inputs
  array Xs = slice(vec(xp), {0}, {2 * n}, {2});
  array Cs = slice(vec(cp), {0}, {2 * n}, {2});
  CHECK(shifts_equal(left_shift(Xs, Cs), L));
  CHECK(shifts_equal(right_shift(Xs, Cs), R));

  // transposed inputs
  array X2 = reshape(X, {nx, nc});
  array C2 = reshape(C, {nx, nc});
  array L2 = reshape(L, {nx, nc});
  array R2 = reshape(R, {nx, nc});
  CHECK(shifts_equal(left_shift(transpose(X2), transpose(C2)), transpose(L2)));
  CHECK(shifts_equal(right_shift(transpose(X2), transpose(C2)), transpose(R2)));

  // a column of values against a row of counts
  array xv = vec(x);
  array cv = vec(c);
  array xcol = reshape(xv, {nx, 1});
  array crow = reshape(cv, {1, nc});
  CHECK(shifts_equal(left_shift(xcol, crow), L2));
  CHECK(shifts_equal(right_shift(xcol, crow), R2));

  // all the values against one 0-d count at a time
  for (int j = 0; j < nc; ++j) {
    std::vector<T> col_left, col_right;
    for (int i = 0; i < nx; ++i) {
      col_left.push_back(want_left[i * nc + j]);
      col_right.push_back(want_right[i * nc + j]);
    }
    array count(c[j], dtype);
    CAPTURE(j);
    CHECK(shifts_equal(left_shift(xv, count), vec(col_left)));
    CHECK(shifts_equal(right_shift(xv, count), vec(col_right)));
  }

  // one 0-d value against all the counts at a time
  for (int i = 0; i < nx; ++i) {
    std::vector<T> row_left(
        want_left.begin() + i * nc, want_left.begin() + (i + 1) * nc);
    std::vector<T> row_right(
        want_right.begin() + i * nc, want_right.begin() + (i + 1) * nc);
    array value(x[i], dtype);
    CAPTURE(i);
    CHECK(shifts_equal(left_shift(value, cv), vec(row_left)));
    CHECK(shifts_equal(right_shift(value, cv), vec(row_right)));
  }

  // 0-d against 0-d
  int bad_left = 0;
  int bad_right = 0;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < nc; ++j) {
      array value(x[i], dtype);
      array count(c[j], dtype);
      bad_left += left_shift(value, count).item<T>() != want_left[i * nc + j];
      bad_right +=
          right_shift(value, count).item<T>() != want_right[i * nc + j];
    }
  }
  CHECK_EQ(bad_left, 0);
  CHECK_EQ(bad_right, 0);
}

} // namespace

TEST_CASE("test shift with counts outside the type width") {
  // A shift count that is negative, or not smaller than the width of the
  // type, is undefined behavior in C++. The result is defined as in numpy: 0
  // for a left shift, and for a right shift 0, or -1 if the value is negative.
  // A count in [0, width) is the usual shift. Under UBSan (USE_UBSAN=ON with
  // UBSAN_OPTIONS=halt_on_error=1) this case also fails if a kernel does the
  // shift with a count outside the width. The grids are longer than the widest
  // SIMD loop (max_size in simd/accelerate_simd.h), so that the vector loop
  // and the scalar tail both run where there is a vector loop.
  const int8_t cmin = std::numeric_limits<int8_t>::min();
  const int8_t cmax = std::numeric_limits<int8_t>::max();
  const int16_t smin = std::numeric_limits<int16_t>::min();
  const int16_t smax = std::numeric_limits<int16_t>::max();
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const uint8_t u8max = std::numeric_limits<uint8_t>::max();
  const uint16_t u16max = std::numeric_limits<uint16_t>::max();
  const uint32_t u32max = std::numeric_limits<uint32_t>::max();
  const uint64_t u64max = std::numeric_limits<uint64_t>::max();

  // int8: 6 values by 18 counts, one row of results per value
  check_shifts<int8_t>(
      {cmin, -1, 0, 1, 85, cmax},
      {cmin, -33, -8, -1, 0, 1, 7, 8, 9, 17, 30, 31, 32, 33, 63, 64, 65, cmax},
      {0, 0, 0, 0, cmin, 0,   0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -1,   -2,  cmin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0,    0,   0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1,    2,   cmin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 85,   -86, cmin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, cmax, -2,  cmin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-1, -1, -1, -1, cmin, -64, -1, -1, -1, -1, -1, -1, -1, -1, -1,   -1,
       -1, -1, -1, -1, -1,   -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,   -1,
       -1, -1, -1, -1, 0,    0,   0,  0,  0,  0,  0,  0,  0,  0,  0,    0,
       0,  0,  0,  0,  0,    0,   0,  0,  0,  0,  1,  0,  0,  0,  0,    0,
       0,  0,  0,  0,  0,    0,   0,  0,  0,  0,  0,  0,  85, 42, 0,    0,
       0,  0,  0,  0,  0,    0,   0,  0,  0,  0,  0,  0,  0,  0,  cmax, 63,
       0,  0,  0,  0,  0,    0,   0,  0,  0,  0,  0,  0});

  // int16: 6 values by 18 counts, one row of results per value
  check_shifts<int16_t>(
      {smin, -1, 0, 1, 21845, smax},
      {smin,
       -33,
       -16,
       -1,
       0,
       1,
       9,
       15,
       16,
       17,
       30,
       31,
       32,
       33,
       63,
       64,
       65,
       smax},
      {0, 0, 0, 0, smin,  0,      0,      0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -1,    -2,     -512,   smin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0,     0,      0,      0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1,     2,      512,    smin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 21845, -21846, -22016, smin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, smax,  -2,     -512,   smin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-1, -1, -1, -1, smin, -16384, -64,   -1,    -1, -1, -1,   -1,    -1, -1,
       -1, -1, -1, -1, -1,   -1,     -1,    -1,    -1, -1, -1,   -1,    -1, -1,
       -1, -1, -1, -1, -1,   -1,     -1,    -1,    0,  0,  0,    0,     0,  0,
       0,  0,  0,  0,  0,    0,      0,     0,     0,  0,  0,    0,     0,  0,
       0,  0,  1,  0,  0,    0,      0,     0,     0,  0,  0,    0,     0,  0,
       0,  0,  0,  0,  0,    0,      21845, 10922, 42, 0,  0,    0,     0,  0,
       0,  0,  0,  0,  0,    0,      0,     0,     0,  0,  smax, 16383, 63, 0,
       0,  0,  0,  0,  0,    0,      0,     0,     0,  0});

  // int32: 6 values by 13 counts, one row of results per value
  check_shifts<int32_t>(
      {imin, -1, 0, 1, 1431655765, imax},
      {imin, -33, -32, -1, 0, 1, 31, 32, 33, 63, 64, 65, imax},
      {0, 0, 0, 0, imin,       0,           0,    0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -1,         -2,          imin, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0,          0,           0,    0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1,          2,           imin, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1431655765, -1431655766, imin, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, imax,       -2,          imin, 0, 0, 0, 0, 0, 0},
      {-1, -1, -1, -1, imin,       -1073741824, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1,         -1,          -1, -1, -1, -1, -1, -1, -1,
       0,  0,  0,  0,  0,          0,           0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  1,          0,           0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  1431655765, 715827882,   0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  imax,       1073741823,  0,  0,  0,  0,  0,  0,  0});

  // int64: 6 values by 15 counts, one row of results per value
  check_shifts<int64_t>(
      {lmin, -1, 0, 1, 6148914691236517205, lmax},
      {lmin,
       -64,
       -33,
       -1,
       0,
       1,
       31,
       32,
       33,
       63,
       64,
       65,
       4294967296,
       4294967297,
       lmax},
      {0,
       0,
       0,
       0,
       lmin,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       -1,
       -2,
       -2147483648,
       -4294967296,
       -8589934592,
       lmin,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       1,
       2,
       2147483648,
       4294967296,
       8589934592,
       lmin,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       6148914691236517205,
       -6148914691236517206,
       -6148914691952345088,
       6148914689804861440,
       -6148914694099828736,
       lmin,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       lmax,
       -2,
       -2147483648,
       -4294967296,
       -8589934592,
       lmin,
       0,
       0,
       0,
       0,
       0},
      {-1,
       -1,
       -1,
       -1,
       lmin,
       -4611686018427387904,
       -4294967296,
       -2147483648,
       -1073741824,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       6148914691236517205,
       3074457345618258602,
       2863311530,
       1431655765,
       715827882,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       lmax,
       4611686018427387903,
       4294967295,
       2147483647,
       1073741823,
       0,
       0,
       0,
       0,
       0,
       0});

  // uint8: 5 values by 15 counts, one row of results per value
  check_shifts<uint8_t>(
      {0, 1, 85, 128, u8max},
      {0, 1, 7, 8, 9, 17, 30, 31, 32, 33, 63, 64, 65, 128, u8max},
      {0, 0, 0, 0,     0,   0,   0, 0,   0, 0, 0, 0,  0,   0,   0, 1, 2, 128, 0,
       0, 0, 0, 0,     0,   0,   0, 0,   0, 0, 0, 85, 170, 128, 0, 0, 0, 0,   0,
       0, 0, 0, 0,     0,   0,   0, 128, 0, 0, 0, 0,  0,   0,   0, 0, 0, 0,   0,
       0, 0, 0, u8max, 254, 128, 0, 0,   0, 0, 0, 0,  0,   0,   0, 0, 0, 0},
      {0, 0, 0, 0,     0,   0, 0, 0,   0,  0, 0, 0,  0,  0, 0, 1, 0, 0, 0,
       0, 0, 0, 0,     0,   0, 0, 0,   0,  0, 0, 85, 42, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0,     0,   0, 0, 128, 64, 1, 0, 0,  0,  0, 0, 0, 0, 0, 0,
       0, 0, 0, u8max, 127, 1, 0, 0,   0,  0, 0, 0,  0,  0, 0, 0, 0, 0});

  // uint16: 5 values by 15 counts, one row of results per value
  check_shifts<uint16_t>(
      {0, 1, 21845, 32768, u16max},
      {0, 1, 9, 15, 16, 17, 30, 31, 32, 33, 63, 64, 65, 32768, u16max},
      {0,      0,     0,     0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1,      2,     512,   32768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       21845,  43690, 43520, 32768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       32768,  0,     0,     0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       u16max, 65534, 65024, 32768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0,      0,     0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1,      0,     0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       21845,  10922, 42,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       32768,  16384, 64,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       u16max, 32767, 127, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  // uint32: 5 values by 10 counts, one row of results per value
  check_shifts<uint32_t>(
      {0, 1, 1431655765, 2147483648, u32max},
      {0, 1, 31, 32, 33, 63, 64, 65, 2147483648, u32max},
      {0,          0,          0,          0, 0, 0, 0, 0, 0, 0,
       1,          2,          2147483648, 0, 0, 0, 0, 0, 0, 0,
       1431655765, 2863311530, 2147483648, 0, 0, 0, 0, 0, 0, 0,
       2147483648, 0,          0,          0, 0, 0, 0, 0, 0, 0,
       u32max,     4294967294, 2147483648, 0, 0, 0, 0, 0, 0, 0},
      {0,          0,          0, 0, 0, 0, 0, 0, 0, 0,
       1,          0,          0, 0, 0, 0, 0, 0, 0, 0,
       1431655765, 715827882,  0, 0, 0, 0, 0, 0, 0, 0,
       2147483648, 1073741824, 1, 0, 0, 0, 0, 0, 0, 0,
       u32max,     2147483647, 1, 0, 0, 0, 0, 0, 0, 0});

  // uint64: 5 values by 12 counts, one row of results per value
  check_shifts<uint64_t>(
      {0, 1, 6148914691236517205, 9223372036854775808ull, u64max},
      {0,
       1,
       31,
       32,
       33,
       63,
       64,
       65,
       4294967296,
       4294967297,
       9223372036854775808ull,
       u64max},
      {0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       1,
       2,
       2147483648,
       4294967296,
       8589934592,
       9223372036854775808ull,
       0,
       0,
       0,
       0,
       0,
       0,
       6148914691236517205,
       12297829382473034410ull,
       12297829381757206528ull,
       6148914689804861440,
       12297829379609722880ull,
       9223372036854775808ull,
       0,
       0,
       0,
       0,
       0,
       0,
       9223372036854775808ull,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       u64max,
       18446744073709551614ull,
       18446744071562067968ull,
       18446744069414584320ull,
       18446744065119617024ull,
       9223372036854775808ull,
       0,
       0,
       0,
       0,
       0,
       0},
      {0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       6148914691236517205,
       3074457345618258602,
       2863311530,
       1431655765,
       715827882,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       9223372036854775808ull,
       4611686018427387904,
       4294967296,
       2147483648,
       1073741824,
       1,
       0,
       0,
       0,
       0,
       0,
       0,
       u64max,
       9223372036854775807,
       8589934591,
       4294967295,
       2147483647,
       1,
       0,
       0,
       0,
       0,
       0,
       0});
}

namespace {

// The counts of the compiled function below. They are read when the function
// is traced, once for each dtype.
std::vector<int64_t> shift_constants;

// Each output shifts its own input by a constant count (the constant is
// written into the generated kernel) and ors the result with 0, so that the
// shift is fused with another operation. Separate inputs and one chain per
// output, because a node with several consumers is not fused.
template <bool Left>
std::vector<array> shift_by_constants(const std::vector<array>& in) {
  std::vector<array> out;
  for (size_t k = 0; k < in.size(); ++k) {
    array count(shift_constants[k], in[k].dtype());
    array zero(0, in[k].dtype());
    array shifted = Left ? left_shift(in[k], count) : right_shift(in[k], count);
    out.push_back(bitwise_or(shifted, zero));
  }
  return out;
}

template <typename T>
void check_compiled_shifts(
    const std::vector<T>& v,
    const std::vector<int64_t>& counts,
    const std::vector<std::vector<T>>& want_left,
    const std::vector<std::vector<T>>& want_right) {
  Dtype dtype = TypeToDtype<T>();
  shift_constants = counts;
  for (bool is_left : {true, false}) {
    auto compiled = is_left ? compile(shift_by_constants<true>)
                            : compile(shift_by_constants<false>);
    std::vector<array> inputs;
    for (size_t k = 0; k < counts.size(); ++k) {
      inputs.push_back(array(v.begin(), {static_cast<int>(v.size())}, dtype));
    }
    auto out = compiled(inputs);
    REQUIRE_EQ(out.size(), counts.size());
    const auto& want = is_left ? want_left : want_right;
    for (size_t k = 0; k < counts.size(); ++k) {
      CAPTURE(k);
      array expected(
          want[k].begin(), {static_cast<int>(want[k].size())}, dtype);
      CHECK(shifts_equal(out[k], expected));
    }
  }
}

} // namespace

TEST_CASE("test compiled shift with counts outside the type width") {
  // The same rule for a count that is a constant of a compiled function: the
  // constant is part of the generated kernel, which the C++ compiler sees, so
  // it may fold a shift by a count outside the width.
  const int32_t imin = std::numeric_limits<int32_t>::min();
  const int32_t imax = std::numeric_limits<int32_t>::max();
  const int64_t lmin = std::numeric_limits<int64_t>::min();
  const int64_t lmax = std::numeric_limits<int64_t>::max();
  const uint32_t u32max = std::numeric_limits<uint32_t>::max();
  const uint64_t u64max = std::numeric_limits<uint64_t>::max();

  // int32: 9 constant counts, expected results per count (rows) and value
  // (columns)
  check_compiled_shifts<int32_t>(
      {imin, -1, 1, 3, 1431655765, imax},
      {32, 33, 63, 64, 65, 100, -1, -32, -33},
      {{0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0}},
      {{-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0}});

  // int64: 9 constant counts, expected results per count (rows) and value
  // (columns)
  check_compiled_shifts<int64_t>(
      {lmin, -1, 1, 3, 6148914691236517205, lmax},
      {64, 65, 127, 128, 129, 100, -1, -64, -65},
      {{0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0}},
      {{-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0},
       {-1, -1, 0, 0, 0, 0}});

  // uint32: 6 constant counts, expected results per count (rows) and value
  // (columns)
  check_compiled_shifts<uint32_t>(
      {0, 1, 3, 1431655765, 2147483648, u32max},
      {32, 33, 63, 64, 65, 100},
      {{0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0}});

  // uint64: 6 constant counts, expected results per count (rows) and value
  // (columns)
  check_compiled_shifts<uint64_t>(
      {0, 1, 3, 6148914691236517205, 9223372036854775808ull, u64max},
      {64, 65, 127, 128, 129, 100},
      {{0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0}});
}

TEST_CASE("test conv_transpose1d with output_padding") {
  auto in = array({1.0, 2.0, 3.0}, {1, 1, 3});
  auto wt = array({1.0, 1.0, 1.0}, {1, 1, 3});
  int stride = 2;
  int padding = 0;
  int dilation = 1;
  int output_padding = 1;
  int groups = 1;

  auto out = conv_transpose1d(
      in, wt, stride, padding, dilation, output_padding, groups);
  auto expected = array({6.0, 0.0}, {1, 2, 1});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test conv_transpose2d with output_padding") {
  auto in = array({1.0, 2.0, 3.0, 4.0}, {1, 1, 2, 2});
  auto wt = array({1.0, 1.0, 1.0, 1.0}, {2, 1, 1, 2});
  std::pair<int, int> stride{2, 2};
  std::pair<int, int> padding{0, 0};
  std::pair<int, int> output_padding{1, 1};
  std::pair<int, int> dilation{1, 1};
  int groups = 1;

  auto out = conv_transpose2d(
      in, wt, stride, padding, dilation, output_padding, groups);
  auto expected = array(
      {3.0,
       3.0,
       0.0,
       0.0,
       7.0,
       7.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0,
       0.0},
      {1, 2, 4, 2});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test conv_transpose3d with output_padding") {
  auto in = array({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, {1, 1, 2, 2, 2});
  auto wt = array({1.0, 1.0}, {1, 1, 1, 1, 2});
  std::tuple<int, int, int> stride{2, 2, 2};
  std::tuple<int, int, int> padding{0, 0, 0};
  std::tuple<int, int, int> output_padding{1, 1, 1};
  std::tuple<int, int, int> dilation{1, 1, 1};
  int groups = 1;

  auto out = conv_transpose3d(
      in, wt, stride, padding, dilation, output_padding, groups);
  auto expected = array(
      {3.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0, 15.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0},
      {1, 2, 4, 4, 1});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test conv shape overflow") {
  // Conv shape arithmetic must not overflow (signed-int UB) for large but
  // otherwise valid int32 parameters; out-of-range results are rejected
  // gracefully. https://github.com/ml-explore/mlx/issues/3611
  const int imax = 2147483647;
  const int imin = -2147483647 - 1;
  auto in = zeros({1, 8, 8, 1});
  auto wt = zeros({1, 3, 3, 1});

  // A kernel dilated past the input reports the spatial-size error.
  CHECK_THROWS_AS(
      conv_general(in, wt, {1, 1}, {0, 0}, {0, 0}, {imax, imax}, {1, 1}),
      std::invalid_argument);

  // Padding sums, input dilation, and negating a padding of INT_MIN raise.
  CHECK_THROWS_AS(
      conv_general(in, wt, {1, 1}, {imax, imax}, {imax, imax}, {1, 1}, {1, 1}),
      std::overflow_error);
  CHECK_THROWS_AS(
      conv_general(in, wt, {1, 1}, {imax, 0}, {0, 0}, {1, 1}, {1, 1}),
      std::overflow_error);
  CHECK_THROWS_AS(
      conv_general(in, wt, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {imax, imax}),
      std::overflow_error);
  CHECK_THROWS_AS(
      conv_general(in, wt, {1, 1}, {imin, imin}, {0, 0}, {1, 1}, {1, 1}),
      std::overflow_error);

  // The transposed padding setup runs before conv_general validates it.
  auto in_t = zeros({1, 4, 4, 1});
  CHECK_THROWS_AS(
      conv_transpose2d(in_t, wt, {1, 1}, {0, 0}, {imax, imax}, {0, 0}),
      std::overflow_error);
  CHECK_THROWS_AS(
      conv_transpose2d(in_t, wt, {1, 1}, {imin, imin}, {1, 1}, {0, 0}),
      std::overflow_error);

  // The dilated input and kernel are both near 4e9 and cancel in the forward
  // output, so only the gradient's own recompute goes out of range.
  auto in_g = zeros({1, 3, 1, 1});
  auto wt_g = zeros({1, 200000, 1, 1});
  auto conv_g = [](const std::vector<array>& primals) {
    return std::vector<array>{conv_general(
        primals[0],
        primals[1],
        {1, 1},
        {0, 0},
        {0, 0},
        {20000, 1},
        {2000000000, 1})};
  };
  auto cotan = ones(conv_g({in_g, wt_g})[0].shape());
  CHECK_THROWS_AS(vjp(conv_g, {in_g, wt_g}, {cotan}), std::overflow_error);

  // The weight gradient pads without dividing by the stride.
  auto in_w = zeros({1, 8, 8, 1});
  auto conv_w = [&in_w, imax](const std::vector<array>& primals) {
    return std::vector<array>{conv_general(
        in_w, primals[0], {imax, 1}, {imax, 0}, {imax, 0}, {1, 1}, {1, 1})};
  };
  auto cotan_w = ones(conv_w({wt})[0].shape());
  CHECK_THROWS_AS(vjp(conv_w, {wt}, {cotan_w}), std::overflow_error);

  // In-range parameters still give the same shapes.
  CHECK_EQ(
      conv_general(in, wt, {1, 1}, {1, 1}, {1, 1}, {2, 2}, {1, 1}).shape(),
      Shape{1, 6, 6, 1});
  CHECK_EQ(
      conv_transpose2d(in_t, wt, {2, 2}, {1, 1}, {1, 1}, {1, 1}).shape(),
      Shape{1, 8, 8, 1});
}

TEST_CASE("test pad with an axes subset") {
  // edge_pad ignored `axes` and indexed the pad sizes by array axis, so it
  // read past the end of those vectors and placed the input on the wrong
  // axes. reflect and symmetric indexed by position but did not normalize a
  // negative axis.
  auto x = reshape(arange(24.0f), {2, 3, 4});
  auto all = {"constant", "edge", "reflect", "symmetric"};

  // Padding a subset of the axes matches the equivalent all-axes spelling.
  for (auto mode : all) {
    CHECK(array_equal(
              pad(x, {1}, Shape{1}, Shape{2}, array(0.0f), mode),
              pad(x,
                  {0, 1, 2},
                  Shape{0, 1, 0},
                  Shape{0, 2, 0},
                  array(0.0f),
                  mode))
              .item<bool>());
  }

  // A negative axis matches its non-negative spelling.
  for (auto mode : all) {
    CHECK(array_equal(
              pad(x, {-2}, Shape{1}, Shape{2}, array(0.0f), mode),
              pad(x, {1}, Shape{1}, Shape{2}, array(0.0f), mode))
              .item<bool>());
  }

  // The order of `axes` does not change the result.
  auto y = reshape(arange(25.0f), {5, 5});
  for (auto mode : all) {
    CHECK(array_equal(
              pad(y, {1, 0}, Shape{1, 2}, Shape{1, 2}, array(0.0f), mode),
              pad(y, {0, 1}, Shape{2, 1}, Shape{2, 1}, array(0.0f), mode))
              .item<bool>());
  }

  // An ndim past SmallVector's inline capacity puts the pad sizes on the heap,
  // where the old read ran off the end of the allocation.
  std::vector<int> wide_axes(11);
  std::iota(wide_axes.begin(), wide_axes.end(), 0);
  auto wide =
      pad(zeros(Shape(34, 1)),
          wide_axes,
          Shape(11, 1),
          Shape(11, 0),
          array(0.0f),
          "edge");
  CHECK_EQ(wide.size(), 2048);

  // An axis outside the array is rejected rather than indexed.
  for (auto mode : all) {
    CHECK_THROWS_AS(
        pad(x, {5}, Shape{1}, Shape{1}, array(0.0f), mode), std::out_of_range);
    CHECK_THROWS_AS(
        pad(x, {-5}, Shape{1}, Shape{1}, array(0.0f), mode), std::out_of_range);
  }
}

TEST_CASE("test pad shape overflow") {
  // A padding sum that overflows int32 is rejected, not wrapped.
  // https://github.com/ml-explore/mlx/issues/3611
  const int imax = 2147483647;
  CHECK_THROWS_AS(
      pad(zeros({8}), {0}, Shape{imax}, Shape{imax}), std::overflow_error);
}

TEST_CASE("test fp8 conversion") {
  for (auto t : {float32, float16, bfloat16}) {
    array in({-1.125, -1.0, 0.0, 1.0, 1.125, 4.5, 448.0}, t);
    auto in_fp8 = to_fp8(in);
    auto out = from_fp8(in_fp8, t);
    CHECK(array_equal(out, in).item<bool>());
  }

  array in({-1.125, -1.0, 0.0, 1.0, 1.125, 4.5, 448.0});
  array noisy_in({-1.135, -1.01, 0.0001, 1.01, 1.135, 4.6, 447.0});
  auto in_fp8 = to_fp8(noisy_in);
  auto out = from_fp8(in_fp8, float32);
  CHECK(array_equal(out, in).item<bool>());

  // Overflow
  in = array({-600.0, 600.0});
  in_fp8 = to_fp8(in);
  out = from_fp8(in_fp8, float32);

  auto expected = array({-448.0f, 448.0f});
  CHECK(array_equal(out, expected, true).item<bool>());
}

TEST_CASE("test max min with nan") {
  // Test maximum and minimum with NaN values
  auto x = array({0.0f, 1.0f, NAN, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  auto y = array({NAN, 1.0f, NAN, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  auto expected_max = array({NAN, 1.0f, NAN, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  auto expected_min = array({NAN, 1.0f, NAN, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  auto max_result = maximum(x, y);
  auto min_result = minimum(x, y);
  CHECK(array_equal(max_result, expected_max, true).item<bool>());
  CHECK(array_equal(min_result, expected_min, true).item<bool>());

  // Test with all NaN values
  x = array({NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
  y = array({NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
  max_result = maximum(x, y);
  min_result = minimum(x, y);
  auto expected = array({NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
  CHECK(array_equal(max_result, expected, true).item<bool>());
  CHECK(array_equal(min_result, expected, true).item<bool>());
}

TEST_CASE("roll and tile shape overflow") {
  // Shape arithmetic must not overflow (signed-int UB) for large but otherwise
  // valid int32 inputs; out-of-range results are rejected gracefully.
  // https://github.com/ml-explore/mlx/issues/3601

  // tile: reps * dim exceeding int32 raises instead of overflowing.
  CHECK_THROWS_AS(tile(zeros({2}), {2147483647}), std::overflow_error);

  // roll: a shift sum exceeding int32 raises instead of overflowing.
  CHECK_THROWS_AS(
      roll(zeros({4}), Shape{2147483647, 2147483647}), std::overflow_error);
  CHECK_THROWS_AS(
      roll(zeros({4}), Shape{2147483647, 2147483647}, 0), std::overflow_error);

  // roll: a shift of INT_MIN must not negate-overflow. INT_MIN mod 4 == 0, so
  // rolling a size-4 axis by INT_MIN is the identity.
  auto x = array({1, 2, 3, 4});
  auto rolled = roll(x, -2147483647 - 1);
  CHECK(array_equal(rolled, x).item<bool>());
}

TEST_CASE("test integer floor_divide and remainder by zero in whole vectors") {
  // 67 elements give whole vectors at the build's static target, then a
  // scalar tail; 67 covers widths up to 16. Zero divisors sit in both.
  StreamOrDevice cpu = Device::cpu;
  std::vector<int> an(67), as(67), bs(67);
  for (int i = 0; i < 67; ++i) {
    an[i] = (7 * i + 3) % 100;
    as[i] = (i % 2 == 0 ? 1 : -1) * an[i];
    bs[i] = (i % 8 == 5 || i == 65) ? 0 : (i % 3 == 0 ? -3 : 5);
  }
  auto apply = [cpu](int op, const array& a, const array& b) {
    return op == 0 ? floor_divide(a, b, cpu) : remainder(a, b, cpu);
  };
  // The integer types that have Highway vectors.
  std::vector<std::pair<std::string, Dtype>> types = {
      {"int32", int32}, {"uint32", uint32}, {"uint8", uint8}};
  for (const auto& type : types) {
    const std::string& name = type.first;
    Dtype t = type.second;
    CAPTURE(name);
    auto a = astype(array(as.begin(), {67}), t, cpu);
    auto n = astype(array(an.begin(), {67}), t, cpu);
    auto b = astype(array(bs.begin(), {67}), t, cpu);
    auto zero = equal(b, array(0, t), cpu);
    // As base_simd.h: a remainder by zero is the dividend, and a quotient by
    // zero is 0 (floor_divide adjusts it for a negative dividend).
    CHECK(array_equal(where(zero, remainder(a, b, cpu), a, cpu), a, cpu)
              .item<bool>());
    CHECK(array_equal(
              where(zero, floor_divide(n, b, cpu), array(0, t), cpu),
              zeros({67}, t, cpu),
              cpu)
              .item<bool>());
    for (int op = 0; op < 2; ++op) {
      auto whole = apply(op, a, b);
      for (int i = 0; i < 67; ++i) {
        CAPTURE(op);
        CAPTURE(i);
        auto one =
            apply(op, slice(a, {i}, {i + 1}, cpu), slice(b, {i}, {i + 1}, cpu));
        CHECK(array_equal(slice(whole, {i}, {i + 1}, cpu), one, cpu)
                  .item<bool>());
      }
    }
  }
}

TEST_CASE(
    "test fp8 weights and nvfp4 scales give NaN for 0x7F and 0xFF in matmuls") {
  StreamOrDevice cpu = Device::cpu;
  // mxfp8: eight rows of 64 E4M3 codes (two groups of 32), scales of 1 (E8M0
  // 127). Row 0 holds 0x7F at column 5, row 1 holds 0xFF at column 40.
  std::vector<uint8_t> codes(8 * 64);
  for (int i = 0; i < 8 * 64; ++i) {
    codes[i] = static_cast<uint8_t>((i * 37) % 0x7E);
  }
  codes[5] = 0x7F;
  codes[64 + 40] = 0xFF;
  auto w = view(array(codes.begin(), {8, 64}, uint8), uint32, cpu);
  auto scales = full({8, 2}, 127, uint8, cpu);
  for (int m : {1, 32}) {
    CAPTURE(m);
    auto y = quantized_matmul(
        ones({m, 64}, float32, cpu),
        w,
        scales,
        std::nullopt,
        true,
        32,
        8,
        "mxfp8",
        cpu);
    CHECK(all(isnan(slice(y, {0, 0}, {m, 2}, cpu), cpu), cpu).item<bool>());
    CHECK_FALSE(
        any(isnan(slice(y, {0, 2}, {m, 8}, cpu), cpu), cpu).item<bool>());
    // dequantize does not read FP8_LUT. Every sum is a multiple of 2^-9 below
    // 2^15, so float32 gives it exactly in any order.
    auto ref = matmul(
        ones({m, 64}, float32, cpu),
        transpose(
            dequantize(
                w,
                scales,
                std::nullopt,
                32,
                8,
                "mxfp8",
                std::nullopt,
                std::nullopt,
                cpu),
            cpu),
        cpu);
    CHECK(
        array_equal(
            slice(y, {0, 2}, {m, 8}, cpu), slice(ref, {0, 2}, {m, 8}, cpu), cpu)
            .item<bool>());
  }
  // nvfp4: E2M1 1.0 everywhere, E4M3 scales of 1 (0x38) in groups of 16, but
  // 0x7F for row 0's second group and 0xFF for row 1's first.
  std::vector<uint8_t> s4(8 * 4, 0x38);
  s4[1] = 0x7F;
  s4[4] = 0xFF;
  auto w4 = full({8, 8}, 0x22222222u, uint32, cpu);
  auto sc4 = array(s4.begin(), {8, 4}, uint8);
  auto y4 = quantized_matmul(
      ones({1, 64}, float32, cpu),
      w4,
      sc4,
      std::nullopt,
      true,
      16,
      4,
      "nvfp4",
      cpu);
  CHECK(all(isnan(slice(y4, {0, 0}, {1, 2}, cpu), cpu), cpu).item<bool>());
  CHECK_FALSE(
      any(isnan(slice(y4, {0, 2}, {1, 8}, cpu), cpu), cpu).item<bool>());
  CHECK(
      array_equal(slice(y4, {0, 2}, {1, 8}, cpu), full({1, 6}, 64.0f, cpu), cpu)
          .item<bool>());
  // mxfp8: every E4M3 code but 0x7F and 0xFF (0 in their places), code c at
  // row c / 64, column c % 64. Identity activations output each weight once.
  std::vector<uint8_t> finite(4 * 64);
  for (int c = 0; c < 4 * 64; ++c) {
    finite[c] = (c & 0x7F) == 0x7F ? 0 : static_cast<uint8_t>(c);
  }
  auto wf = view(array(finite.begin(), {4, 64}, uint8), uint32, cpu);
  auto sf = full({4, 2}, 127, uint8, cpu);
  auto rf = astype(
      dequantize(
          wf,
          sf,
          std::nullopt,
          32,
          8,
          "mxfp8",
          std::nullopt,
          std::nullopt,
          cpu),
      float32,
      cpu);
  auto yf = quantized_matmul(
      eye(64, float32, cpu), wf, sf, std::nullopt, true, 32, 8, "mxfp8", cpu);
  CHECK(array_equal(yf, transpose(rf, cpu), cpu).item<bool>());
  // Not transposed: fp_qmm decodes the weights through FP8_LUT.
  auto yn = quantized_matmul(
      eye(4, float32, cpu), wf, sf, std::nullopt, false, 32, 8, "mxfp8", cpu);
  CHECK(array_equal(yn, rf, cpu).item<bool>());
}

TEST_CASE(
    "test from_fp8 decodes 0x7F and 0xFF as NaN in vectors and the tail") {
  // 256 codes fill whole vectors at every width up to 16; 0x7F, 0xFF and 0x80
  // follow in the scalar tail.
  StreamOrDevice cpu = Device::cpu;
  std::vector<uint8_t> codes(256);
  for (int i = 0; i < 256; ++i) {
    codes[i] = static_cast<uint8_t>(i);
  }
  codes.insert(codes.end(), {0x7F, 0xFF, 0x80});
  // E4M3: a sign, four exponent bits (bias 7), three mantissa bits.
  std::vector<float> want;
  for (auto c : codes) {
    int e = (c >> 3) & 15;
    int m = c & 7;
    float v =
        e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    if (e == 15 && m == 7) {
      v = NAN;
    }
    want.push_back((c & 0x80) ? -v : v);
  }
  auto x = array(codes.begin(), {259}, uint8);
  auto expected = array(want.begin(), {259});
  auto bits = [cpu](const array& a) {
    return where(isnan(a, cpu), array(0u, uint32), view(a, uint32, cpu), cpu);
  };
  for (auto t : {float32, float16, bfloat16}) {
    auto got = astype(from_fp8(x, t, cpu), float32, cpu);
    for (int i : {0x7F, 0xFF, 256, 257}) {
      CAPTURE(t);
      CAPTURE(i);
      CHECK(std::isnan(slice(got, {i}, {i + 1}, cpu).item<float>()));
    }
    // Every other code exactly, the sign of zero included.
    CHECK(array_equal(isnan(got, cpu), isnan(expected, cpu), cpu).item<bool>());
    CHECK(array_equal(bits(got), bits(expected), cpu).item<bool>());
  }
  // dequantize composes from_fp8: mxfp8 weights, and nvfp4 scales.
  std::vector<uint8_t> w8(8 * 64);
  for (int i = 0; i < 8 * 64; ++i) {
    w8[i] = static_cast<uint8_t>((i * 37) % 0x7E);
  }
  w8[5] = 0x7F;
  w8[64 + 40] = 0xFF;
  auto deq = dequantize(
      view(array(w8.begin(), {8, 64}, uint8), uint32, cpu),
      full({8, 2}, 127, uint8, cpu),
      std::nullopt,
      32,
      8,
      "mxfp8",
      std::nullopt,
      float32,
      cpu);
  CHECK(std::isnan(slice(deq, {0, 5}, {1, 6}, cpu).item<float>()));
  CHECK(std::isnan(slice(deq, {1, 40}, {2, 41}, cpu).item<float>()));
  CHECK(astype(sum(isnan(deq, cpu), cpu), int32, cpu).item<int>() == 2);
  std::vector<uint8_t> s4(8 * 4, 0x38);
  s4[1] = 0x7F;
  s4[4] = 0xFF;
  auto deq4 = dequantize(
      full({8, 8}, 0x22222222u, uint32, cpu),
      array(s4.begin(), {8, 4}, uint8),
      std::nullopt,
      16,
      4,
      "nvfp4",
      std::nullopt,
      float32,
      cpu);
  CHECK(all(isnan(slice(deq4, {0, 16}, {1, 32}, cpu), cpu), cpu).item<bool>());
  CHECK(all(isnan(slice(deq4, {1, 0}, {2, 16}, cpu), cpu), cpu).item<bool>());
  CHECK(astype(sum(isnan(deq4, cpu), cpu), int32, cpu).item<int>() == 32);
}

TEST_CASE("test empty arrays in the Highway CPU kernels") {
  // Each input keeps a zero-sized dimension through the collapse, or ends
  // in one, so the kernels' inner extent is 0. Each pair of checks forces
  // one kernel, in this order: binary, copy, rms_norm, layer_norm.
  StreamOrDevice cpu = Device::cpu;
  auto a =
      add(transpose(zeros({0, 3, 3, 3}, float32, cpu), {2, 1, 0, 3}, cpu),
          zeros({3, 3, 0, 3}, float32, cpu),
          cpu);
  CHECK(a.shape() == Shape{3, 3, 0, 3});
  CHECK(array_equal(a, zeros({3, 3, 0, 3}, float32, cpu), cpu).item<bool>());
  auto c = contiguous(
      transpose(zeros({0, 3, 3}, float32, cpu), {1, 0, 2}, cpu), false, cpu);
  CHECK(c.shape() == Shape{3, 0, 3});
  CHECK(array_equal(c, zeros({3, 0, 3}, float32, cpu), cpu).item<bool>());
  auto x = zeros({3, 0}, float32, cpu);
  auto r = fast::rms_norm(x, std::nullopt, 1e-5f, cpu);
  CHECK(r.shape() == Shape{3, 0});
  CHECK(array_equal(r, x, cpu).item<bool>());
  auto l = fast::layer_norm(x, std::nullopt, std::nullopt, 1e-5f, cpu);
  CHECK(l.shape() == Shape{3, 0});
  CHECK(array_equal(l, x, cpu).item<bool>());
}

namespace {
// Sets the buffer cache limit back. With the cache off, a copy freed before
// its tasks run is freed for real, which ASan reports.
struct RestoreCacheLimit {
  size_t limit;
  ~RestoreCacheLimit() {
    set_cache_limit(limit);
  }
};
} // namespace

TEST_CASE("test norms of a non-contiguous input") {
  // A transposed input makes the native norms copy it.
  RestoreCacheLimit restore{set_cache_limit(0)};
  StreamOrDevice cpu = Device::cpu;
  // Values -3 to 3 keep every sum exact, so both layouts give the same bits.
  auto input = [cpu]() {
    auto v =
        subtract(remainder(arange(6144, cpu), array(7), cpu), array(3), cpu);
    return transpose(reshape(astype(v, float32, cpu), {64, 96}, cpu), cpu);
  };
  for (int i = 0; i < 50; i++) {
    auto x = input();
    CHECK(
        array_equal(
            fast::rms_norm(x, std::nullopt, 1e-5f, cpu),
            fast::rms_norm(contiguous(x, false, cpu), std::nullopt, 1e-5f, cpu),
            cpu)
            .item<bool>());
    auto y = input();
    CHECK(array_equal(
              fast::layer_norm(y, std::nullopt, std::nullopt, 1e-5f, cpu),
              fast::layer_norm(
                  contiguous(y, false, cpu),
                  std::nullopt,
                  std::nullopt,
                  1e-5f,
                  cpu),
              cpu)
              .item<bool>());
  }
}

TEST_CASE("test norms of a strided weight and bias") {
  // A strided weight or bias must match its contiguous copy. The broadcast
  // repeats one element of w2, so a stride-1 read stays in w2.
  RestoreCacheLimit restore{set_cache_limit(0)};
  StreamOrDevice cpu = Device::cpu;
  int width = 64;
  auto v =
      subtract(remainder(arange(8 * width, cpu), array(7), cpu), array(3), cpu);
  auto x = reshape(astype(v, float32, cpu), {8, width}, cpu);
  auto w2 = astype(
      add(remainder(arange(2 * width, cpu), array(5), cpu), array(1), cpu),
      float32,
      cpu);
  std::vector<array> params = {
      broadcast_to(slice(w2, {1}, {2}, cpu), {width}, cpu),
      slice(w2, {0}, {2 * width}, {2}, cpu)};
  for (auto& p : params) {
    CAPTURE(p.strides());
    auto pc = contiguous(p, false, cpu);
    CHECK(array_equal(
              fast::rms_norm(x, p, 1e-5f, cpu),
              fast::rms_norm(x, pc, 1e-5f, cpu),
              cpu)
              .item<bool>());
    CHECK(array_equal(
              fast::layer_norm(x, p, std::nullopt, 1e-5f, cpu),
              fast::layer_norm(x, pc, std::nullopt, 1e-5f, cpu),
              cpu)
              .item<bool>());
    CHECK(array_equal(
              fast::layer_norm(x, std::nullopt, p, 1e-5f, cpu),
              fast::layer_norm(x, std::nullopt, pc, 1e-5f, cpu),
              cpu)
              .item<bool>());
  }
}

TEST_CASE("test rope with strided offsets and freqs") {
  // Strided offsets or freqs must match their contiguous copy. Each broadcast
  // repeats one element of a wider array, so a stride-1 read stays in it.
  RestoreCacheLimit restore{set_cache_limit(0)};
  StreamOrDevice cpu = Device::cpu;
  auto v = subtract(remainder(arange(1024, cpu), array(7), cpu), array(3), cpu);
  auto x = reshape(astype(v, float32, cpu), {4, 2, 8, 16}, cpu);
  std::vector<array> offsets = {
      broadcast_to(slice(arange(8, cpu), {3}, {4}, cpu), {4}, cpu),
      slice(arange(8, cpu), {0}, {8}, {2}, cpu)};
  for (auto& off : offsets) {
    CAPTURE(off.strides());
    auto oc = contiguous(off, false, cpu);
    CHECK(array_equal(
              fast::rope(x, 16, false, 10000.0f, 1.0f, off, std::nullopt, cpu),
              fast::rope(x, 16, false, 10000.0f, 1.0f, oc, std::nullopt, cpu),
              cpu)
              .item<bool>());
  }
  auto f2 = astype(
      add(remainder(arange(16, cpu), array(5), cpu), array(1), cpu),
      float32,
      cpu);
  std::vector<array> freqs = {
      broadcast_to(slice(f2, {1}, {2}, cpu), {8}, cpu),
      slice(f2, {0}, {16}, {2}, cpu)};
  for (auto& f : freqs) {
    CAPTURE(f.strides());
    auto fc = contiguous(f, false, cpu);
    CHECK(array_equal(
              fast::rope(x, 16, false, std::nullopt, 1.0f, 0, f, cpu),
              fast::rope(x, 16, false, std::nullopt, 1.0f, 0, fc, cpu),
              cpu)
              .item<bool>());
  }
}

TEST_CASE("test sdpa with strided sinks") {
  // Strided sinks must match their contiguous copy. The broadcast repeats one
  // element of s8, so a stride-1 read stays in s8.
  RestoreCacheLimit restore{set_cache_limit(0)};
  StreamOrDevice cpu = Device::cpu;
  auto input = [cpu](int start, int len) {
    auto a = remainder(arange(start, start + 64 * len, cpu), array(7), cpu);
    auto b = astype(subtract(a, array(3), cpu), float32, cpu);
    return reshape(b, {1, 4, len, 16}, cpu);
  };
  auto q = input(0, 8);
  auto k = input(1, 16);
  auto v = input(2, 16);
  auto s8 = astype(arange(-4, 4, cpu), float32, cpu);
  std::vector<array> sinks = {
      broadcast_to(slice(s8, {1}, {2}, cpu), {4}, cpu),
      slice(s8, {0}, {8}, {2}, cpu)};
  for (auto& sink : sinks) {
    CAPTURE(sink.strides());
    auto sc = contiguous(sink, false, cpu);
    CHECK(array_equal(
              fast::scaled_dot_product_attention(
                  q, k, v, 0.0625f, "", std::nullopt, sink, false, cpu),
              fast::scaled_dot_product_attention(
                  q, k, v, 0.0625f, "", std::nullopt, sc, false, cpu),
              cpu)
              .item<bool>());
  }
}

TEST_CASE("test qqmm with an unquantized weight") {
  // An unquantized w must raise quantized.cpp's error, also with scales,
  // which qqmm accepts since it checks scales only for a quantized w.
  StreamOrDevice cpu = Device::cpu;
  int K = 64;
  int N = 32;
  auto x = ones({1, K}, float32, cpu);
  auto w = ones({N, K}, float32, cpu);
  auto with_scales = qqmm(
      x,
      w,
      zeros({N, K / 16}, uint8, cpu),
      std::nullopt,
      std::nullopt,
      "nvfp4",
      std::nullopt,
      std::nullopt,
      cpu);
  CHECK_THROWS_WITH_AS(
      sum(with_scales, cpu).item<float>(),
      "[QQMatmul] NYI for the general case",
      std::runtime_error);
  auto with_global_scales = qqmm(
      x,
      w,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      "nvfp4",
      array(1.0f),
      array(1.0f),
      cpu);
  CHECK_THROWS_WITH_AS(
      sum(with_global_scales, cpu).item<float>(),
      "[QQMatmul] NYI for the general case",
      std::runtime_error);
  auto without_scales = qqmm(
      x,
      w,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      "nvfp4",
      std::nullopt,
      std::nullopt,
      cpu);
  CHECK_THROWS_WITH_AS(
      sum(without_scales, cpu).item<float>(),
      "[QQMatmul] NYI for the general case",
      std::runtime_error);
}

TEST_CASE("test sdpa with keys and values of different lengths") {
  // Values shorter than the keys must raise the composed path's error, on a
  // shape the native kernel otherwise takes.
  StreamOrDevice cpu = Device::cpu;
  auto q = zeros({1, 4, 8, 16}, float32, cpu);
  auto k = zeros({1, 4, 16, 16}, float32, cpu);
  auto v = zeros({1, 4, 12, 16}, float32, cpu);
  CHECK_THROWS_AS(
      fast::scaled_dot_product_attention(
          q, k, v, 0.0625f, "", std::nullopt, std::nullopt, false, cpu),
      std::invalid_argument);
}

namespace {

// The i-th of a fixed sequence of floats in [0, 1), each with 24 random bits.
float unit_value(uint32_t i) {
  uint32_t h = i * 0x9E3779B9u;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

// The largest relative error of y against a float64 truth, in float64.
double max_relative_error(const array& y, const std::vector<double>& truth) {
  auto s = Device::cpu;
  auto t = array(truth.begin(), y.shape(), float64);
  auto d = divide(subtract(astype(y, float64, s), t, s), t, s);
  return max(abs(d, s), s).item<double>();
}

// The lanes of the CPU's float32 sums (simd::max_size<float>): 8 with
// Accelerate on arm64, else 1.
#if defined(MLX_USE_HIGHWAY_KERNELS)
const int sum_lanes = highway_float_lanes();
#elif defined(MLX_USE_ACCELERATE) && !defined(__x86_64__)
constexpr int sum_lanes = 8;
#else
constexpr int sum_lanes = 1;
#endif

// The CPU's float32 sum of up to 4096 elements: one sum per lane, the lanes
// halved as Apple's simd_reduce_add does, then the tail.
float lane_order_sum(const std::vector<float>& x) {
  const int N = sum_lanes;
  std::vector<float> lanes(N, 0.0f);
  size_t i = 0;
  for (; i + N <= x.size(); i += N) {
    for (int j = 0; j < N; j++) {
      lanes[j] = lanes[j] + x[i + j];
    }
  }
#if defined(MLX_USE_HIGHWAY_KERNELS)
  float total = highway_reduce_sum(lanes.data());
#else
  for (int n = N / 2; n >= 1; n /= 2) {
    for (int j = 0; j < n; j++) {
      lanes[j] = lanes[j] + lanes[j + n];
    }
  }
  float total = lanes[0];
#endif
  for (; i < x.size(); i++) {
    total = total + x[i];
  }
  return total;
}

} // namespace

TEST_CASE("test float sums over long rows on the cpu") {
  auto s = Device::cpu;
  constexpr int n = 3 << 23;
  constexpr double bound = 1e-5;
  std::vector<float> v(n);
  for (int i = 0; i < n; i++) {
    float u = unit_value(i);
    v[i] = u * u;
  }
  auto x = array(v.begin(), {n}, float32);

  // A contiguous row.
  double truth = 0;
  for (auto vi : v) {
    truth += vi;
  }
  double y = sum(x, s).item<float>();
  CHECK_LE(std::abs(y - truth) / truth, bound);

  // Strided rows: the columns of [n / 2, 2].
  std::vector<double> col_truth(2, 0.0);
  for (int i = 0; i < n; i++) {
    col_truth[i % 2] += v[i];
  }
  auto cols = sum(reshape(x, {n / 2, 2}, s), 0, false, s);
  CHECK_LE(max_relative_error(cols, col_truth), bound);

  // The columns of [n / 8, 8]: one column per SIMD lane on macOS.
  std::vector<double> col8_truth(8, 0.0);
  for (int i = 0; i < n; i++) {
    col8_truth[i % 8] += v[i];
  }
  auto cols8 = sum(reshape(x, {n / 8, 8}, s), 0, false, s);
  CHECK_LE(max_relative_error(cols8, col8_truth), bound);

  // A row of one broadcast element, summed one element at a time.
  double z = sum(broadcast_to(array(0.375f), {n}, s), s).item<float>();
  CHECK_LE(std::abs(z - 0.375 * n) / (0.375 * n), bound);
}

TEST_CASE("test softmax over a long row on the cpu") {
  auto s = Device::cpu;
  constexpr int n = 3 << 23;
  std::vector<float> v(n);
  double vmax = -INFINITY;
  for (int i = 0; i < n; i++) {
    v[i] = -8.0f * unit_value(i);
    vmax = std::max(vmax, double(v[i]));
  }
  std::vector<double> truth(n);
  double total = 0;
  for (int i = 0; i < n; i++) {
    truth[i] = std::exp(v[i] - vmax);
    total += truth[i];
  }
  for (auto& t : truth) {
    t /= total;
  }
  auto y = softmax(array(v.begin(), {1, n}, float32), -1, false, s);
  CHECK_LE(max_relative_error(y, truth), 1e-5);
}

TEST_CASE("test logsumexp over a long row on the cpu") {
  // The error of the log of the sum is the relative error of the sum.
  auto s = Device::cpu;
  constexpr int n = 3 << 23;
  std::vector<float> v(n);
  double vmax = -INFINITY;
  for (int i = 0; i < n; i++) {
    v[i] = -8.0f * unit_value(i);
    vmax = std::max(vmax, double(v[i]));
  }
  double total = 0;
  for (auto vi : v) {
    total += std::exp(vi - vmax);
  }
  double truth = std::log(total) + vmax;
  double y = logsumexp(array(v.begin(), {n}, float32), s).item<float>();
  CHECK_LE(std::abs(y - truth), 1e-5);
}

TEST_CASE("test layer_norm fallback over a long row on the cpu") {
  // The values stay about 0.25 or more from the mean, so no output is near 0.
  auto s = Device::cpu;
  constexpr int n = 3 << 23;
  std::vector<float> v(n);
  for (int i = 0; i < n; i++) {
    float u = unit_value(i);
    v[i] = u < 0.5f ? u - 0.25f : u + 0.25f;
  }
  double mean = 0;
  for (auto vi : v) {
    mean += vi;
  }
  mean /= n;
  double var = 0;
  for (auto vi : v) {
    var += (vi - mean) * (vi - mean);
  }
  var /= n;
  std::vector<double> truth(n);
  for (int i = 0; i < n; i++) {
    truth[i] = (v[i] - mean) / std::sqrt(var + 1e-5);
  }
  auto x = array(v.begin(), {1, n}, float32);
  auto y = fast::layer_norm(x, std::nullopt, std::nullopt, 1e-5f, s);
  CHECK_LE(max_relative_error(y, truth), 1e-5);
}

TEST_CASE("test rms_norm over a long row on the cpu") {
  auto s = Device::cpu;
  constexpr int n = 3 << 23;
  std::vector<float> v(n);
  double mean_sq = 0;
  for (int i = 0; i < n; i++) {
    v[i] = unit_value(i) + 0.25f;
    mean_sq += double(v[i]) * v[i];
  }
  mean_sq /= n;
  std::vector<double> truth(n);
  for (int i = 0; i < n; i++) {
    truth[i] = v[i] / std::sqrt(mean_sq + 1e-5);
  }
  auto x = array(v.begin(), {1, n}, float32);
  auto y = fast::rms_norm(x, std::nullopt, 1e-5f, s);
  CHECK_LE(max_relative_error(y, truth), 1e-5);
}

TEST_CASE("test float sums of 4096 elements keep their order on the cpu") {
  // Each expected value takes the order of the sum before the blocks.
  auto s = Device::cpu;
  constexpr int n = 4096;
  std::vector<float> v(2 * n);
  for (int i = 0; i < 2 * n; i++) {
    v[i] = unit_value(i);
  }

  // A contiguous row.
  std::vector<float> row(v.begin(), v.begin() + n);
  auto y = sum(array(row.begin(), {n}, float32), s).item<float>();
  CHECK_EQ(y, lane_order_sum(row));

  // Strided rows: each column of [n, 2] is summed in turn.
  auto cols = sum(array(v.begin(), {n, 2}, float32), 0, false, s);
  for (int j = 0; j < 2; j++) {
    float expected = 0.0f;
    for (int i = 0; i < n; i++) {
      expected = expected + v[2 * i + j];
    }
    CHECK_EQ(slice(cols, {j}, {j + 1}, s).item<float>(), expected);
  }

  // A broadcast element, summed one element at a time.
  float expected = 0.0f;
  for (int i = 0; i < n; i++) {
    expected = expected + 0.1f;
  }
  CHECK_EQ(sum(broadcast_to(array(0.1f), {n}, s), s).item<float>(), expected);

  // softmax: the exponentials from exp, which uses the same simd::exp, summed
  // in the same order, then each times the reciprocal of the sum.
  std::vector<float> w(n);
  for (int i = 0; i < n; i++) {
    w[i] = -8.0f * v[i];
  }
  auto x = array(w.begin(), {n}, float32);
  auto e = exp(subtract(x, max(x, s), s), s);
  CHECK_EQ(max(e, s).item<float>(), 1.0f);
  std::vector<float> ev(e.data<float>(), e.data<float>() + n);
  float r = 1 / lane_order_sum(ev);
  for (auto& ei : ev) {
    ei *= r;
  }
  auto sm = softmax(x, -1, false, s);
  CHECK(array_equal(sm, array(ev.begin(), {n}, float32), s).item<bool>());

  // logsumexp: the log of the same sum. One element is 0 and the rest are
  // near exp(-20), so the sum is near 1 and its log keeps each bit of it.
  w[0] = 0.0f;
  for (int i = 1; i < n; i++) {
    w[i] = -16.0f - 8.0f * v[i];
  }
  auto x2 = array(w.begin(), {n}, float32);
  auto e2 = exp(x2, s);
  CHECK_EQ(max(e2, s).item<float>(), 1.0f);
  std::vector<float> e2v(e2.data<float>(), e2.data<float>() + n);
  CHECK_EQ(logsumexp(x2, s).item<float>(), std::log(lane_order_sum(e2v)));
}

TEST_CASE("test float sums with a short last block on the cpu") {
  // 2 * 4096 + 5 elements. Each result is exact in any order.
  auto s = Device::cpu;
  constexpr int n = 8197;
  std::vector<float> ones(8 * n, 1.0f);
  auto x = array(ones.begin(), {n}, float32);
  CHECK_EQ(sum(x, s).item<float>(), float(n));
  CHECK_EQ(sum(broadcast_to(array(1.0f), {n}, s), s).item<float>(), float(n));
  auto cols = sum(array(ones.begin(), {n, 2}, float32), 0, false, s);
  CHECK(array_equal(cols, full({2}, float(n)), s).item<bool>());
  auto cols8 = sum(array(ones.begin(), {n, 8}, float32), 0, false, s);
  CHECK(array_equal(cols8, full({8}, float(n)), s).item<bool>());

  std::vector<float> zeros_v(n, 0.0f);
  auto z = array(zeros_v.begin(), {n}, float32);
  auto sm = softmax(z, -1, false, s);
  CHECK(array_equal(sm, full({n}, 1.0f / n), s).item<bool>());
  CHECK_EQ(logsumexp(z, s).item<float>(), std::log(float(n)));
}

TEST_CASE("test ops and their gradients keep the caller's stream") {
  // Each node an op adds must run on the stream the op was given, including
  // helper ops and the nodes a primitive's vjp or jvp builds.
  auto check_stream = [](const array& out, const Stream& s) {
    std::vector<array> stack{out};
    std::unordered_set<std::uintptr_t> seen;
    while (!stack.empty()) {
      auto a = stack.back();
      stack.pop_back();
      if (!seen.insert(a.id()).second || !a.has_primitive()) {
        continue;
      }
      CHECK_EQ(a.primitive().stream(), s);
      for (auto& in : a.inputs()) {
        stack.push_back(in);
      }
    }
  };
  auto check_all = [&](const std::vector<array>& outs, const Stream& s) {
    for (auto& out : outs) {
      check_stream(out, s);
    }
  };

  auto s = new_stream(default_device());
  auto a = ones({4, 5}, float32, s);

  SUBCASE("vjps") {
    auto y = full({4, 5}, 2.0f, float32, s);
    auto ct = ones({4, 5}, float32, s);
    auto vjp_of = [&](auto f, const std::vector<array>& primals) {
      return vjp(f, primals, {ct}).second;
    };
    check_all(
        vjp_of(
            [&](const std::vector<array>& in) {
              return std::vector<array>{arctan2(in[0], in[1], s)};
            },
            {a, y}),
        s);
    check_all(
        vjp_of(
            [&](const std::vector<array>& in) {
              return std::vector<array>{logcumsumexp(in[0], 1, false, true, s)};
            },
            {a}),
        s);
    check_all(
        vjp_of(
            [&](const std::vector<array>& in) {
              return std::vector<array>{
                  softmax(in[0], std::vector<int>{-1}, false, s)};
            },
            {a}),
        s);
  }

  SUBCASE("jvps") {
    auto y = full({4, 5}, 2.0f, float32, s);
    auto t = ones({4, 5}, float32, s);
    auto cond = astype(ones({4, 5}, float32, s), bool_, s);
    auto jvp_of = [&](auto f, const std::vector<array>& primals) {
      std::vector<array> tangents(primals.size(), t);
      return jvp(f, primals, tangents).second;
    };
    check_all(
        jvp_of(
            [&](const std::vector<array>& in) {
              return std::vector<array>{where(cond, in[0], in[1], s)};
            },
            {a, y}),
        s);
    check_all(
        jvp_of(
            [&](const std::vector<array>& in) {
              return std::vector<array>{
                  softmax(in[0], std::vector<int>{-1}, false, s)};
            },
            {a}),
        s);
  }

  SUBCASE("jvp of a fast op with a constant input") {
    auto w = ones({5}, float32, s);
    auto t = ones({4, 5}, float32, s);
    check_all(
        jvp(
            [&](const std::vector<array>& in) {
              return std::vector<array>{fast::rms_norm(in[0], w, 1e-5f, s)};
            },
            {a},
            {t})
            .second,
        s);
  }

  SUBCASE("custom_function keeps the stream of its function") {
    auto y = full({4, 5}, 2.0f, float32, s);
    auto ct = ones({4, 5}, float32, s);
    auto t = ones({4, 5}, float32, s);
    std::function<std::vector<array>(const std::vector<array>&)> fun =
        [&](const std::vector<array>& in) {
          return std::vector<array>{multiply(in[0], in[1], s)};
        };
    std::function<std::vector<array>(
        const std::vector<array>&,
        const std::vector<array>&,
        const std::vector<array>&)>
        fun_vjp = [&](const std::vector<array>& primals,
                      const std::vector<array>& cotangents,
                      const std::vector<array>&) {
          return std::vector<array>{
              multiply(cotangents[0], primals[1], s),
              multiply(cotangents[0], primals[0], s)};
        };
    auto f = custom_function(fun, fun_vjp);
    check_all(f({a, y}), s);
    check_all(
        vjp([&](const std::vector<array>& in) { return f(in); }, {a, y}, {ct})
            .second,
        s);
    check_all(
        jvp([&](const std::vector<array>& in) { return f({in[0], y}); },
            {a},
            {t})
            .second,
        s);
  }

  SUBCASE("vmap output fix-ups follow the traced op") {
    auto x = ones({2, 3}, float32, s);
    eval(x);
    auto dflt = default_stream(default_device());
    // Identity with a moved output axis has no traced op: the active default.
    auto id =
        vmap([](const std::vector<array>& in) { return in; }, {0}, {1})({x});
    CHECK_EQ(id[0].primitive().stream(), dflt);
    // An op on s keeps the moved axis on s even with an evaluated input.
    check_all(
        vmap(
            [&](const std::vector<array>& in) {
              return std::vector<array>{abs(in[0], s)};
            },
            {0},
            {1})({x}),
        s);
    // An output with no input dependency follows its own stream.
    check_all(
        vmap([&](const std::vector<array>& in) {
          return std::vector<array>{ones({3}, float32, s)};
        })({x}),
        s);
    // A constant with no stream at all lands on the active default.
    auto other = new_stream(default_device());
    set_default_stream(other);
    auto c = vmap([](const std::vector<array>& in) {
      return std::vector<array>{array({1.0f, 2.0f, 3.0f})};
    })({x});
    set_default_stream(dflt);
    check_all(c, other);
  }

  SUBCASE("vmap passthrough outputs") {
    auto p = full({3}, 2.0f, float32, s);
    auto x = ones({2, 3}, float32, s);
    auto f = [&](const std::vector<array>& in) {
      return std::vector<array>{in[0], abs(in[1], s)};
    };
    // A lazy passthrough keeps its own stream.
    check_all(vmap(f, {-1, 0}, {0, 0})({p, x}), s);
    // An evaluated passthrough has no stream: the active default.
    eval(p);
    auto outs = vmap(f, {-1, 0}, {0, 0})({p, x});
    check_stream(outs[0], default_stream(default_device()));
    check_stream(outs[1], s);
  }
}
