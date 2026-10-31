// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test arange") {
  // Check type is inferred correctly
  {
    auto x = arange(10);
    CHECK_EQ(x.dtype(), int32);

    x = arange(10.0);
    CHECK_EQ(x.dtype(), float32);

    x = arange(10, float32);
    CHECK_EQ(x.dtype(), float32);

    x = arange(10, float16);
    CHECK_EQ(x.dtype(), float16);

    x = arange(10, bfloat16);
    CHECK_EQ(x.dtype(), bfloat16);

    x = arange(10.0, int32);
    CHECK_EQ(x.dtype(), int32);

    x = arange(0, 10);
    CHECK_EQ(x.dtype(), int32);

    x = arange(0.0, 10.0, int32);
    CHECK_EQ(x.dtype(), int32);

    x = arange(0.0, 10.0);
    CHECK_EQ(x.dtype(), float32);

    x = arange(0, 10, float32);
    CHECK_EQ(x.dtype(), float32);

    x = arange(0, 10, 0.1, float32);
    CHECK_EQ(x.dtype(), float32);

    x = arange(0.0, 10.0, 0.5, int32);
    CHECK_EQ(x.dtype(), int32);

    x = arange(10.0, uint32);
    CHECK_EQ(x.dtype(), uint32);
    x = arange(0.0, 10.0, uint32);
    CHECK_EQ(x.dtype(), uint32);
    x = arange(0.0, 10.0, 0.5, uint32);
    CHECK_EQ(x.dtype(), uint32);

    // arange unsupported for bool_
    CHECK_THROWS_AS(arange(10, bool_), std::invalid_argument);
  }

  // Check correct sizes
  {
    auto x = arange(10);
    CHECK_EQ(x.size(), 10);

    x = arange(0.0, 10.0, 0.5);
    CHECK_EQ(x.size(), 20);

    x = arange(0.0, 10.0, 0.45);
    CHECK_EQ(x.size(), 23);

    x = arange(0, 10, 10);
    CHECK_EQ(x.size(), 1);

    x = arange(0, 10, 9);
    CHECK_EQ(x.size(), 2);

    x = arange(0, 10, 100);
    CHECK_EQ(x.size(), 1);

    x = arange(0, -10, 1);
    CHECK_EQ(x.size(), 0);

    x = arange(0, -10, -1);
    CHECK_EQ(x.size(), 10);

    x = arange(0, -10, -10);
    CHECK_EQ(x.size(), 1);
  }

  // Check values
  {
    auto x = arange(0, 3);
    CHECK(array_equal(x, array({0, 1, 2})).item<bool>());

    x = arange(0, 3, 2);
    CHECK(array_equal(x, array({0, 2})).item<bool>());

    x = arange(0, 3, 3);
    CHECK(array_equal(x, array({0})).item<bool>());

    x = arange(0, -3, 1);
    CHECK(array_equal(x, array({})).item<bool>());

    x = arange(0, 3, -1);
    CHECK(array_equal(x, array({})).item<bool>());

    x = arange(0, -3, -1);
    CHECK(array_equal(x, array({0, -1, -2})).item<bool>());

    x = arange(0.0, 5.0, 0.5, int32);
    CHECK(array_equal(x, zeros({10})).item<bool>());

    x = arange(0.0, 5.0, 1.5, int32);
    CHECK(array_equal(x, array({0, 1, 2, 3})).item<bool>());

    x = arange(0.0, 5.0, 1.0, float16);
    CHECK(array_equal(x, array({0, 1, 2, 3, 4}, float16)).item<bool>());

    x = arange(0.0, 5.0, 1.0, bfloat16);
    CHECK(array_equal(x, array({0, 1, 2, 3, 4}, bfloat16)).item<bool>());

    x = arange(0.0, 5.0, 1.5, bfloat16);
    CHECK(array_equal(x, array({0., 1.5, 3., 4.5}, bfloat16)).item<bool>());
  }
}

TEST_CASE("test arange computes each element from start") {
  auto s = Device::cpu;
  // A running sum in float32 stops at 2^24.
  int start = (1 << 24) - 1024;
  std::vector<float> counts(2048);
  for (int i = 0; i < 2048; ++i) {
    counts[i] = static_cast<float>(start + i);
  }
  auto x = arange(static_cast<double>(start), start + 2048.0, float32, s);
  auto expected = array(counts.data(), {2048}, float32);
  auto error = max(abs(subtract(x, expected, s), s), s);
  CHECK_EQ(error.item<float>(), 0.0f);

  // 16-bit types compute start + i * step in double. A running sum drifts.
  x = arange(0.0, 1.0, 0.1, float16, s);
  expected = array(
      {0.0f,
       0.0999755859375f,
       0.199951171875f,
       0.2998046875f,
       0.39990234375f,
       0.5f,
       0.599609375f,
       0.69970703125f,
       0.7998046875f,
       0.89990234375f},
      float16);
  CHECK(array_equal(x, expected, s).item<bool>());
  x = arange(0.0, 1.0, 0.1, bfloat16, s);
  expected = array(
      {0.0f,
       0.10009765625f,
       0.2001953125f,
       0.30078125f,
       0.400390625f,
       0.5f,
       0.6015625f,
       0.69921875f,
       0.80078125f,
       0.90234375f},
      bfloat16);
  CHECK(array_equal(x, expected, s).item<bool>());

  // Element 0 is start, also when the step overflows to inf.
  x = arange(0.0, 1e5, 7e4, float16, s);
  expected = array({0.0f, std::numeric_limits<float>::infinity()}, float16);
  CHECK(array_equal(x, expected, s).item<bool>());

  // float64 rounds i * step, then the sum, as Python does. An FMA rounds
  // once, which gives other values at these indices.
  x = arange(0.3, 14000.0, 0.7, float64, s);
  CHECK(x.shape() == Shape{20000});
  auto indices =
      array({3, 6, 13, 15, 10000, 10005, 19991, 19993, 19996, 19998});
  expected = array(
      {2.3999999999999995,
       4.499999999999999,
       9.4,
       10.8,
       7000.3,
       7003.8,
       13993.999999999998,
       13995.399999999998,
       13997.499999999998,
       13998.899999999998},
      float64);
  CHECK(array_equal(take(x, indices, 0, s), expected, s).item<bool>());
}

TEST_CASE("test astype") {
  // Check type conversions
  {
    auto x = array(1);
    auto y = astype(x, float32);
    CHECK_EQ(y.dtype(), float32);
    CHECK_EQ(y.item<float>(), 1.0f);

    y = astype(x, int32);
    CHECK_EQ(y.dtype(), int32);
    CHECK_EQ(y.item<int>(), 1);

    x = array(-3.0f);
    y = astype(x, int32);
    CHECK_EQ(y.dtype(), int32);
    CHECK_EQ(y.item<int>(), -3);
  }
}

TEST_CASE("test full") {
  // Check throws on bad shape
  {
    CHECK_THROWS(full({-5, 0}, 0));
    CHECK_THROWS(full({0, -5}, 0));
  }

  // Check full works for different types
  {
    auto x = full({}, 0);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(x.item<int>(), 0);

    x = full({}, 0.0);
    CHECK_EQ(x.dtype(), float32);
    CHECK_EQ(x.item<float>(), 0);

    x = full({}, false);
    CHECK_EQ(x.item<bool>(), false);

    x = full({}, 0, int32);
    CHECK_EQ(x.item<int>(), 0);

    x = full({}, 0, float32);
    CHECK_EQ(x.item<float>(), 0);

    x = full({1, 2}, 2, float32);
    CHECK(array_equal(x, array({2.0, 2.0}, {1, 2})).item<bool>());

    x = full({2, 1}, 2, float32);
    CHECK(array_equal(x, array({2.0, 2.0}, {2, 1})).item<bool>());

    x = full({2}, false);
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({false, false})).item<bool>());

    x = full({2}, 1.0, bool_);
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({true, true})).item<bool>());

    x = full({2}, 1.0, uint32);
    CHECK_EQ(x.dtype(), uint32);
    CHECK(array_equal(x, array({1, 1})).item<bool>());

    CHECK_THROWS_AS(full({2}, array({})), std::invalid_argument);
  }

  // Check broadcasting works
  {
    auto x = full({2, 2}, array({3, 4}, {2, 1}));
    CHECK(array_equal(x, array({3, 3, 4, 4}, {2, 2})).item<bool>());
    x = full({2, 2}, array({3, 4}, {1, 2}));
    CHECK(array_equal(x, array({3, 4, 3, 4}, {2, 2})).item<bool>());
  }

  // Check zeros and ones
  {
    auto x = zeros({2, 2}, float32);
    CHECK_EQ(x.shape(), Shape{2, 2});
    CHECK_EQ(x.ndim(), 2);
    CHECK_EQ(x.dtype(), float32);
    auto y = array({0.0, 0.0, 0.0, 0.0}, {2, 2});
    CHECK(array_equal(x, y).item<bool>());

    x = ones({2, 2}, float32);
    CHECK_EQ(x.shape(), Shape{2, 2});
    CHECK_EQ(x.ndim(), 2);
    CHECK_EQ(x.dtype(), float32);
    y = array({1.0, 1.0, 1.0, 1.0}, {2, 2});
    CHECK(array_equal(x, y).item<bool>());

    x = zeros({2, 2}, int32);
    y = zeros_like(x);
    CHECK_EQ(y.dtype(), int32);
    CHECK(array_equal(x, y).item<bool>());

    auto z = zeros_like(x, float16);
    CHECK_EQ(z.dtype(), float16);
    CHECK(array_equal(z, zeros({2, 2}, float16)).item<bool>());

    x = ones({2, 2}, int32);
    y = ones_like(x);
    CHECK_EQ(y.dtype(), int32);
    CHECK(array_equal(x, y).item<bool>());

    z = ones_like(x, float16);
    CHECK_EQ(z.dtype(), float16);
    CHECK(array_equal(z, ones({2, 2}, float16)).item<bool>());
  }

  // Works for empty shape and empty array
  {
    array x = ones({}, int32);
    CHECK_EQ(x.shape(), Shape{});
    CHECK_EQ(x.item<int>(), 1);

    x = full({0}, array({}));
    CHECK_EQ(x.shape(), Shape{0});
    CHECK_EQ(x.size(), 0);

    CHECK_THROWS_AS(full({}, array({})), std::invalid_argument);
  }
}
