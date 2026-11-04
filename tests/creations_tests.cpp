// Copyright © 2023 Apple Inc.

#include <cmath>
#include <sstream>

#include "doctest/doctest.h"

#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/simd/simd.h"
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

namespace {

template <typename T>
struct CastRow {
  double in;
  T out;
};

std::string name_of(Dtype dtype) {
  std::ostringstream os;
  os << dtype;
  return os.str();
}

double max_finite(Dtype dtype) {
  if (dtype == float16) {
    return 65504.0;
  }
  if (dtype == bfloat16) {
    return 3.3895313892515355e38;
  }
  return dtype == float32 ? std::numeric_limits<float>::max()
                          : std::numeric_limits<double>::max();
}

enum class Mode { eager, simd, copy, compiled };

// The Simd constructors convert like astype. Float16 and bfloat16 are only
// converted one value at a time.
template <typename S, typename T>
bool simd_converts(const array& x, const array& expected) {
  // macOS headers also declare a global simd and bfloat16_t.
  namespace simd = mlx::core::simd;
  using mlx::core::bfloat16_t;
  constexpr bool half =
      std::is_same_v<S, float16_t> || std::is_same_v<S, bfloat16_t>;
  constexpr int N = half ? 1 : std::min(simd::max_size<S>, simd::max_size<T>);
  const S* in = x.data<S>();
  const T* want = expected.data<T>();
  int n = x.size();
  std::vector<T> got(n);
  bool ok = true;
  for (int i = 0; i + N <= n; i += N) {
    simd::store(got.data() + i, simd::Simd<T, N>(simd::load<S, N>(in + i)));
  }
  for (int i = 0; i < n / N * N; ++i) {
    ok &= got[i] == want[i];
  }
  for (int i = 0; i < n; ++i) {
    ok &= simd::Simd<T, 1>(simd::Simd<S, 1>(in[i])).value == want[i];
    ok &= simd::Simd<T, 1>(in[i]).value == want[i];
  }
  return ok;
}

template <typename T>
std::vector<array> cast_to(const std::vector<array>& inputs) {
  auto x = multiply(inputs[0], array(1, inputs[0].dtype()), Device::cpu);
  return {astype(x, TypeToDtype<T>(), Device::cpu)};
}

// Check astype to T on the CPU for each float type that holds the inputs
// exactly, with at least 40 values (more than the widest SIMD loop).
template <typename T>
void check_float_to_int(const std::vector<CastRow<T>>& rows, Mode mode) {
  auto dtype = TypeToDtype<T>();
  auto equal = [](const array& a, const array& b) {
    return array_equal(a, b).item<bool>();
  };
  for (auto src : {float16, bfloat16, float32, float64}) {
    INFO("from " << name_of(src) << " to " << name_of(dtype));
    std::vector<double> in;
    std::vector<T> out;
    for (const auto& r : rows) {
      if (std::isfinite(r.in) && std::abs(r.in) > max_finite(src)) {
        continue;
      }
      auto back = astype(
                      astype(array(r.in, float64), src, Device::cpu),
                      float64,
                      Device::cpu)
                      .template item<double>();
      if (std::isnan(r.in) ? std::isnan(back) : back == r.in) {
        in.push_back(r.in);
        out.push_back(r.out);
      }
    }
    // Inputs and outputs with m values, repeating the rows
    auto make = [&](int m) {
      std::vector<double> x;
      std::vector<T> e;
      for (int i = 0; i < m; ++i) {
        x.push_back(in[i % in.size()]);
        e.push_back(out[i % in.size()]);
      }
      return std::make_pair(
          astype(array(x.begin(), {m}, float64), src, Device::cpu),
          array(e.begin(), {m}, dtype));
    };
    int n = (std::max<int>(in.size(), 40) + 19) / 20 * 20;
    auto [x, expected] = make(n);

    if (mode == Mode::compiled) {
      auto f = compile(cast_to<T>);
      CHECK(equal(f({x})[0], expected));
      CHECK(equal(
          f({slice(x, {0}, {n}, {2})})[0], slice(expected, {0}, {n}, {2})));
      continue;
    }
    if (mode == Mode::simd) {
      eval(x, expected);
      if (src == float16) {
        CHECK(simd_converts<float16_t, T>(x, expected));
      } else if (src == bfloat16) {
        CHECK(simd_converts<mlx::core::bfloat16_t, T>(x, expected));
      } else if (src == float32) {
        CHECK(simd_converts<float, T>(x, expected));
      } else {
        CHECK(simd_converts<double, T>(x, expected));
      }
      continue;
    }

    if (mode == Mode::copy) {
      // No public op converts with CopyType::Scalar, and a 0-d array is
      // contiguous, so call the copy kernels of these two cases directly.
      auto stream = default_stream(Device::cpu);
      bool fill_ok = true;
      bool zero_d_ok = true;
      for (int i = 0; i < n; ++i) {
        auto s = astype(array(in[i % in.size()], float64), src, Device::cpu);
        eval(s);
        T want = out[i % in.size()];
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
        fill_ok &= filled.data<T>()[0] == want && filled.data<T>()[2] == want;
        zero_d_ok &= single.data<T>()[0] == want;
      }
      CHECK(fill_ok);
      CHECK(zero_d_ok);
      continue;
    }

    // Contiguous
    CHECK(equal(astype(x, dtype, Device::cpu), expected));

    // 0-d
    bool scalars_ok = true;
    for (int i = 0; i < n; ++i) {
      auto s = astype(array(in[i % in.size()], float64), src, Device::cpu);
      T want = out[i % in.size()];
      scalars_ok &= astype(s, dtype, Device::cpu).template item<T>() == want;
    }
    CHECK(scalars_ok);

    // Strided: arrays with gaps in 1, 2 and 4 axes
    CHECK(equal(
        astype(slice(x, {0}, {n}, {2}), dtype, Device::cpu),
        slice(expected, {0}, {n}, {2})));
    auto x2 = reshape(x, {n / 4, 4});
    auto e2 = reshape(expected, {n / 4, 4});
    CHECK(equal(
        astype(slice(x2, {0, 0}, {n / 4, 4}, {2, 2}), dtype, Device::cpu),
        slice(e2, {0, 0}, {n / 4, 4}, {2, 2})));
    auto [x4, e4] = make(256);
    x4 = reshape(x4, {4, 4, 4, 4});
    e4 = reshape(e4, {4, 4, 4, 4});
    CHECK(equal(
        astype(
            slice(x4, {0, 0, 0, 0}, {4, 4, 4, 4}, {2, 2, 2, 2}),
            dtype,
            Device::cpu),
        slice(e4, {0, 0, 0, 0}, {4, 4, 4, 4}, {2, 2, 2, 2})));
    // Rows of 17 values with a unit stride
    auto x20 = reshape(x, {n / 20, 20});
    auto e20 = reshape(expected, {n / 20, 20});
    CHECK(equal(
        astype(slice(x20, {0, 0}, {n / 20, 17}), dtype, Device::cpu),
        slice(e20, {0, 0}, {n / 20, 17})));
  }
}

} // namespace

TEST_CASE("test astype from float to integer saturates") {
  // Converting a float to an integer is undefined in C++ when the value does
  // not fit, and the result differs between CPUs and between integer widths.
  // The CPU backend gives the same as the GPU: NaN is 0, too large or too
  // small values give the maximum or the minimum, others are truncated. Rows
  // are the input and the expected output for one integer type.
  std::vector<CastRow<int8_t>> int8_rows = {
      {NAN, 0},       {INFINITY, 127}, {-INFINITY, -128}, {0.0, 0},
      {-0.0, 0},      {0.5, 0},        {-0.5, 0},         {1.0, 1},
      {-1.0, -1},     {1e20, 127},     {-1e20, -128},     {127.0, 127},
      {127.5, 127},   {128.0, 127},    {129.0, 127},      {300.0, 127},
      {-128.0, -128}, {-128.5, -128},  {-129.0, -128},    {-300.0, -128},
      {126.5, 126},   {-127.5, -127},
  };
  std::vector<CastRow<uint8_t>> uint8_rows = {
      {NAN, 0},     {INFINITY, 255}, {-INFINITY, 0}, {0.0, 0},     {-0.0, 0},
      {0.5, 0},     {-0.5, 0},       {1.0, 1},       {-1.0, 0},    {1e20, 255},
      {-1e20, 0},   {255.0, 255},    {255.5, 255},   {256.0, 255}, {300.0, 255},
      {254.5, 254}, {-0.9, 0},       {-255.0, 0},    {-300.0, 0},
  };
  std::vector<CastRow<int16_t>> int16_rows = {
      {NAN, 0},           {INFINITY, 32767},  {-INFINITY, -32768},
      {0.0, 0},           {-0.0, 0},          {0.5, 0},
      {-0.5, 0},          {1.0, 1},           {-1.0, -1},
      {1e20, 32767},      {-1e20, -32768},    {32767.0, 32767},
      {32767.5, 32767},   {32768.0, 32767},   {65535.0, 32767},
      {-32768.0, -32768}, {-32768.5, -32768}, {-32769.0, -32768},
      {-65536.0, -32768}, {32766.5, 32766},   {-32767.5, -32767},
      {32752.0, 32752},   {32640.0, 32640},   {-32752.0, -32752},
      {-32640.0, -32640},
  };
  std::vector<CastRow<uint16_t>> uint16_rows = {
      {NAN, 0},
      {INFINITY, 65535},
      {-INFINITY, 0},
      {0.0, 0},
      {-0.0, 0},
      {0.5, 0},
      {-0.5, 0},
      {1.0, 1},
      {-1.0, 0},
      {1e20, 65535},
      {-1e20, 0},
      {65535.0, 65535},
      {65535.5, 65535},
      {65536.0, 65535},
      {65504.0, 65504},
      {131072.0, 65535},
      {-65536.0, 0},
      {65280.0, 65280},
  };
  std::vector<CastRow<int32_t>> int32_rows = {
      {NAN, 0},
      {INFINITY, 2147483647},
      {-INFINITY, -2147483648},
      {0.0, 0},
      {-0.0, 0},
      {0.5, 0},
      {-0.5, 0},
      {1.0, 1},
      {-1.0, -1},
      {1e20, 2147483647},
      {-1e20, -2147483648},
      {2147483520.0, 2147483520},
      {2147483583.0, 2147483583},
      {2147483646.5, 2147483646},
      {2147483647.0, 2147483647},
      {2147483647.5, 2147483647},
      {2147483648.0, 2147483647},
      {2147483904.0, 2147483647},
      {3000000000.0, 2147483647},
      {-2147483648.0, -2147483648},
      {-2147483648.5, -2147483648},
      {-2147483647.5, -2147483647},
      {-2147483649.0, -2147483648},
      {-2147483904.0, -2147483648},
      {-3000000000.0, -2147483648},
      {2139095040.0, 2139095040},
      {-2139095040.0, -2139095040},
  };
  std::vector<CastRow<uint32_t>> uint32_rows = {
      {NAN, 0},
      {INFINITY, 4294967295},
      {-INFINITY, 0},
      {0.0, 0},
      {-0.0, 0},
      {0.5, 0},
      {-0.5, 0},
      {1.0, 1},
      {-1.0, 0},
      {1e20, 4294967295},
      {-1e20, 0},
      {4294967040.0, 4294967040},
      {4294967294.5, 4294967294},
      {4294967295.0, 4294967295},
      {4294967295.5, 4294967295},
      {4294967296.0, 4294967295},
      {4294967552.0, 4294967295},
      {3000000000.0, 3000000000},
      {-4294967296.0, 0},
      {4278190080.0, 4278190080},
  };
  std::vector<CastRow<int64_t>> int64_rows = {
      {NAN, 0},
      {INFINITY, INT64_MAX},
      {-INFINITY, INT64_MIN},
      {0.0, 0},
      {-0.0, 0},
      {0.5, 0},
      {-0.5, 0},
      {1.0, 1},
      {-1.0, -1},
      {1e20, INT64_MAX},
      {-1e20, INT64_MIN},
      {9223371487098961920.0, 9223371487098961920LL},
      {9223372036854774784.0, 9223372036854774784LL},
      {9223372036854775808.0, INT64_MAX},
      {9223372036854777856.0, INT64_MAX},
      {1e19, INT64_MAX},
      {-9223372036854775808.0, INT64_MIN},
      {-9223373136366403584.0, INT64_MIN},
      {-9223372036854777856.0, INT64_MIN},
      {-1e19, INT64_MIN},
      {4294967296.0, 4294967296LL},
      {-4294967296.0, -4294967296LL},
      {1099511627776.5, 1099511627776LL},
      {9187343239835811840.0, 9187343239835811840LL},
      {-9187343239835811840.0, -9187343239835811840LL},
  };
  std::vector<CastRow<uint64_t>> uint64_rows = {
      {NAN, 0},
      {INFINITY, UINT64_MAX},
      {-INFINITY, 0},
      {0.0, 0},
      {-0.0, 0},
      {0.5, 0},
      {-0.5, 0},
      {1.0, 1},
      {-1.0, 0},
      {1e20, UINT64_MAX},
      {-1e20, 0},
      {18446742974197923840.0, 18446742974197923840ULL},
      {18446744073709549568.0, 18446744073709549568ULL},
      {18446744073709551616.0, UINT64_MAX},
      {18446744073709555712.0, UINT64_MAX},
      {9223372036854775808.0, 9223372036854775808ULL},
      {-9223372036854775808.0, 0},
      {4294967296.5, 4294967296ULL},
      {18374686479671623680.0, 18374686479671623680ULL},
  };
  // The checks use float64, which only the CPU supports.
  StreamContext sc(Device::cpu);
  auto run = [&](Mode mode) {
    check_float_to_int(int8_rows, mode);
    check_float_to_int(uint8_rows, mode);
    check_float_to_int(int16_rows, mode);
    check_float_to_int(uint16_rows, mode);
    check_float_to_int(int32_rows, mode);
    check_float_to_int(uint32_rows, mode);
    check_float_to_int(int64_rows, mode);
    check_float_to_int(uint64_rows, mode);
  };
  SUBCASE("eager") {
    run(Mode::eager);
  }
  SUBCASE("simd") {
    run(Mode::simd);
  }
  SUBCASE("copy") {
    run(Mode::copy);
  }
  // The kernels of compile are built with a C++ compiler
  SUBCASE("compiled") {
    run(Mode::compiled);
  }
  // The CPU kernels of exp, sin and cos have lanes that do not fit in an
  // integer (NaN, inf, large values). UBSan reports them if the kernels
  // convert such a lane without a check.
  SUBCASE("special values") {
    auto x =
        array({NAN, INFINITY, -INFINITY, 1e30f, -1e30f, 100.0f, -100.0f, 0.0f});
    auto expected =
        array({NAN, INFINITY, 0.0f, INFINITY, 0.0f, INFINITY, 0.0f, 1.0f});
    CHECK(array_equal(exp(x, Device::cpu), expected, true).item<bool>());
    for (auto y : {sin(x, Device::cpu), cos(x, Device::cpu)}) {
      CHECK(all(isnan(slice(y, {0}, {3}))).item<bool>());
      CHECK(all(less_equal(abs(slice(y, {3}, {8})), array(1.0f))).item<bool>());
    }
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
