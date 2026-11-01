// Copyright © 2026 Osaurus contributors.
#include "doctest/doctest.h"
#include "mlx/mlx.h"
#include "mlx/random.h"
using namespace mlx::core;

TEST_CASE("Metal long batched matmul preserves broadcast and transpose") {
  if (!is_available(Device::gpu)) return;
  random::seed(731);
  for (auto dtype : {float16, bfloat16}) {
    for (int reduction : {65536, 65537}) {
      for (int layout = 0; layout < 4; ++layout) {
        // Distinct values per batch catch offsets accidentally reusing head 0.
        auto a = random::normal({2, 3, 17, reduction}, dtype, 0, .1);
        auto b = random::normal({2, 1, reduction, 19}, dtype, 0, .1);
        if (layout & 1) {
          a = swapaxes(random::normal({2, 3, reduction, 17}, dtype, 0, .1), -1, -2);
        }
        if (layout & 2) {
          b = swapaxes(random::normal({2, 1, 19, reduction}, dtype, 0, .1), -1, -2);
        }
        eval(a, b);
        auto actual = astype(matmul(a, b, Device::gpu), float32);
        auto expected = matmul(astype(a, float32, Device::cpu),
                               astype(b, float32, Device::cpu), Device::cpu);
        eval(actual, expected);
        float relative = divide(sqrt(mean(square(subtract(actual, expected)))),
                                sqrt(mean(square(expected)))).item<float>();
        INFO("dtype=", dtype, " K=", reduction, " layout=", layout);
        CHECK(all(isfinite(actual)).item<bool>());
        CHECK(relative < (dtype == bfloat16 ? .004f : .001f));
      }
    }
  }
}

TEST_CASE("Metal AddMM preserves existing long reduction routing") {
  if (!is_available(Device::gpu)) return;
  random::seed(919);
  for (int batch : {1, 6}) {
    auto a = random::normal({batch, 17, 65537}, bfloat16, 0, .1);
    auto b = random::normal({batch, 65537, 19}, bfloat16, 0, .1);
    auto c = random::normal({batch, 17, 19}, bfloat16, 0, .1);
    eval(a, b, c);
    auto actual = astype(addmm(c, a, b, .7f, .3f, Device::gpu), float32);
    auto expected = addmm(astype(c, float32, Device::cpu),
                         astype(a, float32, Device::cpu),
                         astype(b, float32, Device::cpu), .7f, .3f, Device::cpu);
    eval(actual, expected);
    float relative = divide(sqrt(mean(square(subtract(actual, expected)))),
                            sqrt(mean(square(expected)))).item<float>();
    CHECK(all(isfinite(actual)).item<bool>());
    CHECK(relative < .004f);
  }
}
