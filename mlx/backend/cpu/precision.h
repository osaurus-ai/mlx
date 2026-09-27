// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/api.h"

namespace mlx::core::cpu {

// Whether affine quantized matmul may round activations to int8 per group
// (the fast path of ml-explore/mlx#3019). Off unless MLX_CPU_QUANTIZED_INT8 is
// a nonzero integer at first use, or until set_quantized_int8(true).
// Process-wide.
MLX_API bool quantized_int8();
MLX_API void set_quantized_int8(bool enabled);

// Whether this build has the int8 path at all (builds with Highway kernels).
// Where it does not, the switch changes nothing.
MLX_API bool quantized_int8_available();

// Whether affine 4- and 8-bit quantized matmul sums bf16 and fp16 activations
// in float32 (the Highway kernels) rather than in their own dtype (the scalar
// _qmm_t<T>, which Highway builds still use for 3, 5 and 6 bits).
MLX_API bool quantized_float32_accumulation();

namespace detail {
// As env::get_var reads MLX's variables: an integer, and nonzero is on.
MLX_API bool parse_quantized_int8(const char* value);
} // namespace detail

} // namespace mlx::core::cpu
