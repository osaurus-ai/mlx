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

namespace detail {
// As env::get_var reads MLX's variables: an integer, and nonzero is on.
MLX_API bool parse_quantized_int8(const char* value);
// For a test: whether transposed affine 2-, 4- and 8-bit quantized matmul sums
// bf16 and fp16 in float32, as it does with Highway. Other paths may not.
MLX_API bool quantized_float32_accumulation();
} // namespace detail

} // namespace mlx::core::cpu
