// Copyright © 2023-2024 Apple Inc.

#include <numeric>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/dtype_utils.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#include <algorithm>
#include <cstring>
#endif

namespace mlx::core {

namespace {

#if defined(__aarch64__)
// FCVTZS and FCVTZU give the result of saturating_cast for 32- and 64-bit
// integers, and SQXTN and UQXTN narrow with saturation, which keeps it.
template <typename SrcT, typename DstT>
inline constexpr bool neon_cast_v =
    (std::is_same_v<SrcT, float> || std::is_same_v<SrcT, double> ||
     std::is_same_v<SrcT, float16_t> || std::is_same_v<SrcT, bfloat16_t>) &&
    std::is_integral_v<DstT> && !std::is_same_v<DstT, bool>;

// Four values converted exactly to float32.
inline float32x4_t load_f32(const float* p) {
  return vld1q_f32(p);
}
inline float32x4_t load_f32(const float16_t* p) {
  uint16x4_t bits;
  std::memcpy(&bits, p, sizeof(bits));
  return vcvt_f32_f16(vreinterpret_f16_u16(bits));
}
inline float32x4_t load_f32(const bfloat16_t* p) {
  uint16x4_t bits;
  std::memcpy(&bits, p, sizeof(bits));
  return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

template <typename DstT>
inline auto neon_cvt(float32x4_t v) {
  if constexpr (std::is_signed_v<DstT>) {
    return vcvtq_s32_f32(v);
  } else {
    return vcvtq_u32_f32(v);
  }
}
template <typename DstT>
inline auto neon_cvt(float64x2_t v) {
  if constexpr (std::is_signed_v<DstT>) {
    return vcvtq_s64_f64(v);
  } else {
    return vcvtq_u64_f64(v);
  }
}

inline int32x4_t narrow(int64x2_t a, int64x2_t b) {
  return vqmovn_high_s64(vqmovn_s64(a), b);
}
inline uint32x4_t narrow(uint64x2_t a, uint64x2_t b) {
  return vqmovn_high_u64(vqmovn_u64(a), b);
}
inline int16x8_t narrow(int32x4_t a, int32x4_t b) {
  return vqmovn_high_s32(vqmovn_s32(a), b);
}
inline uint16x8_t narrow(uint32x4_t a, uint32x4_t b) {
  return vqmovn_high_u32(vqmovn_u32(a), b);
}
inline int8x16_t narrow(int16x8_t a, int16x8_t b) {
  return vqmovn_high_s16(vqmovn_s16(a), b);
}
inline uint8x16_t narrow(uint16x8_t a, uint16x8_t b) {
  return vqmovn_high_u16(vqmovn_u16(a), b);
}

template <typename T, typename V>
inline void store(T* p, V v) {
  std::memcpy(p, &v, sizeof(v));
}

// Two 64-bit integers.
inline void store64(int64_t* p, int64x2_t v) {
  vst1q_s64(p, v);
}
inline void store64(uint64_t* p, uint64x2_t v) {
  vst1q_u64(p, v);
}

// Four values converted to 32-bit integers.
template <typename DstT, typename SrcT>
inline auto neon_cvt32(const SrcT* p) {
  if constexpr (std::is_same_v<SrcT, double>) {
    return narrow(
        neon_cvt<DstT>(vld1q_f64(p)), neon_cvt<DstT>(vld1q_f64(p + 2)));
  } else {
    return neon_cvt<DstT>(load_f32(p));
  }
}

// The scalar intrinsics (vcvts_s32_f32 and others) are plain casts in GCC, so
// this uses one lane of the vector form.
template <typename DstT, typename SrcT>
DstT neon_cast(SrcT v) {
  using L = std::numeric_limits<DstT>;
  if constexpr (sizeof(DstT) == 8 || std::is_same_v<SrcT, double>) {
    float64x2_t x;
    if constexpr (std::is_same_v<SrcT, double>) {
      x = vdupq_n_f64(v);
    } else {
      x = vdupq_n_f64(static_cast<float>(v));
    }
    if constexpr (std::is_signed_v<DstT>) {
      int64_t r = vgetq_lane_s64(vcvtq_s64_f64(x), 0);
      return static_cast<DstT>(std::clamp<int64_t>(r, L::lowest(), L::max()));
    } else {
      uint64_t r = vgetq_lane_u64(vcvtq_u64_f64(x), 0);
      return static_cast<DstT>(std::min<uint64_t>(r, L::max()));
    }
  } else {
    auto x = vdupq_n_f32(static_cast<float>(v));
    if constexpr (std::is_signed_v<DstT>) {
      int32_t r = vgetq_lane_s32(vcvtq_s32_f32(x), 0);
      return static_cast<DstT>(std::clamp<int32_t>(r, L::lowest(), L::max()));
    } else {
      uint32_t r = vgetq_lane_u32(vcvtq_u32_f32(x), 0);
      return static_cast<DstT>(std::min<uint32_t>(r, L::max()));
    }
  }
}

template <typename SrcT, typename DstT>
void neon_cast_n(const SrcT* src, DstT* dst, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const SrcT* s = src + i;
    DstT* d = dst + i;
    if constexpr (sizeof(DstT) == 8) {
      for (int j = 0; j < 16; j += 4) {
        float64x2_t lo, hi;
        if constexpr (std::is_same_v<SrcT, double>) {
          lo = vld1q_f64(s + j);
          hi = vld1q_f64(s + j + 2);
          store64(d + j, neon_cvt<DstT>(lo));
          store64(d + j + 2, neon_cvt<DstT>(hi));
        } else {
          float32x4_t f = load_f32(s + j);
          lo = vcvt_f64_f32(vget_low_f32(f));
          hi = vcvt_high_f64_f32(f);
          store(d + j, neon_cvt<DstT>(lo));
          store(d + j + 2, neon_cvt<DstT>(hi));
        }
      }
    } else if constexpr (sizeof(DstT) == 4) {
      for (int j = 0; j < 16; j += 4) {
        store(d + j, neon_cvt32<DstT>(s + j));
      }
    } else if constexpr (sizeof(DstT) == 2) {
      for (int j = 0; j < 16; j += 8) {
        store(
            d + j,
            narrow(neon_cvt32<DstT>(s + j), neon_cvt32<DstT>(s + j + 4)));
      }
    } else {
      auto a = narrow(neon_cvt32<DstT>(s), neon_cvt32<DstT>(s + 4));
      auto b = narrow(neon_cvt32<DstT>(s + 8), neon_cvt32<DstT>(s + 12));
      store(d, narrow(a, b));
    }
  }
  for (; i < n; ++i) {
    dst[i] = neon_cast<DstT>(src[i]);
  }
}

template <typename DstT, typename SrcT>
inline DstT cast_to(SrcT v) {
  if constexpr (neon_cast_v<SrcT, DstT>) {
    return neon_cast<DstT>(v);
  }
  return simd::saturating_cast<DstT>(v);
}
#endif

template <typename SrcT, typename DstT>
void copy_single(const array& src, array& dst) {
  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  auto size = dst.size();
#if defined(__aarch64__)
  auto val = cast_to<DstT>(src_ptr[0]);
#else
  auto val = simd::saturating_cast<DstT>(src_ptr[0]);
#endif
  std::fill_n(dst_ptr, size, val);
}

template <typename SrcT, typename DstT>
void copy_vector(const array& src, array& dst) {
  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  auto size = src.data_size();
  if constexpr (std::is_same_v<SrcT, DstT>) {
    std::copy(src_ptr, src_ptr + size, dst_ptr);
#if defined(__aarch64__)
  } else if constexpr (neon_cast_v<SrcT, DstT>) {
    neon_cast_n(src_ptr, dst_ptr, size);
#endif
  } else {
    for (size_t i = 0; i < size; ++i) {
      dst_ptr[i] = simd::saturating_cast<DstT>(src_ptr[i]);
    }
  }
}

template <typename SrcT, typename DstT, int D>
inline void copy_dims(
    const SrcT* src,
    DstT* dst,
    const Shape& shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int axis) {
  auto stride_src = i_strides[axis];
  auto stride_dst = o_strides[axis];
  auto N = shape[axis];
#if defined(__aarch64__)
  if constexpr (D == 1 && neon_cast_v<SrcT, DstT>) {
    if (stride_src == 1 && stride_dst == 1) {
      neon_cast_n(src, dst, N);
      return;
    }
  }
#endif

  for (int i = 0; i < N; i++) {
    if constexpr (D > 1) {
      copy_dims<SrcT, DstT, D - 1>(
          src, dst, shape, i_strides, o_strides, axis + 1);
    } else {
#if defined(__aarch64__)
      *dst = cast_to<DstT>(*src);
#else
      *dst = simd::saturating_cast<DstT>(*src);
#endif
    }
    src += stride_src;
    dst += stride_dst;
  }
}

template <typename SrcT, typename DstT>
void copy_general_general(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    const std::optional<array>& dynamic_i_offset,
    const std::optional<array>& dynamic_o_offset) {
  auto src_ptr = src.data<SrcT>() + i_offset;
  auto dst_ptr = dst.data<DstT>() + o_offset;
  auto i_offset_ptr =
      dynamic_i_offset ? dynamic_i_offset->data<int64_t>() : nullptr;
  auto o_offset_ptr =
      dynamic_o_offset ? dynamic_o_offset->data<int64_t>() : nullptr;
  if (data_shape.empty()) {
#if defined(__aarch64__)
    auto val = cast_to<DstT>(*src_ptr);
#else
    auto val = simd::saturating_cast<DstT>(*src_ptr);
#endif
    *dst_ptr = val;
    return;
  }
  auto [shape, strides] =
      collapse_contiguous_dims(data_shape, {i_strides, o_strides});

  int ndim = shape.size();
  if (ndim < 3) {
    if (i_offset_ptr) {
      src_ptr += i_offset_ptr[0];
    }
    if (o_offset_ptr) {
      dst_ptr += o_offset_ptr[0];
    }

    if (ndim == 1) {
      copy_dims<SrcT, DstT, 1>(
          src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
    } else if (ndim == 2) {
      copy_dims<SrcT, DstT, 2>(
          src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
    } else if (ndim == 3) {
      copy_dims<SrcT, DstT, 3>(
          src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
    }
    return;
  }
  if (i_offset_ptr) {
    src_ptr += i_offset_ptr[0];
  }
  if (o_offset_ptr) {
    dst_ptr += o_offset_ptr[0];
  }

  auto size = std::accumulate(
      shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  ContiguousIterator in(shape, strides[0], ndim - 3);
  ContiguousIterator out(shape, strides[1], ndim - 3);
  auto stride = std::accumulate(
      shape.end() - 3, shape.end(), 1, std::multiplies<int64_t>());
  for (int64_t elem = 0; elem < size; elem += stride) {
    copy_dims<SrcT, DstT, 3>(
        src_ptr + in.loc,
        dst_ptr + out.loc,
        shape,
        strides[0],
        strides[1],
        ndim - 3);
    in.step();
    out.step();
  }
}

template <typename SrcT, typename DstT>
inline void copy_general_general(const array& src, array& dst) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      src.shape(),
      src.strides(),
      dst.strides(),
      0,
      0,
      std::nullopt,
      std::nullopt);
}

template <typename SrcT, typename DstT>
void copy_general(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides&,
    int64_t i_offset,
    int64_t o_offset,
    const std::optional<array>& dynamic_i_offset,
    const std::optional<array>& dynamic_o_offset) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      data_shape,
      i_strides,
      make_contiguous_strides(data_shape),
      i_offset,
      o_offset,
      dynamic_i_offset,
      dynamic_o_offset);
}

template <typename SrcT, typename DstT>
inline void copy_general(const array& src, array& dst) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      src.shape(),
      src.strides(),
      make_contiguous_strides(src.shape()),
      0,
      0,
      std::nullopt,
      std::nullopt);
}

template <typename SrcT, typename DstT, typename... Args>
void copy(const array& src, array& dst, CopyType ctype, Args&&... args) {
  switch (ctype) {
    case CopyType::Scalar:
      copy_single<SrcT, DstT>(src, dst);
      return;
    case CopyType::Vector:
      copy_vector<SrcT, DstT>(src, dst);
      return;
    case CopyType::General:
      copy_general<SrcT, DstT>(src, dst, std::forward<Args>(args)...);
      return;
    case CopyType::GeneralGeneral:
      copy_general_general<SrcT, DstT>(src, dst, std::forward<Args>(args)...);
      return;
  }
}

template <typename SrcT, typename... Args>
void copy(const array& src, array& dst, CopyType ctype, Args&&... args) {
  dispatch_all_types(dst.dtype(), [&](auto type_tag) {
    using DstT = MLX_GET_TYPE(type_tag);
    copy<SrcT, DstT>(src, dst, ctype, std::forward<Args>(args)...);
  });
}

template <typename... Args>
inline void copy_inplace_dispatch(
    const array& src,
    array& dst,
    CopyType ctype,
    Args&&... args) {
  dispatch_all_types(src.dtype(), [&](auto type_tag) {
    using SrcT = MLX_GET_TYPE(type_tag);
    copy<SrcT>(src, dst, ctype, std::forward<Args>(args)...);
  });
}

} // namespace

void copy_cpu_inplace(
    const array& src,
    array& dst,
    CopyType ctype,
    Stream stream) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_output_array(dst);
  encoder.dispatch(
      [src = array::unsafe_weak_copy(src),
       dst = array::unsafe_weak_copy(dst),
       ctype]() mutable { copy_inplace_dispatch(src, dst, ctype); });
}

void copy_cpu(const array& src, array& dst, CopyType ctype, Stream stream) {
  bool donated = set_copy_output_data(src, dst, ctype);
  if (donated && src.dtype() == dst.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_cpu_inplace(src, dst, ctype, stream);
}

void copy_cpu_inplace(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    Stream stream,
    const std::optional<array>& dynamic_i_offset, /* = std::nullopt */
    const std::optional<array>& dynamic_o_offset /* = std::nullopt */) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_output_array(dst);
  auto weak_copy_if_set = [](auto x) -> std::optional<array> {
    if (x) {
      return array::unsafe_weak_copy(*x);
    } else {
      return std::nullopt;
    }
  };
  encoder.dispatch(
      [src = array::unsafe_weak_copy(src),
       dst = array::unsafe_weak_copy(dst),
       data_shape,
       i_strides,
       o_strides,
       i_offset,
       o_offset,
       ctype,
       dynamic_i_offset = weak_copy_if_set(dynamic_i_offset),
       dynamic_o_offset = weak_copy_if_set(dynamic_o_offset)]() mutable {
        switch (ctype) {
          case CopyType::General:
          case CopyType::GeneralGeneral:
            copy_inplace_dispatch(
                src,
                dst,
                ctype,
                data_shape,
                i_strides,
                o_strides,
                i_offset,
                o_offset,
                dynamic_i_offset,
                dynamic_o_offset);
            break;
          case CopyType::Scalar:
          case CopyType::Vector:
            copy_inplace_dispatch(src, dst, ctype);
        }
      });
}

array contiguous_copy_cpu(const array& arr, Stream stream) {
  array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
  copy_cpu(arr, arr_copy, CopyType::General, stream);
  return arr_copy;
}

} // namespace mlx::core
