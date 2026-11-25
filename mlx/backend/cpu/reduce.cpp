// Copyright © 2023 Apple Inc.

#include <cassert>
#include <functional>
#include <limits>

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "mlx/backend/cpu/threading/common.h"
#endif // MLX_USE_HIGHWAY_KERNELS
#include "mlx/primitives.h"

namespace mlx::core {

template <typename U>
struct Limits {
  static const U max;
  static const U min;
};

#define instantiate_default_limit(type)                           \
  template <>                                                     \
  struct Limits<type> {                                           \
    static constexpr type max = std::numeric_limits<type>::max(); \
    static constexpr type min = std::numeric_limits<type>::min(); \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type) \
  template <>                         \
  struct Limits<type> {               \
    static const type max;            \
    static const type min;            \
  };

instantiate_float_limit(float16_t);
instantiate_float_limit(bfloat16_t);
instantiate_float_limit(float);
instantiate_float_limit(double);
instantiate_float_limit(complex64_t);

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

const float Limits<float>::max = std::numeric_limits<float>::infinity();
const float Limits<float>::min = -std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::max =
    std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::min =
    -std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::max = std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::min =
    -std::numeric_limits<float>::infinity();
const double Limits<double>::max = std::numeric_limits<double>::infinity();
const double Limits<double>::min = -std::numeric_limits<double>::infinity();
const complex64_t Limits<complex64_t>::max =
    std::numeric_limits<float>::infinity();
const complex64_t Limits<complex64_t>::min =
    -std::numeric_limits<float>::infinity();

template <typename T, typename U, typename Op>
struct ReductionAccumulator {
  static constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
#if defined(MLX_USE_HIGHWAY_KERNELS)
  // The facade's float16 arithmetic already runs in float: widen float16 sums
  // and products at every width.
  static constexpr bool widen_to_float =
      std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>;
#else
  // Widen to float32 only if the input is float16 (with N=1) or bfloat16 as it
  // improves performance.
  static constexpr bool widen_to_float =
      (std::is_same_v<T, bfloat16_t> ||
       (N == 1 && std::is_same_v<T, float16_t>));
#endif // MLX_USE_HIGHWAY_KERNELS

  using type = std::conditional_t<widen_to_float, float, U>;
};

// Float sums are made in blocks of sum_block elements, and the block sums are
// added. Rows up to sum_block elements keep their results.
constexpr int sum_block = 4096;

struct SumReduce;
struct ProdReduce;

template <typename U, typename Op>
constexpr bool cascade_sum =
    std::is_same_v<Op, SumReduce> && !std::is_integral_v<U>;

template <typename T, typename U, typename Op>
void strided_reduce_block(
    const T* x,
    U* accumulator,
    int size,
    size_t stride,
    Op op) {
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  for (int i = 0; i < size; i++) {
    U* moving_accumulator = accumulator;
    auto s = stride;
    while (s >= N) {
      auto acc = simd::load<U, N>(moving_accumulator);
      auto v = simd::Simd<U, N>(simd::load<T, N>(x));
      simd::store<U, N>(moving_accumulator, op(acc, v));
      moving_accumulator += N;
      x += N;
      s -= N;
    }
    while (s-- > 0) {
      *moving_accumulator = op(*moving_accumulator, *x);
      moving_accumulator++;
      x++;
    }
  }
};

template <typename T, typename U, typename Op>
void strided_reduce(
    const T* x,
    U* accumulator,
    int size,
    size_t stride,
    Op op) {
  if constexpr (cascade_sum<U, Op>) {
    if (size > sum_block) {
      // The first block goes into the accumulator, each later block into
      // partial from zero.
      strided_reduce_block(x, accumulator, sum_block, stride, op);
      std::vector<U> partial(stride);
      for (int i = sum_block; i < size; i += sum_block) {
        int n = std::min(size - i, sum_block);
        std::fill(partial.begin(), partial.end(), U(0));
        strided_reduce_block(x + i * stride, partial.data(), n, stride, op);
        for (size_t j = 0; j < stride; j++) {
          accumulator[j] = op(accumulator[j], partial[j]);
        }
      }
      return;
    }
  }
  strided_reduce_block(x, accumulator, size, stride, op);
}

#if defined(MLX_USE_HIGHWAY_KERNELS)
// Accumulator type: use float for half-precision types to avoid precision loss
template <typename U>
struct AccumType {
  using type = U;
};
template <>
struct AccumType<bfloat16_t> {
  using type = float;
};
template <>
struct AccumType<float16_t> {
  using type = float;
};

#endif // MLX_USE_HIGHWAY_KERNELS
#if defined(MLX_USE_HIGHWAY_KERNELS)
template <typename T, typename A, typename Op>
A contiguous_accumulate_block(const T* x, int size, Op op, A acc, A init) {
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<A>);
  simd::Simd<A, N> accumulator_v(init);
  while (size >= N) {
    accumulator_v = op(accumulator_v, simd::Simd<A, N>(simd::load<T, N>(x)));
    x += N;
    size -= N;
  }
  A result = op(acc, op(accumulator_v));
  while (size-- > 0) {
    result = op(result, static_cast<A>(*x));
    x++;
  }
  return result;
}

template <typename T, typename A, typename Op>
A contiguous_accumulate(const T* x, int size, Op op, A acc, A init) {
  if constexpr (cascade_sum<A, Op>) {
    for (; size > sum_block; size -= sum_block, x += sum_block) {
      acc = contiguous_accumulate_block(x, sum_block, op, acc, init);
    }
  }
  if constexpr (
      std::is_same_v<Op, ProdReduce> &&
      (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>)) {
    // Multiply half precision values one at a time, in order, in float. The
    // CPU backend without Highway does the same, because it has no bfloat16
    // vectors. A lane that takes each Nth value can overflow or underflow
    // when the product in order does not.
    for (; size > 0; size--, x++) {
      acc = op(acc, static_cast<A>(*x));
    }
    return acc;
  }
  return contiguous_accumulate_block(x, size, op, acc, init);
}

template <typename T, typename U, typename Op>
void contiguous_reduce(const T* x, U* accumulator, int size, Op op, U init) {
  using A = typename AccumType<U>::type;
  *accumulator = static_cast<U>(contiguous_accumulate(
      x, size, op, static_cast<A>(*accumulator), static_cast<A>(init)));
}
#else
template <typename T, typename U, typename Op>
void contiguous_reduce_block(
    const T* x,
    U* accumulator,
    int64_t size,
    Op op,
    U init) {
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  simd::Simd<U, N> accumulator_v(init);
  while (size >= N) {
    accumulator_v = op(accumulator_v, simd::Simd<U, N>(simd::load<T, N>(x)));
    x += N;
    size -= N;
  }
  *accumulator = op(*accumulator, op(accumulator_v));
  while (size-- > 0) {
    *accumulator = op(*accumulator, *x);
    x++;
  }
}

template <typename T, typename U, typename Op>
void contiguous_reduce(
    const T* x,
    U* accumulator,
    int64_t size,
    Op op,
    U init) {
  if constexpr (cascade_sum<U, Op>) {
    // Each block sums from init in the lanes, then adds into the accumulator.
    for (; size > sum_block; size -= sum_block, x += sum_block) {
      contiguous_reduce_block(x, accumulator, sum_block, op, init);
    }
  }
  contiguous_reduce_block(x, accumulator, size, op, init);
}
#endif // MLX_USE_HIGHWAY_KERNELS

#if defined(MLX_USE_HIGHWAY_KERNELS)
// Parallel version for large reductions
template <typename T, typename U, typename Op>
U parallel_contiguous_reduce(const T* x, size_t size, Op op, U init) {
  using A = typename AccumType<U>::type;
  auto& pool = cpu::ThreadPool::instance();
  int max_threads = pool.max_threads();
  int n_threads = cpu::effective_threads(size, max_threads);

  if (n_threads <= 1) {
    U result = init;
    contiguous_reduce(x, &result, static_cast<int>(size), op, init);
    return result;
  }

  // Each thread computes a partial reduction. Keep each partial on its own
  // cache line to avoid false sharing while preserving A's alignment/lifetime.
  struct alignas(64) Partial {
    A value;
  };
  std::vector<Partial> partials(n_threads);

  pool.parallel_for(n_threads, [&](int tid, int nth) {
    size_t chunk = (size + nth - 1) / nth;
    size_t start = chunk * tid;
    size_t end = std::min(start + chunk, size);

    A* partial = &partials[tid].value;
    *partial = static_cast<A>(init);

    if (start < end) {
      *partial = contiguous_accumulate(
          x + start,
          static_cast<int>(end - start),
          op,
          static_cast<A>(init),
          static_cast<A>(init));
    }
  });

  // Final reduction of partial results
  A result = static_cast<A>(init);
  for (int i = 0; i < n_threads; ++i) {
    result = op(result, partials[i].value);
  }
  return static_cast<U>(result);
}

#endif // MLX_USE_HIGHWAY_KERNELS
// Helper for the ndimensional strided loop
void nd_loop(
    std::function<void(int64_t)> callback,
    const Shape& shape,
    const Strides& strides) {
  std::function<void(int, int64_t)> loop_inner;
  loop_inner = [&](int dim, int64_t offset) {
    if (dim < shape.size() - 1) {
      auto size = shape[dim];
      auto stride = strides[dim];
      for (int i = 0; i < size; i++) {
        loop_inner(dim + 1, offset + i * stride);
      }
    } else {
      auto size = shape[dim];
      auto stride = strides[dim];
      for (int i = 0; i < size; i++) {
        callback(offset + i * stride);
      }
    }
  };
  loop_inner(0, 0);
}

template <typename T, typename U, typename Op>
void reduction_op(
    const array& x,
    array& out,
    const std::vector<int>& axes,
    U init) {
  ReductionPlan plan = get_reduction_plan(x, axes);

  auto in_ptr = x.data<T>();
  auto out_ptr = out.data<U>();
  if (plan.type == ContiguousAllReduce) {
#if !defined(MLX_USE_HIGHWAY_KERNELS)
    *out_ptr = init;
    contiguous_reduce(in_ptr, out_ptr, x.size(), Op{}, init);
#else
    // Use parallel reduction for large arrays
    *out_ptr = parallel_contiguous_reduce(in_ptr, x.size(), Op{}, init);
#endif // MLX_USE_HIGHWAY_KERNELS
    return;
  }

  if (plan.type == ContiguousReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape[0];
#if !defined(MLX_USE_HIGHWAY_KERNELS)
    for (int i = 0; i < out.size(); i++, out_ptr++, in_ptr += reduction_size) {
      *out_ptr = init;
      contiguous_reduce(in_ptr, out_ptr, reduction_size, Op{}, init);
#else
    int num_reductions = out.size();

    // Parallelize over output elements (each is an independent reduction)
    // Only parallelize if total work is large enough AND each thread gets
    // enough work
    auto& pool = cpu::ThreadPool::instance();
    size_t total_elements =
        static_cast<size_t>(num_reductions) * reduction_size;
    int n_threads = cpu::effective_threads(total_elements, pool.max_threads());

    // Also ensure each thread handles at least a few reductions to amortize
    // overhead
    n_threads = std::min(n_threads, std::max(1, num_reductions / 4));

    if (n_threads <= 1) {
      for (int i = 0; i < num_reductions;
           i++, out_ptr++, in_ptr += reduction_size) {
        *out_ptr = init;
        contiguous_reduce(in_ptr, out_ptr, reduction_size, Op{}, init);
      }
    } else {
      pool.parallel_for(n_threads, [&, in_ptr, out_ptr](int tid, int nth) {
        int chunk = (num_reductions + nth - 1) / nth;
        int start = chunk * tid;
        int end = std::min(start + chunk, num_reductions);

        const T* my_in = in_ptr + start * reduction_size;
        U* my_out = out_ptr + start;

        for (int i = start; i < end; i++, my_out++, my_in += reduction_size) {
          *my_out = init;
          contiguous_reduce(my_in, my_out, reduction_size, Op{}, init);
        }
      });
#endif // MLX_USE_HIGHWAY_KERNELS
    }
    return;
  }

  if (plan.type == GeneralContiguousReduce || plan.type == ContiguousReduce) {
    int reduction_size = plan.shape.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    // Unrolling the following loop (and implementing it in order for
    // ContiguousReduce) should hold extra performance boost.
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);
    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int64_t offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        contiguous_reduce(in_ptr + offset, out_ptr, reduction_size, Op{}, init);
      }
    } else {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int64_t offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        nd_loop(
            [&](int64_t extra_offset) {
              contiguous_reduce(
                  in_ptr + offset + extra_offset,
                  out_ptr,
                  reduction_size,
                  Op{},
                  init);
            },
            plan.shape,
            plan.strides);
      }
    }
    return;
  }

  if (plan.type == ContiguousStridedReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    for (int i = 0; i < out.size(); i += reduction_stride) {
      std::fill_n(out_ptr, reduction_stride, init);
      strided_reduce(in_ptr, out_ptr, reduction_size, reduction_stride, Op{});
      in_ptr += reduction_stride * reduction_size;
      out_ptr += reduction_stride;
    }
    return;
  }

  if (plan.type == GeneralStridedReduce ||
      plan.type == ContiguousStridedReduce) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);

    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int64_t offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        strided_reduce(
            in_ptr + offset, out_ptr, reduction_size, reduction_stride, Op{});
        out_ptr += reduction_stride;
      }
    } else {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int64_t offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        nd_loop(
            [&](int64_t extra_offset) {
              strided_reduce(
                  in_ptr + offset + extra_offset,
                  out_ptr,
                  reduction_size,
                  reduction_stride,
                  Op{});
            },
            plan.shape,
            plan.strides);
        out_ptr += reduction_stride;
      }
    }
    return;
  }

  if (plan.type == GeneralReduce) {
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);

    if constexpr (cascade_sum<U, Op>) {
      int64_t size = 1;
      for (auto d : plan.shape) {
        size *= d;
      }
      if (size > sum_block) {
        for (int i = 0; i < out.size(); i++, out_ptr++) {
          int64_t offset = elem_to_loc(i, shape, strides);
          // The first block goes into val, each later block into block from 0.
          U val = init;
          U block = U(0);
          int64_t count = 0;
          nd_loop(
              [&](int64_t extra_offset) {
                U& acc = count < sum_block ? val : block;
                acc = Op{}(acc, *(in_ptr + offset + extra_offset));
                if (++count > sum_block && count % sum_block == 0) {
                  val = Op{}(val, block);
                  block = U(0);
                }
              },
              plan.shape,
              plan.strides);
          if (count % sum_block != 0) {
            val = Op{}(val, block);
          }
          *out_ptr = val;
        }
        return;
      }
    }

    for (int i = 0; i < out.size(); i++, out_ptr++) {
      int64_t offset = elem_to_loc(i, shape, strides);
      U val = init;
      nd_loop(
          [&](int64_t extra_offset) {
            val = Op{}(val, *(in_ptr + offset + extra_offset));
          },
          plan.shape,
          plan.strides);
      *out_ptr = val;
    }
  }
}

template <typename T, typename U, typename Op>
void float_reduction(
    const array& x,
    array& out,
    const std::vector<int>& axes,
    U init) {
  using AccT = typename ReductionAccumulator<T, U, Op>::type;
  if constexpr (std::is_same_v<AccT, U>) {
    reduction_op<T, U, Op>(x, out, axes, init);
  } else {
    array temp(out.shape(), TypeToDtype<AccT>(), nullptr, {});
    temp.set_data(allocator::malloc(temp.nbytes()));
    reduction_op<T, AccT, Op>(x, temp, axes, static_cast<AccT>(init));
    std::copy_n(temp.data<AccT>(), out.size(), out.data<U>());
  }
}

struct AndReduce {
  template <typename T>
  bool operator()(bool x, T y) {
    return x & static_cast<bool>(y);
  }

  bool operator()(bool x, bool y) {
    return x & y;
  }

  template <int N, typename T>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<T, N> x) {
    return x & (y != 0);
  };

  template <int N>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<bool, N> x) {
    return x & y;
  };

  template <int N, typename T>
  bool operator()(simd::Simd<T, N> x) {
    return simd::all(x);
  };
};

struct OrReduce {
  template <typename T>
  bool operator()(bool x, T y) {
    return x | static_cast<bool>(y);
  }

  bool operator()(bool x, bool y) {
    return x | y;
  }

  template <int N, typename T>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<T, N> x) {
    return x | (y != 0);
  };

  template <int N>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<bool, N> x) {
    return x | y;
  };

  template <int N, typename T>
  bool operator()(simd::Simd<T, N> x) {
    return simd::any(x);
  };
};

struct MaxReduce {
  template <typename T>
  T operator()(T y, T x) {
    return (*this)(simd::Simd<T, 1>(x), simd::Simd<T, 1>(y)).value;
  };

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) {
    return simd::maximum(x, y);
  };

  template <int N, typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    return simd::max(x);
  };

  template <int N, typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    if (simd::any(x != x)) {
      return static_cast<T>(NAN);
    }
    return simd::max(x);
  };
};

struct MinReduce {
  template <typename T>
  T operator()(T y, T x) {
    return (*this)(simd::Simd<T, 1>(x), simd::Simd<T, 1>(y)).value;
  };

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) {
    return simd::minimum(x, y);
  };

  template <int N, typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    return simd::min(x);
  };

  template <int N, typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    if (simd::any(x != x)) {
      return static_cast<T>(NAN);
    }
    return simd::min(x);
  };
};

// The scalar operators of SumReduce and ProdReduce take the elements after the
// last full vector, add up the totals of separate contiguous segments, and do
// the general (for example broadcast) reductions. Integer sums and products
// wrap around there too (see simd::wrapping).
struct SumReduce {
  template <typename T, typename U>
  U operator()(U y, T x) {
    return simd::wrapping(std::plus<>(), x, y);
  };

  template <int N, typename T, typename U>
  simd::Simd<U, N> operator()(simd::Simd<U, N> y, simd::Simd<T, N> x) {
    return y + x;
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::sum(x);
  };
};

struct ProdReduce {
  template <typename T, typename U>
  U operator()(U y, T x) {
    return simd::wrapping(std::multiplies<>(), x, y);
  };

  template <int N, typename T, typename U>
  simd::Simd<U, N> operator()(simd::Simd<U, N> y, simd::Simd<T, N> x) {
    return x * y;
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::prod(x);
  };
};

template <typename InT>
void reduce_dispatch_and_or(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::And) {
    reduction_op<InT, bool, AndReduce>(in, out, axes, true);
  } else {
    reduction_op<InT, bool, OrReduce>(in, out, axes, false);
  }
}

template <typename InT>
void reduce_dispatch_sum_prod(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::Sum) {
    if constexpr (std::is_integral_v<InT> && sizeof(InT) <= 4) {
      reduction_op<InT, int32_t, SumReduce>(in, out, axes, 0);
    } else {
      float_reduction<InT, InT, SumReduce>(in, out, axes, 0);
    }
  } else {
    if constexpr (std::is_integral_v<InT> && sizeof(InT) <= 4) {
      reduction_op<InT, int32_t, ProdReduce>(in, out, axes, 1);
    } else {
      float_reduction<InT, InT, ProdReduce>(in, out, axes, 1);
    }
  }
}

template <typename InT>
void reduce_dispatch_min_max(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::Max) {
    auto init = Limits<InT>::min;
    reduction_op<InT, InT, MaxReduce>(in, out, axes, init);
  } else {
    auto init = Limits<InT>::max;
    reduction_op<InT, InT, MinReduce>(in, out, axes, init);
  }
}

void Reduce::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    reduce_type_ = reduce_type_,
                    axes_ = axes_]() mutable {
    switch (reduce_type_) {
      case Reduce::And:
      case Reduce::Or: {
        // Integers can be reduced as whatever type has the same width, since
        // only their bits matter. Floats cannot: -0.0 compares equal to zero
        // but has a bit set, so it has to be tested as a float.
        switch (in.dtype()) {
          case bool_:
          case uint8:
          case int8:
            reduce_dispatch_and_or<int8_t>(in, out, reduce_type_, axes_);
            break;
          case int16:
          case uint16:
            reduce_dispatch_and_or<int16_t>(in, out, reduce_type_, axes_);
            break;
          case float16:
            reduce_dispatch_and_or<float16_t>(in, out, reduce_type_, axes_);
            break;
          case bfloat16:
            reduce_dispatch_and_or<bfloat16_t>(in, out, reduce_type_, axes_);
            break;
          case uint32:
          case int32:
            reduce_dispatch_and_or<int32_t>(in, out, reduce_type_, axes_);
            break;
          case float32:
            reduce_dispatch_and_or<float>(in, out, reduce_type_, axes_);
            break;
          case uint64:
          case int64:
            reduce_dispatch_and_or<int64_t>(in, out, reduce_type_, axes_);
            break;
          case complex64:
            reduce_dispatch_and_or<complex64_t>(in, out, reduce_type_, axes_);
            break;
          case float64:
            reduce_dispatch_and_or<double>(in, out, reduce_type_, axes_);
            break;
        }
        break;
      }
      case Reduce::Sum:
      case Reduce::Prod: {
        switch (in.dtype()) {
          case bool_:
          case uint8:
            reduce_dispatch_sum_prod<uint8_t>(in, out, reduce_type_, axes_);
            break;
          case uint16:
            reduce_dispatch_sum_prod<uint16_t>(in, out, reduce_type_, axes_);
            break;
          case uint32:
            reduce_dispatch_sum_prod<uint32_t>(in, out, reduce_type_, axes_);
            break;
          case uint64:
            reduce_dispatch_sum_prod<uint64_t>(in, out, reduce_type_, axes_);
            break;
          case int8:
            reduce_dispatch_sum_prod<int8_t>(in, out, reduce_type_, axes_);
            break;
          case int16:
            reduce_dispatch_sum_prod<int16_t>(in, out, reduce_type_, axes_);
            break;
          case int32:
            reduce_dispatch_sum_prod<int32_t>(in, out, reduce_type_, axes_);
            break;
          case int64:
            reduce_dispatch_sum_prod<int64_t>(in, out, reduce_type_, axes_);
            break;
          case float16:
            reduce_dispatch_sum_prod<float16_t>(in, out, reduce_type_, axes_);
            break;
          case bfloat16:
            reduce_dispatch_sum_prod<bfloat16_t>(in, out, reduce_type_, axes_);
            break;
          case float32:
            reduce_dispatch_sum_prod<float>(in, out, reduce_type_, axes_);
            break;
          case float64:
            reduce_dispatch_sum_prod<double>(in, out, reduce_type_, axes_);
            break;
          case complex64:
            reduce_dispatch_sum_prod<complex64_t>(in, out, reduce_type_, axes_);
            break;
        }
        break;
      }
      case Reduce::Max:
      case Reduce::Min: {
        switch (in.dtype()) {
          case bool_:
            reduce_dispatch_min_max<bool>(in, out, reduce_type_, axes_);
            break;
          case uint8:
            reduce_dispatch_min_max<uint8_t>(in, out, reduce_type_, axes_);
            break;
          case uint16:
            reduce_dispatch_min_max<uint16_t>(in, out, reduce_type_, axes_);
            break;
          case uint32:
            reduce_dispatch_min_max<uint32_t>(in, out, reduce_type_, axes_);
            break;
          case uint64:
            reduce_dispatch_min_max<uint64_t>(in, out, reduce_type_, axes_);
            break;
          case int8:
            reduce_dispatch_min_max<int8_t>(in, out, reduce_type_, axes_);
            break;
          case int16:
            reduce_dispatch_min_max<int16_t>(in, out, reduce_type_, axes_);
            break;
          case int32:
            reduce_dispatch_min_max<int32_t>(in, out, reduce_type_, axes_);
            break;
          case int64:
            reduce_dispatch_min_max<int64_t>(in, out, reduce_type_, axes_);
            break;
          case float16:
            reduce_dispatch_min_max<float16_t>(in, out, reduce_type_, axes_);
            break;
          case float32:
            reduce_dispatch_min_max<float>(in, out, reduce_type_, axes_);
            break;
          case float64:
            reduce_dispatch_min_max<double>(in, out, reduce_type_, axes_);
            break;
          case bfloat16:
            reduce_dispatch_min_max<bfloat16_t>(in, out, reduce_type_, axes_);
            break;
          case complex64:
            reduce_dispatch_min_max<complex64_t>(in, out, reduce_type_, axes_);
            break;
        }
        break;
      }
    }
  });
}

} // namespace mlx::core
