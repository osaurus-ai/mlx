// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/lapack.h"
#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "mlx/backend/cpu/highway_info.h"
#include "mlx/backend/cpu/threading/common.h"
#endif // MLX_USE_HIGHWAY_KERNELS

namespace mlx::core {

#if defined(MLX_USE_HIGHWAY_KERNELS)
// Minimum batches per thread before parallelizing batch matmul
// Higher threshold than unary/binary ops because BLAS calls are heavier
constexpr int MIN_BATCHES_PER_THREAD = 4;

#endif // MLX_USE_HIGHWAY_KERNELS
template <>
void matmul<float>(
    const float* a,
    const float* b,
    float* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

#if !defined(MLX_USE_HIGHWAY_KERNELS)
  for (int i = 0; i < batch_size; ++i) {
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        alpha,
        a + elem_to_loc(M * K * i, a_shape, a_strides),
        lda,
        b + elem_to_loc(K * N * i, b_shape, b_strides),
        ldb,
        beta,
        out + M * N * i,
        ldc);
#else
  // Check if parallelization over batches is beneficial
  auto& pool = cpu::ThreadPool::instance();
  int n_threads = std::min(
      pool.max_threads(),
      static_cast<int>(batch_size / MIN_BATCHES_PER_THREAD));

  if (n_threads > 1) {
    // Parallel path: each thread handles a chunk of batches
    // Note: For many small batches, BLAS threading has minimal benefit,
    // so we parallelize ourselves without adjusting BLAS thread count.
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (batch_size + nth - 1) / nth;
      size_t start = chunk * tid;
      size_t end = std::min(start + chunk, batch_size);
      for (size_t i = start; i < end; ++i) {
        cblas_sgemm(
            CblasRowMajor,
            a_transposed ? CblasTrans : CblasNoTrans,
            b_transposed ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            alpha,
            a + elem_to_loc(M * K * i, a_shape, a_strides),
            lda,
            b + elem_to_loc(K * N * i, b_shape, b_strides),
            ldb,
            beta,
            out + M * N * i,
            ldc);
      }
    });
  } else {
    // Sequential batch path. When BLAS is pinned to 1 thread, parallelize
    // large GEMMs by splitting M rows across our thread pool.
    for (size_t i = 0; i < batch_size; ++i) {
      const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
      const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
      auto* out_ptr = out + M * N * i;

      int m_threads = 1;
      if (M >= 16 && M * N * K >= 65536) {
        m_threads =
            std::min(pool.max_threads(), std::max(1, static_cast<int>(M / 8)));
      }

      // Short inputs (M < 16) split by columns, so that a pinned OpenBLAS never
      // leaves a large GEMM on one core. Each slice writes its own columns.
      int col_threads = 1;
      if (m_threads == 1 && M * N * K >= 65536) {
        col_threads =
            std::min(pool.max_threads(), std::max(1, static_cast<int>(N / 64)));
      }
      if (col_threads > 1) {
        cpu::highway_info::record_sgemm_column_split();
        pool.parallel_for(col_threads, [&](int tid, int nth) {
          size_t n_chunk = (N + nth - 1) / nth;
          size_t n_start = n_chunk * tid;
          size_t n_end = std::min(n_start + n_chunk, N);
          if (n_start < n_end) {
            // Column n of op(B) is row n of a transposed B (stored N x K), and
            // column n of B otherwise (stored K x N).
            size_t b_offset = b_transposed ? n_start * ldb : n_start;
            cblas_sgemm(
                CblasRowMajor,
                a_transposed ? CblasTrans : CblasNoTrans,
                b_transposed ? CblasTrans : CblasNoTrans,
                M,
                n_end - n_start,
                K,
                alpha,
                a_ptr,
                lda,
                b_ptr + b_offset,
                ldb,
                beta,
                out_ptr + n_start,
                ldc);
          }
        });
        continue;
      }

      if (m_threads > 1) {
        pool.parallel_for(m_threads, [&](int tid, int nth) {
          size_t m_chunk = (M + nth - 1) / nth;
          size_t m_start = m_chunk * tid;
          size_t m_end = std::min(m_start + m_chunk, M);
          if (m_start < m_end) {
            // When a_transposed: A stored as KxM (row-major), op(A)=A^T is MxK.
            // Splitting M rows of A^T = splitting columns of stored A.
            // Column offset = m_start; row offset = m_start * lda.
            size_t a_offset = a_transposed ? m_start : m_start * lda;
            cblas_sgemm(
                CblasRowMajor,
                a_transposed ? CblasTrans : CblasNoTrans,
                b_transposed ? CblasTrans : CblasNoTrans,
                m_end - m_start,
                N,
                K,
                alpha,
                a_ptr + a_offset,
                lda,
                b_ptr,
                ldb,
                beta,
                out_ptr + m_start * ldc,
                ldc);
          }
        });
      } else {
        cblas_sgemm(
            CblasRowMajor,
            a_transposed ? CblasTrans : CblasNoTrans,
            b_transposed ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            alpha,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            beta,
            out_ptr,
            ldc);
      }
    }
#endif // MLX_USE_HIGHWAY_KERNELS
  }
}

template <>
void matmul<double>(
    const double* a,
    const double* b,
    double* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

#if !defined(MLX_USE_HIGHWAY_KERNELS)
  for (int i = 0; i < batch_size; ++i) {
    cblas_dgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        alpha,
        a + elem_to_loc(M * K * i, a_shape, a_strides),
        lda,
        b + elem_to_loc(K * N * i, b_shape, b_strides),
        ldb,
        beta,
        out + M * N * i,
        ldc);
#else
  auto& pool = cpu::ThreadPool::instance();
  int n_threads = std::min(
      pool.max_threads(),
      static_cast<int>(batch_size / MIN_BATCHES_PER_THREAD));

  if (n_threads > 1) {
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (batch_size + nth - 1) / nth;
      size_t start = chunk * tid;
      size_t end = std::min(start + chunk, batch_size);
      for (size_t i = start; i < end; ++i) {
        cblas_dgemm(
            CblasRowMajor,
            a_transposed ? CblasTrans : CblasNoTrans,
            b_transposed ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            alpha,
            a + elem_to_loc(M * K * i, a_shape, a_strides),
            lda,
            b + elem_to_loc(K * N * i, b_shape, b_strides),
            ldb,
            beta,
            out + M * N * i,
            ldc);
      }
    });
  } else {
    for (size_t i = 0; i < batch_size; ++i) {
      const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
      const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
      auto* out_ptr = out + M * N * i;

      int m_threads = 1;
      if (M >= 16 && M * N * K >= 65536) {
        m_threads =
            std::min(pool.max_threads(), std::max(1, static_cast<int>(M / 8)));
      }

      if (m_threads > 1) {
        pool.parallel_for(m_threads, [&](int tid, int nth) {
          size_t m_chunk = (M + nth - 1) / nth;
          size_t m_start = m_chunk * tid;
          size_t m_end = std::min(m_start + m_chunk, M);
          if (m_start < m_end) {
            size_t a_offset = a_transposed ? m_start : m_start * lda;
            cblas_dgemm(
                CblasRowMajor,
                a_transposed ? CblasTrans : CblasNoTrans,
                b_transposed ? CblasTrans : CblasNoTrans,
                m_end - m_start,
                N,
                K,
                alpha,
                a_ptr + a_offset,
                lda,
                b_ptr,
                ldb,
                beta,
                out_ptr + m_start * ldc,
                ldc);
          }
        });
      } else {
        cblas_dgemm(
            CblasRowMajor,
            a_transposed ? CblasTrans : CblasNoTrans,
            b_transposed ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            alpha,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            beta,
            out_ptr,
            ldc);
      }
    }
#endif // MLX_USE_HIGHWAY_KERNELS
  }
}

template <>
void matmul<complex64_t>(
    const complex64_t* a,
    const complex64_t* b,
    complex64_t* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];
  auto calpha = static_cast<complex64_t>(alpha);
  auto cbeta = static_cast<complex64_t>(beta);

#if !defined(MLX_USE_HIGHWAY_KERNELS)
  for (int i = 0; i < batch_size; ++i) {
    cblas_cgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        &calpha,
        a + elem_to_loc(M * K * i, a_shape, a_strides),
        lda,
        b + elem_to_loc(K * N * i, b_shape, b_strides),
        ldb,
        &cbeta,
        out + M * N * i,
        ldc);
#else
  auto& pool = cpu::ThreadPool::instance();
  int n_threads = std::min(
      pool.max_threads(),
      static_cast<int>(batch_size / MIN_BATCHES_PER_THREAD));

  if (n_threads > 1) {
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (batch_size + nth - 1) / nth;
      size_t start = chunk * tid;
      size_t end = std::min(start + chunk, batch_size);
      for (size_t i = start; i < end; ++i) {
        cblas_cgemm(
            CblasRowMajor,
            a_transposed ? CblasTrans : CblasNoTrans,
            b_transposed ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            &calpha,
            a + elem_to_loc(M * K * i, a_shape, a_strides),
            lda,
            b + elem_to_loc(K * N * i, b_shape, b_strides),
            ldb,
            &cbeta,
            out + M * N * i,
            ldc);
      }
    });
  } else {
    for (size_t i = 0; i < batch_size; ++i) {
      const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
      const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
      auto* out_ptr = out + M * N * i;

      int m_threads = 1;
      if (M >= 16 && M * N * K >= 65536) {
        m_threads =
            std::min(pool.max_threads(), std::max(1, static_cast<int>(M / 8)));
      }

      if (m_threads > 1) {
        pool.parallel_for(m_threads, [&](int tid, int nth) {
          size_t m_chunk = (M + nth - 1) / nth;
          size_t m_start = m_chunk * tid;
          size_t m_end = std::min(m_start + m_chunk, M);
          if (m_start < m_end) {
            size_t a_offset = a_transposed ? m_start : m_start * lda;
            cblas_cgemm(
                CblasRowMajor,
                a_transposed ? CblasTrans : CblasNoTrans,
                b_transposed ? CblasTrans : CblasNoTrans,
                m_end - m_start,
                N,
                K,
                &calpha,
                a_ptr + a_offset,
                lda,
                b_ptr,
                ldb,
                &cbeta,
                out_ptr + m_start * ldc,
                ldc);
          }
        });
      } else {
        cblas_cgemm(
            CblasRowMajor,
            a_transposed ? CblasTrans : CblasNoTrans,
            b_transposed ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            &calpha,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            &cbeta,
            out_ptr,
            ldc);
      }
    }
#endif // MLX_USE_HIGHWAY_KERNELS
  }
}

} // namespace mlx::core
