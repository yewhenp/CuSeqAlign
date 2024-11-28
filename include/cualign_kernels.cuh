#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template<int NThreads, typename SeqType, typename SType, typename TType>
__global__ void base_align_kern(size_t M, size_t N,
                                const SeqType* __restrict__ const M_seq, const SeqType* __restrict__ const N_seq,
                                SType * __restrict__ s_matrix_ptr, TType * __restrict__ t_matrix_ptr,
                                SType gap_score, SType match_score, SType mismatch_score, size_t step) {

    __shared__ __align__(16) SType shmem[(NThreads + 1) * (NThreads + 1)];

    auto x_pos_thrd = static_cast<int>(threadIdx.x);
    auto y_pos_thrd = -static_cast<int>(threadIdx.x);

    auto x_pos = blockIdx.x * NThreads + x_pos_thrd;
    auto y_pos = step * NThreads - blockIdx.x * NThreads + y_pos_thrd;

    if (step * NThreads - blockIdx.x * NThreads + 2 * NThreads < 0) return;

    auto s_matrix = make_tensor(make_gmem_ptr(s_matrix_ptr), make_layout(make_shape(M + 1, N + 1), make_shape(1, M + 1)));
    auto t_matrix = make_tensor(make_gmem_ptr(t_matrix_ptr), make_layout(make_shape(M + 1, N + 1), make_shape(1, M + 1)));

    auto shmem_matrix = make_tensor(make_smem_ptr(shmem), make_layout(make_shape(NThreads + 1, NThreads + 1), make_shape(1, NThreads + 1)));

    if (x_pos <= M && step * NThreads - blockIdx.x * NThreads >= 0 && step * NThreads - blockIdx.x * NThreads <= N)
        shmem_matrix(threadIdx.x, 0) = s_matrix(x_pos, step * NThreads - blockIdx.x * NThreads);

    if (step * NThreads - blockIdx.x * NThreads + threadIdx.x >= 0 && step * NThreads - blockIdx.x * NThreads + threadIdx.x <= N)
        shmem_matrix(0, threadIdx.x) = s_matrix(blockIdx.x * NThreads, step * NThreads - blockIdx.x * NThreads + threadIdx.x);

    if (threadIdx.x == 0) {
        if (blockIdx.x * NThreads + NThreads <= M && step * NThreads - blockIdx.x * NThreads >= 0 && step * NThreads - blockIdx.x * NThreads <= N)
            shmem_matrix(NThreads, 0) = s_matrix(blockIdx.x * NThreads + NThreads, step * NThreads - blockIdx.x * NThreads);

        if (step * NThreads - blockIdx.x * NThreads + NThreads >= 0 && step * NThreads - blockIdx.x * NThreads + NThreads <= N)
            shmem_matrix(0, NThreads) = s_matrix(blockIdx.x * NThreads, step * NThreads - blockIdx.x * NThreads + NThreads);
    }

    if (x_pos >= M) return;

#pragma unroll
    for (int q = 0; q < 2 * NThreads; q++) {
        __syncthreads();

        if (0 <= y_pos_thrd && y_pos_thrd < NThreads) {
            y_pos = step * NThreads - blockIdx.x * NThreads + y_pos_thrd;

            if (y_pos >= N) {
                return;
            }

            auto compare_score = match_score;
            if (M_seq[x_pos] != N_seq[y_pos]) {
                compare_score = mismatch_score;
            }

            auto diag_score = shmem_matrix(threadIdx.x, y_pos_thrd) + compare_score;
            auto left_score = shmem_matrix(threadIdx.x, y_pos_thrd + 1) + gap_score;
            auto right_score = shmem_matrix(threadIdx.x + 1, y_pos_thrd) + gap_score;

            if (diag_score > max(left_score, right_score)) {
                s_matrix(x_pos + 1, y_pos + 1) = diag_score;
                t_matrix(x_pos + 1, y_pos + 1) = 'D';
                shmem_matrix(threadIdx.x + 1, y_pos_thrd + 1) = diag_score;
            } else if (left_score > right_score) {
                s_matrix(x_pos + 1, y_pos + 1) = left_score;
                t_matrix(x_pos + 1, y_pos + 1) = 'H';
                shmem_matrix(threadIdx.x + 1, y_pos_thrd + 1) = left_score;
            } else {
                s_matrix(x_pos + 1, y_pos + 1) = right_score;
                t_matrix(x_pos + 1, y_pos + 1) = 'V';
                shmem_matrix(threadIdx.x + 1, y_pos_thrd + 1) = right_score;
            }
        }

        y_pos_thrd++;
    }

}


template<int NThreads, typename SType, typename TType>
__global__ void init_trace_matrices_kern(size_t M, size_t N,
                                    SType * __restrict__ s_matrix_ptr, TType * __restrict__ t_matrix_ptr,
                                    SType gap_score) {

    auto coord_thrd = blockIdx.x * NThreads + threadIdx.x;

    auto s_matrix = make_tensor(make_gmem_ptr(s_matrix_ptr), make_layout(make_shape(M + 1, N + 1), make_shape(1, M + 1)));
    auto t_matrix = make_tensor(make_gmem_ptr(t_matrix_ptr), make_layout(make_shape(M + 1, N + 1), make_shape(1, M + 1)));

    if (coord_thrd <= M) {
        s_matrix(coord_thrd, 0) = gap_score * coord_thrd;
        t_matrix(coord_thrd, 0) = 'H';
    }

    if (coord_thrd <= N) {
        s_matrix(0, coord_thrd) = gap_score * coord_thrd;
        t_matrix(0, coord_thrd) = 'V';
    }
}

template<typename SeqType, typename SType, typename TType>
__global__ void traceback_kern(size_t M, size_t N,
                               const SeqType* __restrict__ const M_seq, const SeqType* __restrict__ const N_seq,
                               SType * __restrict__ s_matrix_ptr, TType * __restrict__ t_matrix_ptr,
                               SeqType* __restrict__ M_seq_out, SeqType* __restrict__ N_seq_out) {

    if (thread0()) {
        size_t out_index = 0;
        size_t i = M;
        size_t j = N;

        auto s_matrix = make_tensor(make_gmem_ptr(s_matrix_ptr), make_layout(make_shape(M + 1, N + 1), make_shape(1, M + 1)));
        auto t_matrix = make_tensor(make_gmem_ptr(t_matrix_ptr), make_layout(make_shape(M + 1, N + 1), make_shape(1, M + 1)));

        while (i > 0 || j > 0) {
            auto trace = t_matrix(i, j);
            if (trace == 'D') {
                M_seq_out[out_index] = M_seq[i - 1];
                N_seq_out[out_index] = N_seq[j - 1];
                i--;
                j--;
            } else if (trace == 'H') {
                M_seq_out[out_index] = M_seq[i - 1];
                N_seq_out[out_index] = '-';
                i--;
            } else if (trace == 'V') {
                M_seq_out[out_index] = '-';
                N_seq_out[out_index] = N_seq[j - 1];
                j--;
            }

            out_index++;
        }

        M_seq_out[out_index] = '\0';
        N_seq_out[out_index] = '\0';

//        out_index++;

        for (size_t q = 0; q < (out_index - (out_index % 2)) / 2; q++) {
            auto tmp_M = M_seq_out[q];
            M_seq_out[q] = M_seq_out[out_index - q - 1];
            M_seq_out[out_index - q - 1] = tmp_M;

            auto tmp_N = N_seq_out[q];
            N_seq_out[q] = N_seq_out[out_index - q - 1];
            N_seq_out[out_index - q - 1] = tmp_N;
        }
    }

}

