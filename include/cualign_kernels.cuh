#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template<int NThreads, typename SeqType, typename SType, typename TType>
__global__ void base_align_kern(size_t M, size_t N,
                                const SeqType* __restrict__ const M_seq, const SeqType* __restrict__ const N_seq,
                                SType * __restrict__ s_matrix_ptr, TType * __restrict__ t_matrix_ptr,
                                SType gap_score, SType match_score, SType mismatch_score, size_t step) {

    auto x_pos_thrd = static_cast<int>(threadIdx.x);
    auto y_pos_thrd = -static_cast<int>(threadIdx.x);
//    auto s_matrix = make_tensor(make_gmem_ptr(s_matrix_ptr), make_shape(M + 1, N + 1));
//    auto t_matrix = make_tensor(make_gmem_ptr(t_matrix_ptr), make_shape(M + 1, N + 1));

#define s_matrix(xpos, ypos) (s_matrix_ptr[(ypos) * (M+1) + xpos])
#define t_matrix(xpos, ypos) (t_matrix_ptr[(ypos) * (M+1) + xpos])

#pragma unroll
    for (int q = 0; q < 2 * NThreads; q++) {
        __syncthreads();

        if (0 <= y_pos_thrd && y_pos_thrd < NThreads) {

            auto x_pos = blockIdx.x * NThreads + x_pos_thrd;
            auto y_pos = step * NThreads - blockIdx.x * NThreads + y_pos_thrd;

            if ((x_pos >= M) || (y_pos >= N)) {
                continue;
            }

            auto compare_score = match_score;
            if (M_seq[x_pos] != N_seq[y_pos]) {
                compare_score = mismatch_score;
            }

            auto diag_score = s_matrix(x_pos, y_pos) + compare_score;
            auto left_score = s_matrix(x_pos, y_pos + 1) + gap_score;
            auto right_score = s_matrix(x_pos + 1, y_pos) + gap_score;

            if (diag_score > max(left_score, right_score)) {
                s_matrix(x_pos + 1, y_pos + 1) = diag_score;
                t_matrix(x_pos + 1, y_pos + 1) = 'D';
            } else if (left_score > right_score) {
                s_matrix(x_pos + 1, y_pos + 1) = left_score;
                t_matrix(x_pos + 1, y_pos + 1) = 'H';
            } else {
                s_matrix(x_pos + 1, y_pos + 1) = right_score;
                t_matrix(x_pos + 1, y_pos + 1) = 'V';
            }
        }

        y_pos_thrd++;
    }

}

