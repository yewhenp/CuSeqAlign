#pragma once

#include <vector>
#include <tuple>
#include <fasta_io.hpp>
#include <global_aligner.hpp>
#include <local_aligner.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cualign_kernels.cuh>

class CuAligner {
public:
    ScoreType _gap_score = 0;
    ScoreType _match_score = 1;
    ScoreType _mismatch_score = -1;

    CuAligner(ScoreType gap_score, ScoreType match_score, ScoreType mismatch_score) : _gap_score{gap_score}, _match_score{match_score}, _mismatch_score{mismatch_score} {}

    inline Alignments align(const FastaSeqs& targets, const FastaSeqs& queries) {
        Alignments algns{};

        if (targets.size() != queries.size()) {
            std::cerr << "Error: Mismatch in targets-queries size" << std::endl;
            return algns;
        }

        for (int i = 0; i < targets.size(); ++i) {
            const auto& M_seq = targets.at(i)._seq;
            const auto& M_seq_id = targets.at(i)._id;
            const auto& N_seq = queries.at(i)._seq;
            const auto& N_seq_id = queries.at(i)._id;
            
            const auto M = M_seq.size();
            const auto N = N_seq.size();

            thrust::host_vector<char> M_seq_h(M_seq.data(), M_seq.data() + M);
            thrust::device_vector<char> M_seq_d = M_seq_h;
            thrust::host_vector<char> N_seq_h(N_seq.data(), N_seq.data() + N);
            thrust::device_vector<char> N_seq_d = N_seq_h;

            auto [s_matrix_h, t_matrix_h] = GlobalAlign::initialize_score_trace(M, N, _gap_score);

            ScoreType* s_matrix_d;
            char* t_matrix_d;
            cudaMalloc(&s_matrix_d, (M + 1) * (N + 1) * sizeof(ScoreType));
            cudaMalloc(&t_matrix_d, (M + 1) * (N + 1) * sizeof(char));
            cudaMemcpy(s_matrix_d, s_matrix_h.data(), (M + 1) * (N + 1) * sizeof(ScoreType), cudaMemcpyHostToDevice);
            cudaMemcpy(t_matrix_d, t_matrix_h.data(), (M + 1) * (N + 1) * sizeof(char), cudaMemcpyHostToDevice);

            constexpr int blocksize = 32;
            const int nblocks = std::ceil(static_cast<float>(std::max(M + 1, N + 1)) / static_cast<float>(blocksize));
            for (int j = 0; j < 2 * nblocks + 1; j++) {
                base_align_kern<blocksize><<<1 + j, blocksize>>>(M, N, M_seq_d.data().get(), N_seq_d.data().get(),
                                                             s_matrix_d, t_matrix_d,
                                                             _gap_score, _match_score, _mismatch_score, j);
            }

            cudaDeviceSynchronize();
            cudaMemcpy(s_matrix_h.data(), s_matrix_d, (M + 1) * (N + 1) * sizeof(ScoreType), cudaMemcpyDeviceToHost);
            cudaMemcpy(t_matrix_h.data(), t_matrix_d, (M + 1) * (N + 1) * sizeof(char), cudaMemcpyDeviceToHost);

            auto [M_seq_out, N_seq_out] = GlobalAlign::backtrace(M, N, M_seq, N_seq, s_matrix_h, t_matrix_h);

            const std::string M_seq_out_str = {std::make_move_iterator(M_seq_out.begin()), std::make_move_iterator(M_seq_out.end())};
            const std::string N_seq_out_str = {std::make_move_iterator(N_seq_out.begin()), std::make_move_iterator(N_seq_out.end())};

            FastaSeq target_res{M_seq_id, M_seq_out_str};
            FastaSeq query_res{N_seq_id, N_seq_out_str};
            algns.emplace_back(std::move(target_res), std::move(query_res));
        }

        return algns;
    }

};
