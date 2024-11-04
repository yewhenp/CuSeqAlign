#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

#include <defs.hpp>

namespace GlobalAlign{

    inline std::tuple<ScoreMatrix, TraceMatrix> initialize_score_trace(size_t M, size_t N, ScoreType gap_score) {
        ScoreMatrix s_matrix{M + 1, N + 1};
        TraceMatrix t_matrix{M + 1, N + 1};

        for (size_t i = 0; i < M + 1; ++i) {
            s_matrix(i, 0) = i * gap_score;
            t_matrix(i, 0) = 'H';
        }
        for (size_t i = 0; i < N + 1; ++i) {
            s_matrix(0, i) = i * gap_score;
            t_matrix(0, i) = 'V';
        }
        return {s_matrix, t_matrix};
    }

    inline void fill_score_trace(size_t M, size_t N,
                                 const std::vector<char>& M_seq, const std::vector<char>& N_seq,
                                 ScoreMatrix& s_matrix, TraceMatrix& t_matrix,
                                 ScoreType gap_score, ScoreType match_score, ScoreType mismatch_score) {
        for (size_t i = 1; i < M + 1; ++i) {
            for (size_t j = 1; j < N + 1; ++j) {
                auto compare_score = match_score;
                if (M_seq.at(i - 1) != N_seq.at(j - 1)) {
                    compare_score = mismatch_score;
                }
                auto diag_score = s_matrix(i - 1, j - 1) + compare_score;
                auto left_score = s_matrix(i - 1, j) + gap_score;
                auto right_score = s_matrix(i, j - 1) + gap_score;

                if (diag_score >= std::max(left_score, right_score)) {
                    s_matrix(i, j) = diag_score;
                    t_matrix(i, j) = 'D';
                } else if (left_score >= right_score) {
                    s_matrix(i, j) = left_score;
                    t_matrix(i, j) = 'H';
                } else {
                    s_matrix(i, j) = right_score;
                    t_matrix(i, j) = 'V';
                }
            }
        }
    }

    inline std::tuple<std::vector<char>, std::vector<char>> backtrace(size_t M, size_t N,
                                                                      const std::vector<char>& M_seq, const std::vector<char>& N_seq,
                                                                      ScoreMatrix& s_matrix, TraceMatrix& t_matrix) {
        std::vector<char> M_seq_out{};
        std::vector<char> N_seq_out{};

        size_t i = M;
        size_t j = N;

        while (i > 0 || j > 0) {
            auto trace = t_matrix(i, j);
            if (trace == 'D') {
                M_seq_out.push_back(M_seq.at(i - 1));
                N_seq_out.push_back(N_seq.at(j - 1));
                i--;
                j--;
            } else if (trace == 'H') {
                M_seq_out.push_back(M_seq.at(i - 1));
                N_seq_out.push_back('-');
                i--;
            } else if (trace == 'V') {
                M_seq_out.push_back('-');
                N_seq_out.push_back(N_seq.at(j - 1));
                j--;
            }
        }

        std::reverse(M_seq_out.begin(), M_seq_out.end());
        std::reverse(N_seq_out.begin(), N_seq_out.end());

        return {M_seq_out, N_seq_out};
    }

}

inline std::tuple<std::vector<char>, std::vector<char>> align_global(const std::vector<char>& M_seq,
                                                                     const std::vector<char>& N_seq,
                                                                     ScoreType gap_score,
                                                                     ScoreType match_score,
                                                                     ScoreType mismatch_score) {
    size_t M = M_seq.size();
    size_t N = N_seq.size();

    auto [s_matrix, t_matrix] = GlobalAlign::initialize_score_trace(M, N, gap_score);

    GlobalAlign::fill_score_trace(M, N, M_seq, N_seq, s_matrix, t_matrix, gap_score, match_score, mismatch_score);

    return GlobalAlign::backtrace(M, N, M_seq, N_seq, s_matrix, t_matrix);
}
