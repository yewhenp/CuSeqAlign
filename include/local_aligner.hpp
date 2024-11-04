#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

#include <defs.hpp>

namespace LocalAlign {

    inline std::tuple<ScoreMatrix, TraceMatrix> initialize_score_trace(size_t M, size_t N) {
        ScoreMatrix s_matrix{M + 1, N + 1};
        TraceMatrix t_matrix{M + 1, N + 1};

        for (size_t i = 0; i < M + 1; ++i) {
            s_matrix(i, 0) = 0;
            t_matrix(i, 0) = '*';
        }
        for (size_t i = 0; i < N + 1; ++i) {
            s_matrix(0, i) = 0;
            t_matrix(0, i) = '*';
        }
        return {s_matrix, t_matrix};
    }

    inline std::tuple<ScoreType, size_t, size_t> fill_score_trace(size_t M, size_t N,
                                 const std::vector<char>& M_seq, const std::vector<char>& N_seq,
                                 ScoreMatrix& s_matrix, TraceMatrix& t_matrix,
                                 ScoreType gap_score, ScoreType match_score, ScoreType mismatch_score) {
        ScoreType best_score = -1;
        size_t best_i = 0;
        size_t best_j = 0;

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

                if (s_matrix(i, j) <= 0) {
                    s_matrix(i, j) = 0;
                    t_matrix(i, j) = '*';
                }

                if (s_matrix(i, j) >= best_score) {
                    best_score = s_matrix(i, j);
                    best_i = i;
                    best_j = j;
                }
            }
        }

        return {best_score, best_i, best_j};
    }

    inline std::tuple<std::vector<char>, std::vector<char>> backtrace(size_t M, size_t N,
                                                                      const std::vector<char>& M_seq, const std::vector<char>& N_seq,
                                                                      ScoreMatrix& s_matrix, TraceMatrix& t_matrix,
                                                                      ScoreType best_score,
                                                                      size_t best_i,
                                                                      size_t best_j) {
        std::vector<char> M_seq_out_rev{};
        std::vector<char> N_seq_out_rev{};

        size_t i = best_i;
        size_t j = best_j;

        char trace = t_matrix(i, j);

        while (trace != '*') {
            if (trace == 'D') {
                M_seq_out_rev.push_back(M_seq.at(i - 1));
                N_seq_out_rev.push_back(N_seq.at(j - 1));
                i--;
                j--;
            } else if (trace == 'H') {
                M_seq_out_rev.push_back(M_seq.at(i - 1));
                N_seq_out_rev.push_back('-');
                i--;
            } else {
                M_seq_out_rev.push_back('-');
                N_seq_out_rev.push_back(N_seq.at(j - 1));
                j--;
            }

            trace = t_matrix(i, j);
        }

        std::reverse(M_seq_out_rev.begin(), M_seq_out_rev.end());
        std::reverse(N_seq_out_rev.begin(), N_seq_out_rev.end());

        return {M_seq_out_rev, N_seq_out_rev};
    }

}

inline std::tuple<std::vector<char>, std::vector<char>> align_local(const std::vector<char>& M_seq,
                                                                    const std::vector<char>& N_seq,
                                                                    ScoreType gap_score,
                                                                    ScoreType match_score,
                                                                    ScoreType mismatch_score) {
    size_t M = M_seq.size();
    size_t N = N_seq.size();

    auto [s_matrix, t_matrix] = LocalAlign::initialize_score_trace(M, N);

    auto [best_score, best_i, best_j] = LocalAlign::fill_score_trace(M, N, M_seq, N_seq, s_matrix, t_matrix,
                                                                     gap_score, match_score, mismatch_score);

    return LocalAlign::backtrace(M, N, M_seq, N_seq, s_matrix, t_matrix, best_score, best_i, best_j);
}