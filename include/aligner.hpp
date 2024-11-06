#pragma once

#include <vector>
#include <tuple>
#include <fasta_io.hpp>
#include <global_aligner.hpp>
#include <local_aligner.hpp>

template<std::tuple<std::vector<char>, std::vector<char>>(*align_func)(const std::string&,
                                                                       const std::string&,
                                                                       ScoreType,
                                                                       ScoreType,
                                                                       ScoreType)>
class Aligner {
public:
    ScoreType _gap_score = 0;
    ScoreType _match_score = 1;
    ScoreType _mismatch_score = -1;

    Aligner(ScoreType gap_score, ScoreType match_score, ScoreType mismatch_score) : _gap_score{gap_score}, _match_score{match_score}, _mismatch_score{mismatch_score} {}

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

            auto [M_seq_out, N_seq_out] = align_func(M_seq, N_seq, _gap_score, _match_score, _mismatch_score);
            const std::string M_seq_out_str = {std::make_move_iterator(M_seq_out.begin()), std::make_move_iterator(M_seq_out.end())};
            const std::string N_seq_out_str = {std::make_move_iterator(N_seq_out.begin()), std::make_move_iterator(N_seq_out.end())};

            FastaSeq target_res{M_seq_id, M_seq_out_str};
            FastaSeq query_res{N_seq_id, N_seq_out_str};
            algns.emplace_back(std::move(target_res), std::move(query_res));
        }

        return algns;
    }

};

using GlobalAligner = Aligner<align_global>;
using LocalAligner = Aligner<align_local>;