#pragma once

#include <fasta_io.hpp>

inline double compare_alignment_accuracy(const Alignments& y_true, const Alignments& y_pred) {
    if (y_true.size() != y_pred.size()) {
        std::cerr << "Error: Mismatch in y_true-y_pred size" << std::endl;
        return 0;
    }

    double tot_accuracy = 0.0;

    for (int i = 0; i < y_true.size(); ++i) {
        const auto& ref_alignment = y_true.at(i);
        const auto& pred_alignment = y_pred.at(i);

        const auto& ref_target = ref_alignment.first._seq;
        const auto& ref_query = ref_alignment.second._seq;

        const auto& pred_target = pred_alignment.first._seq;
        const auto& pred_query = pred_alignment.second._seq;

        double corrects_target = 0.0;
        for (int j = 0; j < std::min(ref_target.size(), pred_target.size()); ++j) {
            if (ref_target.at(j) == pred_target.at(j)) corrects_target += 1.0;
        }

        double corrects_query = 0.0;
        for (int j = 0; j < std::min(ref_query.size(), pred_query.size()); ++j) {
            if (ref_query.at(j) == pred_query.at(j)) corrects_query += 1.0;
        }

        if (std::min(ref_target.size(), pred_target.size()) + std::min(ref_query.size(), pred_query.size()) > 0)
            tot_accuracy += (corrects_target + corrects_query) / static_cast<double>(std::min(ref_target.size(), pred_target.size()) + std::min(ref_query.size(), pred_query.size()));
    }
    return tot_accuracy / static_cast<double>(y_true.size());
}