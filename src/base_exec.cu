#include <iostream>

#include <aligner.hpp>
#include <fasta_io.hpp>
#include <cualigner.hpp>
#include <chrono>
#include <thread>
#include <random>

std::string generate_random_string(size_t length) {
    const std::string characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, characters.size() - 1);

    std::string randomString;
    for (size_t i = 0; i < length; ++i) {
        randomString += characters[dist(gen)];
    }

    return randomString;
}

int main() {
    std::string M_seq_str = "TCACGCCTGTAATTCCAAAAAAAAAAAAAAAATCACG";
    FastaSeqs targets = {{M_seq_str}};

    std::string N_seq_str = "TTAATTTGTTGAAAAAAAAAAAAAAAAAAAAATTAAA";
    FastaSeqs queries = {{N_seq_str}};

    ScoreType gap_score = -1;
    ScoreType match_score = 2;
    ScoreType mismatch_score = -1;

    {
        auto begin = std::chrono::steady_clock::now();
        auto aligner = GlobalAligner{gap_score, match_score, mismatch_score};
        auto alignments = aligner.align(targets, queries);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Host Global alignment " << "(time spent = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]):" << std::endl;
        for (const auto &alignment: alignments) {
            FastaSeq::print_alignment(alignment);
        }
    }

    {
        auto begin = std::chrono::steady_clock::now();
        auto aligner = CuAligner{gap_score, match_score, mismatch_score};
        auto alignments = aligner.align(targets, queries);
        auto end = std::chrono::steady_clock::now();

        std::cout << "GPU Global alignment " << "(time spent = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]):" << std::endl;
        for (const auto &alignment: alignments) {
            FastaSeq::print_alignment(alignment);
        }
    }
//
//    {
//        auto begin = std::chrono::steady_clock::now();
//        auto aligner = LocalAligner{gap_score, match_score, mismatch_score};
//        auto alignments = aligner.align(targets, queries);
//        auto end = std::chrono::steady_clock::now();
//
//        std::cout << "Local alignment " << "(time spent = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]):" << std::endl;
//        for (const auto &alignment: alignments) {
//            FastaSeq::print_alignment(alignment);
//        }
//    }

    return 0;
}