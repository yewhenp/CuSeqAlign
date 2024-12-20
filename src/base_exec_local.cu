#include <iostream>

#include <aligner.hpp>
#include <fasta_io.hpp>
#include <cualigner.hpp>
#include <cualigner_local.hpp>
#include <chrono>
#include <thread>
#include <random>
#include <comparator.hpp>
#include <argparse/argparse.hpp>

std::string generate_random_string(size_t length) {
//    const std::string characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const std::string characters = "ATGC";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, characters.size() - 1);

    std::string randomString;
    for (size_t i = 0; i < length; ++i) {
        randomString += characters[dist(gen)];
    }

    return randomString;
}

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("OutAlignerGlobal");

    program.add_argument("--seq_size")
            .help("Seq size")
            .default_value(1000)
            .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    cudaSetDevice(0);

    int rand_len = program.get<int>("seq_size");
//    std::string M_seq_str = "TCACGCCTGTAATTCCAAAAAAAAAAAAAAAATCACG";
    std::string M_seq_str = generate_random_string(rand_len);
    FastaSeqs targets = {{M_seq_str}};

//    std::string N_seq_str = "TTAATTTGTTGAAAAAAAAAAAAAAAAAAAAATTAAA";
    std::string N_seq_str = generate_random_string(rand_len);
    FastaSeqs queries = {{N_seq_str}};

    std::cout << "Querry: " << N_seq_str << std::endl;
    std::cout << "Target: " << M_seq_str << std::endl;

    ScoreType gap_score = -1;
    ScoreType match_score = 2;
    ScoreType mismatch_score = -1;

    Alignments reference_al, cuda_al;

    {
        auto begin = std::chrono::steady_clock::now();
        auto aligner = LocalAligner{gap_score, match_score, mismatch_score};
        auto alignments = aligner.align(targets, queries);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Host Local alignment " << "(time spent = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]):" << std::endl;
        for (const auto &alignment: alignments) {
            FastaSeq::print_alignment(alignment);
        }

        reference_al = alignments;
    }

    {
        auto begin = std::chrono::steady_clock::now();
        auto aligner = CuLocalAligner{gap_score, match_score, mismatch_score};
        auto alignments = aligner.align(targets, queries, false);
        auto end = std::chrono::steady_clock::now();

        std::cout << "GPU Local alignment " << "(time spent = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]):" << std::endl;
        for (const auto &alignment: alignments) {
            FastaSeq::print_alignment(alignment);
        }

        cuda_al = alignments;
    }

    std::cout << "Accuracy = " << compare_alignment_accuracy(reference_al, cuda_al) << std::endl;
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
