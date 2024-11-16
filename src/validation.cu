#include <iostream>

#include <aligner.hpp>
#include <fasta_io.hpp>
#include <comparator.hpp>
#include <cualigner.hpp>
#include <chrono>
#include <thread>

#include <argparse/argparse.hpp>

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("OutAlignerValidate");

    program.add_argument("--gap_score")
            .help("Gap score")
            .default_value(-2)
            .scan<'i', int>();
    program.add_argument("--match_score")
            .help("Match score")
            .default_value(1)
            .scan<'i', int>();
    program.add_argument("--mismatch_score")
            .help("Mismatch score")
            .default_value(-1)
            .scan<'i', int>();

    program.add_argument("-t", "--target")
            .default_value(std::string(""))
            .required()
            .help("Targets file .fasta");
    program.add_argument("-q", "--query")
            .default_value(std::string(""))
            .required()
            .help("Query file .fasta");
    program.add_argument("-r", "--reference")
            .default_value(std::string(""))
            .required()
            .help("Reference file .fasta");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

//    auto aligner = GlobalAligner{program.get<int>("gap_score"), program.get<int>("match_score"), program.get<int>("mismatch_score")};
    auto aligner = CuAligner{program.get<int>("gap_score"), program.get<int>("match_score"), program.get<int>("mismatch_score")};

    auto begin_read = std::chrono::steady_clock::now();
    auto targets = FastaSeq::read_fasta_seqs(program.get<std::string>("target"));
    auto queries = FastaSeq::read_fasta_seqs(program.get<std::string>("query"));
    auto reference = FastaSeq::read_alignments(program.get<std::string>("reference"));
    auto end_read = std::chrono::steady_clock::now();

    auto begin = std::chrono::steady_clock::now();
    auto alignments = aligner.align(targets, queries);
    auto end = std::chrono::steady_clock::now();

    std::cout << "(time spent read = " << std::chrono::duration_cast<std::chrono::microseconds>(end_read - begin_read).count() / 1000000.0 << " [seconds]) " <<
                 "(time spent align = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]) " <<
                 " total accuracy = " <<compare_alignment_accuracy(reference, alignments) << std::endl << std::endl;

    return 0;
}
