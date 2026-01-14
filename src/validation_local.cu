#include <iostream>

#include <aligner.hpp>
#include <fasta_io.hpp>
#include <comparator.hpp>
#include <cualigner.hpp>
#include <cualigner_local.hpp>
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
    program.add_argument("-sktr", "--skip_traceback")
            .help("do traceback")
            .default_value(false)
            .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto aligner_ref = LocalAligner{static_cast<ScoreType>(program.get<int>("gap_score")),
                                     static_cast<ScoreType>(program.get<int>("match_score")),
                                     static_cast<ScoreType>(program.get<int>("mismatch_score"))};
    auto aligner = CuLocalAligner{static_cast<ScoreType>(program.get<int>("gap_score")),
                             static_cast<ScoreType>(program.get<int>("match_score")),
                             static_cast<ScoreType>(program.get<int>("mismatch_score"))};

    auto begin_read = std::chrono::steady_clock::now();
    auto targets = FastaSeq::read_fasta_seqs(program.get<std::string>("target"));
    auto queries = FastaSeq::read_fasta_seqs(program.get<std::string>("query"));
    auto end_read = std::chrono::steady_clock::now();

    bool do_cpu = true;

    Alignments alignments_ref = do_cpu ? aligner_ref.align(targets, queries) : Alignments{};
    std::cout << "Ref alignment done" << std::endl;
    auto alignments = aligner.align(targets, queries, program.get<bool>("skip_traceback"));
    std::cout << "Cuda alignment done" << std::endl;

    auto n_scale = 5;

    auto begin_n_ref = std::chrono::steady_clock::now();
    if (do_cpu) {
        for (int n = 0; n < n_scale; n++) {
            auto al_tmp = aligner_ref.align(targets, queries);
        }
    }
    auto end_n_ref = std::chrono::steady_clock::now();

    auto begin_n = std::chrono::steady_clock::now();
    for (int n = 0; n < n_scale; n++) {
        auto al_tmp = aligner.align(targets, queries, program.get<bool>("skip_traceback"));
    }
    auto end_n = std::chrono::steady_clock::now();

    if (n_scale > 0) {
        std::cout << "(time spent read = " << std::chrono::duration_cast<std::chrono::microseconds>(end_read - begin_read).count() / 1000000.0 << " [seconds]) " <<
                  "(time spent align (n scaled) = " << std::chrono::duration_cast<std::chrono::microseconds>(end_n - begin_n).count() / n_scale / 1000000.0 << " [seconds]) " <<
                  "(time spent align reference (n scaled) = " << std::chrono::duration_cast<std::chrono::microseconds>(end_n_ref - begin_n_ref).count() / n_scale / 1000000.0 << " [seconds]) " <<
                  " total accuracy gpu = " << compare_alignment_accuracy(alignments_ref, alignments) << std::endl << std::endl;
    } else {
        std::cout << " total accuracy gpu = " << compare_alignment_accuracy(alignments_ref, alignments) << std::endl << std::endl;
    }


    return 0;
}
