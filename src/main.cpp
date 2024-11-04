#include <iostream>
#include <global_aligner.hpp>
#include <local_aligner.hpp>
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
//    std::string M_seq_str = "MKVAGWFLA";
//    std::string N_seq_str = "TGKWAGWFQ";

    size_t N = 8000;
    size_t M = 8000;
    std::string M_seq_str = generate_random_string(N);
    std::string N_seq_str = generate_random_string(M);

    ScoreType gap_score = -2;
    ScoreType match_score = 1;
    ScoreType mismatch_score = -1;

    std::vector<char> M_seq(M_seq_str.begin(), M_seq_str.end());
    std::vector<char> N_seq(N_seq_str.begin(), N_seq_str.end());

    {

        auto begin = std::chrono::steady_clock::now();
        auto [M_seq_out, N_seq_out] = align_global(M_seq, N_seq, gap_score, match_score, mismatch_score);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Global alignment " << "(time spent = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]):" << std::endl;

        for (int i = 0; i < M_seq_out.size(); ++i) {
            std::cout << M_seq_out.at(i) << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < N_seq_out.size(); ++i) {
            std::cout << N_seq_out.at(i) << " ";
        }
        std::cout << std::endl;
    }
    {
        auto begin = std::chrono::steady_clock::now();
        auto [M_seq_out, N_seq_out] = align_local(M_seq, N_seq, gap_score, match_score, mismatch_score);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Local alignment " << "(time spent = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " [seconds]):" << std::endl;

        for (int i = 0; i < M_seq_out.size(); ++i) {
            std::cout << M_seq_out.at(i) << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < N_seq_out.size(); ++i) {
            std::cout << N_seq_out.at(i) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
