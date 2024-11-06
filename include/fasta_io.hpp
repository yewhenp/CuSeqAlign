#pragma once

#include <vector>
#include <fstream>

class FastaSeq;
using FastaSeqs = std::vector<FastaSeq>;
using Alignment = std::pair<FastaSeq, FastaSeq>;
using Alignments = std::vector<Alignment>;


class FastaSeq {
public:
    std::string _id;
    std::string _seq;

    FastaSeq(const std::string& id, const std::string& seq) : _id{id}, _seq{seq} {}
    FastaSeq(const std::string& seq) : _id{""}, _seq{seq} {}
    FastaSeq() : _id{}, _seq{} {}

    inline std::string to_string() const {
        return ">" + _id + "\n" + _seq + "\n";
    }

    static inline FastaSeqs read_fasta_seqs(const std::string& filepath) {
        std::ifstream file(filepath);
        FastaSeqs sequences;
        std::string line, sequence_id, sequence;

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            return sequences;
        }

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == ';' || line[0] == ' ') continue;

            if (line[0] == '>') {
                if (!sequence_id.empty()) {
                    sequences.emplace_back(sequence_id, sequence);
                }
                sequence_id = line.substr(1);
                sequence.clear();
            } else {
                sequence += line;
            }
        }

        if (!sequence_id.empty()) {
            sequences.emplace_back(sequence_id, sequence);
        }

        file.close();
        return sequences;
    }

    static inline Alignments read_alignments(const std::string& filepath) {
        auto all_aligns = FastaSeq::read_fasta_seqs(filepath);

        Alignments algns;

        FastaSeq temp_1, temp_2;

        for (int i = 0; i < all_aligns.size(); ++i) {
            if (i % 2 == 1) {
                temp_2 = all_aligns.at(i);
                algns.emplace_back(temp_1, temp_2);
            } else {
                temp_1 = all_aligns.at(i);
            }
        }

        return algns;
    }

    static inline void write_fasta_seqs(const std::string& filepath, const FastaSeqs& sequences) {
        std::ofstream file(filepath);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << " for writing." << std::endl;
            return;
        }

        for (const auto &entry : sequences) {
            file << entry.to_string();
        }

        file.close();
    }

    static inline void write_alignments(const std::string& filepath, const Alignments& alignments) {
        FastaSeqs all_seq{};

        for (const auto &entry : alignments) {
            all_seq.push_back(entry.first);
            all_seq.push_back(entry.second);
        }

        FastaSeq::write_fasta_seqs(filepath, all_seq);
    }

    static inline void print_alignment(const Alignment& alignment) {
        std::cout << alignment.first.to_string() << alignment.second.to_string() << std::endl;
    }

};






