#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <tuple>
#include <fasta_io.hpp>
#include <global_aligner.hpp>
#include <local_aligner.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cualign_kernels.cuh>

class CudaThreadPooledAligner {
public:
    CudaThreadPooledAligner(size_t numThreads_, size_t maxM, size_t maxN) : numThreads(numThreads_) {
        streams.resize(numThreads);

        for (int i = 0; i < numThreads; i++) {
            ScoreType* s_matrix_d;
            char* t_matrix_d;
            char* M_seq_d;
            char* N_seq_d;
            char* M_seq_d_out;
            char* N_seq_d_out;

            cudaMalloc(&s_matrix_d, (maxM + 1) * (maxN + 1) * sizeof(ScoreType));
            cudaMalloc(&t_matrix_d, (maxM + 1) * (maxN + 1) * sizeof(char));
            cudaMalloc(&M_seq_d, (maxM + 1) * sizeof(char));
            cudaMalloc(&N_seq_d, (maxN + 1) * sizeof(char));
            cudaMalloc(&M_seq_d_out, (maxM + 1 + maxN + 1) * sizeof(char));
            cudaMalloc(&N_seq_d_out, (maxM + 1 + maxN + 1) * sizeof(char));

            s_matrix_d_s.push_back(s_matrix_d);
            t_matrix_d_s.push_back(t_matrix_d);
            M_seq_d_s.push_back(M_seq_d);
            N_seq_d_s.push_back(N_seq_d);
            M_seq_d_out_s.push_back(M_seq_d_out);
            N_seq_d_out_s.push_back(N_seq_d_out);
        }

        for (size_t i = 0; i < numThreads; ++i) {
            cudaStreamCreate(&streams[i]);
            workers.emplace_back([this, i]() {
                cudaStream_t stream = streams[i];
                ScoreType* s_matrix_d = s_matrix_d_s[i];
                char* t_matrix_d = t_matrix_d_s[i];
                char* M_seq_d = M_seq_d_s[i];
                char* N_seq_d = N_seq_d_s[i];
                char* M_seq_d_out = M_seq_d_out_s[i];
                char* N_seq_d_out = N_seq_d_out_s[i];
                while (true) {
                    std::function<void(cudaStream_t, ScoreType*, char*, char*, char*, char*, char*)> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });

                        if (stop && tasks.empty())
                            return;

                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task(stream, s_matrix_d, t_matrix_d, M_seq_d, N_seq_d, M_seq_d_out, N_seq_d_out);
                    cudaStreamSynchronize(stream);
                }
            });
        }
    }

    ~CudaThreadPooledAligner() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (auto &worker : workers) {
            worker.join();
        }
        for (auto &stream : streams) {
            cudaStreamDestroy(stream);
        }

        for (int i = 0; i < numThreads; i++) {
            cudaFree(s_matrix_d_s[i]);
            cudaFree(t_matrix_d_s[i]);
            cudaFree(M_seq_d_s[i]);
            cudaFree(N_seq_d_s[i]);
            cudaFree(M_seq_d_out_s[i]);
            cudaFree(M_seq_d_out_s[i]);
        }
    }

    void enqueue(std::function<void(cudaStream_t, ScoreType*, char*, char*, char*, char*, char*)> task) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::move(task));
        }
        condition.notify_one();
    }

    std::vector<std::thread> workers;
    std::vector<cudaStream_t> streams;
    std::queue<std::function<void(cudaStream_t, ScoreType*, char*, char*, char*, char*, char*)>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop = false;
    size_t numThreads;

    std::vector<ScoreType*> s_matrix_d_s;
    std::vector<char*> t_matrix_d_s;
    std::vector<char*> M_seq_d_s;
    std::vector<char*> N_seq_d_s;
    std::vector<char*> M_seq_d_out_s;
    std::vector<char*> N_seq_d_out_s;
};

class CuAligner {
public:
    ScoreType _gap_score = 0;
    ScoreType _match_score = 1;
    ScoreType _mismatch_score = -1;

    CuAligner(ScoreType gap_score, ScoreType match_score, ScoreType mismatch_score) : _gap_score{gap_score}, _match_score{match_score}, _mismatch_score{mismatch_score} {}

    inline Alignments align(const FastaSeqs& targets, const FastaSeqs& queries) {
        Alignments algns{};

        constexpr int n_workers = 8;  // TODO: tune

        if (targets.size() != queries.size()) {
            std::cerr << "Error: Mismatch in targets-queries size" << std::endl;
            return algns;
        }

        size_t maxM = 0;
        size_t maxN = 0;

        for (int i = 0; i < targets.size(); ++i) {
            if (maxM < targets.at(i)._seq.size()) maxM = targets.at(i)._seq.size();
            if (maxN < queries.at(i)._seq.size()) maxN = queries.at(i)._seq.size();
        }

        CudaThreadPooledAligner pool {n_workers, maxM, maxN};
        algns.resize(targets.size());

        for (int i = 0; i < targets.size(); ++i) {

            pool.enqueue([i, &targets, &queries, &algns, this](cudaStream_t stream, ScoreType* s_matrix_d, char* t_matrix_d,
                                                               char* M_seq_d, char* N_seq_d, char* M_seq_d_out, char* N_seq_d_out) {
                const auto& M_seq = targets.at(i)._seq;
                const auto& M_seq_id = targets.at(i)._id;
                const auto& N_seq = queries.at(i)._seq;
                const auto& N_seq_id = queries.at(i)._id;

                const auto M = M_seq.size();
                const auto N = N_seq.size();

                constexpr int blocksize = 128;  // TODO: tune
                const int nblocks = std::ceil(static_cast<float>(std::max(M + 1, N + 1)) / static_cast<float>(blocksize));

                init_trace_matrices_kern<blocksize><<<nblocks, blocksize, 0, stream>>>(M, N, s_matrix_d, t_matrix_d, _gap_score);

                cudaMemcpyAsync(M_seq_d, M_seq.data(), (M + 1) * sizeof(char), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(N_seq_d, N_seq.data(), (N + 1) * sizeof(char), cudaMemcpyHostToDevice, stream);


                for (int j = 0; j < 2 * nblocks + 1; j++) {
                    base_align_kern<blocksize><<<1 + j, blocksize, 0, stream>>>(M, N, M_seq_d, N_seq_d,
                                                                     s_matrix_d, t_matrix_d,
                                                                     _gap_score, _match_score, _mismatch_score, j);
                }

                traceback_kern<<<1,1,0,stream>>>(M, N, M_seq_d, N_seq_d,
                                                 s_matrix_d, t_matrix_d,
                                                 M_seq_d_out, N_seq_d_out);

                std::vector<char> M_seq_out, N_seq_out;
                M_seq_out.resize(M + 1 + N + 1);
                N_seq_out.resize(M + 1 + N + 1);

                cudaMemcpyAsync(M_seq_out.data(), M_seq_d_out, (M + 1 + N + 1) * sizeof(char), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(N_seq_out.data(), N_seq_d_out, (M + 1 + N + 1) * sizeof(char), cudaMemcpyDeviceToHost, stream);

                cudaStreamSynchronize(stream);

                const std::string M_seq_out_str = {std::make_move_iterator(M_seq_out.begin()), std::make_move_iterator(M_seq_out.end())};
                const std::string N_seq_out_str = {std::make_move_iterator(N_seq_out.begin()), std::make_move_iterator(N_seq_out.end())};

                FastaSeq target_res{M_seq_id, M_seq_out_str};
                FastaSeq query_res{N_seq_id, N_seq_out_str};
                algns[i] = Alignment{std::move(target_res), std::move(query_res)};

            });

        }

        return algns;
    }

};
