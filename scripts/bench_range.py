import argparse

import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib


N_REPEATS = 5
SEQ_SIZES = [100, 500, 1000, 3000, 5000, 10000, 12000, 15000, 20000, 23000, 26000, 30000]


def run_single_benchmark(seq_size):
    command_line = (os.path.dirname(os.path.abspath(__file__)) + f"/../bin/OutAligner --seq_size {seq_size}").split(" ")

    time_cuda = 0
    time_cpu = 0

    for q in range(N_REPEATS):
        sub_res = subprocess.run(command_line, capture_output=True, text=True)
        outs = sub_res.stdout

        try:
            time_str_cpu = outs.split("\n")[0].split(" ")[-2]
            time_seconds_cpu = float(time_str_cpu)
            time_str_cuda = outs.split("\n")[1].split(" ")[-2]
            time_seconds_cuda = float(time_str_cuda)
        except:
            time_seconds_cpu = 0
            time_seconds_cuda = 0
            print(outs)
            print(sub_res.stderr)

        time_cuda += time_seconds_cuda
        time_cpu += time_seconds_cpu

        print(f"Seq size {seq_size} Iter {q}: time_seconds_cuda = {time_seconds_cuda} time_seconds_cpu = {time_seconds_cpu}")

    time_cuda = time_cuda / N_REPEATS
    time_cpu = time_cpu / N_REPEATS

    print(f"Res(seq_size={seq_size}) = cpu {time_cpu} sec cuda {time_cuda} sec")

    return time_cpu, time_cuda


def main():
    times_cuda = []
    times_cpu = []

    for seq_size in SEQ_SIZES:
        cur_time_cpu, cur_time_cuda = run_single_benchmark(seq_size)
        times_cuda.append(cur_time_cuda)
        times_cpu.append(cur_time_cpu)


    fig, ax = plt.subplots(figsize=(11, 8), tight_layout=True)
    plt.plot(SEQ_SIZES, times_cpu, label="CPU execution")
    plt.plot(SEQ_SIZES, times_cuda, label="CUDA execution")

    ax.set_xlabel('Sequence size')
    ax.set_ylabel('Execution time, seconds')
    plt.legend()
    plt.yscale("log")

    fig.savefig('perf_compare.png', dpi=300)
    plt.cla()


if __name__ == '__main__':
    main()
