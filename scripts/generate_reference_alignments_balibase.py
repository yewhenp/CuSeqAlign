import os

import random
import argparse
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--out_one")
    parser.add_argument("--out_two")
    args = parser.parse_args()

    all_seqs = []

    for filename in os.listdir(args.folder):
        if filename.endswith(".tfa"):
            file_path = os.path.join(args.folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:

                last_name = None
                last_seq = None

                for line in tqdm.tqdm(file.readlines()):
                    if line.startswith(">"):
                        if last_name is not None:
                            all_seqs.append([last_name, last_seq])

                        last_seq = ""
                        last_name = line[1:].strip()
                    else:
                        last_seq += line.strip()

    print("read done")

    random.shuffle(all_seqs)

    print("shuffle done")

    len_key = int(len(all_seqs) / 2)

    out_one = []
    out_two = []

    for i in tqdm.trange(len_key):
        out_one.append(all_seqs[i])
        out_two.append(all_seqs[len_key + i])

    print("split done")

    with open(args.out_one, 'w', encoding='utf-8') as file:
        out_lines = []
        for elem in tqdm.tqdm(out_one):
            enrt = ">" + elem[0] + "\n" + elem[1] + "\n"
            out_lines.append(enrt)
        file.writelines(out_lines)

    with open(args.out_two, 'w', encoding='utf-8') as file:
        out_lines = []
        for elem in tqdm.tqdm(out_two):
            enrt = ">" + elem[0] + "\n" + elem[1] + "\n"
            out_lines.append(enrt)
        file.writelines(out_lines)


if __name__ == '__main__':
    main()
