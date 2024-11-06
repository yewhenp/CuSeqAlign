from Bio import Align
from Bio import SeqIO
import argparse
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file")
    parser.add_argument("--target_file")
    parser.add_argument("--output_file", default="data/reference_alignment.fasta")
    args = parser.parse_args()

    queries = []
    targets = []

    for record in SeqIO.parse(args.query_file, "fasta"):
        queries.append(record)

    for record in SeqIO.parse(args.target_file, "fasta"):
        targets.append(record)

    aligner = Align.PairwiseAligner(match_score=1.0, mismatch_score=-1.0,
                                    open_gap_score=-2.0, extend_gap_score=-2.0,
                                    mode="global")

    alignments = []
    for target, query in tqdm.tqdm(zip(targets, queries)):
        alignments.append(aligner.align(target, query)[0])

    Align.write(alignments, args.output_file, "fasta")


if __name__ == '__main__':
    main()
