# this is supposed to prcoess the saved samples in white_dataset_samples.pkl and the black one too , to fit the to_tensor fn 
# @title
import pickle
import copy
import numpy as np


DATASETS = [
    "white_dataset_samples.pkl",
    "black_dataset_samples.pkl",
]

OUTPUT_SUFFIX = "_processed"
EVAL_SCALE = 400.0
EVAL_DROPOUT = 1
NUM_CANDIDATES = 13
DUMMY_MOVE = "0000"

def normalize_eval(eval_cp):
    if eval_cp is None:
        return 0.0
    return float(np.tanh(eval_cp / EVAL_SCALE))


def process_dataset(dataset):
    samples = copy.deepcopy(dataset)
    dropout_count = 0
    padding_count = 0

    for sample_index, sample in enumerate(samples):
        moves = sample["candidate_moves_uci"]
        raw_evals = sample["eval"]

        real_evals = raw_evals[:len(moves)]

        if len(real_evals) != len(moves):
            print(
                f"WARNING sample {sample_index}: "
                f"{len(moves)} moves, "
                f"{len(real_evals)} usable evaluations"
            )
            continue

        normalized_evals = []

        for evaluation in real_evals:
            normalized_evals.append(
                normalize_eval(evaluation)
            )


        if np.random.random() < EVAL_DROPOUT:

            normalized_evals = [0.0 for _ in normalized_evals]
            dropout_count += 1

        while len(moves) < NUM_CANDIDATES:

            moves.append(DUMMY_MOVE)
            normalized_evals.append(-10000.0)
            padding_count += 1

        sample["candidate_moves_uci"] = moves
        sample["eval"] = normalized_evals

    print(f"Samples: {len(samples)}")
    print(f"Evaluation dropout: {dropout_count}")
    print(f"Dummy candidates added: {padding_count}")

    return samples

for filename in DATASETS:

    print(f"\nProcessing {filename}...")

    with open(filename, "rb") as f:
        dataset = pickle.load(f)

    processed_dataset = process_dataset(dataset)

    output_filename = filename.replace(
        ".pkl",
        f"{OUTPUT_SUFFIX}.pkl"
    )

    with open(output_filename, "wb") as f:
        pickle.dump(processed_dataset, f)

    print(f"Saved: {output_filename}")