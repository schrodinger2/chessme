import pickle
import copy
import numpy as np

DATASETS = [
    "white_dataset_samples.pkl",
    "black_dataset_samples.pkl",
]

OUTPUT_SUFFIX = "_processed"

EVAL_SCALE = 400.0
EVAL_DROPOUT = 0.50
NUM_CANDIDATES = 13
DUMMY_MOVE = "0000"


def normalize_eval(x):
    if x is None:
        return 0.0
    return float(np.tanh(x / EVAL_SCALE))


def process_dataset(dataset):
    processed = copy.deepcopy(dataset)

    dropout_count = 0
    padding_count = 0

    for sample in processed:
        moves = sample["candidate_moves_uci"][:NUM_CANDIDATES]
        raw_evals = sample["eval"][:len(moves)]

        evals = [normalize_eval(x) for x in raw_evals]

        # Real candidates
        is_padding = [0.0] * len(moves)

        # Mate cannot be recovered reliably from these saved samples
        is_mate = [0.0] * len(moves)

        # Evaluation dropout
        if np.random.random() < EVAL_DROPOUT:
            evals = [0.0] * len(evals)
            dropout_count += 1

        # Pad to 13 candidates
        while len(moves) < NUM_CANDIDATES:
            moves.append(DUMMY_MOVE)
            evals.append(0.0)
            is_padding.append(1.0)
            is_mate.append(0.0)
            padding_count += 1

        sample["candidate_moves_uci"] = moves
        sample["eval"] = evals
        sample["is_padding"] = is_padding
        sample["is_mate"] = is_mate

    print(f"Samples: {len(processed)}")
    print(f"Evaluation dropout: {dropout_count}")
    print(f"Dummy candidates added: {padding_count}")

    return processed


for filename in DATASETS:
    print(f"\nProcessing {filename}...")

    with open(filename, "rb") as f:
        dataset = pickle.load(f)

    processed = process_dataset(dataset)

    output_filename = filename.replace(
        ".pkl",
        f"{OUTPUT_SUFFIX}.pkl"
    )

    with open(output_filename, "wb") as f:
        pickle.dump(processed, f)

    print(f"Saved: {output_filename}")