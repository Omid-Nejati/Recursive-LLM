from typing import Optional, List
import os
import json
import numpy as np

from datasets import load_dataset
from argdantic import ArgParser
from pydantic import BaseModel


# -------------------------------------------------
# Fallback metadata class so you do NOT depend on
# importing PuzzleDatasetMetadata from your repo.
# -------------------------------------------------
class PuzzleDatasetMetadata(BaseModel):
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: int
    total_puzzles: int
    sets: List[str]


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "yuruny/futoshiki_generated_dataset_5x5_6-8"
    output_dir: str = "data/futoshiki-5x5-generated"

    # Hugging Face split to read from
    source_split: str = "train"

    # Optional quick debug subset
    subsample_size: Optional[int] = None

    # Local split
    train_ratio: float = 0.8
    seed: int = 42
    shuffle: bool = True


# -------------------------------------------------
# Token encoding
# -------------------------------------------------
PAD_ID = 0

# Board cell encoding
# blank cell -> 1
# digits 1..5 -> 2..6
BLANK_ID = 1
DIGIT_OFFSET = 1

# Constraint encoding
# no constraint -> 7
# horizontal: <, >
# vertical normalized as:
#   ^ means top > bottom
#   v means top < bottom
NO_CONSTRAINT_ID = 7
LT_ID = 8
GT_ID = 9
UP_ID = 10
DOWN_ID = 11

VOCAB_SIZE = 12


def encode_digit(x: int) -> int:
    if x == 0:
        return BLANK_ID
    if 1 <= x <= 5:
        return x + DIGIT_OFFSET
    raise ValueError(f"Unexpected cell value: {x}")


def encode_solution_digit(x: int) -> int:
    if 1 <= x <= 5:
        return x + DIGIT_OFFSET
    raise ValueError(f"Unexpected solution value: {x}")


def build_constraint_planes(board_size: int, constraints: dict):
    """
    Build two planes:

    horiz: shape (N, N-1)
        relation between (r,c) and (r,c+1)

    vert: shape (N-1, N)
        relation between (r,c) and (r+1,c)

    Encoded as tokens:
      NO_CONSTRAINT_ID, LT_ID, GT_ID, UP_ID, DOWN_ID
    """
    n = board_size
    horiz = np.full((n, n - 1), NO_CONSTRAINT_ID, dtype=np.uint8)
    vert = np.full((n - 1, n), NO_CONSTRAINT_ID, dtype=np.uint8)

    for key, symbol in constraints.items():
        r1, c1, r2, c2 = map(int, key.split(","))

        # Horizontal adjacent cells
        if r1 == r2 and abs(c1 - c2) == 1:
            left_c = min(c1, c2)
            left_is_first = (c1 < c2)

            # Normalize relation to: left ? right
            if left_is_first:
                norm_symbol = symbol
            else:
                norm_symbol = "<" if symbol == ">" else ">"

            if norm_symbol == "<":
                horiz[r1, left_c] = LT_ID
            elif norm_symbol == ">":
                horiz[r1, left_c] = GT_ID
            else:
                raise ValueError(f"Unexpected horizontal symbol: {symbol}")

        # Vertical adjacent cells
        elif c1 == c2 and abs(r1 - r2) == 1:
            top_r = min(r1, r2)
            top_is_first = (r1 < r2)

            # Normalize relation to: top ? bottom
            if top_is_first:
                norm_symbol = symbol
            else:
                norm_symbol = "<" if symbol == ">" else ">"

            # top < bottom  => DOWN_ID
            # top > bottom  => UP_ID
            if norm_symbol == "<":
                vert[top_r, c1] = DOWN_ID
            elif norm_symbol == ">":
                vert[top_r, c1] = UP_ID
            else:
                raise ValueError(f"Unexpected vertical symbol: {symbol}")

        else:
            raise ValueError(
                f"Constraint must be between adjacent cells, got {key} -> {symbol}"
            )

    return horiz, vert


def encode_example(example: dict):
    """
    Sequence layout for 5x5 Futoshiki:
      [25 grid cells] + [20 horizontal constraints] + [20 vertical constraints]
    Total seq_len = 65
Labels:
      first 25 positions = solved grid digits
      last 40 positions = 0 (ignored in loss)
    """
    n = example["board_size"]
    if n != 5:
        raise ValueError(f"This script expects board_size=5, got {n}")

    grid = np.array(example["grid"], dtype=np.int64)
    solution = np.array(example["solution"], dtype=np.int64)
    constraints = example["constraints"]

    horiz, vert = build_constraint_planes(n, constraints)

    # Encode input
    grid_tokens = np.vectorize(encode_digit)(grid).reshape(-1)
    horiz_tokens = horiz.reshape(-1)
    vert_tokens = vert.reshape(-1)

    input_tokens = np.concatenate([grid_tokens, horiz_tokens, vert_tokens], axis=0)
    assert input_tokens.shape[0] == 65

    # Encode labels: only solve the 25 board cells
    solution_tokens = np.vectorize(encode_solution_digit)(solution).reshape(-1)
    label_tokens = np.concatenate(
        [
            solution_tokens,
            np.zeros(horiz_tokens.shape[0] + vert_tokens.shape[0], dtype=np.uint8),
        ],
        axis=0,
    )
    assert label_tokens.shape[0] == 65

    return input_tokens.astype(np.uint8), label_tokens.astype(np.uint8)


def save_split(split_name: str, inputs: np.ndarray, labels: np.ndarray, output_dir: str):
    num_examples = inputs.shape[0]

    puzzle_indices = np.arange(num_examples + 1, dtype=np.int32)
    group_indices = np.arange(num_examples + 1, dtype=np.int32)
    puzzle_identifiers = np.zeros(num_examples, dtype=np.int32)

    results = {
        "inputs": inputs,
        "labels": labels,
        "group_indices": group_indices,
        "puzzle_indices": puzzle_indices,
        "puzzle_identifiers": puzzle_identifiers,
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=65,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=num_examples,
        mean_puzzle_examples=1,
        total_puzzles=num_examples,
        sets=["all"],
    )

    save_dir = os.path.join(output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    print(f"Saved {num_examples} examples to: {save_dir}")
    print(f"{split_name} inputs shape: {inputs.shape}")
    print(f"{split_name} labels shape: {labels.shape}")


def convert_subset(config: DataProcessConfig):
    ds = load_dataset(config.source_repo, split=config.source_split)

    if config.subsample_size is not None and config.subsample_size < len(ds):
        ds = ds.select(range(config.subsample_size))

    inputs = []
    labels = []

    for ex in ds:
        # Some datasets may store puzzle as a JSON string
        if isinstance(ex.get("puzzle"), str) and "grid" not in ex:
            parsed = json.loads(ex["puzzle"])
            inp, lab = encode_example(parsed)
        else:
            structured = {
                "board_size": ex["board_size"] if "board_size" in ex else ex["puzzle"]["board_size"],
                "grid": ex["grid"] if "grid" in ex else ex["puzzle"]["grid"],
                "constraints": ex["constraints"] if "constraints" in ex else ex["puzzle"]["constraints"],
                "solution": ex["solution"] if "solution" in ex else ex["puzzle"]["solution"],
            }
            inp, lab = encode_example(structured)

        inputs.append(inp)
        labels.append(lab)

    inputs = np.stack(inputs, axis=0)
    labels = np.stack(labels, axis=0)

    num_examples = inputs.shape[0]
    indices = np.arange(num_examples)

    if config.shuffle:
        rng = np.random.default_rng(config.seed)
        rng.shuffle(indices)

    inputs = inputs[indices]
    labels = labels[indices]

    train_size = int(config.train_ratio * num_examples)

    train_inputs = inputs[:train_size]
    train_labels = labels[:train_size]

    test_inputs = inputs[train_size:]
    test_labels = labels[train_size:]

    save_split("train", train_inputs, train_labels, config.output_dir)
    save_split("test", test_inputs, test_labels, config.output_dir)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print("\nDone.")
    print(f"Total examples: {num_examples}")
    print(f"Train examples: {train_inputs.shape[0]}")
    print(f"Test examples: {test_inputs.shape[0]}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset(config)


if __name__ == "__main__":
    cli()