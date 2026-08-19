import os
import torch
import chess
import chess.pgn
import chess.engine
from tensor import to_tensor
from model import ChessPolicyNet

MODEL_PATH = "models/chess_policy_net_weights4.pth"

STOCKFISH_PATH = "./stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"

PGN_FILES = [
    "./val/white_games.pgn",
    "./val/black_games.pgn"
]

IGNORE_PLIES = 10

STOCKFISH_DEPTH = 8

NUM_CANDIDATE_MOVES = 13

# Which Top-K values we want to report.
TOP_K_VALUES = [1, 3, 5, 10, 13]

print("Loading model...")

model = ChessPolicyNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded.")

def predict_position(board, engine):
    """
    Predict the move for the current position.

    Returns a dictionary containing:

        predicted_move
        probabilities
        candidate_moves
        true_move_probability
        true_move_rank
        true_move_in_candidates
    """

    info = engine.analyse(board, chess.engine.Limit( depth=STOCKFISH_DEPTH ), multipv=NUM_CANDIDATE_MOVES )

    if not info:
        return None

    candidate_moves = []
    candidate_evals = []

    for item in info:

        if "pv" not in item or not item["pv"]:
            continue

        move = item["pv"][0]

        candidate_moves.append(move)

        score = item["score"].pov(board.turn)

        if score.is_mate():

            mate = score.mate()

            if mate is not None and mate > 0:
                evaluation = 10000
            else:
                evaluation = -10000

        else:

            evaluation = score.score()
        candidate_evals.append(evaluation)

    if not candidate_moves:
        return None

    is_black = board.turn == chess.BLACK

    board_tensor, moves_tensor , candidate_moves= to_tensor( board, candidate_moves, candidate_evals, black=is_black)

    board_tensor = board_tensor.unsqueeze(0)
    moves_tensor = moves_tensor.unsqueeze(0)

    with torch.no_grad():

        scores = model( board_tensor, moves_tensor)

        probabilities = torch.softmax(scores, dim=1)[0]

    ranked_indices = torch.argsort( probabilities, descending=True)

    ranked_moves = [candidate_moves[i] for i in ranked_indices]

    ranked_probabilities = [ probabilities[i].item() for i in ranked_indices]

    predicted_move = ranked_moves[0]

    return {
        "predicted_move": predicted_move,

        "candidate_moves": candidate_moves,

        "ranked_moves": ranked_moves,

        "ranked_probabilities": ranked_probabilities,

        "probabilities": probabilities.tolist(),
    }


def validate_game(game, engine, game_number, filename):
    """
    Validate an entire PGN game using teacher forcing.

    IMPORTANT:

    The model prediction is NEVER played.

    The actual PGN move is ALWAYS played.

    Therefore a wrong prediction does not corrupt
    subsequent positions.
    """

    board = game.board()

    results = []

    moves = list(game.mainline_moves())

    for ply, actual_move in enumerate(moves):

        if ply < IGNORE_PLIES:

            board.push(actual_move)

            continue

        prediction = predict_position(board,engine)

        if prediction is None:

            print(f"WARNING: Could not predict " f"ply {ply + 1}")

            board.push(actual_move)

            continue

        ranked_moves = prediction["ranked_moves"]

        ranked_probabilities = ( prediction["ranked_probabilities"])

        predicted_move = prediction["predicted_move"]

        actual_rank = None
        actual_probability = 0.0

        for rank, move in enumerate(ranked_moves,start=1):
            if move == actual_move:
                actual_rank = rank
                actual_probability = (ranked_probabilities[rank - 1])
                break

        top_k = {}

        for k in TOP_K_VALUES:

            top_k[k] = ( actual_rank is not None and actual_rank <= k)

        results.append({
            "ply": ply + 1,
            "actual_move": actual_move,
            "predicted_move": predicted_move,
            "actual_rank": actual_rank,
            "actual_probability": actual_probability,
            "top_k": top_k,
        })
        board.push(actual_move)

    return results


def load_games(filename):

    games = []

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        while True:

            game = chess.pgn.read_game(f)

            if game is None:
                break

            games.append(game)

    return games


def print_game_results(results,game_number,filename):

    print()
    print("=" * 75)

    print(
        f"Game {game_number} "
        f"({filename})"
    )

    print("=" * 75)

    if not results:

        print("No positions were evaluated.")

        return

    for result in results:

        actual = result["actual_move"]

        predicted = result["predicted_move"]

        rank = result["actual_rank"]

        probability = result[
            "actual_probability"
        ]

        actual_uci = actual.uci()

        predicted_uci = predicted.uci()

        if rank is None:

            rank_text = "NOT IN CANDIDATES"

        else:

            rank_text = f"rank #{rank}"

        print(
            f"Ply {result['ply']:3d} | "
            f"Actual: {actual_uci:5s} | "
            f"Predicted: {predicted_uci:5s} | "
            f"{rank_text:20s} | "
            f"P(actual): {probability:.4f}"
        )


def print_summary(all_results):

    print()
    print("=" * 75)
    print("VALIDATION SUMMARY")
    print("=" * 75)

    total = len(all_results)

    print(
        f"Positions evaluated: {total}"
    )

    in_candidates = sum(
        r["actual_rank"] is not None
        for r in all_results
    )

    coverage = (in_candidates / total)

    print(
        f"Actual move in Stockfish "
        f"candidate set: "
        f"{in_candidates}/{total} "
        f"({coverage:.2%})"
    )

    print()

    print("Top-K accuracy:")

    for k in TOP_K_VALUES:

        correct = sum(
            r["top_k"][k]
            for r in all_results
        )

        accuracy = correct / total

        print(
            f"  Top-{k:2d}: "
            f"{correct:5d}/{total:5d} "
            f"({accuracy:.2%})"
        )

    probabilities = [
        r["actual_probability"]
        for r in all_results
        if r["actual_rank"] is not None
    ]

    if probabilities:
        average_probability = (
            sum(probabilities)
            / len(probabilities)
        )

        print(
            f"Average P(actual move) "
            f"when candidate exists: "
            f"{average_probability:.4f}"
        )

    if probabilities:

        sorted_probs = sorted(
            probabilities
        )

        middle = len(
            sorted_probs
        ) // 2

        if len(sorted_probs) % 2 == 0:
            median_probability = (
                sorted_probs[middle - 1]
                + sorted_probs[middle]
            ) / 2

        else:
            median_probability = (
                sorted_probs[middle]
            )

        print(
            f"Median P(actual move): "
            f"{median_probability:.4f}"
        )

    reciprocal_ranks = [
        1 / r["actual_rank"]
        for r in all_results
        if r["actual_rank"] is not None
    ]

    if reciprocal_ranks:
        mrr = (
            sum(reciprocal_ranks)
            / len(reciprocal_ranks)
        )

        print(
            f"Mean Reciprocal Rank: "
            f"{mrr:.4f}"
        )

    print()


def main():

    print("Starting Stockfish...")

    engine = chess.engine.SimpleEngine.popen_uci(
        STOCKFISH_PATH
    )

    print("Stockfish ready.")

    all_results = []

    try:

        # Process every PGN file

        for filename in PGN_FILES:

            print(
                f"Loading games from {filename}..."
            )

            games = load_games(
                filename
            )

            print(
                f"Loaded {len(games)} games."
            )

            for game_number, game in enumerate(
                games,
                start=1
            ):

                print(
                    f"\rValidating game "
                    f"{game_number}/{len(games)}...",
                    end=""
                )

                results = validate_game(
                    game,
                    engine,
                    game_number,
                    filename
                )

                print_game_results(
                    results,
                    game_number,
                    filename
                )

                all_results.extend(
                    results
                )

    finally:

        engine.quit()

    print_summary(
        all_results
    )

if __name__ == "__main__":
    main()