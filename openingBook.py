import chess
import chess.pgn
import random
from collections import defaultdict
from tqdm import tqdm

class OpeningBook:
    def __init__(self, min_occurrences: int = 2):
        """
        Initializes the OpeningBook.

        Args:
            min_occurrences: The minimum number of times a move transition (sequence -> next_move)
                             must occur to be considered part of the book for sampling.
        """
        # self.transitions stores counts for (sequence_tuple -> next_move_uci -> count)
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.min_occurrences = min_occurrences
        print(f"OpeningBook initialized with min_occurrences={self.min_occurrences}")

    def add(self, game: chess.pgn.Game):
        """
        Adds a game's mainline moves to the opening book.
        """
        current_sequence_ucis = []
        for move in game.mainline_moves():
            sequence_tuple = tuple(current_sequence_ucis)
            next_move_uci = move.uci()

            # Increment the count for this specific transition
            self.transitions[sequence_tuple][next_move_uci] += 1

            # Update the current sequence for the next iteration
            current_sequence_ucis.append(next_move_uci)

    def next(self, starting_move_seq: list[chess.Move]) -> chess.Move | None:
        """
        Returns a probability-sampled next_move for the given starting_move_seq.
        Returns None if the sequence is not in the book (or doesn't meet min_occurrences).
        """
        current_sequence_ucis = tuple(move.uci() for move in starting_move_seq)

        if current_sequence_ucis in self.transitions:
            possible_next_moves_counts = self.transitions[current_sequence_ucis]

            # Filter for moves that meet the minimum occurrence threshold
            valid_moves_with_counts = {
                move_uci: count
                for move_uci, count in possible_next_moves_counts.items()
                if count >= self.min_occurrences
            }

            if not valid_moves_with_counts:
                return None # No moves meet the minimum occurrence criteria

            next_moves_ucis = list(valid_moves_with_counts.keys())
            counts = list(valid_moves_with_counts.values())

            # Calculate probabilities
            total_count = sum(counts)
            if total_count == 0:
                return None # Should not happen if valid_moves_with_counts is not empty

            probabilities = [count / total_count for count in counts]

            # Sample a move
            sampled_uci = random.choices(next_moves_ucis, weights=probabilities, k=1)[0]
            return chess.Move.from_uci(sampled_uci)
        else:
            return None


PGN_FILES = ["schrodinger404-white.pgn", "schrodinger404-black.pgn"]

opening_book = OpeningBook(min_occurrences=2)

# Load games and add them to the opening book
all_games_to_add = []
for pgn_file_path in PGN_FILES:
    print(f"Loading games from {pgn_file_path}...")
    with open(pgn_file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            all_games_to_add.append(game)

for game in tqdm(all_games_to_add, desc="Populating Opening Book"):
    opening_book.add(game)


