import chess.pgn

MY_USERNAME = "schrodinger404"
INPUT_FILE = "val/validation.pgn"
WHITE_FILE = "val/white_games.pgn"
BLACK_FILE = "val/black_games.pgn"

# Open output files
with open(WHITE_FILE, "w", encoding="utf-8") as white_out, \
     open(BLACK_FILE, "w", encoding="utf-8") as black_out, \
     open(INPUT_FILE, "r", encoding="utf-8") as pgn_in:

    white_count = 0
    black_count = 0

    while True:
        game = chess.pgn.read_game(pgn_in)
        if game is None:
            break  # End of file

        white_player = game.headers.get("White", "")
        black_player = game.headers.get("Black", "")

        # Case-insensitive username check
        if white_player.lower() == MY_USERNAME.lower():
            white_out.write(str(game) + "\n\n")
            white_count += 1
        elif black_player.lower() == MY_USERNAME.lower():
            black_out.write(str(game) + "\n\n")
            black_count += 1

print(f"Splitting complete!")
print(f"- Saved {white_count} games to '{WHITE_FILE}'")
print(f"- Saved {black_count} games to '{BLACK_FILE}'")