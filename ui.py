import os
import pickle
import pygame
import chess
import chess.engine
import torch
from tensor import to_tensor
from model import ChessPolicyNet
from openingBook import OpeningBook


MODEL_PATH = "chess_policy_net_weights.pth"
OPENING_BOOK_PATH = "custom_opening_book.pkl"

STOCKFISH_PATH = "./stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 760
BOARD_SIZE = 700
SQUARE_SIZE = BOARD_SIZE // 8

BOARD_X = 30
BOARD_Y = 30

SIDE_PANEL_X = BOARD_X + BOARD_SIZE + 30

LIGHT = (235, 235, 235)
DARK = (90, 90, 90)

SELECTED_COLOR = (100, 180, 100)

MOVE_DOT_COLOR = (40, 40, 40)

# Text colors
TEXT_COLOR = (240, 240, 240)
BACKGROUND = (25, 25, 25)

# Piece colors
WHITE_PIECE_COLOR = (120, 75, 35)   # brown
BLACK_PIECE_COLOR = (45, 120, 70)   # green

PIECE_IMAGES = {}

PIECE_NAMES = {
    "P": "wP",
    "N": "wN",
    "B": "wB",
    "R": "wR",
    "Q": "wQ",
    "K": "wK",

    "p": "bP",
    "n": "bN",
    "b": "bB",
    "r": "bR",
    "q": "bQ",
    "k": "bK",
}














print("Loading model...")

model = ChessPolicyNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Loading opening book...")

try:
    with open(OPENING_BOOK_PATH, "rb") as f:
        loaded_opening_book = pickle.load(f)

    print("Opening book loaded.")

except FileNotFoundError:
    print("WARNING: Opening book not found.")
    loaded_opening_book = None


engine = None

if os.path.exists(STOCKFISH_PATH):

    try:
        engine = chess.engine.SimpleEngine.popen_uci(
            STOCKFISH_PATH
        )

        print("Stockfish loaded.")

    except Exception as e:

        print("WARNING: Could not start Stockfish.")
        print(e)

else:

    print(
        f"WARNING: Stockfish not found at '{STOCKFISH_PATH}'."
    )





def predict_next_move(
    board,
    stockfish_depth=8,
    num_candidate_moves=13
):
    """
    Use Stockfish to generate candidate moves and let the
    neural network select one of them.
    """

    if engine is None:

        print("No Stockfish engine available.")

        return None

    try:
        # Ask Stockfish for candidate moves

        info = engine.analyse(
            board,
            chess.engine.Limit(
                depth=stockfish_depth
            ),
            multipv=num_candidate_moves
        )

        stockfish_candidate_moves = [item['pv'][0] for item in info]
        stockfish_candidate_eval = [item['score'].pov(board.turn).score() for item in info]

        is_black_turn = (
            board.turn == chess.BLACK
        )

        board_tensor, moves_tensor = to_tensor(
            board,
            stockfish_candidate_moves,
            stockfish_candidate_eval,
            black=is_black_turn
        )

        # Add batch dimension
        board_tensor = board_tensor.unsqueeze(0)
        moves_tensor = moves_tensor.unsqueeze(0)

        # ----------------------------------------------------
        # Neural network prediction
        # ----------------------------------------------------

        with torch.no_grad():

            scores = model(
                board_tensor,
                moves_tensor
            )

            probabilities = torch.softmax(
                scores,
                dim=1
            )

            best_move_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

        # Safety check
        if best_move_index >= len(stockfish_candidate_moves):

            print("Model returned invalid candidate index.")

            return None

        chosen_move = (
            stockfish_candidate_moves[
                best_move_index
            ]
        )

        print()
        print("Neural network candidates:")

        for i, move in enumerate(stockfish_candidate_moves):

            probability = (
                probabilities[0, i].item()
                if i < probabilities.shape[1]
                else 0
            )

            print(
                f"{i + 1:2d}. "
                f"{board.san(move):6s} "
                f"{probability:.3f}"
            )

        print(
            "Selected:",
            board.san(chosen_move)
        )

        return chosen_move

    except Exception as e:

        print()
        print("ERROR during model prediction:")
        print(e)

        return None


# ============================================================
# CHOOSE ENGINE MOVE
# ============================================================

def choose_move(board):
    """
    First try the opening book.

    If there is no book move, use the neural network.
    """

    starting_move_seq = list(board.move_stack)
    book_move = loaded_opening_book.next(starting_move_seq)

    if book_move is not None:

        print()
        print(
            "Opening book:",
            board.san(book_move)
        )

        return book_move, "Opening Book"

    # Neural network
    
    print()
    print("No opening-book move.")

    move = predict_next_move(board)

    return move, "Neural Network"


def draw_board(screen, board, selected_square=None):

    for row in range(8):

        for col in range(8):

            # Convert screen row to chess rank.
            # Screen row 0 = rank 8.
            rank = 7 - row
            file = col

            square = chess.square(
                file,
                rank
            )

            # Board color
            if (row + col) % 2 == 0:
                color = LIGHT
            else:
                color = DARK

            # Highlight selected square
            if square == selected_square:
                color = SELECTED_COLOR

            x = BOARD_X + col * SQUARE_SIZE
            y = BOARD_Y + row * SQUARE_SIZE

            pygame.draw.rect(
                screen,
                color,
                (
                    x,
                    y,
                    SQUARE_SIZE,
                    SQUARE_SIZE
                )
            )

            # ------------------------------------------------
            # Draw piece
            # ------------------------------------------------

            piece = board.piece_at(square)

            if piece is not None:

                draw_piece(
                    screen,
                    piece,
                    x,
                    y
                )


def draw_piece(screen, piece, x, y):

    symbol = piece.symbol()

    image = PIECE_IMAGES[symbol]

    screen.blit(
        image,
        (x, y)
    )


def draw_side_panel(
    screen,
    board,
    engine_source,
    selected_square
):

    font = pygame.font.SysFont(
        "Arial",
        22
    )

    small_font = pygame.font.SysFont(
        "Arial",
        17
    )

    x = SIDE_PANEL_X
    y = 40

    # --------------------------------------------------------
    # Turn
    # --------------------------------------------------------

    if board.turn == chess.WHITE:

        turn_text = "White to move"

    else:

        turn_text = "Black to move"

    text = font.render(
        turn_text,
        True,
        TEXT_COLOR
    )

    screen.blit(
        text,
        (x, y)
    )

    y += 45

    # --------------------------------------------------------
    # Engine source
    # --------------------------------------------------------

    text = small_font.render(
        f"Engine: {engine_source}",
        True,
        TEXT_COLOR
    )

    screen.blit(
        text,
        (x, y)
    )

    y += 40

    # --------------------------------------------------------
    # Move count
    # --------------------------------------------------------

    text = small_font.render(
        f"Moves: {len(board.move_stack)}",
        True,
        TEXT_COLOR
    )

    screen.blit(
        text,
        (x, y)
    )

    y += 50

    # --------------------------------------------------------
    # Last move
    # --------------------------------------------------------

    if board.move_stack:

        last_move = board.peek()

        text = small_font.render(
            f"Last move: {last_move.uci()}",
            True,
            TEXT_COLOR
        )

        screen.blit(
            text,
            (x, y)
        )

        y += 40

    # --------------------------------------------------------
    # Selected square
    # --------------------------------------------------------

    if selected_square is not None:

        square_name = chess.square_name(
            selected_square
        )

        text = small_font.render(
            f"Selected: {square_name}",
            True,
            TEXT_COLOR
        )

        screen.blit(
            text,
            (x, y)
        )


# ============================================================
# MOUSE → CHESS SQUARE
# ============================================================

def mouse_to_square(mouse_x, mouse_y):

    # Check if outside board
    if not (
        BOARD_X <= mouse_x < BOARD_X + BOARD_SIZE
        and
        BOARD_Y <= mouse_y < BOARD_Y + BOARD_SIZE
    ):

        return None

    col = (
        mouse_x - BOARD_X
    ) // SQUARE_SIZE

    row = (
        mouse_y - BOARD_Y
    ) // SQUARE_SIZE

    # Convert screen coordinates to chess coordinates
    file = col
    rank = 7 - row

    return chess.square(
        file,
        rank
    )


# ============================================================
# DRAW LEGAL MOVE DOTS
# ============================================================

def draw_legal_moves(
    screen,
    board,
    selected_square
):

    if selected_square is None:
        return

    for move in board.legal_moves:

        if move.from_square != selected_square:
            continue

        file = chess.square_file(
            move.to_square
        )

        rank = chess.square_rank(
            move.to_square
        )

        col = file
        row = 7 - rank

        x = (
            BOARD_X
            + col * SQUARE_SIZE
            + SQUARE_SIZE // 2
        )

        y = (
            BOARD_Y
            + row * SQUARE_SIZE
            + SQUARE_SIZE // 2
        )

        pygame.draw.circle(
            screen,
            MOVE_DOT_COLOR,
            (x, y),
            10
        )


# ============================================================
# RESET GAME
# ============================================================

def reset_game():

    return chess.Board()


# ============================================================
# MAIN
# ============================================================

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
    )

    for symbol, filename in PIECE_NAMES.items():

        image = pygame.image.load(
            f"chess_pieces/{filename}.png"
        ).convert_alpha()

        image = pygame.transform.smoothscale(
            image,
            (SQUARE_SIZE, SQUARE_SIZE)
        )

        PIECE_IMAGES[symbol] = image


    pygame.display.set_caption(
        "Chess Neural Network"
    )

    clock = pygame.time.Clock()

    board = chess.Board()

    selected_square = None

    engine_source = "Waiting"

    running = True

    # --------------------------------------------------------
    # Player is WHITE
    # AI is BLACK
    # --------------------------------------------------------

    player_color = chess.WHITE

    while running:

        # ====================================================
        # EVENTS
        # ====================================================

        for event in pygame.event.get():

            # ------------------------------------------------
            # Close window
            # ------------------------------------------------

            if event.type == pygame.QUIT:

                running = False

            # ------------------------------------------------
            # Mouse click
            # ------------------------------------------------

            elif event.type == pygame.MOUSEBUTTONDOWN:

                # Don't allow moves while game is over
                if board.is_game_over():
                    continue

                # Only allow player to move when it is
                # their turn.
                if board.turn != player_color:
                    continue

                mouse_x, mouse_y = event.pos

                clicked_square = mouse_to_square(
                    mouse_x,
                    mouse_y
                )

                if clicked_square is None:
                    continue

                # ============================================
                # Nothing selected yet
                # ============================================

                if selected_square is None:

                    piece = board.piece_at(
                        clicked_square
                    )

                    # Only select player's pieces
                    if (
                        piece is not None
                        and
                        piece.color == player_color
                    ):

                        selected_square = (
                            clicked_square
                        )

                # ============================================
                # Already selected something
                # ============================================

                else:

                    move = chess.Move(
                        selected_square,
                        clicked_square
                    )

                    # Promotion
                    if (
                        chess.square_rank(
                            clicked_square
                        ) in (0, 7)
                        and
                        board.piece_at(
                            selected_square
                        )
                        and
                        board.piece_at(
                            selected_square
                        ).piece_type
                        == chess.PAWN
                    ):

                        move = chess.Move(
                            selected_square,
                            clicked_square,
                            promotion=chess.QUEEN
                        )

                    # ========================================
                    # Legal move
                    # ========================================

                    if move in board.legal_moves:

                        print()
                        print(
                            "You played:",
                            board.san(move)
                        )

                        board.push(move)

                        selected_square = None

                        engine_source = "Thinking..."

                        # ====================================
                        # Check game over
                        # ====================================

                        if board.is_game_over():

                            print()
                            print(
                                "Game over:",
                                board.result()
                            )

                            engine_source = (
                                "Game Over"
                            )

                        else:

                            # =================================
                            # AI MOVE
                            # =================================

                            ai_move, source = choose_move(
                                board
                            )

                            if ai_move is not None:

                                print()
                                print(
                                    "AI played:",
                                    board.san(ai_move)
                                )

                                board.push(
                                    ai_move
                                )

                                engine_source = source

                            else:

                                print(
                                    "AI could not find a move."
                                )

                                engine_source = (
                                    "ERROR"
                                )

                    # ========================================
                    # Clicked another own piece
                    # ========================================

                    else:

                        piece = board.piece_at(
                            clicked_square
                        )

                        if (
                            piece is not None
                            and
                            piece.color == player_color
                        ):

                            selected_square = (
                                clicked_square
                            )

                        else:

                            selected_square = None


        screen.fill(BACKGROUND)

        draw_board(
            screen,
            board,
            selected_square
        )

        draw_legal_moves(
            screen,
            board,
            selected_square
        )

        draw_side_panel(
            screen,
            board,
            engine_source,
            selected_square
        )

        # ----------------------------------------------------
        # Game over text
        # ----------------------------------------------------

        if board.is_game_over():

            font = pygame.font.SysFont(
                "Arial",
                28
            )

            text = font.render(
                f"Game Over: {board.result()}",
                True,
                (255, 200, 80)
            )

            screen.blit(
                text,
                (
                    SIDE_PANEL_X,
                    WINDOW_HEIGHT - 100
                )
            )

        pygame.display.flip()

        clock.tick(60)

    # ========================================================
    # CLEANUP
    # ========================================================

    if engine is not None:

        engine.quit()

    pygame.quit()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()