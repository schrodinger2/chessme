import os
import random
import pickle
import pygame
import chess
import chess.engine
import torch
from tensor import to_tensor
from model import ChessPolicyNet
from openingBook import OpeningBook


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = ".\\models\\chess_policy_net_weights11.pth"
OPENING_BOOK_PATH = "custom_opening_book.pkl"

STOCKFISH_PATH = "./stockfish\\stockfish\\stockfish-windows-x86-64-avx2.exe"

BOARD_IMAGE_FOLDER = "board"
PIECE_IMAGE_FOLDER = "chess_pieces"

# Single static images.
BACKGROUND_IMAGE_PATH = "animations/background.jpg"
NEUTRAL_IMAGE_PATH = "animations/neutral.png"
MAD_IMAGE_PATH = "animations/mad.png"


# ============================================================
# LAYOUT (filled in at runtime once we know the screen size)
# ============================================================

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 760
BOARD_SIZE = 700
SQUARE_SIZE = BOARD_SIZE // 8

BOARD_X = 30
BOARD_Y = 30

SIDE_PANEL_X = BOARD_X + BOARD_SIZE + 30
SIDE_PANEL_WIDTH = 240

MARGIN = 40
SIDE_PANEL_MIN_WIDTH = 380

MOVE_DOT_COLOR = (40, 40, 40)

TEXT_COLOR = (240, 240, 240)
MUTED_TEXT_COLOR = (190, 190, 190)
BACKGROUND = (25, 25, 25)

PANEL_BG = (20, 20, 20, 165)
BUBBLE_BG = (245, 245, 245, 235)
BUBBLE_BORDER = (25, 25, 25, 255)
BUBBLE_TEXT = (20, 20, 20)

PIECE_IMAGES = {}

PIECE_NAMES = {
    "P": "wP", "N": "wN", "B": "wB", "R": "wR", "Q": "wQ", "K": "wK",
    "p": "bP", "n": "bN", "b": "bB", "r": "bR", "q": "bQ", "k": "bK",
}

candidate_info = []

arrow_start = None
arrow_end = None

# ============================================================
# MODEL / OPENING BOOK / ENGINE SETUP
# ============================================================

print("Loading model...")

model = ChessPolicyNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
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
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("Stockfish loaded.")
    except Exception as e:
        print("WARNING: Could not start Stockfish.")
        print(e)
else:
    print(f"WARNING: Stockfish not found at '{STOCKFISH_PATH}'.")


# ============================================================
# STATIC IMAGE LOADING
# ============================================================

def load_static_image(path, size=None, use_alpha=True):
    """
    Load a single image as a pygame Surface, optionally scaled to
    `size` = (w, h). Returns None (and prints a warning) if the file
    is missing, so the game keeps running even without the art yet.
    """

    if not os.path.isfile(path):
        print(f"WARNING: image not found: '{path}'")
        return None

    try:
        if use_alpha:
            image = pygame.image.load(path).convert_alpha()
        else:
            image = pygame.image.load(path).convert()
    except Exception as e:
        print(f"WARNING: could not load image '{path}': {e}")
        return None

    if size is not None:
        image = pygame.transform.smoothscale(image, size)

    return image


# ============================================================
# NEURAL NETWORK MOVE PREDICTION
# ============================================================

def predict_next_move(board, stockfish_depth=8, num_candidate_moves=13):
    """
    Use Stockfish to generate candidate moves and let the neural
    network select one of them. Returns (move, eval) or None.
    """

    if engine is None:
        print("No Stockfish engine available.")
        return None

    try:
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=stockfish_depth),
            multipv=num_candidate_moves,
        )

        stockfish_candidate_moves = [item["pv"][0] for item in info]
        stockfish_candidate_eval = [
            item["score"].pov(board.turn).score() for item in info
        ]

        is_black_turn = board.turn == chess.BLACK

        board_tensor, moves_tensor = to_tensor(
            board,
            stockfish_candidate_moves,
            stockfish_candidate_eval,
            black=is_black_turn,
        )

        board_tensor = board_tensor.unsqueeze(0)
        moves_tensor = moves_tensor.unsqueeze(0)

        with torch.no_grad():
            scores = model(board_tensor, moves_tensor)
            probabilities = torch.softmax(scores, dim=1)
            global candidate_info
            candidate_info = [(board.san(move), probabilities[0, i].item()) for i, move in enumerate(stockfish_candidate_moves)]
            best_move_index = torch.argmax(probabilities, dim=1).item()

        if best_move_index >= len(stockfish_candidate_moves):
            print("Model returned invalid candidate index.")
            return None

        chosen_move = stockfish_candidate_moves[best_move_index]

        print()
        print("Neural network candidates:")

        for i, move in enumerate(stockfish_candidate_moves):
            probability = (
                probabilities[0, i].item() if i < probabilities.shape[1] else 0
            )
            print(f"{i + 1:2d}. {board.san(move):6s} {probability:.3f}")

        print("Selected:", board.san(chosen_move))

        return chosen_move, stockfish_candidate_eval[best_move_index]

    except Exception as e:
        print()
        print("ERROR during model prediction:")
        print(e)
        return None

def trashtalker(eval, last_eval=0, opening=False, move=30):
    diff = last_eval - eval  # because computer playing black

    if opening:
        talk = ["may the better saif win"]
    elif eval > 5:
        talk = ["you are walking the green mile now", "last words?", "متعيطش", "jeder versagt so $#&!$& manchmel"]
        if move > 50:
            talk.append("took longer than expected")
        elif move > 15:
            talk.append("what? do i really have to continue? this is beneath me")
    elif eval > 2:
        talk = ["slipping away too fast", "ich habe mir bessers vorgestellt", "can you last another 10 moves"]
        if move > 50:
            talk.append("finally")
        elif move > 15:
            talk.append("already?")
    elif eval > 0:
        talk = ["i am machine i never sleep, i keep my eyes wide open", "عاش", "i gotta wake up now"]
        if move > 50:
            talk.append("wont let you survive that long again")
        elif move > 20:
            talk.append("do you feel it ?")
    elif eval > -2:
        talk = ["wie viel sprachen sprichest du", "kein foreplay mehr", "jetzt reicht es"]
        if move > 50:
            talk.append("are you using an engine")
        elif move > 20:
            talk.append("not so fast, have a lil respect")
    elif eval > -5:
        talk = ["oh , cheats", "who cares about a board game anyways", "you should have played me when i still cared for this stupid game"]
        if move > 50:
            talk.append("it's cool to lose long games anyways only try hards focus that long")
        elif move > 20:
            talk.append("if i just stop bludering , haaa stupid rigorous game")
    else:  # eval <= -5
        talk = ["u got some odds on u", "i have always been into the cooler hobbies", "just how many chess books have you eaten"]
        if move > 50:
            talk.append("lowkey sus that you can't finish the game already")
        elif move > 20:
            talk.append("touch some grass nerd")

    if diff > 3:
        talk += [
            "didnt think i still have it in me", "take your time", "le piece de resistance",
            "was furs wunder was furs drama", "bit you didnt see it comming", "bout time u witness sum fun",
        ]
    if diff > 1:
        talk += [
            "calm down", "it hurts to mess up like this", "i tell you what , why not try ballet instead",
            "is that all you got", "brother what is even that", "maybe you should use some engine cheating isn't that bad",
        ]
    if diff > -1 and diff > 0:
        talk += [
            "nein nein nein nein", "whatever", "es hat sich alles gedreht",
            "das leben ist nur ein phase und der rest ist die holle", "if i only cared enough",
        ]
    if diff > -3:
        talk += [
            "i am PLAYING against myself", "i got a rope just in case", "let's hope i stop playing around",
            "you are sooooo lucky", "verdammt", "life fits perfectly as a missing piece to all the escaped souls that never wish to comeback",
        ]
    if diff < -3:
        talk += [
            "what a nice daaay", "my troops are being idiots now", "je ne regrette rien",
            "i am playing for the crowd ,really", "hope the give away doesn't seem too obvious", "take this game , you look like you need it",
        ]

    return talk[random.randint(0, len(talk) - 1)]



def draw_arrow(screen, start_square, end_square):
    if start_square is None or end_square is None:
        return
    start_file = chess.square_file(start_square)
    start_rank = chess.square_rank(start_square)
    end_file = chess.square_file(end_square)
    end_rank = chess.square_rank(end_square)

    start = (
        BOARD_X + start_file * SQUARE_SIZE + SQUARE_SIZE // 2,
        BOARD_Y + (7 - start_rank) * SQUARE_SIZE + SQUARE_SIZE // 2
    )

    end = (
        BOARD_X + end_file * SQUARE_SIZE + SQUARE_SIZE // 2,
        BOARD_Y + (7 - end_rank) * SQUARE_SIZE + SQUARE_SIZE // 2
    )

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx ** 2 + dy ** 2) ** 0.5, 1)

    ux = dx / length
    uy = dy / length

    end_x = end[0] - ux * 10
    end_y = end[1] - uy * 10

    pygame.draw.line(screen, (220, 60, 60), start, (end_x, end_y), 10)

    perp_x = -uy
    perp_y = ux

    tip = (end[0], end[1])
    left = (end_x - ux * 22 + perp_x * 14, end_y - uy * 22 + perp_y * 14)
    right = (end_x - ux * 22 - perp_x * 14, end_y - uy * 22 - perp_y * 14)

    pygame.draw.polygon(screen, (220, 60, 60), [tip, left, right])

# ============================================================
# CHOOSE ENGINE MOVE
# ============================================================

last_eval = 0


def choose_move(board):
    """
    First try the opening book. If there is no book move, use the
    neural network.

    Returns: (move, source, text, mood)
      source: "Opening Book" or "Neural Network"
      mood:   "neutral" or "mad" -- "mad" when the bot's evaluation
              got worse compared to its previous move.
    """

    global last_eval

    starting_move_seq = list(board.move_stack)
    book_move = (
        loaded_opening_book.next(starting_move_seq)
        if loaded_opening_book is not None
        else None
    )

    if book_move is not None:
        print()
        print("Opening book:", board.san(book_move))
        text = trashtalker(0, 0, True, len(board.move_stack)/2)
        return book_move, "Opening Book", text, "neutral"

    print()
    print("No opening-book move.")

    result = predict_next_move(board)

    if result is None:
        return None, "Neural Network", "...", "neutral"

    move, eval_score = result

    # Mate scores can come back as None from python-chess; fall back
    # to the last known eval so the mood/trash-talk logic never crashes.
    if eval_score is None:
        eval_score = last_eval

    text = trashtalker(eval_score, last_eval, False, len(board.move_stack)/2)

    mood = "mad" if eval_score < 0 else "neutral"

    last_eval = eval_score

    return move, "Neural Network", text, mood


# ============================================================
# BOARD / PIECE DRAWING
# ============================================================

def draw_board(screen, board, selected_square=None):
    for row in range(8):
        for col in range(8):
            rank = 7 - row
            file = col
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            x = BOARD_X + col * SQUARE_SIZE
            y = BOARD_Y + row * SQUARE_SIZE

            if piece is not None:
                draw_piece(screen, piece, x, y)


def load_random_board():
    """Pick a random board texture. Only call this on real transitions,
    not on every move, so the board stays put during a game phase."""

    board_num = random.randint(1, 6)

    image = pygame.image.load(
        os.path.join(BOARD_IMAGE_FOLDER, f"{board_num}.jpg")
    ).convert()

    return pygame.transform.scale(image, (8 * SQUARE_SIZE, 8 * SQUARE_SIZE))


def draw_piece(screen, piece, x, y):
    symbol = piece.symbol()
    image = PIECE_IMAGES[symbol]
    screen.blit(image, (x, y))


# ============================================================
# TEXT / SPEECH BUBBLE HELPERS
# ============================================================

def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        test = (current + " " + word).strip()

        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def draw_speech_bubble(screen, font, text, x, y, width):
    """Draws a rounded speech bubble with a small tail. Returns the
    total height used (bubble + tail) so callers can lay out below it."""

    if not text:
        return 0

    padding = 14
    lines = wrap_text(text, font, width - 2 * padding)
    line_height = font.get_height() + 4
    bubble_height = line_height * len(lines) + 2 * padding

    bubble_surface = pygame.Surface((width, bubble_height), pygame.SRCALPHA)
    bubble_rect = bubble_surface.get_rect()

    pygame.draw.rect(bubble_surface, BUBBLE_BG, bubble_rect, border_radius=16)
    pygame.draw.rect(bubble_surface, BUBBLE_BORDER, bubble_rect, width=2, border_radius=16)

    screen.blit(bubble_surface, (x, y))

    ty = y + padding
    for line in lines:
        rendered = font.render(line, True, BUBBLE_TEXT)
        screen.blit(rendered, (x + padding, ty))
        ty += line_height

    tail_w, tail_h = 22, 16
    tail_x = x + 34
    tail_points = [
        (tail_x, y + bubble_height),
        (tail_x + tail_w, y + bubble_height),
        (tail_x, y + bubble_height + tail_h),
    ]

    pygame.draw.polygon(screen, BUBBLE_BG[:3], tail_points)
    pygame.draw.polygon(screen, BUBBLE_BORDER[:3], tail_points, width=2)

    return bubble_height + tail_h


def draw_panel_backdrop(screen, rect, radius=18):
    """Semi-transparent rounded panel so text stays readable over the
    animated background."""

    panel_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel_surface, PANEL_BG, panel_surface.get_rect(), border_radius=radius)
    screen.blit(panel_surface, (rect.x, rect.y))


# ============================================================
# SIDE PANEL (character, speech bubble, game info)
# ============================================================

def draw_side_panel(
    screen,
    board,
    engine_source,
    selected_square,
    character_frame,
    speech_text,
):
    font = pygame.font.SysFont("Arial", 24, bold=True)
    small_font = pygame.font.SysFont("Arial", 18)
    bubble_font = pygame.font.SysFont("Arial", 17)

    x = SIDE_PANEL_X
    panel_width = SIDE_PANEL_WIDTH

    y = 30

    # --------------------------------------------------------
    # Character + speech bubble
    # --------------------------------------------------------

    char_size = min(panel_width, 280)

    if character_frame is not None:
        frame = pygame.transform.smoothscale(character_frame, (char_size, char_size))
        char_rect = frame.get_rect(topleft=(x, y))
        draw_panel_backdrop(screen, char_rect.inflate(16, 16))
        screen.blit(frame, char_rect.topleft)
    else:
        char_rect = pygame.Rect(x, y, char_size, char_size)
        draw_panel_backdrop(screen, char_rect)

    bubble_y = y + char_size + 14
    used = draw_speech_bubble(screen, bubble_font, speech_text, x, bubble_y, panel_width)

    y = bubble_y + used + 24

    # --------------------------------------------------------
    # Game info panel
    # --------------------------------------------------------

    info_top = y
    info_lines = []

    turn_text = "White to move" if board.turn == chess.WHITE else "Black to move"
    info_lines.append((turn_text, font, TEXT_COLOR))
    info_lines.append((f"Engine: {engine_source}", small_font, MUTED_TEXT_COLOR))
    info_lines.append((f"Moves: {len(board.move_stack)}", small_font, MUTED_TEXT_COLOR))

    if board.move_stack:
        info_lines.append((f"Last move: {board.peek().uci()}", small_font, MUTED_TEXT_COLOR))

    if selected_square is not None:
        info_lines.append((f"Selected: {chess.square_name(selected_square)}", small_font, MUTED_TEXT_COLOR))
    if candidate_info:
        info_lines.append(("Candidates:", font, TEXT_COLOR))
        for move, probability in candidate_info[:5]:
            info_lines.append((f"{move}: {probability:.1%}", small_font, MUTED_TEXT_COLOR))

    line_gap = 12
    info_height = sum(f.get_height() for _, f, _ in info_lines) + line_gap * (len(info_lines) - 1) + 24

    draw_panel_backdrop(screen, pygame.Rect(x, info_top, panel_width, info_height))

    ty = info_top + 12
    for text, f, color in info_lines:
        rendered = f.render(text, True, color)
        screen.blit(rendered, (x + 14, ty))
        ty += f.get_height() + line_gap


# ============================================================
# MOUSE -> CHESS SQUARE
# ============================================================

def mouse_to_square(mouse_x, mouse_y):
    if not (
        BOARD_X <= mouse_x < BOARD_X + BOARD_SIZE
        and BOARD_Y <= mouse_y < BOARD_Y + BOARD_SIZE
    ):
        return None

    col = (mouse_x - BOARD_X) // SQUARE_SIZE
    row = (mouse_y - BOARD_Y) // SQUARE_SIZE

    file = col
    rank = 7 - row

    return chess.square(file, rank)


def draw_legal_moves(screen, board, selected_square):
    if selected_square is None:
        return

    for move in board.legal_moves:
        if move.from_square != selected_square:
            continue

        file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.to_square)

        col = file
        row = 7 - rank

        x = BOARD_X + col * SQUARE_SIZE + SQUARE_SIZE // 2
        y = BOARD_Y + row * SQUARE_SIZE + SQUARE_SIZE // 2

        pygame.draw.circle(screen, MOVE_DOT_COLOR, (x, y), 10)


def reset_game():
    return chess.Board()


# ============================================================
# MAIN
# ============================================================

def main():
    global WINDOW_WIDTH, WINDOW_HEIGHT
    global BOARD_SIZE, SQUARE_SIZE, BOARD_X, BOARD_Y
    global SIDE_PANEL_X, SIDE_PANEL_WIDTH

    pygame.init()

    # --------------------------------------------------------
    # 1) Fullscreen, using the real desktop resolution
    # --------------------------------------------------------

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    WINDOW_WIDTH, WINDOW_HEIGHT = screen.get_size()

    pygame.display.set_caption("Chess Neural Network")

    # --------------------------------------------------------
    # Recompute layout for the real screen size. Board fills the
    # available height (minus margins) but leaves room for the
    # side panel; keep it a clean multiple of 8 for crisp squares.
    # --------------------------------------------------------

    available_height = WINDOW_HEIGHT - 2 * MARGIN
    available_width_for_board = WINDOW_WIDTH - SIDE_PANEL_MIN_WIDTH - 3 * MARGIN

    BOARD_SIZE = max(320, min(available_height, available_width_for_board))
    BOARD_SIZE -= BOARD_SIZE % 8
    SQUARE_SIZE = BOARD_SIZE // 8

    BOARD_X = MARGIN
    BOARD_Y = (WINDOW_HEIGHT - BOARD_SIZE) // 2

    SIDE_PANEL_X = BOARD_X + BOARD_SIZE + MARGIN
    SIDE_PANEL_WIDTH = WINDOW_WIDTH - SIDE_PANEL_X - MARGIN

    # --------------------------------------------------------
    # Load piece art now that SQUARE_SIZE is known
    # --------------------------------------------------------

    for symbol, filename in PIECE_NAMES.items():
        image = pygame.image.load(
            os.path.join(PIECE_IMAGE_FOLDER, f"{filename}.png")
        ).convert_alpha()

        image = pygame.transform.smoothscale(image, (SQUARE_SIZE, SQUARE_SIZE))
        PIECE_IMAGES[symbol] = image

    # --------------------------------------------------------
    # 1) Static background, sized to the full screen
    # --------------------------------------------------------

    background_image = load_static_image(
        BACKGROUND_IMAGE_PATH,
        size=(WINDOW_WIDTH, WINDOW_HEIGHT),
        use_alpha=False,
    )

    # --------------------------------------------------------
    # 2) Character faces (neutral / mad), sized for the panel
    # --------------------------------------------------------

    char_size = min(SIDE_PANEL_WIDTH, 280)

    mood_images = {
        "neutral": load_static_image(NEUTRAL_IMAGE_PATH, size=(char_size, char_size)),
        "mad": load_static_image(MAD_IMAGE_PATH, size=(char_size, char_size)),
    }

    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None
    arrow_start = None
    arrow_end = None
    engine_source = "Waiting"
    running = True

    player_color = chess.WHITE

    board_image = load_random_board()

    # 4) Only regenerate the board when the source flips from the
    # opening book to the neural network -- never on every move.
    previous_source = None

    # 2) Current mood driving which animation plays.
    current_mood = "neutral"

    # 3) Speech bubble only updates every random(4, 7) AI moves.
    displayed_speech = "may the better saif win"
    moves_since_last_speech = 0
    speech_threshold = random.randint(4, 7)

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Fullscreen has no window chrome -- give the user a
                # reliable way out.
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                arrow_start = mouse_to_square(*event.pos)
                arrow_end = arrow_start

            elif event.type == pygame.MOUSEMOTION and arrow_start is not None:
                arrow_end = mouse_to_square(*event.pos)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                arrow_start = None
                arrow_end = None
                if board.is_game_over():
                    continue

                if board.turn != player_color:
                    continue

                mouse_x, mouse_y = event.pos
                clicked_square = mouse_to_square(mouse_x, mouse_y)

                if clicked_square is None:
                    continue

                if selected_square is None:
                    piece = board.piece_at(clicked_square)

                    if piece is not None and piece.color == player_color:
                        selected_square = clicked_square

                else:
                    move = chess.Move(selected_square, clicked_square)

                    if (
                        chess.square_rank(clicked_square) in (0, 7)
                        and board.piece_at(selected_square)
                        and board.piece_at(selected_square).piece_type == chess.PAWN
                    ):
                        move = chess.Move(
                            selected_square, clicked_square, promotion=chess.QUEEN
                        )

                    if move in board.legal_moves:
                        print()
                        print("You played:", board.san(move))

                        board.push(move)
                        selected_square = None
                        engine_source = "Thinking..."

                        if board.is_game_over():
                            print()
                            print("Game over:", board.result())
                            engine_source = "Game Over"

                        else:
                            ai_move, source, text, mood = choose_move(board)

                            if ai_move is not None:
                                print()
                                print("AI played:", board.san(ai_move))

                                board.push(ai_move)
                                engine_source = source
                                current_mood = mood

                                # 4) Board only changes on the opening
                                # book -> neural network transition.
                                if previous_source == "Opening Book" and source == "Neural Network":
                                    board_image = load_random_board()

                                previous_source = source

                                # 3) Speech bubble text refreshes only
                                # every speech_threshold AI moves.
                                moves_since_last_speech += 1

                                if moves_since_last_speech >= speech_threshold:
                                    displayed_speech = text
                                    moves_since_last_speech = 0
                                    speech_threshold = random.randint(4, 7)

                            else:
                                print("AI could not find a move.")
                                engine_source = "ERROR"

                    else:
                        piece = board.piece_at(clicked_square)

                        if piece is not None and piece.color == player_color:
                            selected_square = clicked_square
                        else:
                            selected_square = None

        # ------------------------------------------------------
        # Draw
        # ------------------------------------------------------

        if background_image is not None:
            screen.blit(background_image, (0, 0))
        else:
            screen.fill(BACKGROUND)

        board_shadow = pygame.Rect(BOARD_X - 6, BOARD_Y - 6, BOARD_SIZE + 12, BOARD_SIZE + 12)
        shadow_surface = pygame.Surface(board_shadow.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 120), shadow_surface.get_rect(), border_radius=10)
        screen.blit(shadow_surface, board_shadow.topleft)

        screen.blit(board_image, (BOARD_X, BOARD_Y))
        draw_board(screen, board, selected_square)
        draw_legal_moves(screen, board, selected_square)
        draw_arrow(screen, arrow_start, arrow_end)
        draw_side_panel(
            screen,
            board,
            engine_source,
            selected_square,
            mood_images[current_mood],
            displayed_speech,
        )

        if board.is_game_over():
            font = pygame.font.SysFont("Arial", 30, bold=True)
            text = font.render(f"Game Over: {board.result()}", True, (255, 200, 80))
            screen.blit(text, (SIDE_PANEL_X, WINDOW_HEIGHT - 60))

        pygame.display.flip()

    if engine is not None:
        engine.quit()

    pygame.quit()


if __name__ == "__main__":
    main()